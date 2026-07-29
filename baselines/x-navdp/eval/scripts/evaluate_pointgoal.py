"""
NavDP Evaluation Script for Point-Goal Navigation.

This script runs evaluation of the NavDP diffusion-based navigation policy
using Isaac Lab as the simulation environment.
"""

import sys
import os
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

import cv2
import numpy as np
import imageio
import torch
import argparse
import time
import csv
import threading
import traceback

from eval.src import navigator_reset, navigator_shutdown, pointgoal_step
from eval.config_utils import load_default_config
from eval.environment import create_environment, BatchMPCNEWController, namespace_to_dict


input_obs = None
input_lock = threading.Lock()
output_action = None
output_action_version = 0
output_lock = threading.Lock()
stop_event = threading.Event()
simulation_app = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate; defaults to all samples in the scene.",
    )
    parser.add_argument(
        "--scene_index",
        type=int,
        default=None,
        help="Evaluation scene index; defaults to environment.scene_index in the YAML config.",
    )
    parser.add_argument("--server_port", type=int, default=19999)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Stop cleanly after this many simulation steps (useful for smoke tests).",
    )
    args = parser.parse_args()
    return args


def parse_observations(observation, obs_mapping, return_tensor=False):
    """Parse observations according to the mapping configuration."""
    return_observations = {}
    if hasattr(obs_mapping, "__dict__"):
        obs_mapping = vars(obs_mapping)
    for key, value in obs_mapping.items():
        if value not in observation:
            continue
        if return_tensor:
            return_observations[key] = observation[value]
        else:
            return_observations[key] = observation[value].cpu().numpy()
    return return_observations


def add_robot_state(observation, env):
    """Attach world-frame robot pose used by temporal guidance and stuck detection."""
    observation = dict(observation)
    robot = env.unwrapped.scene["robot"]
    observation["robot_pose"] = robot.data.root_pos_w
    # Isaac Lab uses wxyz quaternions; SciPy guidance code expects xyzw.
    observation["robot_rot"] = robot.data.root_quat_w[:, [1, 2, 3, 0]]
    return observation


def write_metrics(metrics, path="exploration.csv"):
    """Write evaluation metrics to CSV file."""
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def planning_thread(server_port, mpc_controller):
    """Thread for running navigation planning asynchronously."""
    global input_obs, output_action, output_action_version
    while not stop_event.is_set():
        obs_to_process = None
        with input_lock:
            if input_obs is not None:
                obs_to_process = input_obs
                input_obs = None

        if obs_to_process is not None:
            try:
                output_trajectory, _, _ = pointgoal_step(**obs_to_process, port=server_port)
                output_trajectory = np.concatenate((np.zeros_like(output_trajectory[:, 0:1]), output_trajectory), axis=1)
                mpc_controls, _, _, _ = mpc_controller.solve(output_trajectory)
                with output_lock:
                    output_action = mpc_controls
                    output_action_version += 1
            except Exception:
                print("[planning_thread] policy/MPC step failed:")
                print(traceback.format_exc(), flush=True)
                with output_lock:
                    output_action = None

        time.sleep(0.01)


def main():
    global input_obs, input_lock, output_action, output_lock, simulation_app

    args = get_args()
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=True,
        enable_cameras=True,
        distributed=True
    )
    simulation_app = app_launcher.app

    device = args.device
    cfg = load_default_config(args.config_file)
    scene_dir = cfg.environment.scene_dir
    scene_index = (
        args.scene_index
        if args.scene_index is not None
        else getattr(cfg.environment, "scene_index", 0)
    )
    mpc_controller = BatchMPCNEWController(batch=cfg.environment.num_envs, **namespace_to_dict(cfg.mpc))

    planning_thread_obj = threading.Thread(target=planning_thread, args=(args.server_port, mpc_controller,))
    planning_thread_obj.daemon = True
    planning_thread_obj.start()

    env, controller, house_id = create_environment(cfg, scene_index=scene_index, device=device)

    reset_outputs = env.reset()
    if isinstance(reset_outputs, tuple) and len(reset_outputs) == 2:
        raw_obs, infos = reset_outputs
        obs = infos.get("observations", raw_obs)
    else:
        obs, infos = reset_outputs, {}
    obs = add_robot_state(obs, env)

    camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]
    sample_indices = [env.unwrapped._sample_idx[i].item() for i in range(env.num_envs)]
    server_algo = navigator_reset(
        camera_intrinsic.cpu().numpy(),
        batch_size=env.num_envs,
        port=args.server_port,
        sample_indices=sample_indices,
        scene_name=house_id
    )

    scene_split = scene_dir.split('/')[-1]
    save_prefix = cfg.run_root_dir
    embodiment_name = {
        "dingo": "wheeled",
        "unitree_g1": "humanoid",
        "unitree_go2": "quadruped",
    }.get(getattr(cfg.environment, "embodiment", ""), getattr(cfg.environment, "embodiment", "unknown"))
    scene_kind = getattr(cfg.environment, "scene_type", scene_split)
    output_group = f"{embodiment_name}_{scene_kind}"
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = os.path.join(save_prefix, output_group, server_algo + time_stamp, scene_split, house_id)
    os.makedirs(save_dir, exist_ok=True)

    done_sample_indices = set()
    total_episode_count = env.unwrapped._total_episode_count
    if args.num_episodes is not None:
        total_episode_count = min(args.num_episodes, total_episode_count)

    fps_writers = []
    current_episode_idx = []
    for i in range(env.num_envs):
        sample_idx = env.unwrapped._sample_idx[i].item()
        if sample_idx not in done_sample_indices:
            fps_writers.append(imageio.get_writer(save_dir + f"/fps_{sample_idx}.mp4", fps=20))
            current_episode_idx.append(sample_idx)
        else:
            print(f"Warning: sample_idx {sample_idx} already done, skipping env {i}")
            fps_writers.append(None)
            current_episode_idx.append(None)

    with input_lock:
        input_obs = parse_observations(obs, cfg.obs_mapping)
    euclidean = np.sqrt(np.square(obs['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1))
    trajectory_length = np.zeros((env.num_envs))
    evaluation_metrics = []

    output_action_version_last = 0
    step_count = 0

    while simulation_app.is_running():
        with output_lock:
            if output_action_version > output_action_version_last:
                output_action_version_last = output_action_version
                print("action update")
            if output_action is not None and output_action.shape[1] != 0:
                first_action = output_action[:, 0].copy()
                output_action = output_action[:, 1:]
            else:
                first_action = None

        if first_action is not None:
            robot_action = controller.forward_batch(obs['policy'], first_action)
        else:
            robot_action = controller.forward_batch(obs['policy'], np.zeros((env.num_envs, 2)))

        planar_speed = torch.linalg.vector_norm(obs['policy'][:, :2], dim=-1)
        trajectory_length += (planar_speed * env.unwrapped.step_dt).cpu().numpy()

        step_outputs = env.step(robot_action)
        if len(step_outputs) == 5:
            obs, rewards, terminated, truncated, infos = step_outputs
            dones = terminated | truncated
        else:
            raw_obs, rewards, dones, infos = step_outputs
            obs = infos.get("observations", raw_obs)
        obs = add_robot_state(obs, env)
        step_count += 1

        for i in range(env.num_envs):
            if fps_writers[i] is None:
                continue
            resize_raw_image = cv2.resize(obs['raw_rgb'][i].detach().cpu().numpy(), (384, 384))
            if 'birdeye_rgb' in obs:
                bev_image = obs['birdeye_rgb'][i].detach().cpu().numpy()
            else:
                bev_image = np.zeros_like(obs['raw_rgb'][i].detach().cpu().numpy())
            resize_bev_image = cv2.resize(bev_image, (384, 384))
            fps_writers[i].append_data(np.concatenate([resize_raw_image, resize_bev_image], axis=1))

        for i in range(env.num_envs):
            if dones[i] == True:
                current_sample_idx = current_episode_idx[i]
                new_sample_idx = env.unwrapped._sample_idx[i].item()
                with input_lock:
                    input_obs = None
                with output_lock:
                    output_action = None
                navigator_reset(env_id=i, port=args.server_port, sample_idx=new_sample_idx)

                if fps_writers[i] is not None and current_sample_idx is not None:
                    success_flag = (1 - float(infos["time_outs"][i].item()))
                    fps_writers[i].close()

                    if current_sample_idx not in done_sample_indices:
                        evaluation_metrics.append({
                            'success': success_flag,
                            'spl': (
                                euclidean[i] / max(trajectory_length[i], euclidean[i], 1e-8)
                            ) * success_flag,
                            'distance': euclidean[i],
                            'episode_idx': current_sample_idx
                        })
                        done_sample_indices.add(current_sample_idx)
                        write_metrics(evaluation_metrics, save_dir + "/metric.csv")
                    else:
                        print(f"Warning: sample_idx {current_sample_idx} already done, skipping metrics")

                if len(done_sample_indices) >= total_episode_count:
                    print(f"All {total_episode_count} episodes completed!")
                    break

                if new_sample_idx not in done_sample_indices:
                    fps_writers[i] = imageio.get_writer(save_dir + f"/fps_{new_sample_idx}.mp4", fps=20)
                    current_episode_idx[i] = new_sample_idx
                else:
                    print(f"Warning: new sample_idx {new_sample_idx} already done, skipping env {i}")
                    fps_writers[i] = None
                    current_episode_idx[i] = None

                euclidean[i] = np.sqrt(np.square(obs['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1))[i]
                trajectory_length[i] = 0.0

        if len(done_sample_indices) >= total_episode_count:
            print(f"All {total_episode_count} episodes completed!")
            break
        if args.max_steps is not None and step_count >= args.max_steps:
            print(f"Reached --max_steps={args.max_steps}; stopping evaluation cleanly")
            break

        with input_lock:
            input_obs = parse_observations(obs, cfg.obs_mapping)

    stop_event.set()
    planning_thread_obj.join(timeout=10.0)
    navigator_shutdown(port=args.server_port)
    for w in fps_writers:
        if w is not None:
            try:
                w.close()
            except (RuntimeError, ValueError, OSError):
                pass
    try:
        env.close()
    except (RuntimeError, AttributeError, OSError):
        pass
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
