"""Isaac Lab worker process utilities for distributed X-NavDP training."""

import os
import sys

import numpy as np

from src.training.scene_assets import is_clutter_scene


def configure_mdl_system_path(scene_dir: str) -> None:
    """Configure Isaac Sim MDL material search paths before AppLauncher starts."""
    if os.environ.get("MDL_SYSTEM_PATH"):
        return

    parts = []
    isaac_materials = os.environ.get("ISAAC_SIM_MATERIALS", "/isaac-sim/materials/")
    if isaac_materials:
        parts.append(isaac_materials.rstrip("/"))

    if scene_dir:
        for rel in (
            os.path.join("internscenes_home", "scenes_home"),
            os.path.join("internscenes_commercial", "scenes_commercial"),
            os.path.join("internscenes_home", "scenes_home_train"),
            os.path.join("internscenes_commercial", "scenes_commercial_train"),
            "scenes",
            "Materials",
        ):
            candidate = os.path.join(scene_dir, rel)
            if os.path.isdir(candidate):
                parts.append(candidate)

    extra = os.environ.get(
        "X_NAVDP_MDL_EXTRA_PATHS",
        os.environ.get("NAVRL_MDL_EXTRA_PATHS", ""),
    )
    if extra:
        parts.extend(p for p in extra.split(":") if p)

    if parts:
        os.environ["MDL_SYSTEM_PATH"] = ":".join(parts)


def wait_worker_result(result_queue, timeout=None, rank=None):
    """Read a subprocess result and convert queue timeouts into actionable errors."""
    import queue

    if timeout is None:
        return result_queue.get()
    try:
        return result_queue.get(timeout=timeout)
    except queue.Empty:
        raise RuntimeError(f"Worker did not respond within timeout (rank={rank})")


def extract_env_metadata(env, num_envs: int) -> dict:
    """Collect lightweight environment state needed by rewards, logging, and videos."""
    unwrapped = env.unwrapped
    robot_pos = unwrapped.scene["robot"].data.root_pos_w[:, :2].cpu().numpy()
    robot_quat = unwrapped.scene["robot"].data.root_quat_w.cpu().numpy()

    grid_matrix = unwrapped._grid_matrix
    occ_min_bound = getattr(unwrapped, "_occ_min_bound", np.zeros(3))
    grid_size = getattr(unwrapped, "_grid_size", 0.1)

    episode_length_buf = env.episode_length_buf.cpu().numpy()
    is_eval = np.array([bool(x) for x in unwrapped._is_eval], dtype=bool)

    return {
        "robot_pos": robot_pos,
        "robot_quat": robot_quat,
        "grid_matrix": grid_matrix,
        "occ_min_bound": occ_min_bound,
        "grid_size": grid_size,
        "episode_length_buf": episode_length_buf,
        "_is_eval": is_eval,
        "_eval_interval": getattr(unwrapped, "_eval_interval", 4),
        "_episodes_num_list": getattr(unwrapped, "_episodes_num_list", None),
    }


def tensor_dict_to_numpy(d):
    """Detach tensors from the Isaac worker before sending them through multiprocessing."""
    out = {}
    for k, v in d.items():
        if hasattr(v, "cpu"):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.array(v)
    return out


def run_env_worker(
    cmd_queue,
    result_queue,
    scene_list,
    scene_index,
    num_envs,
    scene_scale,
    device,
    stage,
    embodiment,
):
    """Run Isaac Lab in a subprocess and service reset/step/reward commands."""
    simulation_app = None
    env = None
    try:
        from isaaclab.app import AppLauncher

        ujitso_debug = os.environ.get(
            "X_NAVDP_UJITSO_DEBUG",
            os.environ.get("NAVRL_UJITSO_DEBUG", "0"),
        )
        if ujitso_debug == "1":
            ujitso_cache_path = os.environ.get(
                "X_NAVDP_UJITSO_CACHE_PATH",
                os.environ.get(
                    "NAVRL_UJITSO_CACHE_PATH",
                    os.path.expanduser("~/.cache/ov/DerivedDataCache"),
                ),
            )
            ujitso_args = [
                "--/UJITSO/enabled=true",
                "--/UJITSO/materials=true",
                "--/UJITSO/textures=true",
                f"--/UJITSO/datastore/localCachePath={ujitso_cache_path}",
            ]
            ujitso_log = os.environ.get(
                "X_NAVDP_UJITSO_LOG",
                os.environ.get("NAVRL_UJITSO_LOG", "1"),
            )
            if ujitso_log != "0":
                ujitso_args.extend([
                    "--/UJITSO/failedDepLoadingLogging=true",
                    "--/UJITSO/logBuildResults=true",
                ])
            sys.argv.extend(ujitso_args)
            print(f"[Rank {int(os.environ.get('RANK', '0'))}] UJITSO enabled, cache={ujitso_cache_path}", flush=True)

        app_launcher = AppLauncher(headless=True, enable_cameras=True, distributed=True)
        simulation_app = app_launcher.app

        print(
            f"[Rank {int(os.environ.get('RANK', '0'))}] AppLauncher initialized, "
            f"Stage={stage}, embodiment={embodiment}",
            flush=True,
        )
        import torch
        from isaaclab.managers import SceneEntityCfg

        from src.environment import create_dingonav_environment
        from src.environment.tasks.curriculum_utils import update_curriculum_success
        from src.environment.tasks.reward_utils import (
            relative_distance_reward_dir,
            relative_distance_reward_dir_clutter,
        )

        clutter_scene = is_clutter_scene(scene_list, scene_index)
        path_reward_fn = relative_distance_reward_dir_clutter if clutter_scene else relative_distance_reward_dir
        print(
            f"[Rank {int(os.environ.get('RANK', '0'))}] path_reward_fn="
            f"{'clutter_euclidean' if clutter_scene else 'astar'}",
            flush=True,
        )

        def create_env_for_scene(idx: int, stage: int, embodiment: str):
            scene_scale_value = scene_list[idx].get("scene_scale", scene_scale)
            return create_dingonav_environment(
                scene_list=scene_list,
                scene_index=idx,
                num_envs=num_envs,
                scene_scale=scene_scale_value,
                device=device,
                embodiment=embodiment,
            )

        print(
            f"[Rank {int(os.environ.get('RANK', '0'))}] Creating env scene_index={scene_index}, "
            f"scene_name={scene_list[scene_index].get('scene_name')}, embodiment={embodiment}",
            flush=True,
        )
        env, controller = create_env_for_scene(scene_index, stage, embodiment)
        print(f"[Rank {int(os.environ.get('RANK', '0'))}] Env created", flush=True)

        while True:
            cmd = cmd_queue.get()

            if cmd[0] == "close":
                break

            if cmd[0] == "path_reward":
                try:
                    env_indices = list(cmd[1])
                    reward = path_reward_fn(env.unwrapped, SceneEntityCfg("robot"), env_indices)
                    result_queue.put(
                        {
                            "type": "path_reward",
                            "reward": reward.detach().cpu().numpy().astype(np.float32),
                        }
                    )
                except Exception as e:
                    result_queue.put({"type": "error", "error": f"path_reward: {e!r}"})

            elif cmd[0] == "step":
                actions = np.asarray(cmd[1], dtype=np.float32)
                obs_policy = cmd[2] if len(cmd) > 2 else None
                control_tensor = controller.forward_batch(obs_policy, actions)
                step_outputs = env.step(control_tensor)
                if len(step_outputs) == 5:
                    obs_raw, rewards, terminated, truncated, infos = step_outputs
                    dones = terminated | truncated
                else:
                    obs_raw, rewards, dones, infos = step_outputs
                obs_np = tensor_dict_to_numpy({
                    "obs_rgb": obs_raw["obs_rgb"],
                    "obs_depth": obs_raw["obs_depth"],
                    "goal_pose": obs_raw["goal_pose"],
                    "raw_rgb": obs_raw["raw_rgb"],
                    "policy": obs_raw["policy"],
                    **({"birdeye_rgb": obs_raw["birdeye_rgb"]} if "birdeye_rgb" in obs_raw else {}),
                })
                metadata = extract_env_metadata(env, num_envs)
                result_queue.put({
                    "type": "step",
                    "obs_raw": obs_np,
                    "rewards": rewards.cpu().numpy(),
                    "dones": dones.cpu().numpy(),
                    "infos": {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in infos.items()},
                    "metadata": metadata,
                })

            elif cmd[0] == "reset":
                obs_raw, infos = env.reset()
                unwrapped = env.unwrapped
                for attr in (
                    "_prev_distance_to_goal",
                    "_prev_pose",
                    "_prev_euclidean_distance_to_goal",
                    "_prev_euclidean_pose",
                ):
                    if hasattr(unwrapped, attr):
                        delattr(unwrapped, attr)
                metadata = extract_env_metadata(env, num_envs)
                obs_np = tensor_dict_to_numpy({
                    "obs_rgb": obs_raw["obs_rgb"],
                    "obs_depth": obs_raw["obs_depth"],
                    "goal_pose": obs_raw["goal_pose"],
                    "raw_rgb": obs_raw["raw_rgb"],
                    "policy": obs_raw["policy"],
                    **({"birdeye_rgb": obs_raw["birdeye_rgb"]} if "birdeye_rgb" in obs_raw else {}),
                })
                result_queue.put({
                    "type": "reset",
                    "obs_raw": obs_np,
                    "infos": infos,
                    "metadata": metadata,
                })

            elif cmd[0] == "episode_end":
                env_idx, episodes_num = cmd[1], cmd[2]
                unwrapped = env.unwrapped
                success = unwrapped._is_last_success[env_idx]
                was_eval = bool(unwrapped._is_eval[env_idx])
                unwrapped._episodes_num_list[env_idx] += 1
                unwrapped._is_eval[env_idx] = unwrapped._episodes_num_list[env_idx] % unwrapped._eval_interval == 0
                total_success_rate, stage_success_rate = update_curriculum_success(
                    unwrapped, bool(success), episodes_num, record_success=was_eval
                )
                result_queue.put({
                    "type": "episode_end",
                    "success": success,
                    "total_success_rate": total_success_rate,
                    "stage_success_rate": stage_success_rate,
                })

        env.close()
        simulation_app.close()
        result_queue.put({"type": "closed"})
    except Exception:
        import traceback

        result_queue.put({"type": "error", "error": traceback.format_exc()})
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass
