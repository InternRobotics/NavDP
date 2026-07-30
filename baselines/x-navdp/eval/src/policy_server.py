"""
X-NavDP policy server for navigation policy inference.

This module provides a server that accepts RGB-D images and goals via HTTP POST,
runs the X-NavDP policy to generate navigation trajectories, and returns the results.
"""

from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
import cv2
import imageio
import time
import json
import os
import threading
from functools import wraps

try:
    from .policy_agent import NavDP_Agent
except ImportError:
    from policy_agent import NavDP_Agent

app = Flask(__name__)
navdp_navigator = None
navdp_fps_writer = None
navdp_fps_writer_lock = threading.Lock()
navdp_state_lock = threading.RLock()
embodiment_idx = 2  # Default: unitree_go2 (0=wheeled, 1=humanoid, 2=quadruped)
vis_output_dir = None
enable_visualization = True
policy_device = "cuda:0"
policy_checkpoint = None
policy_embodiment = "quadruped"
policy_is_real = False


def synchronized_navdp_route(func):
    """Serialize policy server access to mutable navigator state."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with navdp_state_lock:
            return func(*args, **kwargs)
    return wrapper


def init_app(
    embodiment,
    no_visualization=False,
    device="cuda:0",
    checkpoint=None,
    real=False,
):
    """Initialize the global app configuration."""
    global embodiment_idx, enable_visualization, policy_device
    global policy_checkpoint, policy_embodiment, policy_is_real

    EMBODIMENT_NAME_TO_IDX = {
        "quadruped": 2,
        "humanoid": 1,
        "wheeled": 0,
    }
    if embodiment not in EMBODIMENT_NAME_TO_IDX:
        raise ValueError(
            f"embodiment must be one of {list(EMBODIMENT_NAME_TO_IDX)}, got {embodiment!r}"
        )
    embodiment_idx = EMBODIMENT_NAME_TO_IDX[embodiment]
    enable_visualization = not no_visualization
    policy_device = device
    policy_checkpoint = checkpoint
    policy_embodiment = embodiment
    policy_is_real = real

    return app


@app.route("/navigator_reset", methods=['POST'])
@synchronized_navdp_route
def navdp_reset():
    """Reset the navigator with initial camera intrinsics and batch size."""
    global navdp_navigator, navdp_fps_writer, vis_output_dir

    intrinsic = np.array(request.get_json().get('intrinsic'))
    batchsize = np.array(request.get_json().get('batch_size'))
    sample_indices = request.get_json().get('sample_indices', None)
    scene_name = request.get_json().get('scene_name', None)

    vis_output_dir = os.path.join(
        "./vis_output", policy_embodiment,
        scene_name if scene_name else "no_scene"
    )
    os.makedirs(vis_output_dir, exist_ok=True)

    if navdp_navigator is None:
        if not policy_checkpoint:
            raise RuntimeError("Initialize the policy server with a checkpoint before reset.")
        navdp_navigator = NavDP_Agent(
            intrinsic,
            image_size=224,
            memory_size=8,
            predict_size=24,
            temporal_depth=16,
            heads=8,
            token_dim=384,
            navi_model=policy_checkpoint,
            device=policy_device,
            embodiment=embodiment_idx,
            is_real=policy_is_real,
        )
        navdp_navigator.reset(batchsize)
    else:
        navdp_navigator.reset(batchsize)

    navdp_navigator.sample_idx_list = sample_indices if sample_indices is not None else list(range(batchsize))
    navdp_navigator.current_scene_name = scene_name
    navdp_navigator._occ_cache = {}

    if enable_visualization:
        with navdp_fps_writer_lock:
            if navdp_fps_writer is None:
                navdp_fps_writer = []
                os.makedirs(vis_output_dir, exist_ok=True)
                for i in range(batchsize):
                    if sample_indices is not None and i < len(sample_indices):
                        sample_idx = sample_indices[i]
                        navdp_fps_writer.append(imageio.get_writer(
                            os.path.join(vis_output_dir, f"fps_{sample_idx}.mp4"), fps=7))
                    else:
                        navdp_fps_writer.append(imageio.get_writer(
                            os.path.join(vis_output_dir, f"fps_env{i}.mp4"), fps=7))
            else:
                for writer in navdp_fps_writer:
                    if writer is not None:
                        writer.close()
                navdp_fps_writer = []
                os.makedirs(vis_output_dir, exist_ok=True)
                for i in range(batchsize):
                    if sample_indices is not None and i < len(sample_indices):
                        sample_idx = sample_indices[i]
                        navdp_fps_writer.append(imageio.get_writer(
                            os.path.join(vis_output_dir, f"fps_{sample_idx}.mp4"), fps=7))
                    else:
                        navdp_fps_writer.append(imageio.get_writer(
                            os.path.join(vis_output_dir, f"fps_env{i}.mp4"), fps=7))
    else:
        with navdp_fps_writer_lock:
            if navdp_fps_writer is not None:
                for writer in navdp_fps_writer:
                    if writer is not None:
                        writer.close()
                navdp_fps_writer = None

    return jsonify({"algo": "navdp-rl"})


@app.route("/navigator_reset_env", methods=['POST'])
@synchronized_navdp_route
def navdp_reset_env():
    """Reset a specific environment in the navigator."""
    global navdp_navigator, navdp_fps_writer, vis_output_dir

    env_id = int(request.get_json().get('env_id'))
    sample_idx = request.get_json().get('sample_idx', None)
    scene_name = request.get_json().get('scene_name', None)

    navdp_navigator.reset_env(env_id)

    if sample_idx is not None:
        navdp_navigator.sample_idx_list[env_id] = sample_idx
    if scene_name is not None:
        navdp_navigator.current_scene_name = scene_name
        navdp_navigator._occ_cache = {}
        vis_output_dir = os.path.join(
            "./vis_output", policy_embodiment,
            scene_name if scene_name else "no_scene"
        )

    if enable_visualization:
        with navdp_fps_writer_lock:
            if navdp_fps_writer is None:
                navdp_fps_writer = [None] * navdp_navigator.batch_size

            if navdp_fps_writer[env_id] is not None:
                navdp_fps_writer[env_id].close()

            os.makedirs(vis_output_dir, exist_ok=True)
            if sample_idx is not None:
                navdp_fps_writer[env_id] = imageio.get_writer(
                    os.path.join(vis_output_dir, f"fps_{sample_idx}.mp4"), fps=7)
            else:
                navdp_fps_writer[env_id] = imageio.get_writer(
                    os.path.join(vis_output_dir, f"fps_env{env_id}.mp4"), fps=7)

    return jsonify({"algo": "navdp-rl"})


@app.route("/pointgoal_step", methods=['POST'])
@synchronized_navdp_route
def navdp_step_xy():
    """Process a point goal navigation step (with robot state guidance)."""
    global navdp_navigator, navdp_fps_writer

    start_time = time.time()
    image_file = request.files['image']
    depth_file = request.files['depth']

    goal_data = json.loads(request.form.get('goal_data'))
    goal_x = np.array(goal_data['goal_x'])
    goal_y = np.array(goal_data['goal_y'])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
    batch_size = navdp_navigator.batch_size

    state_data = json.loads(request.form.get('state_data', '{}'))
    robot_pos = np.array(state_data['robot_pos']) if 'robot_pos' in state_data else None
    robot_quat = np.array(state_data['robot_quat']) if 'robot_quat' in state_data else None
    # The client sends quaternions in SciPy-compatible xyzw order.

    phase1_time = time.time()
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)[:, :, np.newaxis]
    depth = depth.astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    phase2_time = time.time()
    execute_trajectory, all_trajectory, all_values, trajectory_mask = \
        navdp_navigator.step_pointgoal_with_guidance(goal, image, depth, robot_pos, robot_quat)
    phase3_time = time.time()

    if navdp_fps_writer is not None:
        img_width_per_env = trajectory_mask.shape[1] // batch_size
        with navdp_fps_writer_lock:
            for i in range(batch_size):
                if i < len(navdp_fps_writer) and navdp_fps_writer[i] is not None:
                    try:
                        start_col = i * img_width_per_env
                        end_col = (i + 1) * img_width_per_env
                        env_mask = trajectory_mask[:, start_col:end_col, :]
                        navdp_fps_writer[i].append_data(env_mask)
                    except (RuntimeError, ValueError):
                        pass

    phase4_time = time.time()
    print("phase1:%.3f, phase2:%.3f, phase3:%.3f, phase4:%.3f, all:%.3f" % (
        phase1_time - start_time, phase2_time - phase1_time,
        phase3_time - phase2_time, phase4_time - phase3_time,
        time.time() - start_time))

    return jsonify({
        'trajectory': execute_trajectory.tolist(),
        'all_trajectory': all_trajectory.tolist(),
        'all_values': all_values.tolist()
    })


@app.route("/shutdown", methods=["POST"])
def shutdown_server():
    """Shutdown the server gracefully."""
    _close_fps_writers()
    shutdown = request.environ.get("werkzeug.server.shutdown")

    def _exit_after_delay():
        time.sleep(0.15)
        if shutdown is not None:
            shutdown()
        else:
            os._exit(0)

    threading.Thread(target=_exit_after_delay, daemon=True).start()
    return jsonify({"status": "ok", "message": "server shutting down"})


def _close_fps_writers():
    """Close all FPS writers."""
    global navdp_fps_writer
    with navdp_fps_writer_lock:
        if navdp_fps_writer is not None:
            for writer in navdp_fps_writer:
                if writer is not None:
                    try:
                        writer.close()
                    except (RuntimeError, ValueError, OSError):
                        pass
            navdp_fps_writer = None


def run_server(port=8888, host='127.0.0.1'):
    """Run the Flask server."""
    app.run(host=host, port=port, threaded=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--embodiment",
        type=str,
        required=True,
        choices=("wheeled", "humanoid", "quadruped"),
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--no-visualization", action='store_true')
    args = parser.parse_args()

    init_app(
        args.embodiment,
        args.no_visualization,
        args.device,
        checkpoint=args.checkpoint,
        real=args.real,
    )
    run_server(port=args.port)
