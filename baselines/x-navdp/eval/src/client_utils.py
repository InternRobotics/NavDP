"""
Client utilities for communicating with the NavDP Policy Server.

This module provides functions to send RGB-D images and goals to the server,
receive trajectory predictions, and manage the server lifecycle.
"""

import requests
import numpy as np
import cv2
import io
import json
import time

RESET_TIMEOUT = (5.0, 180.0)
STEP_TIMEOUT = (5.0, 60.0)


def navigator_reset(intrinsic=None, batch_size=1, port=8888,
                    env_id=None, sample_indices=None, sample_idx=None, scene_name=None):
    """Reset the navigator on the server."""
    if env_id is None:
        url = f"http://localhost:{port}/navigator_reset"
        json_data = {
            'intrinsic': intrinsic.tolist() if hasattr(intrinsic, 'tolist') else intrinsic,
            'batch_size': batch_size
        }
        if sample_indices is not None:
            json_data['sample_indices'] = sample_indices
        if scene_name is not None:
            json_data['scene_name'] = scene_name
        response = requests.post(url, json=json_data, timeout=RESET_TIMEOUT)
    else:
        url = f"http://localhost:{port}/navigator_reset_env"
        json_data = {'env_id': env_id}
        if sample_idx is not None:
            json_data['sample_idx'] = sample_idx
        if scene_name is not None:
            json_data['scene_name'] = scene_name
        response = requests.post(url, json=json_data, timeout=RESET_TIMEOUT)
    response.raise_for_status()
    return json.loads(response.text)['algo']


def navigator_shutdown(port=8888, timeout=3.0):
    """Shutdown the policy server gracefully."""
    url = f"http://localhost:{port}/shutdown"
    try:
        requests.post(url, json={}, timeout=timeout)
    except requests.RequestException:
        pass


def pointgoal_step(pointgoal, rgb, depth, port=8888, **kwargs):
    """
    Perform a point-goal navigation step.

    Args:
        pointgoal: Array of shape (batch, 2) or (batch, 3) with (x, y) or (x, y, z) goal coordinates
        rgb: RGB images, can be (batch, H, W, C) or (batch, T, H, W, C)
        depth: Depth images, same shape as rgb
        port: Server port
        **kwargs: Optional robot_pos and robot_quat for state-guided navigation

    Returns:
        trajectory: Selected trajectory
        all_trajectory: All sampled trajectories
        all_value: Q-values for all trajectories
    """
    if rgb.ndim == 5:
        concat_images = rgb.reshape(-1, rgb.shape[3], rgb.shape[4])
    else:
        concat_images = np.concatenate([img for img in rgb], axis=0)

    if depth.ndim == 5:
        concat_depths = depth.reshape(-1, depth.shape[3], depth.shape[4]).squeeze(-1)
        if concat_depths.ndim == 2:
            concat_depths = concat_depths[..., np.newaxis]
    else:
        concat_depths = np.concatenate([img for img in depth], axis=0)

    url = f"http://localhost:{port}/pointgoal_step"

    _, rgb_image = cv2.imencode('.jpg', concat_images)
    image_bytes = io.BytesIO()
    image_bytes.write(rgb_image)

    depth_image = np.clip(concat_depths * 10000.0, 0, 65535.0).astype(np.uint16)
    _, depth_image = cv2.imencode('.png', depth_image)
    depth_bytes = io.BytesIO()
    depth_bytes.write(depth_image)

    files = {
        'image': ('image.jpg', image_bytes.getvalue(), 'image/jpeg'),
        'depth': ('depth.png', depth_bytes.getvalue(), 'image/png'),
    }

    goal_data_dict = {
        'goal_x': pointgoal[:, 0].tolist(),
        'goal_y': pointgoal[:, 1].tolist()
    }
    state_data_dict = {}
    if 'robot_pos' in kwargs:
        state_data_dict['robot_pos'] = kwargs['robot_pos'].tolist() if hasattr(kwargs['robot_pos'], 'tolist') else kwargs['robot_pos']
    if 'robot_quat' in kwargs:
        state_data_dict['robot_quat'] = kwargs['robot_quat'].tolist() if hasattr(kwargs['robot_quat'], 'tolist') else kwargs['robot_quat']

    data = {
        'goal_data': json.dumps(goal_data_dict),
        'depth_time': time.time(),
        'rgb_time': time.time(),
    }
    if state_data_dict:
        data['state_data'] = json.dumps(state_data_dict)

    response = requests.post(url, files=files, data=data, timeout=STEP_TIMEOUT)
    response.raise_for_status()
    result = json.loads(response.text)
    return np.array(result['trajectory']), np.array(result['all_trajectory']), np.array(result['all_values'])
