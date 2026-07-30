"""Reward terms for point-goal progress, arrival, and collision behavior."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaacsim.core.prims import XFormPrim
import isaaclab.envs.mdp as mdp
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R
from .curriculum_utils import (
    ENABLE_CURRICULUM_LEARNING,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_VELOCITY_THRESHOLD,
    scaled_arrival_stop_delay_steps,
)
from .event_utils import path_finder

DEFAULT_PATH_PROGRESS_THRESHOLD = 4.0
STATIC_TURN_THRESHOLD_RAD1 = float(np.deg2rad(20.0))
STATIC_TURN_THRESHOLD_RAD2 = float(np.deg2rad(60.0))


def _angle_delta_rad(current: float, previous: float) -> float:
    """Return wrapped absolute yaw change in radians."""
    return abs((current - previous + np.pi) % (2.0 * np.pi) - np.pi)


def _progress_reward_with_static_turn(
    distance_change: torch.Tensor,
    is_same_dir: float,
    turn_delta: float,
    progress_threshold: float,
) -> torch.Tensor:
    """Reward forward progress, penalize backward motion and idle non-turning plans."""
    if distance_change > 2:
        env_reward = distance_change * is_same_dir
        return torch.clamp(env_reward, 0.0, 15.0)

    if distance_change < -1:
        if turn_delta > STATIC_TURN_THRESHOLD_RAD2:
            return torch.zeros_like(distance_change)
        penalized_change = distance_change - progress_threshold
        env_reward = penalized_change * 1.5
        return torch.clamp(env_reward, -15.0, 0.0)

    if turn_delta > STATIC_TURN_THRESHOLD_RAD1:
        return torch.zeros_like(distance_change)

    penalized_change = distance_change - progress_threshold
    env_reward = penalized_change * 1.5
    return torch.clamp(env_reward, -15.0, 0.0)

""" Reward based on relative distance change to goal (progress towards goal)."""
def relative_distance_reward_dir(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg,
    env_indices: list,
    weight: float = 0.075,
    progress_threshold: float = DEFAULT_PATH_PROGRESS_THRESHOLD,
) -> torch.Tensor:
    """Reward selected envs for A* progress aligned with robot heading."""

    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w[:, :2]
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0][:, :2]
    device = robot_pos.device
    num_envs = env.num_envs
    reward = torch.zeros(num_envs, dtype=torch.float32, device=device)

    if not hasattr(env, '_prev_distance_to_goal'):
        env._prev_distance_to_goal = torch.full((num_envs,), -1.0, dtype=torch.float32, device=device)
        env._prev_pose = robot_pos.clone()

    robot_quat = robot_asset.data.root_quat_w.cpu().numpy()
    scipy_quats = robot_quat[:, [1, 2, 3, 0]]
    dir_2d = None
    try:
        rot = R.from_quat(scipy_quats)
        rot_matrix = rot.as_matrix()
        dir_3d = rot_matrix @ np.array([1, 0, 0])
        dir_2d = dir_3d[:, :2]
    except (ValueError, TypeError) as e:
        print(f"[relative_distance_reward_dir] Quat conversion failed: {e}")
    heading_yaw = np.arctan2(dir_2d[:, 1], dir_2d[:, 0]) if dir_2d is not None else None
    if heading_yaw is not None and not hasattr(env, '_prev_heading_yaw'):
        env._prev_heading_yaw = heading_yaw.copy()

    grid = env._grid
    occ_min_bound, grid_size = env._occ_min_bound, env._grid_size

    for i in env_indices:
        start_index = ((robot_pos[i].cpu().numpy() - occ_min_bound[0:2]) / grid_size).astype(np.int32)
        end_index = ((goal_pos[i].cpu().numpy() - occ_min_bound[0:2]) / grid_size).astype(np.int32)
        start_oob = (start_index[0] < 0 or start_index[0] > grid.height - 1 or
                     start_index[1] < 0 or start_index[1] > grid.width - 1)
        end_oob = (end_index[0] < 0 or end_index[0] > grid.height - 1 or
                   end_index[1] < 0 or end_index[1] > grid.width - 1)
        if start_oob or end_oob:
            is_out_of_bounds = True
            current_distance = env._prev_distance_to_goal[i]
        else:
            is_out_of_bounds = False
            start = grid.node(start_index[1], start_index[0])
            goal = grid.node(end_index[1], end_index[0])
            dist, _ = path_finder.find_path(start, goal, grid)
            try:
                assert len(dist) > 0
                current_distance = torch.tensor(float(len(dist)), dtype=torch.float32, device=device)
            except Exception:
                current_distance = env._prev_distance_to_goal[i]

        if env._prev_distance_to_goal[i] < 0:
            env._prev_distance_to_goal[i] = current_distance
            env._prev_pose[i] = robot_pos[i]
            if heading_yaw is not None:
                env._prev_heading_yaw[i] = heading_yaw[i]
            continue

        if dir_2d is not None:
            robot_pos_delta = robot_pos[i] - env._prev_pose[i]
            robot_pos_change = robot_pos_delta.cpu().numpy()
            is_same_dir = float(np.sum(dir_2d[i] * robot_pos_change) > 0)
            turn_delta = _angle_delta_rad(heading_yaw[i], env._prev_heading_yaw[i])
        else:
            robot_pos_delta = robot_pos[i] - env._prev_pose[i]
            is_same_dir = 1.0
            turn_delta = 0.0

        distance_change = env._prev_distance_to_goal[i] - current_distance
        env._prev_distance_to_goal[i] = current_distance
        env._prev_pose[i] = robot_pos[i]
        if heading_yaw is not None:
            env._prev_heading_yaw[i] = heading_yaw[i]

        env_reward = _progress_reward_with_static_turn(
            distance_change,
            is_same_dir,
            turn_delta,
            progress_threshold,
        )
        if not is_out_of_bounds:
            reward[i] = env_reward * weight

    return reward


"""Clutter scenes: reward from Euclidean distance change, with heading-direction filter."""
def relative_distance_reward_dir_clutter(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg,
    env_indices: list,
    weight: float = 0.075,
    progress_threshold: float = DEFAULT_PATH_PROGRESS_THRESHOLD,
) -> torch.Tensor:
    """Reward clutter scenes using Euclidean progress and heading direction."""

    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w[:, :2]
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0][:, :2]
    device = robot_pos.device
    num_envs = env.num_envs
    reward = torch.zeros(num_envs, dtype=torch.float32, device=device)

    if not hasattr(env, '_prev_euclidean_distance_to_goal'):
        env._prev_euclidean_distance_to_goal = torch.full(
            (num_envs,), -1.0, dtype=torch.float32, device=device
        )
        env._prev_euclidean_pose = robot_pos.clone()

    robot_quat = robot_asset.data.root_quat_w.cpu().numpy()
    scipy_quats = robot_quat[:, [1, 2, 3, 0]]
    dir_2d = None
    try:
        rot = R.from_quat(scipy_quats)
        rot_matrix = rot.as_matrix()
        dir_3d = rot_matrix @ np.array([1, 0, 0])
        dir_2d = dir_3d[:, :2]
    except (ValueError, TypeError) as e:
        print(f"[relative_distance_reward_dir_clutter] Quat conversion failed: {e}")
    heading_yaw = np.arctan2(dir_2d[:, 1], dir_2d[:, 0]) if dir_2d is not None else None
    if heading_yaw is not None and not hasattr(env, '_prev_euclidean_heading_yaw'):
        env._prev_euclidean_heading_yaw = heading_yaw.copy()

    grid_size = 0.1

    for i in env_indices:
        current_distance = torch.norm(robot_pos[i] - goal_pos[i]) / grid_size

        if env._prev_euclidean_distance_to_goal[i] < 0:
            env._prev_euclidean_distance_to_goal[i] = current_distance
            env._prev_euclidean_pose[i] = robot_pos[i]
            if heading_yaw is not None:
                env._prev_euclidean_heading_yaw[i] = heading_yaw[i]
            continue

        if dir_2d is not None:
            robot_pos_delta = robot_pos[i] - env._prev_euclidean_pose[i]
            robot_pos_change = robot_pos_delta.cpu().numpy()
            is_same_dir = float(np.sum(dir_2d[i] * robot_pos_change) > 0)
            turn_delta = _angle_delta_rad(heading_yaw[i], env._prev_euclidean_heading_yaw[i])
        else:
            robot_pos_delta = robot_pos[i] - env._prev_euclidean_pose[i]
            is_same_dir = 1.0
            turn_delta = 0.0

        raw_delta = env._prev_euclidean_distance_to_goal[i] - current_distance
        distance_change = torch.where(
            raw_delta >= 0,
            torch.floor(raw_delta),
            torch.round(raw_delta),
        )
        env._prev_euclidean_distance_to_goal[i] = current_distance
        env._prev_euclidean_pose[i] = robot_pos[i]
        if heading_yaw is not None:
            env._prev_euclidean_heading_yaw[i] = heading_yaw[i]

        env_reward = _progress_reward_with_static_turn(
            distance_change,
            is_same_dir,
            turn_delta,
            progress_threshold,
        )

        reward[i] = env_reward * weight

    return reward


def _contact_force_magnitude(net_forces_w: torch.Tensor) -> torch.Tensor:
    """Aggregate per-body contact forces; use max norm when multiple bodies are monitored."""
    if net_forces_w.ndim == 3 and net_forces_w.shape[1] > 1:
        return torch.norm(net_forces_w, dim=-1).max(dim=1).values
    return torch.norm(net_forces_w[:, 0], dim=-1)

""" Reward based on the dangerous behavior."""
def collision_reward(env: ManagerBasedRLEnv,
                     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
                     robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     force_threshold: float = 8.0,
                     delta_threshold: float = 10.0) -> torch.Tensor:
    """Penalize dangerous contact impacts, suppressing repeated same-place hits."""
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w[:, :2]
    sensor = env.scene[sensor_cfg.name]
    # net_forces_w: [num_envs, num_bodies, 3]
    net_forces_w = sensor.data.net_forces_w
    force_mag = _contact_force_magnitude(net_forces_w)
    dec_scale = 10.0 / float(getattr(env, "_decimation", 10))
    if not hasattr(env, "_prev_contact_force_mag"):
        env._prev_contact_force_mag = force_mag.clone()
        return torch.zeros_like(force_mag)

    is_start = env._prev_contact_force_mag <= 0
    delta_mag = force_mag - env._prev_contact_force_mag
    env._prev_contact_force_mag = force_mag.clone()
    impact = (force_mag > force_threshold) | (delta_mag > delta_threshold)
    impact = impact.float() * (1 - is_start.float())

    if not hasattr(env, "_prev_contact_position"):
        reward = torch.ones_like(impact) * impact
        env._prev_contact_position = impact[:,None] * robot_pos
        return reward * dec_scale

    distance = (env._prev_contact_position - robot_pos).square().sum(dim=-1).sqrt()
    distance_flag = (distance > 1.0).float()
    reward = -torch.ones_like(impact) * impact * distance_flag

    for i in range(reward.shape[0]):
        if reward[i] < 0:
            env._prev_contact_position[i] = robot_pos[i]
    return reward * dec_scale

""" Reward based on the dangerous behavior without _prev_contact_position"""
def collision_reward_new(env: ManagerBasedRLEnv,
                     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
                     robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     force_threshold: float = 8.0,
                     delta_threshold: float = 10.0) -> torch.Tensor:
    """Penalize new and repeated contact impacts without an initial positive reward."""

    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w[:, :2]
    sensor = env.scene[sensor_cfg.name]

    net_forces_w = sensor.data.net_forces_w
    force_mag = _contact_force_magnitude(net_forces_w)
    dec_scale = 10.0 / float(getattr(env, "_decimation", 10))
    if not hasattr(env, "_prev_contact_force_mag"):
        env._prev_contact_force_mag = force_mag.clone()
        return torch.zeros_like(force_mag)

    is_start = env._prev_contact_force_mag <= 0
    delta_mag = force_mag - env._prev_contact_force_mag
    env._prev_contact_force_mag = force_mag.clone()
    impact = (force_mag > force_threshold) | (delta_mag > delta_threshold)
    impact = impact.float() * (1 - is_start.float())

    if not hasattr(env, "_prev_contact_position"):
        reward = -torch.ones_like(impact) * impact
        env._prev_contact_position = impact[:, None] * robot_pos
        env._collision_repeat_count = torch.zeros(env.num_envs, dtype=torch.long, device=robot_pos.device)
        return reward * dec_scale

    distance = (env._prev_contact_position - robot_pos).square().sum(dim=-1).sqrt()
    mask_new = impact * (distance > 1.0).float()
    mask_old = impact * (1 - (distance > 1.0).float())
    count = env._collision_repeat_count.float()
    reward = mask_new * (-1.0) + mask_old * (-torch.pow(0.90, count + 1)) # 629 good model
    env._prev_contact_position = torch.where(mask_new.unsqueeze(-1).bool(), robot_pos, env._prev_contact_position)
    # 新地点: 0; 旧地方: count+1; 无碰撞: count
    env._collision_repeat_count = ((1 - mask_new) * (env._collision_repeat_count.float() + mask_old)).long()
    return reward * dec_scale


"""Reward when stuck with longer window and interval sampling."""
def stuck_reward_long_window(env: ManagerBasedRLEnv,
                              robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                              window_size: int = 100,
                              interval: int = 2,
                              threshold: float = 4.0):
    """Penalize long-duration stagnation using interval-sampled positions."""
    if not hasattr(env, '_recent_positions_long'):
        env._recent_positions_long = [deque(maxlen=window_size) for i in range(env.num_envs)]
    if not hasattr(env, '_frame_counter_long'):
        env._frame_counter_long = torch.zeros(env.num_envs, dtype=torch.int32, device=env.scene.device)
    if not hasattr(env, '_is_stuck_long'):
        env._is_stuck_long = torch.zeros(env.num_envs, dtype=torch.bool, device=env.scene.device)

    embodiment_interval = int(interval * (10 / env._decimation))
    robot_asset = env.scene[robot_asset_cfg.name]
    pos = robot_asset.data.root_pos_w[:, :2]
    reward = torch.zeros_like(pos[:,0])

    for i in range(env.num_envs):
        if env._arrival_timer_started[i]:
            env._is_stuck_long[i] = False
            continue

        env._frame_counter_long[i] += 1
        if env._frame_counter_long[i] >= embodiment_interval:
            env._recent_positions_long[i].append(pos[i].cpu().numpy())
            env._frame_counter_long[i] = 0

        if len(env._recent_positions_long[i]) >= window_size:
            current = env._recent_positions_long[i][-1]
            history_pos = np.array(env._recent_positions_long[i])[:-1]
            max_dist = max(np.sqrt(np.square(history_pos - current[None,:]).sum(axis=-1)))
            if max_dist < threshold:
                reward[i] = -1
                env._is_stuck_long[i] = True
            else:
                env._is_stuck_long[i] = False

    return reward

def stuck_penalty(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_size: int = 30,
    interval: int = 3,
    threshold: float = 0.25,
) -> torch.Tensor:
    """调用 ``stuck_reward_long_window``，直接返回其 reward（stuck 时为 -1）。"""
    return stuck_reward_long_window(
        env,
        robot_asset_cfg=robot_asset_cfg,
        window_size=window_size,
        interval=interval,
        threshold=threshold,
    )

def humanoid_low_height_fall_penalty(
    env: ManagerBasedRLEnv,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.4,
) -> torch.Tensor:
    """仅 unitree_g1：根高度低于阈值时每步惩罚 -1（再乘 RewTerm weight），并锁存跌倒标志。"""
    robot_asset = env.scene[robot_asset_cfg.name]
    device = robot_asset.data.root_pos_w.device
    if getattr(env, "_embodiment", None) != "unitree_g1":
        return torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    z = robot_asset.data.root_pos_w[:, 2]
    low = z < height_threshold
    if not hasattr(env, "_humanoid_fell_latched"):
        env._humanoid_fell_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env._humanoid_fell_latched = env._humanoid_fell_latched | low
    return torch.where(
        env._humanoid_fell_latched,
        torch.full((env.num_envs,), 1.0, device=device, dtype=torch.float32),
        torch.zeros(env.num_envs, device=device, dtype=torch.float32),
    )



"""Reward based the task success"""
def goal_arrival_reward(env: ManagerBasedRLEnv,
                        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    距离/速度阈值由课程阶段或 DEFAULT_* 决定；`_arrival_stop_timer` 与终止里的 `env._arrival_stop_delay`
    比较，后者按 embodiment 的 env._decimation 与 stop_delay 初始化（见 scaled_arrival_stop_delay_steps）。
    """
    enable_curriculum = getattr(env, '_enable_curriculum_learning', ENABLE_CURRICULUM_LEARNING)
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    robot_vel = robot_asset.data.root_lin_vel_w
    distance = torch.square(robot_pos[:,0:2] - goal_pos[:,0:2]).sum(axis=1).sqrt()
    velocity = torch.abs(robot_vel).sum(axis=1)

    if enable_curriculum:

        current_stage = env._curriculum_current_stage
        distance_threshold = current_stage["distance"]
        velocity_threshold = current_stage["velocity"]

        distance_ok = distance < distance_threshold
        velocity_ok = velocity < velocity_threshold
        condition_ok = distance_ok & velocity_ok

        env._arrival_timer_started[condition_ok] = True
        env._arrival_stop_timer[env._arrival_timer_started] += 1
        if env._arrival_timer_started.any():
            success_reward = env._arrival_timer_started.float()
            velocity_penalty = torch.where(
                env._arrival_timer_started,
                -velocity,
                torch.zeros_like(velocity)
            )
            epsilon = 0.5
            distance_inv_reward = torch.where(
                env._arrival_timer_started,
                1.0 / (distance + epsilon),
                torch.zeros_like(distance)
            )
            reward = success_reward + distance_inv_reward + 5 * velocity_penalty
        else:
            reward = torch.zeros_like(distance)
    else:

        condition_ok = (distance < DEFAULT_DISTANCE_THRESHOLD) & (velocity < DEFAULT_VELOCITY_THRESHOLD)
        env._arrival_timer_started[condition_ok] = True
        env._arrival_stop_timer[env._arrival_timer_started] += 1

        if env._arrival_timer_started.any():
            success_reward = env._arrival_timer_started.float()
            velocity_penalty = torch.where(
                env._arrival_timer_started,
                -velocity,
                torch.zeros_like(velocity)
            )
            epsilon = 0.5
            distance_inv_reward = torch.where(
                env._arrival_timer_started,
                1.0 / (distance + epsilon),
                torch.zeros_like(distance)
            )
            reward = success_reward + distance_inv_reward + 5 * velocity_penalty
        else:
            reward = torch.zeros_like(distance)

    return reward

def goal_arrival_reward_eval(env: ManagerBasedRLEnv,
                        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return an evaluation success reward once the arrival timer finishes."""

    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    robot_vel = robot_asset.data.root_lin_vel_w

    distance = torch.square(robot_pos[:,0:2] - goal_pos[:,0:2]).sum(axis=1).sqrt()
    velocity = torch.abs(robot_vel).sum(axis=1)
    condition_ok = (distance < DEFAULT_DISTANCE_THRESHOLD) & (velocity < DEFAULT_VELOCITY_THRESHOLD)
    env._arrival_timer_started[condition_ok] = True
    env._arrival_stop_timer[env._arrival_timer_started] += 1
    delay_steps = scaled_arrival_stop_delay_steps(env)
    timer_finished = env._arrival_stop_timer >= delay_steps
    success_reward = timer_finished.float()
    reward = success_reward

    return reward

@configclass
class NavigationOffRewardsStageFinalCfg:
    """Reward terms for the MDP."""
    alive = RewTerm(func=mdp.is_alive, weight=-0.025)
    arrive_goal = RewTerm(
        func=goal_arrival_reward,
        params={"robot_asset_cfg": SceneEntityCfg("robot")},
        weight=2.0 #stage1
    )
    collision_reward = RewTerm(
        func=collision_reward_new,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"),
                "force_threshold":6.5, "delta_threshold":2.0},
        weight=2.0
    )
    humanoid_low_height_fall = RewTerm(
        func=humanoid_low_height_fall_penalty,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "height_threshold": 0.4,
        },
        weight=-0.4,
    )
    stuck_penalty = RewTerm(
        func=stuck_penalty,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "window_size": 30,
            "interval": 2,
            "threshold": 0.2,
        },
        weight=0.2,
    )



@configclass
class EvalNavigationOffRewardsCfg:
    """Reward terms for the MDP."""
    arrive_goal = RewTerm(
        func=goal_arrival_reward_eval,
        params={"robot_asset_cfg": SceneEntityCfg("robot")},
        weight=2.0
    )
