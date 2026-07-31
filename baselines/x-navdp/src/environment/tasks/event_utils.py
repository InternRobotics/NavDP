"""Reset events, path sampling, and scene randomization helpers."""

import hashlib
import torch
import numpy as np
from pxr import UsdLux, Gf
from isaaclab.utils import configclass
from isaacsim.core.utils.prims import create_prim
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaacsim.core.utils.prims import get_prim_at_path
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import random
import math
import isaaclab.envs.mdp as mdp

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover
    distance_transform_edt = None

# 与 tool/vis_occ_grid_path_from_scene 对齐：栅格外沿为障 + EDT 加权 A*
PATHFIND_COST_BASE = 1.0
PATHFIND_COST_ALPHA = 0.1
PATHFIND_COST_EPS_M = 0.05
PATHFIND_COST_MAX = 50
GRID_BORDER_AS_OBSTACLE_CELLS = 1

# Import curriculum learning utilities
from .curriculum_utils import (
    _init_curriculum_attributes,
    _reset_arrival_timer,
    _reset_reward_related_state,
    _get_random_reset_prob,
)

reset_counter = 0
eval_random_orientation = False

def filter_outlier_points(matrix, neighborhood_size=3, min_neighbors=7):
    """
    过滤二值矩阵中的离群离散点
    :param matrix: 输入二值矩阵（0/1）
    :param neighborhood_size: 邻域大小（奇数，如3=3x3邻域，5=5x5邻域）
    :param min_neighbors: 保留点所需的最小邻域内1的数量（小于此值则判定为离群点）
    :return: 过滤后的矩阵
    """
    # 复制矩阵避免修改原数据
    filtered_matrix = matrix.copy()
    h, w = matrix.shape
    # 计算邻域半宽（如3x3邻域的半宽为1）
    half_nh = neighborhood_size // 2

    for y in range(h):
        for x in range(w):
            if matrix[y][x] == 0:
                y_start = max(0, y - half_nh)
                y_end = min(h, y + half_nh + 1)
                x_start = max(0, x - half_nh)
                x_end = min(w, x + half_nh + 1)

                neighbor_sum = np.sum(matrix[y_start:y_end, x_start:x_end])
                if neighbor_sum >= min_neighbors:
                    filtered_matrix[y][x] = 1
    return filtered_matrix


def erode_map_border_as_obstacle(
    decision_map: np.ndarray, border_cells: int = GRID_BORDER_AS_OBSTACLE_CELLS
) -> np.ndarray:
    """把栅格外沿 border_cells 层标为障碍（0）。"""
    dm = np.asarray(decision_map, dtype=np.int32).copy()
    if border_cells <= 0:
        return dm
    h, w = dm.shape
    if h == 0 or w == 0:
        return dm
    b = min(int(border_cells), h, w)
    if b <= 0:
        return dm
    dm[:b, :] = 0
    dm[h - b :, :] = 0
    dm[:, :b] = 0
    dm[:, w - b :] = 0
    return dm


def distance_weighted_cost_map(
    decision_map: np.ndarray,
    grid_cell_m: float,
    *,
    base: float = PATHFIND_COST_BASE,
    alpha: float = PATHFIND_COST_ALPHA,
    epsilon_m: float = PATHFIND_COST_EPS_M,
    cost_max: int = PATHFIND_COST_MAX,
) -> np.ndarray:
    """Convert a free-space mask into A* traversal costs using obstacle distance."""
    if distance_transform_edt is None:
        raise ImportError(
            "distance_weighted_cost_map 需要 scipy.ndimage.distance_transform_edt"
        )
    dm = np.asarray(decision_map)
    if dm.ndim != 2:
        raise ValueError(f"decision_map 须为 2D，got shape {dm.shape}")
    dist_cells = distance_transform_edt(dm)
    dist_m = dist_cells * float(grid_cell_m)
    cost_f = base + alpha / (dist_m + float(epsilon_m))
    cost_f = np.where(dm > 0, cost_f, 0.0)
    cost_i = np.round(cost_f).astype(np.int32)
    cost_i = np.clip(cost_i, 1, int(cost_max))
    cost_i = np.where(dm > 0, cost_i, 0)
    return cost_i


def decision_map_to_weighted_planning_grid(
    decision_map_filtered: np.ndarray, grid_cell_m: float
):
    """filter_outlier 后的二值图 → 外沿侵蚀 → EDT 权重 Grid；_grid_matrix 为侵蚀后可走掩膜。"""
    dm = erode_map_border_as_obstacle(np.asarray(decision_map_filtered, dtype=np.int32))
    if distance_transform_edt is None:
        _grid = Grid(matrix=dm.astype(np.int32))
    else:
        weighted = distance_weighted_cost_map(dm, float(grid_cell_m))
        _grid = Grid(matrix=weighted)
    _grid_matrix = (dm > 0).astype(np.int32)
    return _grid, _grid_matrix


scale = 1
path_finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
_occ_cache_path = None
_occ_cache_data = None
_sample_points_cache_path = None
_sample_points_cache_data = None

def hide_entity(prim_path: str):
    """Hide a USD prim by setting its visibility attribute to invisible."""
    prim = get_prim_at_path(prim_path)
    if prim.IsValid():
        prim.GetAttribute("visibility").Set("invisible")
    else:
        print(f"Warning: Cannot find the prim with path {prim_path}")

def add_point_light(
    position: torch.Tensor,
    intensity: float = 20000.0,
    color: tuple = (1.0, 1.0, 1.0),
    radius: float = 0.1,
    prim_path: str = None) -> UsdLux.SphereLight:
    """Create a point light at the requested world position."""
    if isinstance(position, torch.Tensor):
        position = position.cpu().numpy()
    if prim_path is None:
        prim_path = "/World/Lights/point_light"
        count = 0
        while create_prim(prim_path).IsValid():
            count += 1
            prim_path = f"/World/Lights/point_light_{count}"
    light_prim = create_prim(
        prim_path=prim_path,
        prim_type="SphereLight"
    )
    point_light = UsdLux.SphereLight(light_prim)
    point_light.CreateIntensityAttr(intensity)
    point_light.CreateColorAttr(Gf.Vec3f(*color))
    point_light.CreateRadiusAttr(radius)
    xform_prim = XFormPrim(prim_path)
    xform_prim.set_world_pose(position=position)
    return point_light

def pointnav_reset(env: ManagerBasedEnv,
                   env_ids: torch.Tensor,
                   init_point_path:str,
                   global_occ_path:str,
                   height_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   embodiment:str,
                   scale_factor: float = 1.0,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset point-goal episodes and refresh cached occupancy-grid metadata."""

    global sample_points, _occ_cache_path, _occ_cache_data, reset_counter
    if _occ_cache_path != global_occ_path:
        _gs = 0.1 * scale_factor
        occ_pcd = o3d.io.read_point_cloud(global_occ_path)
        occ_point = np.array(occ_pcd.points) * scale_factor
        _occ_min, _occ_max = occ_point.min(axis=0), occ_point.max(axis=0)

        grid_dimension = np.ceil((_occ_max[0:2] - _occ_min[0:2]) / _gs).astype(int)
        decision_map = np.zeros(grid_dimension, dtype=np.int32)
        free_pt = np.floor((occ_point - _occ_min) / _gs).astype(np.int32)
        decision_map[free_pt[:, 0], free_pt[:, 1]] = 1
        decision_map = filter_outlier_points(decision_map)

        _grid, _grid_matrix = decision_map_to_weighted_planning_grid(decision_map, _gs)
        _occ_cache_path = global_occ_path
        _occ_cache_data = (_grid, _occ_min, _occ_max, _gs, _grid_matrix)

        grid, occ_min_bound, occ_max_bound, grid_size, grid_matrix = _occ_cache_data
        env._last_occ_path = global_occ_path
        env._grid = grid
        env._occ_min_bound = occ_min_bound
        env._occ_max_bound = occ_max_bound
        env._grid_size = grid_size
        env._grid_matrix = grid_matrix

        reset_counter = 0

    global _sample_points_cache_path, _sample_points_cache_data
    if _sample_points_cache_path != init_point_path:
        sp = np.load(init_point_path)
        _sample_points_cache_path = init_point_path
        _sample_points_cache_data = sp
    sample_points = _sample_points_cache_data

    _init_curriculum_attributes(env, embodiment)
    random_reset_prob = _get_random_reset_prob(env)

    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")
    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")

    random_robot_points = []
    random_goal_points = []
    random_init_orientions = []
    count = reset_counter

    for i in range(env_ids.shape[0]):
        env_id = env_ids[i]
        if random.random() < random_reset_prob:
            start_idx = random.randint(0, sample_points.shape[0] - 1)
            end_idx = random.randint(0, sample_points.shape[0] - 1)
            start_point = sample_points[start_idx].copy()
            start_point[:4] = start_point[:4] * scale_factor
            end_point = sample_points[end_idx].copy()
            end_point[:4] = end_point[:4] * scale_factor
            start_points = np.array([start_point[0], start_point[1], 0])
            goal_points = np.array([end_point[2], end_point[3], 0])
            init_orientions = random.uniform(-math.pi, math.pi)
        else:
            count += 1
            idx = int(count % sample_points.shape[0])
            start_goal_pair = sample_points[idx].copy()
            start_goal_pair[:4] = start_goal_pair[:4] * scale_factor
            start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
            goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
            init_orientions = start_goal_pair[4]

        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientions.append(init_orientions)

        metadata = {
            'start_points': start_points,
            'goal_points': goal_points,
            'init_orientions': init_orientions,
            'use_failed_config': False
        }
        env._episode_metadata[env_id].append(metadata)
        env._is_last_success[env_id] = env._has_arrived_goal[env_id]

    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5

    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]

    _reset_arrival_timer(env, env_ids)
    _reset_reward_related_state(env, env_ids, robot_asset.data.root_pos_w.device)

def pointnav_reset_eval(env: ManagerBasedEnv,
                   env_ids: torch.Tensor,
                   init_point_path:str,
                   global_occ_path: str = None,
                   height_offset:float=0.1,
                   robot_visible:bool=False,
                   light_enabled:bool=False,
                   embodiment: str = None,
                   scale_factor: float = 1.0,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset evaluation episodes in a deterministic sample order."""

    global sample_points, _sample_points_cache_path, _sample_points_cache_data
    if _sample_points_cache_path != init_point_path:
        sp = np.load(init_point_path)
        _sample_points_cache_path = init_point_path
        _sample_points_cache_data = sp
    sample_points = _sample_points_cache_data

    if global_occ_path is not None:
        global _occ_cache_path, _occ_cache_data
        if _occ_cache_path != global_occ_path:
            _gs = 0.1 * scale_factor
            occ_pcd = o3d.io.read_point_cloud(global_occ_path)
            occ_point = np.array(occ_pcd.points) * scale_factor
            _occ_min, _occ_max = occ_point.min(axis=0), occ_point.max(axis=0)
            grid_dimension = np.ceil((_occ_max[0:2] - _occ_min[0:2]) / _gs).astype(int)
            decision_map = np.zeros(grid_dimension, dtype=np.int32)
            free_pt = np.floor((occ_point - _occ_min) / _gs).astype(np.int32)
            decision_map[free_pt[:, 0], free_pt[:, 1]] = 1
            decision_map = filter_outlier_points(decision_map)
            _grid, _grid_matrix = decision_map_to_weighted_planning_grid(decision_map, _gs)
            _occ_cache_path = global_occ_path
            _occ_cache_data = (_grid, _occ_min, _occ_max, _gs, _grid_matrix)
        g, omin, omax, gs, gmat = _occ_cache_data
        env._grid, env._occ_min_bound, env._occ_max_bound, env._grid_size, env._grid_matrix = g, omin, omax, gs, gmat

    _init_curriculum_attributes(env, embodiment)
    _reset_arrival_timer(env, env_ids)
    if not hasattr(env, '_sample_idx'):
        env._sample_idx = np.zeros(env.num_envs, dtype=np.int32)
    global reset_counter
    np.random.seed(1234)
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    env._total_episode_count = sample_points.shape[0]

    if not robot_visible:
        for i in range(env_ids.shape[0]):
            hide_entity(f"/World/envs/env_{env_ids[i]}/Robot")

    if light_enabled:
        if reset_counter == 0:
            for light_idx,pts in enumerate(sample_points[:,0]):
                pts = pts + np.array([0.0, 0.0, 1.5])
                add_point_light(torch.as_tensor(pts, dtype=torch.float32, device=robot_asset.data.root_pos_w.device),
                                prim_path= f"/World/envs/env_{env_ids[0]}/point_light_{light_idx}")

    random_robot_points = []
    random_goal_points = []
    random_init_orientions = []
    count = reset_counter - env.num_envs

    for i in range(env_ids.shape[0]):
        idx = int(count % sample_points.shape[0])
        start_goal_pair = sample_points[idx].copy()
        start_goal_pair[:4] = start_goal_pair[:4] * scale_factor
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])

        global eval_random_orientation
        if eval_random_orientation:
            rng = np.random.RandomState(seed=idx)
            init_orientions = rng.uniform(0, 2 * np.pi)
        else:
            init_orientions = start_goal_pair[4]

        env_id = env_ids[i]
        env._sample_idx[env_id] = idx

        count += 1
        random_robot_points.append(start_points)
        random_goal_points.append(goal_points)
        random_init_orientions.append(init_orientions)

    random_robot_points = np.array(random_robot_points)
    random_goal_points = np.array(random_goal_points)
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
    tensor_goal_points = torch.tensor(random_goal_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_goal_points[:, 2] = tensor_goal_points[:, 2] + 1.5

    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrim(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]

@configclass
class PointNavEventCfg:
    """Configuration for events."""
    reset_pose = EventTerm(func=pointnav_reset,
                           mode='reset',
                           params={})
    reset_joint = EventTerm(func=mdp.reset_joints_by_scale,
                        mode='reset',
                        params={'position_range': (1.0, 1.0),
                                'velocity_range': (1.0, 1.0)})
@configclass
class EvalPointNavEventCfg:
    """Configuration for events."""
    reset_pose = EventTerm(func=pointnav_reset_eval,
                           mode='reset',
                           params={})
    reset_joint = EventTerm(func=mdp.reset_joints_by_scale,
                    mode='reset',
                    params={'position_range': (1.0, 1.0),
                            'velocity_range': (1.0, 1.0)})
