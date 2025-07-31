import math
import random
import torch
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import MISSING
from typing import Literal
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg,ManagerBasedEnvCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.spawners import materials
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, patterns, CameraCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.core.prims import XFormPrimView
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.assets import Articulation, RigidObject
from configs.robots import *
from .usd_utils import *

reset_counter = 0
def camera_rgb_data(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.output['rgb']
def camera_depth_data(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.output['distance_to_image_plane']
def oracle_imu_pose_data(env: ManagerBasedEnv, 
                         robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_rot = math_utils.matrix_from_quat(robot_asset.data.root_quat_w)
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrimView(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    rel_pos = torch.zeros((goal_pos.shape[0], 3))
    for i in range(rel_pos.shape[0]):
        rel_pos[i] = torch.matmul(torch.inverse(robot_rot[i]),(goal_pos[i] - robot_pos[i]).T)
    return rel_pos

def pixel_projection_data(env: ManagerBasedEnv,
                          robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")):
    camera_asset = env.scene[robot_asset_cfg.name]
    camera_w_pos = camera_asset._data.pos_w 
    camera_w_rot = math_utils.matrix_from_quat(camera_asset._data.quat_w_world)
    camera_intrinsic = camera_asset._data.intrinsic_matrices
    goal_primview = XFormPrimView(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    pixel_coords = torch.zeros((goal_pos.shape[0], 2))
    for i in range(camera_intrinsic.shape[0]):
        frame_coord = torch.matmul(torch.inverse(camera_w_rot[i]),(goal_pos[i] - camera_w_pos[i]).T)
        pixel_coord_x =  -frame_coord[1] * camera_intrinsic[i,0,0] / frame_coord[0] + camera_intrinsic[i,0,2]
        pixel_coord_y =  -frame_coord[2] * camera_intrinsic[i,1,1] / frame_coord[0] + camera_intrinsic[i,1,2]
        pixel_coords[i] = torch.as_tensor([pixel_coord_x,pixel_coord_y],dtype=torch.float32,device=camera_w_pos.device)
    return pixel_coords
    
def stuck_terminal_check(env: ManagerBasedEnv,
                         robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                         window_size: int = 10,
                         threshold: float = 0.05):
    if not hasattr(env, '_recent_positions'):
        env._recent_positions = deque(maxlen=window_size)
    robot_asset = env.scene[robot_asset_cfg.name]
    pos = robot_asset.data.root_pos_w[0, :2].cpu().numpy()  # 只看x, y
    env._recent_positions.append(pos)
    if len(env._recent_positions) < window_size:
        return False 
    current = env._recent_positions[-1]
    max_dist = max(np.linalg.norm(current - np.array(p)) for p in list(env._recent_positions)[:-1])
    return bool(max_dist < threshold)

def arrival_terminal_check(env: ManagerBasedEnv,
                           robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrimView(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    robot_vel = robot_asset.data.root_lin_vel_w
    distance = torch.square(robot_pos[:,0:2] - goal_pos[:,0:2]).sum(axis=1).sqrt()
    velocity = torch.abs(robot_vel).sum(axis=1)
    return (distance < 1.0) & (velocity < 0.5)

def exploration_reset(env: ManagerBasedEnv, 
                      env_ids: torch.Tensor, 
                      init_point_path:str,
                      height_offset:float,
                      robot_visible:bool,
                      light_enabled:bool,
                      robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    np.random.seed(1234)
    sample_points = np.load(init_point_path)
    
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
    random_init_orientions = []
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        init_orientions = start_goal_pair[4]
        random_robot_points.append(start_points)
        random_init_orientions.append(init_orientions)
        
    random_robot_points = np.array(random_robot_points)
    tensor_robot_points = torch.tensor(random_robot_points, dtype=torch.float32, device=robot_asset.data.root_pos_w.device) + env.scene.env_origins[env_ids]
    tensor_robot_points[:, 2] = tensor_robot_points[:, 2] + height_offset
    random_init_orientions = np.array(random_init_orientions)
    random_init_orientions = torch.tensor(random_init_orientions, dtype=torch.float32, device=robot_asset.data.root_pos_w.device)
    if len(tensor_robot_points.shape) == 1:
        tensor_robot_points = tensor_robot_points.unsqueeze(0)
        
    angle = random_init_orientions
    angle = angle.unsqueeze(-1).cpu().numpy()
    batch_init_rotation = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0, angle*0.0, angle), axis=-1))).to(robot_asset.data.root_pos_w.device)
    robot_asset.write_root_pose_to_sim(torch.concat((tensor_robot_points, batch_init_rotation.to(torch.float32)),dim=-1),env_ids)
    reset_counter += env_ids.shape[0]

def pointnav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    np.random.seed(1234)
    sample_points = np.load(init_point_path)
    
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
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
        init_orientions = start_goal_pair[4]
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
        goal_primview = XFormPrimView(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]
    
def imagenav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   camera_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    np.random.seed(1234)
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    sample_points = np.load(init_point_path)
    
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
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
        init_orientions = start_goal_pair[4]
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
        goal_primview = XFormPrimView(prim_paths_expr=f"/World/envs/env_{env_id}/goal_cam", name="xform_view")
        goal_image_point = tensor_goal_points[i]
        goal_image_point[2] = robot_asset.data.root_pos_w[i,2] + camera_offset
        goal_image_rot = torch.tensor(rot_utils.euler_angles_to_quats(np.concatenate((angle*0.0 + np.pi/2, angle*0.0, angle - np.pi/2), axis=-1))).to(robot_asset.data.root_pos_w.device)
        goal_primview.set_world_poses(goal_image_point.unsqueeze(0),goal_image_rot)
        
    for i, env_id in enumerate(env_ids):
        goal_primview = XFormPrimView(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]
 
def pixelnav_reset(env: ManagerBasedEnv, 
                   env_ids: torch.Tensor, 
                   init_point_path:str,
                   height_offset:float,
                   robot_visible:bool,
                   light_enabled:bool,
                   robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    global reset_counter
    robot_asset: RigidObject | Articulation = env.scene[robot_asset_cfg.name]
    np.random.seed(1234)
    sample_points = np.load(init_point_path)
    
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
    for i in range(env_ids.shape[0]):
        idx = int((i + reset_counter) % sample_points.shape[0])
        start_goal_pair = sample_points[idx]
        start_points = np.array([start_goal_pair[0], start_goal_pair[1], 0])
        goal_points = np.array([start_goal_pair[2], start_goal_pair[3], 0])
        init_orientions = start_goal_pair[4]
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
        goal_primview = XFormPrimView(prim_paths_expr=f"/World/envs/env_{env_id}/Goal", name="xform_view")
        goal_primview.set_world_poses(tensor_goal_points[i].unsqueeze(0),batch_init_rotation[i].unsqueeze(0))
    reset_counter += env_ids.shape[0]

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

@configclass
class ExploreObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class MetricRGBImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("metric_sensor")}
        )
    @configclass
    class MetricDepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("metric_sensor")}
        )
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    metric_rgb: MetricRGBImageCfg=MetricRGBImageCfg()
    metric_depth: MetricDepthImageCfg=MetricDepthImageCfg()

@configclass
class PointNavObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class GoalPoseCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()

@configclass
class ImageNavObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class GoalImageCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("goal_camera")}
        )
    
    @configclass
    class GoalPoseCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    goal_image: GoalImageCfg = GoalImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()

@configclass
class PixelNavObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_pos = ObsTerm(func=mdp.root_pos_w)
        base_rot = ObsTerm(func=mdp.root_quat_w)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class RGBImageCfg(ObsGroup):
        rgb_measurement = ObsTerm(
            func = camera_rgb_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class DepthImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func = camera_depth_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class GoalPoseCfg(ObsGroup):
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )
    
    @configclass
    class GoalPixelCfg(ObsGroup):
        pixel_measurement = ObsTerm(
            func = pixel_projection_data,
            params = {'robot_asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    policy: PolicyCfg = PolicyCfg()
    rgb: RGBImageCfg = RGBImageCfg()
    depth: DepthImageCfg = DepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()
    goal_pixel: GoalPixelCfg = GoalPixelCfg()

@configclass
class ExploreEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=exploration_reset,
                           mode='reset',
                           params={})

@configclass
class PointNavEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=pointnav_reset,
                           mode='reset',
                           params={})

@configclass
class ImageNavEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=imagenav_reset,
                           mode='reset',
                           params={})

@configclass
class PixelNavEventCfg:
    """Configuration for events.""" 
    reset_pose = EventTerm(func=pixelnav_reset,
                           mode='reset',
                           params={})
    
@configclass
class DingoActionsCfg:
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=DINGO_WHEEL_JOINTS, scale=1.0, use_default_offset=True, debug_vis=True)
     
@configclass
class DingoExploreTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=DINGO_BASE_LINK), "threshold": DINGO_THRESHOLD},
    )
    struck = DoneTerm(func=stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})
    
@configclass
class PointNavTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(func=arrival_terminal_check,
                           params={"robot_asset_cfg":SceneEntityCfg("robot")})
    stuck = DoneTerm(func=stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})
    
@configclass
class ImageNavTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(func=arrival_terminal_check,
                           params={"robot_asset_cfg":SceneEntityCfg("robot")})
    stuck = DoneTerm(func=stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})

@configclass
class PixelNavTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    arrive_goal = DoneTerm(func=arrival_terminal_check,
                           params={"robot_asset_cfg":SceneEntityCfg("robot")})
    stuck = DoneTerm(func=stuck_terminal_check,
                      params={"robot_asset_cfg": SceneEntityCfg("robot"), 
                              "window_size": 30, 
                              "threshold": 0.1})
    
@configclass
class DingoPointNavCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    observations = PointNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = PointNavTerminationsCfg()
    events = PointNavEventCfg()
    rewards = RewardsCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        
@configclass
class DingoImageNavCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    observations = ImageNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = ImageNavTerminationsCfg()
    events = ImageNavEventCfg()
    rewards = RewardsCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True

@configclass
class DingoPixelNavCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    observations = PixelNavObservationsCfg()
    actions = DingoActionsCfg()
    terminations = PixelNavTerminationsCfg()
    events = PixelNavEventCfg()
    rewards = RewardsCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
        
@configclass
class DingoExplorationCfg(ManagerBasedRLEnvCfg):
    scene: InteractiveSceneCfg = MISSING
    observations = ExploreObservationsCfg()
    actions = DingoActionsCfg()
    terminations = DingoExploreTerminationsCfg()
    events = ExploreEventCfg()
    def __post_init__(self):
        self.sim.render_interval = 15
        self.decimation = 15
        self.episode_length_s = 120.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True

        

