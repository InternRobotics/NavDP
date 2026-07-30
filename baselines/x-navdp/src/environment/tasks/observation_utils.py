"""Observation terms and image preprocessing for point-goal navigation."""

import torch
import torch.nn.functional as F
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaacsim.core.prims import XFormPrim
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

@torch.no_grad()
def process_image_torch(images: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Resize and letterbox RGB/RGBA images to the policy input size."""
    assert images.ndim == 4, "images must be [N, H, W, C]"
    device = images.device
    N, H, W, C = images.shape
    assert C in (3, 4), "expect 3/4 channels (RGB/RGBA)"
    imgs = images.float()
    if imgs.max() > 1.0:
        imgs = imgs / 255.0

    out_list = []
    for i in range(N):
        img = imgs[i]  # [H, W, C]
        h, w = img.shape[0], img.shape[1]
        prop = image_size / max(h, w)

        new_h = max(int(round(h * prop)), 1)
        new_w = max(int(round(w * prop)), 1)
        img_nchw = img.permute(2, 0, 1).unsqueeze(0)            # [1, C, H, W]
        img_scaled = F.interpolate(img_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_h_total = max(image_size - new_h, 0)
        pad_w_total = max(image_size - new_w, 0)
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        img_padded = F.pad(img_scaled, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)  # [1, C, H', W']

        img_final = F.interpolate(img_padded, size=(image_size, image_size), mode="bilinear", align_corners=False)
        out_list.append(img_final.squeeze(0).permute(1, 2, 0))  # [H, W, C]
    return torch.stack(out_list, dim=0).to(device)

@torch.no_grad()
def process_depth_torch(depths: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Resize, letterbox, and clamp depth images for policy input."""
    assert depths.ndim == 4 and depths.shape[-1] == 1, "depths must be [N, H, W, 1]"
    device = depths.device
    N, H, W, C = depths.shape
    d = depths.clone()
    d[torch.isinf(d)] = 0.0

    out_list = []
    for i in range(N):
        dep = d[i]  # [H, W, 1]
        h, w = dep.shape[0], dep.shape[1]
        prop = image_size / max(h, w)

        new_h = max(int(round(h * prop)), 1)
        new_w = max(int(round(w * prop)), 1)
        dep_nchw = dep.permute(2, 0, 1).unsqueeze(0)            # [1, 1, H, W]
        dep_scaled = F.interpolate(dep_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

        pad_h_total = max(image_size - new_h, 0)
        pad_w_total = max(image_size - new_w, 0)
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        dep_padded = F.pad(dep_scaled, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        dep_final = F.interpolate(dep_padded, size=(image_size, image_size), mode="bilinear", align_corners=False)
        dep_final = dep_final.squeeze(0)  # [1, H, W] -> [1, H, W]
        dep_final = dep_final.permute(1, 2, 0)  # -> [H, W, 1]

        dep_final = torch.where(dep_final > 5.0, torch.zeros_like(dep_final), dep_final)
        dep_final = torch.where(dep_final < 0.1, torch.zeros_like(dep_final), dep_final)
        out_list.append(dep_final)
    return torch.stack(out_list, dim=0).to(device)

def camera_rgb_raw_data(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    """Return raw RGB camera output as a uint8 tensor batch."""
    # return uint8 batch images with shape is (B,H,W,3)
    asset = env.scene[asset_cfg.name]
    return asset.data.output['rgb'][:,:,:,0:3]

def camera_depth_raw_data(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    """Return raw distance-to-plane camera output."""
    # return float batch depths with shape is (B,H,W,1)
    asset = env.scene[asset_cfg.name]
    return asset.data.output['distance_to_image_plane'][:,:,:,:]

def camera_rgb_policy_data_long_new(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera"), history_window:int=8, image_size:int=224, frame_interval:int=4) -> torch.Tensor:
    """Build embodiment-aware RGB history for NavDP policy input."""
    asset = env.scene[asset_cfg.name]
    current_image = asset.data.output['rgb'][:,:,:,0:3]
    B,H,W,C = current_image.shape
    num_history_samples = 7
    processed_current_image = process_image_torch(current_image, image_size=image_size)

    if 'history_rgb' in asset.data.output.keys():
        B,T,H,W,C = asset.data.output['history_rgb'].shape
        asset.data.output['history_rgb'] = asset.data.output['history_rgb'] * (env.episode_length_buf > 0).reshape(B,1,1,1,1)
        asset.data.output['history_rgb_valid_count'] = asset.data.output['history_rgb_valid_count'] * (env.episode_length_buf > 0)
    if 'history_rgb' not in asset.data.output.keys():
        asset.data.output['history_rgb'] = torch.zeros((B,history_window,image_size,image_size,C),device=current_image.device,dtype=torch.float32)
        asset.data.output['history_rgb_valid_count'] = torch.zeros(B, device=current_image.device, dtype=torch.long)

    history_rgb = asset.data.output['history_rgb']
    indices = torch.linspace(0, history_window - 3, num_history_samples, device=history_rgb.device).long()
    history_part = history_rgb[:, indices]  # (B, num_history_samples, H, W, C)
    sampled_frames = torch.cat([history_part, processed_current_image.unsqueeze(1)], dim=1)

    if hasattr(env, '_decimation'):
        embodiment_frame_interval = int(frame_interval * (10 / env._decimation))
    else:
        embodiment_frame_interval = frame_interval
    should_save = ((env.episode_length_buf) % embodiment_frame_interval == 0) & (env.episode_length_buf >= 4)
    for b in range(B):
        if should_save[b].item():
            asset.data.output['history_rgb'][b, 0:-1] = asset.data.output['history_rgb'][b, 1:].clone()
            asset.data.output['history_rgb'][b, -1] = processed_current_image[b]
            asset.data.output['history_rgb_valid_count'][b] = torch.clamp(asset.data.output['history_rgb_valid_count'][b] + 1, max=history_window)
    return sampled_frames

def camera_depth_policy_data(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("camera"),history_window:int=1, image_size:int=224) -> torch.Tensor:
    """Build depth history for policy observations."""
    # return preprocessed batch depths with history, shape is (B,T,H,W,1)
    asset = env.scene[asset_cfg.name]
    current_depth = asset.data.output['distance_to_image_plane'][:,:,:,:]
    B,H,W,C = current_depth.shape
    if 'history_depth' not in asset.data.output.keys():
        asset.data.output['history_depth'] = torch.zeros((B,history_window,image_size,image_size,C),device=current_depth.device)
        asset.data.output['history_depth'][:,-1] = process_depth_torch(current_depth,image_size=image_size)
    else:
        asset.data.output['history_depth'][:,0:-1] = asset.data.output['history_depth'][:,1:].clone()
        asset.data.output['history_depth'][:,-1] = process_depth_torch(current_depth,image_size=image_size)
    return asset.data.output['history_depth']

def oracle_imu_pose_data(env: ManagerBasedEnv,
                         robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Return goal position expressed in each robot's local frame."""
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_rot = math_utils.matrix_from_quat(robot_asset.data.root_quat_w)
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrim(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    rel_pos = torch.zeros((goal_pos.shape[0], 3),device=robot_pos.device)
    for i in range(rel_pos.shape[0]):
        rel_pos[i] = torch.matmul(torch.inverse(robot_rot[i]), goal_pos[i] - robot_pos[i])
    return rel_pos

def nonhead_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return non-head joint position offsets from defaults."""
    asset = env.scene[asset_cfg.name]
    all_names = list(asset.data.joint_names)
    ids = [i for i, n in enumerate(all_names) if "head" not in n.lower()]
    ids = torch.tensor(ids, device=asset.data.joint_pos.device, dtype=torch.long)
    return asset.data.joint_pos[:,ids] - asset.data.default_joint_pos[:,ids]

def nonhead_joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return non-head joint velocity offsets from defaults."""
    asset = env.scene[asset_cfg.name]
    all_names = list(asset.data.joint_names)
    ids = [i for i, n in enumerate(all_names) if "head" not in n.lower()]
    ids = torch.tensor(ids, device=asset.data.joint_pos.device, dtype=torch.long)
    return asset.data.joint_vel[:,ids] - asset.data.default_joint_vel[:,ids]

def nonhead_last_action(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the previous low-level joint-position action."""
    return env.action_manager.get_term('robot_joint').raw_actions

@configclass
class PointNavObservationsDingoCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        def __post_init__(self):
            """Concatenate Dingo base-velocity observations without corruption."""
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RawRGBImageCfg(ObsGroup):
        """Raw RGB observation group."""
        rgb_measurement = ObsTerm(
            func = camera_rgb_raw_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class RawDepthImageCfg(ObsGroup):
        """Raw depth observation group."""
        depth_measurement = ObsTerm(
            func = camera_depth_raw_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )

    @configclass
    class ObsRGBImageCfg(ObsGroup):
        """Policy RGB-history observation group."""
        rgb_measurement = ObsTerm(
            func = camera_rgb_policy_data_long_new,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor"),'history_window':48,'frame_interval':2})

    @configclass
    class ObsDepthImageCfg(ObsGroup):
        """Policy depth-history observation group."""
        depth_measurement = ObsTerm(
            func = camera_depth_policy_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor"),'history_window':1})

    @configclass
    class GoalPoseCfg(ObsGroup):
        """Goal pose observation group."""
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )

    policy: PolicyCfg = PolicyCfg()
    raw_rgb: RawRGBImageCfg = RawRGBImageCfg()
    raw_depth: RawDepthImageCfg = RawDepthImageCfg()
    obs_rgb: ObsRGBImageCfg = ObsRGBImageCfg()
    obs_depth: ObsDepthImageCfg = ObsDepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()

@configclass
class PointNavObservationsUnitreeG1Cfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=nonhead_joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=nonhead_joint_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=nonhead_last_action)
        def __post_init__(self):
            """Concatenate G1 proprioception observations without corruption."""
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RawRGBImageCfg(ObsGroup):
        """Raw RGB observation group."""
        rgb_measurement = ObsTerm(
            func = camera_rgb_raw_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class RawDepthImageCfg(ObsGroup):
        """Raw depth observation group."""
        depth_measurement = ObsTerm(
            func = camera_depth_raw_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )

    @configclass
    class ObsRGBImageCfg(ObsGroup):
        """Policy RGB-history observation group."""
        rgb_measurement = ObsTerm(
            func = camera_rgb_policy_data_long_new,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor"),'history_window':48,'frame_interval':2})

    @configclass
    class ObsDepthImageCfg(ObsGroup):
        """Policy depth-history observation group."""
        depth_measurement = ObsTerm(
            func = camera_depth_policy_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor"),'history_window':1})

    @configclass
    class GoalPoseCfg(ObsGroup):
        """Goal pose observation group."""
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )

    policy: PolicyCfg = PolicyCfg()
    raw_rgb: RawRGBImageCfg = RawRGBImageCfg()
    raw_depth: RawDepthImageCfg = RawDepthImageCfg()
    obs_rgb: ObsRGBImageCfg = ObsRGBImageCfg()
    obs_depth: ObsDepthImageCfg = ObsDepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()

@configclass
class PointNavObservationsUnitreeGo2Cfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=nonhead_joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=nonhead_joint_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=nonhead_last_action)
        def __post_init__(self):
            """Concatenate Go2 proprioception observations without corruption."""
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class RawRGBImageCfg(ObsGroup):
        """Raw RGB observation group."""
        rgb_measurement = ObsTerm(
            func = camera_rgb_raw_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )
    @configclass
    class RawDepthImageCfg(ObsGroup):
        """Raw depth observation group."""
        depth_measurement = ObsTerm(
            func = camera_depth_raw_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor")}
        )

    @configclass
    class ObsRGBImageCfg(ObsGroup):
        """Policy RGB-history observation group."""
        rgb_measurement = ObsTerm(
            func = camera_rgb_policy_data_long_new,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor"),'history_window':48,'frame_interval':2})

    @configclass
    class ObsDepthImageCfg(ObsGroup):
        """Policy depth-history observation group."""
        depth_measurement = ObsTerm(
            func = camera_depth_policy_data,
            params = {'asset_cfg':SceneEntityCfg("camera_sensor"),'history_window':1})

    @configclass
    class GoalPoseCfg(ObsGroup):
        """Goal pose observation group."""
        pose_measurement = ObsTerm(
            func = oracle_imu_pose_data,
            params = {'robot_asset_cfg':SceneEntityCfg("robot")}
        )

    policy: PolicyCfg = PolicyCfg()
    raw_rgb: RawRGBImageCfg = RawRGBImageCfg()
    raw_depth: RawDepthImageCfg = RawDepthImageCfg()
    obs_rgb: ObsRGBImageCfg = ObsRGBImageCfg()
    obs_depth: ObsDepthImageCfg = ObsDepthImageCfg()
    goal_pose: GoalPoseCfg = GoalPoseCfg()

@configclass
class EvalBirdeyeRGBImageCfg(ObsGroup):
    """Bird-eye RGB observation group used only by evaluation videos."""
    rgb_measurement = ObsTerm(
        func = camera_rgb_raw_data,
        params = {'asset_cfg':SceneEntityCfg("birdeye_camera")}
    )

@configclass
class PointNavEvalObservationsDingoCfg(PointNavObservationsDingoCfg):
    """Dingo evaluation observations with an extra bird-eye camera stream."""
    birdeye_rgb: EvalBirdeyeRGBImageCfg = EvalBirdeyeRGBImageCfg()

@configclass
class PointNavEvalObservationsUnitreeG1Cfg(PointNavObservationsUnitreeG1Cfg):
    """G1 evaluation observations with an extra bird-eye camera stream."""
    birdeye_rgb: EvalBirdeyeRGBImageCfg = EvalBirdeyeRGBImageCfg()

@configclass
class PointNavEvalObservationsUnitreeGo2Cfg(PointNavObservationsUnitreeGo2Cfg):
    """Go2 evaluation observations with an extra bird-eye camera stream."""
    birdeye_rgb: EvalBirdeyeRGBImageCfg = EvalBirdeyeRGBImageCfg()
