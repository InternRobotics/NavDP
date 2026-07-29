"""Action manager configs for wheel, humanoid, and quadruped control."""

from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from src.environment.robots import DINGO_WHEEL_JOINTS

@configclass
class BaseVelocityActionsCfg:
    """Wheel velocity action space for the Dingo base."""
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot",
                                           joint_names=DINGO_WHEEL_JOINTS,
                                           scale=1.0,
                                           use_default_offset=True,
                                           debug_vis=False)
@configclass
class BasePositionActionsCfg:
    """Joint position action space for humanoid embodiments."""
    robot_joint = mdp.JointPositionActionCfg(asset_name="robot",
                                                 joint_names=['.*joint.*'],
                                                 scale=0.5,
                                                 use_default_offset=True,
                                                 debug_vis=False)

@configclass
class QuadrupedPositionActionsCfg:
    """Joint position action space for quadruped embodiments."""
    robot_joint = mdp.JointPositionActionCfg(asset_name="robot",
                                                 joint_names=['.*joint.*'],
                                                 scale=0.25,
                                                 use_default_offset=True,
                                                 debug_vis=False)
