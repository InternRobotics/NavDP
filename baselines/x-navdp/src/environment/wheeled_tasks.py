"""IsaacLab manager-based task configs for Dingo, G1, and Go2 navigation."""

from .tasks.action_utils import *
from .tasks.event_utils import *
from .tasks.observation_utils import *
from .tasks.reward_utils import *
from .tasks.terminal_utils import *
from dataclasses import MISSING
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg

@configclass
class DingoRLPointNavOffStage1Cfg(ManagerBasedRLEnvCfg):
    """Off-policy point-goal training config for stage 1 rewards."""
    scene: InteractiveSceneCfg = MISSING
    observations = PointNavObservationsDingoCfg()
    actions = BaseVelocityActionsCfg()
    terminations = NavigationOffTerminationsCfg()
    events = PointNavEventCfg()
    rewards = NavigationOffRewardsStageFinalCfg()
    def __post_init__(self):
        """Set simulation timing and episode length for off-policy stage 1."""
        self.sim.render_interval = 10
        self.episode_length_s = 122.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True

@configclass
class DingoEvalPointNavCfg(ManagerBasedRLEnvCfg):
    """Evaluation config for point-goal navigation episodes."""
    scene: InteractiveSceneCfg = MISSING
    observations = PointNavObservationsDingoCfg()
    actions = BaseVelocityActionsCfg()
    terminations = EvalTerminationsCfg()
    events = EvalPointNavEventCfg()
    rewards = EvalNavigationOffRewardsCfg()
    def __post_init__(self):
        """Set deterministic evaluation timing and episode length."""
        self.sim.render_interval = 10
        self.decimation = 10
        self.episode_length_s = 122.0
        self.sim.dt = 0.01
        self.sim.disable_contact_processing = True
