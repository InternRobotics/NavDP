"""Default IsaacLab scene configs used by point-goal navigation tasks."""

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg,AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.sim.spawners import materials
from dataclasses import MISSING
import isaaclab.sim as sim_utils

GOAL_CFG = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Goal",\
    spawn = sim_utils.SphereCfg(visual_material=materials.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),visible=False,radius=0.25),
)

BENCH_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/Scene",
    terrain_type="usd",
    usd_path=f"",
)

@configclass
class PointNavSceneCfg(InteractiveSceneCfg):
    """Scene assets required by point-goal navigation tasks."""
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal: AssetBaseCfg = MISSING

@configclass
class PointNavEvalSceneCfg(PointNavSceneCfg):
    """Evaluation scene assets with an extra bird-eye camera for videos."""
    birdeye_camera: CameraCfg = MISSING
