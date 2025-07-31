from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg,AssetBaseCfg
from omni.isaac.lab.sensors import ContactSensorCfg, CameraCfg, RayCasterCfg
from dataclasses import MISSING
from omni.isaac.lab.sim.spawners import materials
import omni.isaac.lab.sim as sim_utils

GOAL_CFG = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Goal",\
    spawn = sim_utils.SphereCfg(visual_material=materials.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),visible=False,radius=0.25),
)

BENCH_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/Scene",
    terrain_type="usd",
    usd_path=f"",
)

@configclass
class ExplorationSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    metric_sensor: CameraCfg = MISSING

@configclass
class PointNavSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal: AssetBaseCfg = MISSING
    
@configclass
class ImageNavSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal_camera: CameraCfg = MISSING
    goal_marker: AssetBaseCfg = MISSING

@configclass
class PixelNavSceneCfg(InteractiveSceneCfg):
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = MISSING
    camera_sensor: CameraCfg = MISSING
    goal_marker: AssetBaseCfg = MISSING
    
@configclass
class QuadrupedPointNavSceneCfg(PointNavSceneCfg):
    height_sensor: RayCasterCfg = MISSING

@configclass
class QuadrupedImageNavSceneCfg(PointNavSceneCfg):
    height_sensor: RayCasterCfg = MISSING
    
@configclass
class QuadrupedExplorationSceneCfg(ExplorationSceneCfg):
    height_sensor: RayCasterCfg = MISSING

@configclass
class HumanoidPointNavSceneCfg(PointNavSceneCfg):
    height_sensor: RayCasterCfg = MISSING

@configclass
class HumanoidImageNavSceneCfg(PointNavSceneCfg):
    height_sensor: RayCasterCfg = MISSING
    
@configclass
class HumanoidExplorationSceneCfg(ExplorationSceneCfg):
    height_sensor: RayCasterCfg = MISSING
    
    
    


    


        
    
    
    
    
    
    
    