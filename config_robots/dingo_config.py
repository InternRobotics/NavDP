import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sensors import ContactSensorCfg, patterns, CameraCfg, RayCasterCfg, OffsetCfg

DINGO_CFG = ArticulationCfg(
    prim_path = "{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="./asset_robots/dingo.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint","right_wheel_joint"],
            velocity_limit=100.0,
            effort_limit=20.0,
            stiffness=0.0,
            damping=1.0,
        ),
    },
)
DINGO_BASE_LINK = 'base_link'
DINGO_WHEEL_JOINTS = ["left_wheel_joint","right_wheel_joint"]
DINGO_WHEEL_RADIUS = 0.0851 #0.0984/1.666 #0.0495
DINGO_WHEEL_BASE = 0.22616 * 2
DINGO_THRESHOLD = 15.0
DINGO_CAMERA_TRANS = [0.0,0.0,0.3]
DINGO_CAMERA_ROTS = [0.5, -0.5, 0.5, -0.5]
DINGO_IMAGEGOAL_TRANS = [5.0,0.0,0.3]
DINGO_IMAGEGOAL_ROTS = [0.5, -0.5, 0.5, -0.5]

DINGO_ContactCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/%s"%DINGO_BASE_LINK, 
                                    history_length=10, 
                                    track_air_time=True,
                                    update_period=0.02)

DINGO_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/front_cam"%DINGO_BASE_LINK,
    update_period=0.05,
    height=360,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.4, focus_distance=0.205, horizontal_aperture=1.88, clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=DINGO_CAMERA_TRANS, rot=DINGO_CAMERA_ROTS, convention="ros"),
)

DINGO_ImageGoal_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/goal_cam",
    update_period=0.05,
    height=360,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.4, focus_distance=0.205, horizontal_aperture=1.88, clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=DINGO_IMAGEGOAL_TRANS, rot=DINGO_IMAGEGOAL_ROTS, convention="ros"),
)

DINGO_MetricCameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/narritor_cam"%DINGO_BASE_LINK,
    update_period=0.05,
    height=90,
    width=160,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.4, focus_distance=0.205, horizontal_aperture=1.88, clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=[0.0,0.0,1.0], rot=DINGO_CAMERA_ROTS, convention="ros"),
)