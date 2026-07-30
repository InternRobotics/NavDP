"""Dingo robot, camera, and contact sensor IsaacLab configs."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg

DINGO_BASE_LINK = 'base_link'
DINGO_WHEEL_JOINTS = ["left_wheel_joint","right_wheel_joint"]
DINGO_WHEEL_RADIUS = 0.049 * 2.5 * 0.5
DINGO_WHEEL_BASE = 0.22616
DINGO_CAMERA_TRANS = [0.28618,0.0,0.62532]
DINGO_CAMERA_ROTS = [-0.54168, -0.45452, 0.45452, 0.54168]

# RealSense D455 depth @ 640x360 / Z16 (device calibration intrinsics)
#   fx=fy=326.398560, cx=321.792145, cy=181.007690
# PinholeCameraCfg: fx = width * focal_length / horizontal_aperture
D455_DEPTH_WIDTH = 640
D455_DEPTH_HEIGHT = 360
D455_DEPTH_FOCAL_LENGTH = 1.93
D455_DEPTH_HORIZONTAL_APERTURE = 640 * D455_DEPTH_FOCAL_LENGTH / 326.398559570312
D455_DEPTH_VERTICAL_APERTURE = 360 * D455_DEPTH_FOCAL_LENGTH / 326.398559570312

DINGO_CFG = ArticulationCfg(
    prim_path = "{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="./data/robots/dingo.usd",
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

DINGO_ContactCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/%s"%DINGO_BASE_LINK,
                                    history_length=10,
                                    track_air_time=True,
                                    update_period=0.02)

DINGO_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/front_cam"%DINGO_BASE_LINK,
    update_period=0.05,
    height=D455_DEPTH_HEIGHT,
    width=D455_DEPTH_WIDTH,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=D455_DEPTH_FOCAL_LENGTH,
        focus_distance=0.205,
        horizontal_aperture=D455_DEPTH_HORIZONTAL_APERTURE,
        vertical_aperture=D455_DEPTH_VERTICAL_APERTURE,
        clipping_range=(0.01, 100.0),
    ),
    offset=CameraCfg.OffsetCfg(pos=DINGO_CAMERA_TRANS, rot=DINGO_CAMERA_ROTS, convention="usd"),
)

DINGO_BirdEye_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/birdeye_cam"%DINGO_BASE_LINK,
    update_period=0.05,
    height=480,
    width=480,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=18.14,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        vertical_aperture=15.29,
        clipping_range=(0.01, 100.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=[-2.0, 0.0, 1.5],
        rot=[0.6123, 0.35355, -0.35355, -0.61237],
        convention="usd",
    ),
)
