"""Unitree Go2 robot, camera, and contact sensor IsaacLab configs."""

import inspect

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

GO2_CAMERA_LINK = 'base'
GO2_CAMERA_TRANS = [0.32, 0.0, 0.20]
GO2_CAMERA_ROTS = [0.54168, 0.45452, -0.45452, -0.54168]

# RealSense D455 depth @ 640x360 / Z16 (device calibration intrinsics)
#   fx=fy=326.398560, cx=321.792145, cy=181.007690
D455_DEPTH_WIDTH = 640
D455_DEPTH_HEIGHT = 360
D455_DEPTH_FOCAL_LENGTH = 1.93
D455_DEPTH_HORIZONTAL_APERTURE = 640 * D455_DEPTH_FOCAL_LENGTH / 326.398559570312
D455_DEPTH_VERTICAL_APERTURE = 360 * D455_DEPTH_FOCAL_LENGTH / 326.398559570312
GO2_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"

GO2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_USD_PATH,
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
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            velocity_limit=30.0,
            effort_limit=23.5,
            stiffness=25.0,
            damping=0.5,
        ),
    },
)

# Contact sensor: base + legs (hip/thigh/calf; exclude foot — ground contact would false-trigger).
GO2_COLLISION_LINK_REGEX = (
    "{ENV_REGEX_NS}/Robot/"
    "(base|.*_(hip|thigh|calf))"
)
_contact_sensor_compat_kwargs = {}
if "max_contact_data_count_per_prim" in inspect.signature(ContactSensorCfg).parameters:
    _contact_sensor_compat_kwargs["max_contact_data_count_per_prim"] = 8

GO2_ContactCfg = ContactSensorCfg(
    prim_path=GO2_COLLISION_LINK_REGEX,
    history_length=10,
    track_air_time=True,
    update_period=0.02,
    **_contact_sensor_compat_kwargs,
)

GO2_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/front_cam" % GO2_CAMERA_LINK,
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
    offset=CameraCfg.OffsetCfg(pos=GO2_CAMERA_TRANS, rot=GO2_CAMERA_ROTS, convention="usd"),
)

GO2_BirdEye_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/birdeye_cam" % GO2_CAMERA_LINK,
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
