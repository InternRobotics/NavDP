"""Unitree G1 robot, camera, and contact sensor IsaacLab configs."""

import inspect

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg

G1_HEAD_LINK = 'yaw_head_01'
G1_CAMERA_TRANS = [-0.1, 0.0, 0.05]
G1_CAMERA_ROTS = [0.53163, 0.46623, 0.46623, 0.53163]

# RealSense D455 depth @ 640x360 / Z16 (device calibration intrinsics)
#   fx=fy=326.398560, cx=321.792145, cy=181.007690
D455_DEPTH_WIDTH = 640
D455_DEPTH_HEIGHT = 360
D455_DEPTH_FOCAL_LENGTH = 1.93
D455_DEPTH_HORIZONTAL_APERTURE = 640 * D455_DEPTH_FOCAL_LENGTH / 326.398559570312
D455_DEPTH_VERTICAL_APERTURE = 360 * D455_DEPTH_FOCAL_LENGTH / 326.398559570312

G1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="./data/robots/unitreeg1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit_sim=300,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[".*head.*"],
            velocity_limit=100.0,
            effort_limit=20.0,
            stiffness=40.0,
            damping=10.0,
        ),
    },
)

# Contact sensor: pelvis/torso + arms/legs (exclude ankle — ground contact would false-trigger).
G1_COLLISION_LINK_REGEX = (
    "{ENV_REGEX_NS}/Robot/"
    "(pelvis|torso_link|"
    ".*_(shoulder|elbow).*link|.*_palm_link|"
    ".*_(hip|knee).*link)"
)
_contact_sensor_compat_kwargs = {}
if "max_contact_data_count_per_prim" in inspect.signature(ContactSensorCfg).parameters:
    _contact_sensor_compat_kwargs["max_contact_data_count_per_prim"] = 8

G1_ContactCfg = ContactSensorCfg(
    prim_path=G1_COLLISION_LINK_REGEX,
    history_length=10,
    track_air_time=True,
    update_period=0.02,
    **_contact_sensor_compat_kwargs,
)


def _camera_cfg(name, rot):
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/%s/%s" % (G1_HEAD_LINK, name),
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
        offset=CameraCfg.OffsetCfg(pos=G1_CAMERA_TRANS, rot=rot, convention="usd"),
    )


G1_CameraCfg = _camera_cfg("front_cam", G1_CAMERA_ROTS)

G1_BirdEye_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/birdeye_cam" % G1_HEAD_LINK,
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
