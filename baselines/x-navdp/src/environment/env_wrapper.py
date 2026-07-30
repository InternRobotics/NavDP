"""Scene, robot, and environment assembly helpers for X-NavDP."""

import os
from typing import Union
from .scenes import PointNavSceneCfg, PointNavEvalSceneCfg, GOAL_CFG, BENCH_TERRAIN_CFG
from .robots import (
    DINGO_CFG, DINGO_ContactCfg, DINGO_CameraCfg, DINGO_BirdEye_CameraCfg,
    DINGO_WHEEL_BASE, DINGO_WHEEL_RADIUS,
    G1_CFG, G1_ContactCfg, G1_CameraCfg, G1_BirdEye_CameraCfg,
    GO2_CFG, GO2_ContactCfg, GO2_CameraCfg, GO2_BirdEye_CameraCfg,
)
from .wheeled_tasks import DingoEvalPointNavCfg, DingoRLPointNavOffStage1Cfg, BaseVelocityActionsCfg, BasePositionActionsCfg, QuadrupedPositionActionsCfg
from .tasks.observation_utils import (
    PointNavObservationsDingoCfg,
    PointNavObservationsUnitreeG1Cfg,
    PointNavObservationsUnitreeGo2Cfg,
    PointNavEvalObservationsDingoCfg,
    PointNavEvalObservationsUnitreeG1Cfg,
    PointNavEvalObservationsUnitreeGo2Cfg,
)
from .tasks.curriculum_utils import _EMBODIMENT_DECIMATION_CFG
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from .controllers import DifferentialController, G1VelocityController, Go2VelocityController
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema

def find_usd_path(dir,task='pointgoal'):
    """Locate USD, point-goal sample, and occupancy files in one legacy scene dir."""
    paths = os.listdir(dir)
    usd_candidates = []
    init_path = ""
    occ_path = ""
    for p in paths:
        if ".usd" in p and "_noMDL" not in p and "scale" not in p:
            usd_candidates.append(os.path.join(dir, p))
        if ".npy" in p and task in p:
            init_path = os.path.join(dir,p)
        if ".ply" in p:
            occ_path = os.path.join(dir,p)
    usd_path = ""
    preferred_markers = (
        "mesh-model.usdz",
        "_visible_collmesh",
        "_merged_zup_flat_mesh_hidden.usdz",
        "model.usdz",
    )
    for marker in preferred_markers:
        for candidate in sorted(usd_candidates):
            if marker in os.path.basename(candidate):
                usd_path = candidate
                break
        if usd_path:
            break
    if not usd_path and usd_candidates:
        usd_path = sorted(usd_candidates)[0]
    return usd_path,init_path,occ_path

def find_blanket_prim_recursively(current_prim):
    """Collect USD prims whose names contain 'blanket' under a root prim."""
    blanket_prims = []
    prim_name = current_prim.GetName().lower()
    if "blanket" in prim_name:
        blanket_prims.append(current_prim)
        print(f"找到毛毯Prim：<{current_prim.GetPath()}>")

    for child_prim in current_prim.GetChildren():
        child_blanket_prims = find_blanket_prim_recursively(child_prim)
        blanket_prims.extend(child_blanket_prims)

    return blanket_prims

SceneScale = Union[float, tuple[float, float, float], list[float]]


def _normalize_scale_xyz(scale: SceneScale) -> tuple[float, float, float]:
    """将 scene_scale 统一为 (x, y, z)；标量则三轴同值。"""
    if isinstance(scale, (tuple, list)):
        if len(scale) != 3:
            raise ValueError(f"scene_scale 元组须为 (x, y, z)，got {scale!r}")
        return float(scale[0]), float(scale[1]), float(scale[2])
    s = float(scale)
    return s, s, s


def get_home_commercial_collision_child_prims(parent_prim, target_list):
    """home/commercial 场景：通过 collection:collisionmeshes 属性查找碰撞体。"""
    if not parent_prim.IsValid():
        return
    direct_children = parent_prim.GetAllChildren()
    for child in direct_children:
        if child.GetAttribute("collection:collisionmeshes").IsValid():
            target_list.append(child)
        get_home_commercial_collision_child_prims(child, target_list)


def get_clutter_collision_child_prims(parent_prim, target_list):
    """clutter 场景：递归查找名称含 collision 的子 prim。"""
    if not parent_prim.IsValid():
        return
    for child in parent_prim.GetAllChildren():
        if child.GetAttribute("physics:collisionEnabled").IsValid():
            target_list.append(child)
        get_clutter_collision_child_prims(child, target_list)


def _is_clutter_cylinder_collision(collision_prim) -> bool:
    """Return True when a clutter collision prim contains cylinder geometry."""
    if not collision_prim.IsValid():
        return False
    if collision_prim.IsA(UsdGeom.Cylinder):
        return True
    for child in collision_prim.GetAllChildren():
        if child.IsA(UsdGeom.Cylinder):
            return True
    return False


def _is_clutter_sphere_collision(collision_prim) -> bool:
    """Return True when a clutter collision prim contains sphere geometry."""
    if not collision_prim.IsValid():
        return False
    if collision_prim.IsA(UsdGeom.Sphere):
        return True
    for child in collision_prim.GetAllChildren():
        if child.IsA(UsdGeom.Sphere):
            return True
    return False


def _is_clutter_primitive_shape_collision(collision_prim) -> bool:
    """Detect simple clutter primitives that need dimension normalization."""
    return _is_clutter_cylinder_collision(collision_prim) or _is_clutter_sphere_collision(collision_prim)


def _translate_prim_along_world_z(prim, delta_world_z: float) -> None:
    """沿世界坐标 Z 平移 prim（delta 可正可负）。"""
    if abs(delta_world_z) < 1e-9 or not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return

    local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_pos = local_to_world.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    new_world_pos = Gf.Vec3d(world_pos[0], world_pos[1], world_pos[2] + delta_world_z)

    parent = prim.GetParent()
    if parent.IsValid():
        parent_xformable = UsdGeom.Xformable(parent)
        if parent_xformable:
            parent_l2w = parent_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            new_local = parent_l2w.GetInverse().Transform(new_world_pos)
        else:
            new_local = new_world_pos
    else:
        new_local = new_world_pos

    xform_api = UsdGeom.XformCommonAPI(prim)
    if xform_api:
        xform_api.SetTranslate(new_local)
        return

    translate_attr = prim.GetAttribute("xformOp:translate")
    if translate_attr.IsValid():
        translate_attr.Set(Gf.Vec3d(new_local[0], new_local[1], new_local[2]))
    else:
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "").Set(new_local)


CLUTTER_CYLINDER_MIN_TOP_Z = 1.0
CLUTTER_SPHERE_MAX_BOTTOM_Z = 0.15
CLUTTER_SPHERE_MIN_HEIGHT = 0.5


def _read_prim_world_z_extent(prim, bbox_cache: UsdGeom.BBoxCache):
    """Read bottom, top, and height of a prim in world Z coordinates."""
    bound = bbox_cache.ComputeWorldBound(prim)
    if bound.GetRange().IsEmpty():
        return None
    bottom_z = float(bound.GetRange().GetMin()[2])
    top_z = float(bound.GetRange().GetMax()[2])
    return bottom_z, top_z, top_z - bottom_z


def _adjust_clutter_sphere(prim, bbox_cache: UsdGeom.BBoxCache) -> None:
    """球体：底部离地 <= 0.15m，高度 >= 0.5m；不足则调整半径与位置。"""
    sphere = UsdGeom.Sphere(prim)
    radius_attr = sphere.GetRadiusAttr()
    if not radius_attr.HasValue():
        return

    for _ in range(6):
        dims = _read_prim_world_z_extent(prim, bbox_cache)
        if dims is None:
            return
        bottom_z, _top_z, height_world = dims
        changed = False

        if height_world < CLUTTER_SPHERE_MIN_HEIGHT - 1e-6:
            radius = radius_attr.Get()
            radius_attr.Set(radius * (CLUTTER_SPHERE_MIN_HEIGHT / max(height_world, 1e-8)))
            bbox_cache.Clear()
            changed = True
            continue

        if bottom_z > CLUTTER_SPHERE_MAX_BOTTOM_Z + 1e-6:
            _translate_prim_along_world_z(prim, CLUTTER_SPHERE_MAX_BOTTOM_Z - bottom_z)
            bbox_cache.Clear()
            changed = True
            continue

        if not changed:
            break


def _ensure_clutter_primitive_min_top_z(collision_prim) -> None:
    """检查圆柱/球体碰撞体尺寸；圆柱顶至少 1.0m，球体满足底部/高度约束。"""
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
    )
    prims_to_visit = [collision_prim]
    prims_to_visit.extend(collision_prim.GetAllChildren())

    for prim in prims_to_visit:
        if not prim.IsValid():
            continue
        is_cylinder = prim.IsA(UsdGeom.Cylinder)
        is_sphere = prim.IsA(UsdGeom.Sphere)
        if not is_cylinder and not is_sphere:
            continue

        if is_sphere:
            _adjust_clutter_sphere(prim, bbox_cache)
            continue

        bound = bbox_cache.ComputeWorldBound(prim)
        if bound.GetRange().IsEmpty():
            continue

        top_z = float(bound.GetRange().GetMax()[2])
        if top_z >= CLUTTER_CYLINDER_MIN_TOP_Z:
            continue

        bottom_z = float(bound.GetRange().GetMin()[2])
        height_world = top_z - bottom_z
        if height_world <= 1e-8:
            continue
        target_height_world = CLUTTER_CYLINDER_MIN_TOP_Z - bottom_z
        scale_up = target_height_world / height_world
        height_attr = UsdGeom.Cylinder(prim).GetHeightAttr()
        if height_attr.HasValue():
            height_attr.Set(height_attr.Get() * scale_up)


def adjust_usd_scale(
    prim_path: str = "/World/Scene/terrain",
    scale: SceneScale = 1.0,
    is_clutter: bool = False,
):
    """设置场景 terrain prim 的 USD 缩放；scale 可为标量或 (x, y, z) 元组。"""
    import omni
    from pxr import UsdGeom, Usd, Sdf, Gf
    stage = omni.usd.get_context().get_stage()

    scale_xyz = _normalize_scale_xyz(scale)

    scene_prim = stage.GetPrimAtPath(prim_path)
    if scene_prim.IsValid():
        print(f"Directly setting scale for prim: <{scene_prim.GetPath()}>")
        # 1. Get or create the scale attribute and set its value.
        scale_attr = scene_prim.GetAttribute("xformOp:scale")
        if not scale_attr:
            scale_attr = scene_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3, False)
        scale_attr.Set(Gf.Vec3d(*scale_xyz))
        # 2. Ensure 'xformOp:scale' is in the transformation order.
        order_attr = scene_prim.GetAttribute("xformOpOrder")
        if not order_attr.HasValue():
            # If order doesn't exist, create it with a default that includes scale.
            scene_prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, False).Set(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
        else:
            order = list(order_attr.Get())
            if "xformOp:scale" not in order:
                order.append("xformOp:scale")
                order_attr.Set(order)
        print(f"Successfully set scale for prim <{scene_prim.GetPath()}>")
    else:
        print("Warning: Could not find prim at /World/Scene to apply scale.")

    print(f"\nStart searching for blanket prim under <{scene_prim.GetPath()}>...")
    blanket_scope = find_blanket_prim_recursively(scene_prim)
    blanket_prim_list = []
    for blanket in blanket_scope:
        blanket_prim_list.extend(blanket.GetAllChildren())
    for blanket_prim in blanket_prim_list:
        if blanket_prim is not None:
            print(f"Found blanket prim: <{blanket_prim.GetPath()}>")
            blanket_scale_attr = blanket_prim.GetAttribute("xformOp:scale")
            if not blanket_scale_attr:
                blanket_scale_attr = blanket_prim.CreateAttribute(
                    "xformOp:scale", Sdf.ValueTypeNames.Double3, False
                )
                blanket_scale_attr.Set(Gf.Vec3d(1.0, 1.0, 1.0))

            current_blanket_scale = blanket_scale_attr.Get()
            target_blanket_scale = Gf.Vec3d(
                0.01,
                0.01,
                0.01
            )
            blanket_scale_attr.Set(target_blanket_scale)
            blanket_order_attr = blanket_prim.GetAttribute("xformOpOrder")
            if not blanket_order_attr.HasValue():
                blanket_prim.CreateAttribute(
                    "xformOpOrder", Sdf.ValueTypeNames.TokenArray, False
                ).Set(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
            else:
                blanket_order = list(blanket_order_attr.Get())
                if "xformOp:scale" not in blanket_order:
                    blanket_order.append("xformOp:scale")
                    blanket_order_attr.Set(blanket_order)
            print(f"Successfully corrected blanket prim: only Z-axis scale changed to 0")
            print(f"Blanket current scale: X={current_blanket_scale[0]}, Y={current_blanket_scale[1]}, Z=0.0")
        else:
            print(f"Did not find any blanket prim under <{scene_prim.GetPath()}>.")

    target_list = []
    if is_clutter:
        get_clutter_collision_child_prims(scene_prim, target_list)
        for collision_target in target_list:
            if _is_clutter_primitive_shape_collision(collision_target):
                _ensure_clutter_primitive_min_top_z(collision_target)
            collision_target.RemoveAPI(UsdPhysics.CollisionAPI)
            UsdPhysics.CollisionAPI.Apply(collision_target)
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_target)
            collision_api.CreateContactOffsetAttr().Set(0.03)  # important,不然人形会摔倒，性能上不去
    else:
        get_home_commercial_collision_child_prims(scene_prim, target_list)
        for collision_target in target_list:
            schema_list = collision_target.GetAppliedSchemas()
            if "PhysicsCollisionAPI" in schema_list:
                collision_target.RemoveAPI(UsdPhysics.CollisionAPI)
            if "PhysicsMeshCollisionAPI" in schema_list:
                collision_target.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            if "PhysxConvexHullCollisionAPI" in schema_list:
                collision_target.RemoveAPI(PhysxSchema.PhysxConvexHullCollisionAPI)
            if "PhysxConvexDecompositionCollisionAPI" in schema_list:
                collision_target.RemoveAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
            if "PhysxSDFMeshCollisionAPI" in schema_list:
                collision_target.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
            if "PhysxTriangleMeshCollisionAPI" in schema_list:
                collision_target.RemoveAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI)
            if "PhysxMeshMergeCollisionAPI" in schema_list:
                collision_target.RemoveAPI(PhysxSchema.PhysxMeshMergeCollisionAPI)

            UsdPhysics.CollisionAPI.Apply(collision_target)
            collisionMeshAPI = UsdPhysics.MeshCollisionAPI.Apply(collision_target)
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(collision_target)
            meshMergeCollision = PhysxSchema.PhysxMeshMergeCollisionAPI.Apply(collision_target)
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_target)
            collision_api.CreateContactOffsetAttr().Set(0.03)

# 具身形态到配置的映射

_EMBODIMENT_ROBOT_CFG = {'dingo': DINGO_CFG, 'unitree_g1': G1_CFG, 'unitree_go2': GO2_CFG}
_EMBODIMENT_CONTACT_CFG = {'dingo': DINGO_ContactCfg, 'unitree_g1': G1_ContactCfg, 'unitree_go2': GO2_ContactCfg}
_EMBODIMENT_CAMERA_CFG = {'dingo': DINGO_CameraCfg, 'unitree_g1': G1_CameraCfg, 'unitree_go2': GO2_CameraCfg}
_EMBODIMENT_BIRDEYE_CAMERA_CFG = {
    'dingo': DINGO_BirdEye_CameraCfg,
    'unitree_g1': G1_BirdEye_CameraCfg,
    'unitree_go2': GO2_BirdEye_CameraCfg,
}
_EMBODIMENT_ACTION_CFG = {'dingo': BaseVelocityActionsCfg, 'unitree_g1': BasePositionActionsCfg, 'unitree_go2': QuadrupedPositionActionsCfg}
_EMBODIMENT_OBSERVATION_CFG = {'dingo': PointNavObservationsDingoCfg, 'unitree_g1': PointNavObservationsUnitreeG1Cfg, 'unitree_go2': PointNavObservationsUnitreeGo2Cfg}
_EMBODIMENT_EVAL_OBSERVATION_CFG = {
    'dingo': PointNavEvalObservationsDingoCfg,
    'unitree_g1': PointNavEvalObservationsUnitreeG1Cfg,
    'unitree_go2': PointNavEvalObservationsUnitreeGo2Cfg,
}
_EMBODIMENT_HEIGHT_OFFSET_CFG = {'dingo': 0.1, 'unitree_g1': 0.74, 'unitree_go2': 0.4}
HOME_COMMERCIAL_SCENE_SCALE = (0.01, 0.01, 0.01)
HOME_COMMERCIAL_SCALE_FACTOR = 1.0

_EMBODIMENT_CLUTTER_SCENE_SCALE = {
    'dingo': 0.5,
    'unitree_go2': 0.5,
    'unitree_g1': 0.75,
}


def resolve_scene_scale(scene_type: str, embodiment: str) -> tuple[tuple[float, float, float], float]:
    """按场景类型与 embodiment 返回 (scene_scale_xyz, scale_factor)。"""
    if str(scene_type).startswith('cluttered'):
        if embodiment not in _EMBODIMENT_CLUTTER_SCENE_SCALE:
            raise ValueError(
                f"embodiment 必须是 {list(_EMBODIMENT_CLUTTER_SCENE_SCALE.keys())} 之一，当前为 {embodiment}"
            )
        scale = _EMBODIMENT_CLUTTER_SCENE_SCALE[embodiment]
        return (scale, scale, scale), scale
    if scene_type in ('home', 'commercial'):
        return HOME_COMMERCIAL_SCENE_SCALE, HOME_COMMERCIAL_SCALE_FACTOR
    raise ValueError(f"未知 scene_type: {scene_type}")


def _make_dingo_controller(device, controller_config=None):
    """Build the wheel-speed controller for Dingo."""
    config = dict(controller_config or {})
    config.pop("type", None)
    config.pop("device", None)
    config.setdefault("name", "simple_control")
    config.setdefault("wheel_radius", DINGO_WHEEL_RADIUS)
    config.setdefault("wheel_base", DINGO_WHEEL_BASE)
    return DifferentialController(**config)


def _make_g1_controller(device, controller_config=None):
    """Build the learned velocity controller for Unitree G1."""
    config = dict(controller_config or {})
    config.pop("type", None)
    config.pop("device", None)
    config.setdefault("name", "g1_control")
    return G1VelocityController(device=device, **config)


def _make_go2_controller(device, controller_config=None):
    """Build the learned velocity controller for Unitree Go2."""
    config = dict(controller_config or {})
    config.pop("type", None)
    config.pop("device", None)
    config.setdefault("name", "go2_control")
    return Go2VelocityController(device=device, **config)

_EMBODIMENT_CONTROLLER_CFG = {'dingo': _make_dingo_controller, 'unitree_g1': _make_g1_controller, 'unitree_go2': _make_go2_controller}

def create_dingonav_environment(scene_list: list,
                                scene_index: int,
                                num_envs: int,
                                scene_scale: SceneScale = 1.0,
                                device: str = 'cuda:0',
                                embodiment: str = 'dingo'):
    """
    Create DingoNav environment using scene_list[scene_index].
    Args:
        embodiment: 具身形态，可选 'dingo'（轮式）或 'unitree_g1'（人形）
    """
    if embodiment not in _EMBODIMENT_ROBOT_CFG:
        raise ValueError(f"embodiment 必须是 {list(_EMBODIMENT_ROBOT_CFG.keys())} 之一，当前为 {embodiment}")
    scene_data = scene_list[scene_index]
    usd_path = scene_data['usd_path']
    init_path = scene_data['pointgoal_path']
    occ_path = scene_data['esdf_path']
    scene_type = scene_data.get('scene_type', '')
    is_clutter = str(scene_type).startswith('cluttered')
    scene_scale_value, scale_factor = resolve_scene_scale(scene_type, embodiment)
    if scene_scale is not None:
        scene_scale_value = _normalize_scale_xyz(scene_scale)
        if is_clutter and len(set(scene_scale_value)) == 1:
            scale_factor = scene_scale_value[0]
    print(
        f"[create_dingonav_environment] scene_type={scene_type}, scene_scale={scene_scale_value}, "
        f"scale_factor={scale_factor}, embodiment={embodiment}"
    )

    scene_config = PointNavSceneCfg()
    scene_config.num_envs = num_envs
    scene_config.env_spacing = 0.0
    scene_config.terrain = BENCH_TERRAIN_CFG
    scene_config.terrain.usd_path = usd_path
    scene_config.goal = GOAL_CFG
    scene_config.robot = _EMBODIMENT_ROBOT_CFG[embodiment]
    scene_config.contact_sensor = _EMBODIMENT_CONTACT_CFG[embodiment]
    scene_config.camera_sensor = _EMBODIMENT_CAMERA_CFG[embodiment]

    env_config = DingoRLPointNavOffStage1Cfg()
    env_config.actions = _EMBODIMENT_ACTION_CFG[embodiment]()
    env_config.observations = _EMBODIMENT_OBSERVATION_CFG[embodiment]()
    env_config.decimation = _EMBODIMENT_DECIMATION_CFG[embodiment]
    env_config.scene = scene_config
    env_config.sim.device = device

    env_config.events.reset_pose.params = {"init_point_path": init_path,
                                           "global_occ_path": occ_path,
                                           'height_offset': _EMBODIMENT_HEIGHT_OFFSET_CFG[embodiment],
                                           'robot_visible': False,
                                           'light_enabled': False,
                                           'embodiment': embodiment,
                                           'scale_factor': scale_factor}

    env = ManagerBasedRLEnv(env_config)
    env = RslRlVecEnvWrapper(env)
    adjust_usd_scale(scale=scene_scale_value, is_clutter=is_clutter)
    controller = _EMBODIMENT_CONTROLLER_CFG[embodiment](device)
    print("current rank={}".format(int(os.environ['RANK'])))
    return env, controller

def create_dingoeval_environment(scene_dir: str,
                                scene_index: int,
                                num_envs: int,
                                scene_scale: SceneScale | None = None,
                                device: str = 'cuda:0',
                                embodiment: str = 'dingo',
                                scene_data: dict | None = None,
                                controller_config: dict | None = None):
    """
    Args:
        embodiment: 具身形态，可选 'dingo'（轮式）或 'unitree_g1'（人形）
    """
    if embodiment not in _EMBODIMENT_ROBOT_CFG:
        raise ValueError(f"embodiment must be {list(_EMBODIMENT_ROBOT_CFG.keys())} 之一，当前为 {embodiment}")
    if scene_data is None:
        scene_names = sorted(
            name for name in os.listdir(scene_dir)
            if os.path.isdir(os.path.join(scene_dir, name))
        )
        scene_path = os.path.join(scene_dir, scene_names[scene_index])
        usd_path, init_path, occ_path = find_usd_path(scene_path, 'pointgoal')
        is_clutter = 'cluttered' in scene_dir.lower()
        scene_type = 'cluttered' if is_clutter else 'home'
    else:
        usd_path = scene_data['usd_path']
        init_path = scene_data['pointgoal_path']
        occ_path = scene_data['esdf_path']
        scene_type = scene_data.get('scene_type', 'home')
        is_clutter = str(scene_type).startswith('cluttered')

    scene_scale_value, scale_factor = resolve_scene_scale(scene_type, embodiment)
    if scene_scale is not None:
        scene_scale_value = _normalize_scale_xyz(scene_scale)
        if is_clutter and len(set(scene_scale_value)) == 1:
            scale_factor = scene_scale_value[0]
    height_offset = _EMBODIMENT_HEIGHT_OFFSET_CFG[embodiment]
    scene_config = PointNavEvalSceneCfg()
    scene_config.num_envs = num_envs
    scene_config.env_spacing = 0.0
    scene_config.terrain = BENCH_TERRAIN_CFG
    scene_config.terrain.usd_path = usd_path
    scene_config.goal = GOAL_CFG
    scene_config.robot = _EMBODIMENT_ROBOT_CFG[embodiment]
    scene_config.contact_sensor = _EMBODIMENT_CONTACT_CFG[embodiment]
    scene_config.camera_sensor = _EMBODIMENT_CAMERA_CFG[embodiment]
    scene_config.birdeye_camera = _EMBODIMENT_BIRDEYE_CAMERA_CFG[embodiment]

    env_config = DingoEvalPointNavCfg()
    env_config.actions = _EMBODIMENT_ACTION_CFG[embodiment]()
    env_config.observations = _EMBODIMENT_EVAL_OBSERVATION_CFG[embodiment]()
    env_config.decimation = _EMBODIMENT_DECIMATION_CFG[embodiment]
    env_config.scene = scene_config
    env_config.sim.device = device
    env_config.events.reset_pose.params = {"init_point_path": init_path,
                                           "global_occ_path": occ_path,
                                           'height_offset': height_offset,
                                           'robot_visible': True,
                                           'light_enabled': False,
                                           'embodiment': embodiment,
                                           'scale_factor': scale_factor}

    env = ManagerBasedRLEnv(env_config)
    env = RslRlVecEnvWrapper(env)
    adjust_usd_scale(scale=scene_scale_value, is_clutter=is_clutter)
    controller = _EMBODIMENT_CONTROLLER_CFG[embodiment](device, controller_config)
    print("current rank={}".format(int(os.environ.get('LOCAL_RANK', '0'))))
    return env, controller
