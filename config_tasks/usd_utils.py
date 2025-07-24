from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdLux, Gf
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import create_prim
import torch
# 辅助函数：设置实体不可见
def hide_entity(prim_path: str):
    prim = get_prim_at_path(prim_path)
    if prim.IsValid():
        prim.GetAttribute("visibility").Set("invisible")
    else:
        print(f"警告: 找不到路径为 {prim_path} 的prim")

def add_point_light(
    position: torch.Tensor,
    intensity: float = 20000.0,
    color: tuple = (1.0, 1.0, 1.0),
    radius: float = 0.1,
    prim_path: str = None
) -> UsdLux.SphereLight:

    if isinstance(position, torch.Tensor):
        position = position.cpu().numpy()

    if prim_path is None:
        prim_path = "/World/Lights/point_light"
        count = 0
        while create_prim(prim_path).IsValid():
            count += 1
            prim_path = f"/World/Lights/point_light_{count}"
    
    light_prim = create_prim(
        prim_path=prim_path,
        prim_type="SphereLight"
    )
    
    point_light = UsdLux.SphereLight(light_prim)
    point_light.CreateIntensityAttr(intensity)
    point_light.CreateColorAttr(Gf.Vec3f(*color))
    point_light.CreateRadiusAttr(radius)
    
    xform_prim = XFormPrim(prim_path)
    xform_prim.set_world_pose(position=position)
    
    return point_light