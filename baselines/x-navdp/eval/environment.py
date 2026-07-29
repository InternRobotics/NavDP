"""Evaluation environment adapter used by point-goal scripts."""

import json
import os

from src.training.scene_assets import find_usd_path
from src.utils import BatchMPCController


class BatchMPCNEWController(BatchMPCController):
    """Compatibility alias for the legacy evaluation script."""

    def __init__(self, batch=1, **kwargs):
        super().__init__(batch=batch, **kwargs)


def namespace_to_dict(obj):
    """Convert argparse-style namespaces from eval YAML into plain dictionaries."""
    if hasattr(obj, "__dict__"):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    return obj


def find_eval_pointgoal_path(scene_pointgoal_dir: str) -> str | None:
    """Return the point-goal file used by evaluation, excluding training-safe files."""
    preferred_names = (
        "pointgoal_start_goal_pairs.npy",
        "pointgoal_start_pair_samples.npy",
    )
    for name in preferred_names:
        path = os.path.join(scene_pointgoal_dir, name)
        if os.path.isfile(path):
            return path

    if not os.path.isdir(scene_pointgoal_dir):
        return None
    candidates = [
        os.path.join(scene_pointgoal_dir, name)
        for name in sorted(os.listdir(scene_pointgoal_dir))
        if name.endswith(".npy")
        and "pointgoal" in name.lower()
        and "safe" not in name.lower()
    ]
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple evaluation point-goal files found in {scene_pointgoal_dir}: "
            f"{', '.join(os.path.basename(path) for path in candidates)}"
        )
    return candidates[0] if candidates else None


def _scene_entries(cfg) -> list[dict]:
    """Build evaluation scenes without changing the configured list or order."""
    scene_dir = cfg.environment.scene_dir
    dataset_dir = getattr(cfg.environment, "dataset_dir", None)
    scene_type = getattr(cfg.environment, "scene_type", None)

    if dataset_dir:
        scene_type = scene_type or "home"
        if scene_type == "home":
            scene_subdir = os.path.join("internscenes_home", "scenes_home")
        elif scene_type == "commercial":
            scene_subdir = os.path.join("internscenes_commercial", "scenes_commercial")
        else:
            raise ValueError(f"Unsupported metadata-backed evaluation scene_type: {scene_type!r}")

        split_file = getattr(
            cfg.environment,
            "scene_split_file",
            os.path.join(scene_dir, "scene_split.json"),
        )
        if not os.path.isfile(split_file):
            split_file = os.path.join(os.path.dirname(dataset_dir), "scene_split.json")
        split = getattr(cfg.environment, "scene_split", "eval")
        if split not in ("train", "eval"):
            raise ValueError(f"environment.scene_split must be 'train' or 'eval', got {split!r}")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(
                f"Evaluation requires a scene split file, but none was found at {split_file}"
            )
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        split_key = f"{scene_type}_{split}"
        if split_key not in split_data:
            raise KeyError(f"Scene split {split_key!r} is missing from {split_file}")
        scene_names = list(split_data[split_key])

        if os.path.isdir(os.path.join(scene_dir, scene_subdir)):
            scene_root = os.path.join(scene_dir, scene_subdir)
        else:
            scene_root = scene_dir
        esdf_root = os.path.join(dataset_dir, "esdf")
        pointgoal_root = os.path.join(dataset_dir, "pointgoal_start_pair")
        if not os.path.isdir(esdf_root) or not os.path.isdir(pointgoal_root):
            raise FileNotFoundError(
                f"Evaluation metadata is incomplete under {dataset_dir}: "
                "expected esdf/ and pointgoal_start_pair/"
            )
    else:
        scene_root = scene_dir
        scene_type = scene_type or ("cluttered" if "cluttered" in scene_dir.lower() else "home")
        scene_names = sorted(
            name for name in os.listdir(scene_root)
            if os.path.isdir(os.path.join(scene_root, name))
        )

    entries = []
    for scene_name in scene_names:
        scene_path = os.path.join(scene_root, scene_name)
        if dataset_dir:
            esdf_path = os.path.join(dataset_dir, "esdf", scene_name, "navigable.ply")
            pointgoal_path = find_eval_pointgoal_path(
                os.path.join(dataset_dir, "pointgoal_start_pair", scene_name)
            )
            usd_variant = getattr(cfg.environment, "usd_variant", "navigation")
        else:
            esdf_path = os.path.join(scene_path, "occupancy.ply")
            pointgoal_path = find_eval_pointgoal_path(scene_path)
            usd_variant = "auto"

        scene_data = {
            "scene_name": scene_name,
            "scene_type": scene_type,
            "usd_path": find_usd_path(scene_path, usd_variant),
            "esdf_path": esdf_path,
            "pointgoal_path": pointgoal_path,
        }
        entries.append(scene_data)

    return entries


def _resolve_scene(cfg, scene_index: int) -> tuple[dict, str]:
    scene_data_list = _scene_entries(cfg)
    if not scene_data_list:
        raise ValueError(f"No evaluation scenes are configured for {cfg.environment.scene_dir}")
    if not 0 <= scene_index < len(scene_data_list):
        raise IndexError(
            f"scene_index {scene_index} is out of range for "
            f"{len(scene_data_list)} configured evaluation scenes"
        )

    scene_data = scene_data_list[scene_index]
    scene_name = scene_data["scene_name"]
    missing = [
        key for key in ("usd_path", "esdf_path", "pointgoal_path")
        if not scene_data[key] or not os.path.exists(scene_data[key])
    ]
    if missing:
        raise FileNotFoundError(
            f"Configured evaluation scene {scene_name!r} is missing required assets: "
            f"{', '.join(missing)}. The scene list and index were left unchanged."
        )
    return scene_data, scene_name


def create_environment(cfg, scene_index: int, device: str):
    """Create the X-NavDP Isaac evaluation environment from an eval config."""
    from src.environment import create_dingoeval_environment

    scene_data, scene_name = _resolve_scene(cfg, scene_index)
    controller_config = namespace_to_dict(cfg.controller) if hasattr(cfg, "controller") else None
    expected_controller_types = {
        "dingo": "differential",
        "unitree_g1": "unitree_g1",
        "unitree_go2": "unitree_go2",
    }
    embodiment = cfg.environment.embodiment
    expected_type = expected_controller_types.get(embodiment)
    configured_type = controller_config.get("type") if controller_config else None
    if expected_type is None:
        raise ValueError(f"Unsupported evaluation embodiment: {embodiment!r}")
    if configured_type is not None and configured_type != expected_type:
        raise ValueError(
            f"controller.type {configured_type!r} is incompatible with "
            f"environment.embodiment {embodiment!r}; expected {expected_type!r}"
        )
    env, controller = create_dingoeval_environment(
        scene_dir=cfg.environment.scene_dir,
        scene_index=0,
        num_envs=cfg.environment.num_envs,
        scene_scale=getattr(cfg.environment, "scene_scale", None),
        device=device,
        embodiment=embodiment,
        scene_data=scene_data,
        controller_config=controller_config,
    )
    return env, controller, scene_name
