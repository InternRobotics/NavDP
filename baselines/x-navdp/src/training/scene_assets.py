"""Scene asset discovery and mixing utilities."""

import json
import os

CLUTTER_EASY_LIMIT = 6
SCENE_MIX_ORDER = ("cluttered_hard", "cluttered_easy", "commercial", "home")

# Scenes with known asset or navigation issues are skipped without mutating the dataset.
EXCLUDED_SCENE_SUBSTRINGS = (
    "MWHLEPQKTIFZIAABAAAAAAA8",
    "MWAX5JYKTKJZ2AABAAAAAAQ8",
)


def first_existing_path(paths):
    """Return the first existing path from an ordered candidate list."""
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def find_usd_path(scene_dir: str, usd_variant: str = "navigation") -> str | None:
    """Find the preferred USD file inside one scene directory."""
    variant_candidates = {
        "navigation": ["start_result_navigation.usd", "start_result_raw.usd", "start_result_interaction.usd"],
        "raw": ["start_result_raw.usd", "start_result_navigation.usd", "start_result_interaction.usd"],
        "interaction": ["start_result_interaction.usd", "start_result_navigation.usd", "start_result_raw.usd"],
        "auto": ["start_result_navigation.usd", "start_result_raw.usd", "start_result_interaction.usd"],
    }
    if usd_variant not in variant_candidates:
        raise ValueError(f"usd_variant must be one of {list(variant_candidates)}, got {usd_variant!r}")

    found = first_existing_path(os.path.join(scene_dir, name) for name in variant_candidates[usd_variant])
    if found:
        return found
    if not os.path.isdir(scene_dir):
        return None

    usd_files = [
        f for f in sorted(os.listdir(scene_dir))
        if f.endswith(".usd") and "noMDL" not in f and "scale" not in f
    ]
    return os.path.join(scene_dir, usd_files[0]) if usd_files else None


def scene_name_allowed(scene_name: str) -> bool:
    """Filter known problematic scenes without changing dataset files."""
    return not any(excluded in scene_name for excluded in EXCLUDED_SCENE_SUBSTRINGS)


def append_scene_entry(
    scene_data_list: list,
    *,
    scene_name: str,
    scene_type: str,
    usd_path: str | None,
    esdf_path: str | None,
    pointgoal_path: str | None,
) -> None:
    """Append a scene only when all files required by the training env exist."""
    missing = [
        name for name, value in (
            ("usd", usd_path if usd_path and os.path.exists(usd_path) else None),
            ("esdf", esdf_path if esdf_path and os.path.exists(esdf_path) else None),
            ("pointgoal", pointgoal_path if pointgoal_path and os.path.exists(pointgoal_path) else None),
        )
        if value is None
    ]
    if missing:
        print(f"[scene_loader] skip {scene_type}/{scene_name}: missing {', '.join(missing)}")
        return

    scene_data_list.append(
        {
            "scene_name": scene_name,
            "scene_type": scene_type,
            "usd_path": usd_path,
            "esdf_path": esdf_path,
            "pointgoal_path": pointgoal_path,
        }
    )


def _read_scene_split(split_file: str | None) -> dict | None:
    if not split_file or not os.path.exists(split_file):
        return None
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _split_scene_items(
    split_file: str | None,
    split: str,
    scene_type: str,
    scenes_dir: str,
) -> list[tuple[str, str | None]]:
    """Return (scene_name, split_name) items for train/eval aware metadata lookup."""
    if split not in ("train", "eval", "all"):
        raise ValueError(f"--scene_split must be train/eval/all, got {split!r}")

    scene_split = _read_scene_split(split_file)
    if scene_split:
        suffixes = ["train", "eval"] if split == "all" else [split]
        items = []
        for suffix in suffixes:
            items.extend((scene_name, suffix) for scene_name in scene_split.get(f"{scene_type}_{suffix}", []))
        return items

    if not os.path.isdir(scenes_dir):
        return []
    return [
        (name, None)
        for name in sorted(os.listdir(scenes_dir))
        if os.path.isdir(os.path.join(scenes_dir, name))
    ]


def find_split_pointgoal_path(pointgoal_dir: str, split_name: str | None) -> str | None:
    """Find the point-goal file matching the current metadata contract."""
    if split_name == "eval":
        preferred_names = ("pointgoal_start_goal_pairs.npy",)
    elif split_name == "train":
        preferred_names = ("pointgoal_start_pair_samples_safe.npy",)
    else:
        preferred_names = (
            "pointgoal_start_pair_samples_safe.npy",
            "pointgoal_start_goal_pairs.npy",
            "pointgoal_start_pair_samples.npy",
        )
    return first_existing_path(os.path.join(pointgoal_dir, name) for name in preferred_names)


def _internscene_dirs(scene_type: str) -> tuple[str, str]:
    if scene_type == "home":
        return "internscenes_home", "scenes_home"
    return "internscenes_commercial", "scenes_commercial"


def load_grscenes_split_data(
    grscenes_dir: str,
    dataset_dir: str,
    split_file: str,
    split: str = "train",
    usd_variant: str = "navigation",
) -> list[dict]:
    """Load the older split layout with USDs under GRScenes-100/scenes."""
    scene_data_list = []
    for scene_type in ("commercial", "home"):
        subset_dir, _ = _internscene_dirs(scene_type)
        scenes_dir = os.path.join(grscenes_dir, "scenes")
        for scene_name, split_name in _split_scene_items(split_file, split, scene_type, scenes_dir):
            if not scene_name_allowed(scene_name):
                continue
            append_scene_entry(
                scene_data_list,
                scene_name=scene_name,
                scene_type=scene_type,
                usd_path=find_usd_path(os.path.join(scenes_dir, scene_name), usd_variant=usd_variant),
                esdf_path=os.path.join(dataset_dir, subset_dir, "esdf", scene_name, "navigable.ply"),
                pointgoal_path=find_split_pointgoal_path(
                    os.path.join(dataset_dir, subset_dir, "pointgoal_start_pair", scene_name),
                    split_name,
                ),
            )

    return scene_data_list


def load_x_navdp_scene_layout(
    data_root: str,
    split_file: str | None = None,
    split: str = "train",
    usd_variant: str = "navigation",
) -> list[dict]:
    """Load the current NavDP-style single-root layout used for release assets."""
    scene_data_list = []
    metadata_root = os.path.join(data_root, "navigation_metadata")
    for scene_type in ("commercial", "home"):
        subset_dir, scenes_dir_name = _internscene_dirs(scene_type)
        scenes_dir = os.path.join(data_root, subset_dir, scenes_dir_name)
        esdf_dir = os.path.join(metadata_root, subset_dir, "esdf")
        pointgoal_dir = os.path.join(metadata_root, subset_dir, "pointgoal_start_pair")
        if not os.path.isdir(scenes_dir):
            continue
        for scene_name, split_name in _split_scene_items(split_file, split, scene_type, scenes_dir):
            scene_dir = os.path.join(scenes_dir, scene_name)
            if not os.path.isdir(scene_dir) or not scene_name_allowed(scene_name):
                continue
            append_scene_entry(
                scene_data_list,
                scene_name=scene_name,
                scene_type=scene_type,
                usd_path=find_usd_path(scene_dir, usd_variant=usd_variant),
                esdf_path=os.path.join(esdf_dir, scene_name, "navigable.ply"),
                pointgoal_path=find_split_pointgoal_path(os.path.join(pointgoal_dir, scene_name), split_name),
            )

    return scene_data_list


def load_legacy_internscenes_data(data_root: str, usd_variant: str = "auto") -> list[dict]:
    """Load the older layout where USDs also live under dataset/internscenes_*_train."""
    scene_data_list = []
    for scene_type, subset_dir, scenes_dir_name in (
        ("commercial", "internscenes_commercial", "scenes_commercial_train"),
        ("home", "internscenes_home", "scenes_home_train"),
    ):
        scenes_dir = os.path.join(data_root, subset_dir, scenes_dir_name)
        esdf_dir = os.path.join(data_root, subset_dir, "esdf")
        pointgoal_dir = os.path.join(data_root, subset_dir, "pointgoal_start_pair")
        if not os.path.isdir(scenes_dir):
            continue
        for scene_name in sorted(os.listdir(scenes_dir)):
            scene_usd_dir = os.path.join(scenes_dir, scene_name)
            if not os.path.isdir(scene_usd_dir) or not scene_name_allowed(scene_name):
                continue
            append_scene_entry(
                scene_data_list,
                scene_name=scene_name,
                scene_type=scene_type,
                usd_path=find_usd_path(scene_usd_dir, usd_variant=usd_variant),
                esdf_path=os.path.join(esdf_dir, scene_name, "navigable.ply"),
                pointgoal_path=find_split_pointgoal_path(os.path.join(pointgoal_dir, scene_name), "train"),
            )

    return scene_data_list


def load_clutter_data(dataset_dir: str, usd_variant: str = "auto") -> list[dict]:
    """Load clutter scenes from the scene folder and point-goals from navigation_metadata."""
    scene_data_list = []
    metadata_root = os.path.join(dataset_dir, "navigation_metadata")
    for clutter_type in ("cluttered_easy", "cluttered_hard"):
        clutter_dir = os.path.join(dataset_dir, clutter_type)
        pointgoal_root = os.path.join(metadata_root, clutter_type, "pointgoal_start_pair")
        if not os.path.isdir(clutter_dir):
            continue
        for scene_name in sorted(os.listdir(clutter_dir)):
            scene_dir = os.path.join(clutter_dir, scene_name)
            if not os.path.isdir(scene_dir) or not scene_name_allowed(scene_name):
                continue
            pointgoal_path = find_split_pointgoal_path(
                os.path.join(pointgoal_root, scene_name),
                "train",
            ) or first_existing_path(
                [
                    os.path.join(scene_dir, "pointgoal_start_pair_samples.npy"),
                    os.path.join(scene_dir, "pointgoal_start_goal_pairs.npy"),
                ]
            )
            append_scene_entry(
                scene_data_list,
                scene_name=scene_name,
                scene_type=clutter_type,
                usd_path=find_usd_path(scene_dir, usd_variant=usd_variant),
                esdf_path=os.path.join(scene_dir, "occupancy.ply"),
                pointgoal_path=pointgoal_path,
            )

    return scene_data_list


def _interleave_clutter_with_internscenes(clutter_scenes: list[dict], intern_scenes: list[dict]) -> list[dict]:
    """Evenly place clutter scenes among home/commercial scenes while preserving per-group order."""
    if not clutter_scenes:
        return list(intern_scenes)
    if not intern_scenes:
        return list(clutter_scenes)

    total = len(clutter_scenes) + len(intern_scenes)
    clutter_positions = {
        min(total - 1, int((idx + 0.5) * total / len(clutter_scenes)))
        for idx in range(len(clutter_scenes))
    }
    mixed = []
    clutter_idx = 0
    intern_idx = 0
    for pos in range(total):
        if pos in clutter_positions and clutter_idx < len(clutter_scenes):
            mixed.append(clutter_scenes[clutter_idx])
            clutter_idx += 1
        elif intern_idx < len(intern_scenes):
            mixed.append(intern_scenes[intern_idx])
            intern_idx += 1
        else:
            mixed.append(clutter_scenes[clutter_idx])
            clutter_idx += 1
    return mixed


def load_and_mix_scene_data(
    base_data_dir: str,
    dataset_dir: str | None = None,
    scene_split_file: str | None = None,
    split: str = "train",
    usd_variant: str = "navigation",
    uniform_scene_mix: bool = False,
):
    """Build the training scene list from clutter data plus InternScenes home/commercial data."""
    data_root = dataset_dir or base_data_dir
    scene_split_file = scene_split_file or os.path.join(data_root, "scene_split.json")

    scene_data_list = load_clutter_data(data_root)
    if os.path.isdir(os.path.join(data_root, "internscenes_home", "scenes_home")) or os.path.isdir(
        os.path.join(data_root, "internscenes_commercial", "scenes_commercial")
    ):
        scene_data_list.extend(
            load_x_navdp_scene_layout(
                data_root,
                split_file=scene_split_file,
                split=split,
                usd_variant=usd_variant,
            )
        )
    elif scene_split_file and os.path.exists(scene_split_file) and os.path.isdir(os.path.join(base_data_dir, "scenes")):
        scene_data_list.extend(
            load_grscenes_split_data(
                base_data_dir,
                data_root,
                scene_split_file,
                split=split,
                usd_variant=usd_variant,
            )
        )
    else:
        scene_data_list.extend(load_legacy_internscenes_data(data_root, usd_variant=usd_variant))

    scenes_by_type = {
        scene_type: [s for s in scene_data_list if s["scene_type"] == scene_type]
        for scene_type in SCENE_MIX_ORDER
    }
    clutter_scenes = scenes_by_type["cluttered_hard"] + scenes_by_type["cluttered_easy"][:CLUTTER_EASY_LIMIT]
    intern_scenes = scenes_by_type["commercial"] + scenes_by_type["home"]
    if uniform_scene_mix:
        mixed_scene_list = _interleave_clutter_with_internscenes(clutter_scenes, intern_scenes)
    else:
        mixed_scene_list = clutter_scenes + intern_scenes
    for scene_order, scene_item in enumerate(mixed_scene_list):
        scene_item["scene_order"] = scene_order

    return mixed_scene_list


def scene_output_tag(scene_entry: dict) -> str:
    """Build a stable output tag that includes scene order and scene name."""
    name = scene_entry["scene_name"]
    order = scene_entry.get("scene_order")
    if order is None:
        return name
    return f"{order}_{name}"


def is_clutter_scene(scene_list: list, scene_index: int) -> bool:
    """Return whether the selected scene uses the clutter reward branch."""
    return str(scene_list[scene_index].get("scene_type", "")).startswith("cluttered")
