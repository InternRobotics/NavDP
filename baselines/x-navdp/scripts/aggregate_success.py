#!/usr/bin/env python3
"""Aggregate X-NavDP training-time evaluation success logs.

Each input row is expected to follow the format written by ``train.py``::

    rank,step,episode,success,trainer_success_rate

Global and per-embodiment curves use the boolean ``success`` column. Per-scene
curves prefer ``trainer_success_rate`` when it is available, matching the
original experiment analysis. Events from all matching files are merged by
``(step, filename, row)`` before applying one exponential moving average.
"""

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENE_CATEGORIES = ("home_commercial", "clutter")
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "result"
MILESTONE_INTERVAL = 1250
MILESTONES = list(range(MILESTONE_INTERVAL, 200001, MILESTONE_INTERVAL))
MAX_MILESTONE_STEP_DIFF = MILESTONE_INTERVAL
EMA_ALPHA_NEW = 0.01
EMA_ALPHA_OLD = 1.0 - EMA_ALPHA_NEW
EMA_SCENE_ALPHA_NEW = 0.01
EMA_SCENE_ALPHA_OLD = 1.0 - EMA_SCENE_ALPHA_NEW
MAKE_PLOTS = True

_EMBODIMENT_TOKENS = ("unitree_g1", "unitree_go2", "dingo")
_CLUTTER_SCENE_RE = re.compile(r"^\d+_(?:hard|easy)_\d+$")


def _parse_usd_embodiment_and_scene(stem: str) -> tuple[str | None, str | None]:
    """home/commercial: 10_..._usd_unitree_go2_10"""
    if "_usd_" not in stem:
        return None, None
    left, right = stem.split("_usd_", 1)
    i = right.rfind("_")
    if i <= 0:
        return None, None
    emb, suffix = right[:i], right[i + 1 :]
    if not suffix.isdigit():
        return None, None
    return emb, left


def _parse_clutter_embodiment_and_scene(stem: str) -> tuple[str | None, str | None]:
    """clutter: 0_hard_0_dingo_0 / 12_easy_2_unitree_g1_12"""
    for emb in _EMBODIMENT_TOKENS:
        marker = f"_{emb}_"
        pos = stem.rfind(marker)
        if pos <= 0:
            continue
        scene = stem[:pos]
        tail = stem[pos + 1 :]
        rank = tail.rsplit("_", 1)[-1]
        if tail != f"{emb}_{rank}" or not rank.isdigit():
            continue
        if _CLUTTER_SCENE_RE.fullmatch(scene):
            return emb, scene
    return None, None


def parse_embodiment_and_scene(filename: str) -> tuple[str | None, str | None]:
    """
    解析 embodiment 与场景标签：
    - home/commercial: 10_MV7J6..._usd_unitree_go2_10.txt → scene=10_MV7J6...
    - clutter hard/easy: 0_hard_0_dingo_0.txt → scene=0_hard_0
    """
    stem = Path(filename).stem
    emb, scene = _parse_usd_embodiment_and_scene(stem)
    if emb is not None and scene is not None:
        return emb, scene
    return _parse_clutter_embodiment_and_scene(stem)


def parse_embodiment(filename: str) -> str | None:
    """从文件名解析 embodiment（兼容旧接口）。"""
    emb, _ = parse_embodiment_and_scene(filename)
    return emb


def parse_embodiment_classic(filename: str) -> str | None:
    """解析 embodiment（含 home/commercial 与 clutter hard/easy）。"""
    return parse_embodiment(filename)


def get_scene_category(filename: str) -> str | None:
    """返回 home_commercial 或 clutter。"""
    stem = Path(filename).stem
    if _parse_usd_embodiment_and_scene(stem)[0] is not None:
        return "home_commercial"
    if _parse_clutter_embodiment_and_scene(stem)[0] is not None:
        return "clutter"
    return None


def category_output_dir(category: str) -> Path:
    return RESULT_DIR / category


def _parse_success_cell_classic(cell: str) -> bool | None:
    """与 aggregate_txt_success.py 一致（仅 True/False、1.0/0.0）。"""
    s = cell.strip().lower()
    if s == "true" or s == "1.0":
        return True
    if s == "false" or s == "0.0":
        return False
    return None


def parse_file_events_classic(filepath: Path) -> list[tuple[int, bool]]:
    """与 aggregate_txt_success.py 一致：只用第 4 列布尔。"""
    out: list[tuple[int, bool]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                try:
                    step_num = int(parts[1])
                except (ValueError, IndexError):
                    continue
                ok = _parse_success_cell_classic(parts[3])
                if ok is None:
                    continue
                out.append((step_num, ok))
    except (OSError, IOError) as e:
        print(f"[WARN] 跳过 {filepath.name}: {e}")
        return []
    return out


def merged_ema_series_classic(
    file_events: list[tuple[Path, list[tuple[int, bool]]]],
) -> list[tuple[int, float]]:
    """与 aggregate_txt_success.py 一致：布尔转 0/1 后 EMA。"""
    keyed: list[tuple[int, str, int, bool]] = []
    for fp, events in file_events:
        fp_s = str(fp)
        for i, (step_num, ok) in enumerate(events):
            keyed.append((step_num, fp_s, i, ok))
    if not keyed:
        return []
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    ema = 0.0
    series: list[tuple[int, float]] = []
    for step_num, _fp_s, _i, ok in keyed:
        x = 1.0 if ok else 0.0
        ema = EMA_ALPHA_OLD * ema + EMA_ALPHA_NEW * x
        series.append((step_num, ema))
    return series


def _parse_success_cell(cell: str) -> bool | None:
    s = cell.strip().lower()
    if s in ("true", "1.0", "1"):
        return True
    if s in ("false", "0.0", "0"):
        return False
    return None


def _parse_trainer_rate_column(parts: list[str]) -> float | None:
    """第 5 列（若存在）为训练端已平滑的成功率，优先使用。"""
    if len(parts) < 5:
        return None
    try:
        v = float(parts[4].strip())
    except ValueError:
        return None
    if 0.0 <= v <= 1.0:
        return v
    return None


def parse_file_events(filepath: Path) -> list[tuple[int, float]]:
    """读取文件，返回按行顺序的 (step_num, success_or_rate)；优先用第 5 列浮点率，否则用第 4 列布尔转 0/1。"""
    out: list[tuple[int, float]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                try:
                    step_num = int(parts[1])
                except (ValueError, IndexError):
                    continue
                rate = _parse_trainer_rate_column(parts)
                if rate is not None:
                    out.append((step_num, rate))
                    continue
                ok = _parse_success_cell(parts[3])
                if ok is None:
                    continue
                out.append((step_num, 1.0 if ok else 0.0))
    except (OSError, IOError) as e:
        print(f"[WARN] 跳过 {filepath.name}: {e}")
        return []
    return out


def merged_ema_series(
    file_events: list[tuple[Path, list[tuple[int, float]]]],
    alpha_old: float | None = None,
    alpha_new: float | None = None,
) -> list[tuple[int, float]]:
    """
    将多文件事件按 (step_num, 文件名, 行序) 排序后，整条序列上做 EMA，
    返回每次更新后的 [(step_num, ema), ...]（保留每次观测一点，供里程碑最近邻查询）。
    每行 x 为第 5 列训练端成功率（若有）或第 4 列布尔对应的 0/1。
    """
    ao = EMA_ALPHA_OLD if alpha_old is None else alpha_old
    an = EMA_ALPHA_NEW if alpha_new is None else alpha_new
    keyed: list[tuple[int, str, int, float]] = []
    for fp, events in file_events:
        fp_s = str(fp)
        for i, (step_num, x) in enumerate(events):
            keyed.append((step_num, fp_s, i, x))
    if not keyed:
        return []
    keyed.sort(key=lambda t: (t[0], t[1], t[2]))
    ema = 0.0
    series: list[tuple[int, float]] = []
    for step_num, _fp_s, _i, x in keyed:
        ema = ao * ema + an * x
        series.append((step_num, ema))
    return series


def find_nearest(rows: list[tuple[int, float]], target: int) -> float | None:
    """在 rows 中找 step_num 最接近 target 的点，返回其 EMA 成功率。
    若最近 step 与 target 差距大于 MAX_MILESTONE_STEP_DIFF 则返回 None（不参与统计）。
    """
    if not rows:
        return None
    best_diff = float("inf")
    best_rate: float | None = None
    for step_num, rate in rows:
        diff = abs(step_num - target)
        if diff < best_diff:
            best_diff = diff
            best_rate = rate
    if best_diff > MAX_MILESTONE_STEP_DIFF:
        return None
    return best_rate


def aggregate_milestones(
    ema_series: list[tuple[int, float]],
    n_source_files: int,
) -> list[tuple[int, int, float | None]]:
    """对合并后的 EMA 序列，每个里程碑返回 (step, n_source_files, rate_or_none)。"""
    out: list[tuple[int, int, float | None]] = []
    for milestone in MILESTONES:
        r = find_nearest(ema_series, milestone)
        if r is not None:
            out.append((milestone, n_source_files, r))
        else:
            out.append((milestone, n_source_files, None))
    return out


def write_csv_all(path: Path, rows: list[tuple[int, int, float | None]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "n_files_merged", "ema_success_rate"])
        for step, n, avg in rows:
            w.writerow([step, n, "" if avg is None else f"{avg:.10g}"])


def write_csv_by_emb(
    path: Path,
    emb_to_results: dict[str, list[tuple[int, int, float | None]]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["embodiment", "step", "n_files_merged", "ema_success_rate"])
        for emb in sorted(emb_to_results.keys()):
            for step, n, avg in emb_to_results[emb]:
                w.writerow([emb, step, n, "" if avg is None else f"{avg:.10g}"])


def write_csv_by_emb_scene(
    path: Path,
    emb_scene_to_results: dict[tuple[str, str], list[tuple[int, int, float | None]]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["embodiment", "scene", "step", "n_files_merged", "ema_success_rate"])
        for emb, scene in sorted(emb_scene_to_results.keys(), key=lambda t: (t[0], t[1])):
            for step, n, avg in emb_scene_to_results[(emb, scene)]:
                w.writerow([emb, scene, step, n, "" if avg is None else f"{avg:.10g}"])


def _last_valid_ema(rows: list[tuple[int, int, float | None]]) -> float | None:
    """里程碑序列从末尾找第一个非空的 EMA。"""
    for _step, _n, avg in reversed(rows):
        if avg is not None:
            return avg
    return None


def _scene_rgb(scene: str) -> tuple[float, float, float]:
    """场景名稳定映射到 RGB，便于多曲线区分。"""
    h = hashlib.md5(scene.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def plot_emb_scene_per_emb_subplots(
    emb_scene_results: dict[tuple[str, str], list[tuple[int, int, float | None]]],
    out_path: Path,
) -> None:
    """每个 embodiment 一行子图，该机体下各场景成功率曲线。"""
    embs_sorted = sorted({e for e, _s in emb_scene_results})
    if not embs_sorted:
        return
    n = len(embs_sorted)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(4.0, 4.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, emb in zip(axes, embs_sorted):
        nlines = 0
        for (e, scene), res in sorted(emb_scene_results.items(), key=lambda x: (x[0][0], x[0][1])):
            if e != emb:
                continue
            nlines += 1
            xs = [r[0] for r in res]
            ys = [r[2] if r[2] is not None else float("nan") for r in res]
            ax.plot(xs, ys, "-", linewidth=1.2, alpha=0.88, color=_scene_rgb(scene), label=scene[:48])
        ax.set_ylabel("EMA success", fontsize=10)
        ax.set_title(f"{emb} — 各场景", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if nlines <= 15:
            ax.legend(loc="best", fontsize=5, ncol=2)
    axes[-1].set_xlabel("Step", fontsize=12)
    fig.suptitle(
        f"按 embodiment × 场景 — EMA ({EMA_SCENE_ALPHA_OLD}/{EMA_SCENE_ALPHA_NEW})，里程碑最近邻",
        fontsize=12,
        y=1.002,
    )
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_emb_scene_heatmap_last(
    emb_scene_results: dict[tuple[str, str], list[tuple[int, int, float | None]]],
    out_path: Path,
) -> None:
    """行=场景，列=embodiment，值为该组合里程碑序列上最后一个有效 EMA。"""
    scenes = sorted({s for _e, s in emb_scene_results})
    embs = sorted({e for e, _s in emb_scene_results})
    if not scenes or not embs:
        return
    mat = np.full((len(scenes), len(embs)), np.nan, dtype=np.float64)
    for j, emb in enumerate(embs):
        for i, scene in enumerate(scenes):
            lr = _last_valid_ema(emb_scene_results.get((emb, scene), []))
            if lr is not None:
                mat[i, j] = lr
    fig_w = max(10.0, len(embs) * 1.4)
    fig_h = max(8.0, min(48.0, len(scenes) * 0.22))
    plt.figure(figsize=(fig_w, fig_h))
    # 低→蓝、高→红（经白过渡），便于一眼区分好坏
    plt.imshow(mat, aspect="auto", cmap="coolwarm", vmin=0.0, vmax=1.0)
    plt.colorbar(label="最后有效里程碑 EMA")
    plt.xticks(range(len(embs)), embs, rotation=25, ha="right")
    plt.yticks(
        range(len(scenes)),
        [s if len(s) <= 40 else s[:37] + "..." for s in scenes],
        fontsize=7,
    )
    plt.xlabel("Embodiment", fontsize=11)
    plt.ylabel("Scene", fontsize=11)
    plt.title(
        f"各 (embodiment × scene) 合并 EMA ({EMA_SCENE_ALPHA_OLD}/{EMA_SCENE_ALPHA_NEW}) — 热力图（末点）",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_emb_scene_boxplot_final(
    emb_scene_results: dict[tuple[str, str], list[tuple[int, int, float | None]]],
    out_path: Path,
) -> None:
    """每个 embodiment 下，各场景「最后有效 EMA」的箱线图。"""
    embs = sorted({e for e, _s in emb_scene_results})
    data: list[list[float]] = []
    ticks: list[str] = []
    for emb in embs:
        vals: list[float] = []
        for (e, _scene), res in emb_scene_results.items():
            if e != emb:
                continue
            v = _last_valid_ema(res)
            if v is not None:
                vals.append(v)
        if vals:
            data.append(vals)
            ticks.append(emb)
    if not data:
        return
    plt.figure(figsize=(max(8.0, len(ticks) * 1.2), 6))
    bp = plt.boxplot(data, patch_artist=True)
    plt.gca().set_xticklabels(ticks)
    for box in bp["boxes"]:
        box.set_facecolor("lightblue")
    plt.ylabel("EMA success（最后有效里程碑）", fontsize=11)
    plt.xlabel("Embodiment", fontsize=11)
    plt.title(
        f"各 embodiment：跨场景最终 EMA ({EMA_SCENE_ALPHA_OLD}/{EMA_SCENE_ALPHA_NEW}) — 箱线",
        fontsize=13,
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_category(
    category: str,
    all_loaded_classic: list[tuple[Path, list[tuple[int, bool]]]],
    by_emb_classic: dict[str, list[tuple[Path, list[tuple[int, bool]]]]],
    by_emb_scene: dict[tuple[str, str], list[tuple[Path, list[tuple[int, float]]]]],
) -> None:
    """对单一场景类别输出 CSV 与 PNG。"""
    out_dir = category_output_dir(category)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_label = "home/commercial" if category == "home_commercial" else "clutter (hard/easy)"
    print(f"========== {cat_label} ==========")
    print(f"  文件数(classic): {len(all_loaded_classic)}")
    emb_file_counts = {k: len(v) for k, v in sorted(by_emb_classic.items())}
    print(
        f"  embodiment: {sorted(by_emb_classic.keys())}，每类文件数: "
        f"{emb_file_counts}"
    )
    print(f"  embodiment×场景 组合数: {len(by_emb_scene)}")
    if not all_loaded_classic:
        print("  [SKIP] 无可用 txt")
        print()
        return

    out_csv_all = out_dir / "success_rate_subprocess_embodiment_all.csv"
    out_csv_by_emb = out_dir / "success_rate_subprocess_embodiment_by_embodiment.csv"
    out_csv_by_emb_scene = out_dir / "success_rate_subprocess_embodiment_by_scene.csv"
    out_fig_all = out_dir / "success_rate_subprocess_embodiment_all.png"
    out_fig_by_emb = out_dir / "success_rate_subprocess_embodiment_by_embodiment.png"
    out_fig_emb_scene_sub = out_dir / "success_rate_subprocess_embodiment_by_scene_per_emb.png"
    out_fig_emb_scene_heat = out_dir / "success_rate_subprocess_embodiment_by_scene_heatmap_last.png"
    out_fig_emb_scene_box = out_dir / "success_rate_subprocess_embodiment_by_scene_boxplot_final.png"

    series_all = merged_ema_series_classic(all_loaded_classic)
    results_all = aggregate_milestones(series_all, len(all_loaded_classic))
    write_csv_all(out_csv_all, results_all)
    print(f"  表（全部）: {out_csv_all}")

    emb_results: dict[str, list[tuple[int, int, float | None]]] = {}
    for emb, flist in sorted(by_emb_classic.items()):
        ser = merged_ema_series_classic(flist)
        emb_results[emb] = aggregate_milestones(ser, len(flist))
    write_csv_by_emb(out_csv_by_emb, emb_results)
    print(f"  表（按 embodiment）: {out_csv_by_emb}")

    emb_scene_results: dict[tuple[str, str], list[tuple[int, int, float | None]]] = {}
    for (emb, scene), flist in sorted(by_emb_scene.items(), key=lambda x: (x[0][0], x[0][1])):
        ser = merged_ema_series(
            flist,
            alpha_old=EMA_SCENE_ALPHA_OLD,
            alpha_new=EMA_SCENE_ALPHA_NEW,
        )
        emb_scene_results[(emb, scene)] = aggregate_milestones(ser, len(flist))
    write_csv_by_emb_scene(out_csv_by_emb_scene, emb_scene_results)
    print(
        f"  表（按 embodiment × 场景）: {out_csv_by_emb_scene}，共 {len(emb_scene_results)} 条组合"
    )

    if MAKE_PLOTS and emb_scene_results:
        plot_emb_scene_per_emb_subplots(emb_scene_results, out_fig_emb_scene_sub)
        print(f"  图（场景-分 embodiment 子图）: {out_fig_emb_scene_sub}")
        plot_emb_scene_heatmap_last(emb_scene_results, out_fig_emb_scene_heat)
        print(f"  图（场景×embodiment 热力图-末点）: {out_fig_emb_scene_heat}")
        plot_emb_scene_boxplot_final(emb_scene_results, out_fig_emb_scene_box)
        print(f"  图（场景末 EMA 箱线-按机体）: {out_fig_emb_scene_box}")

    if MAKE_PLOTS:
        steps = [r[0] for r in results_all]
        rates_all = [r[2] if r[2] is not None else float("nan") for r in results_all]
        plot_series(
            steps,
            rates_all,
            f"EMA success ({cat_label}) — merged events, every {MILESTONE_INTERVAL} steps",
            out_fig_all,
        )
        print(f"  图（全部）: {out_fig_all}")

        plt.figure(figsize=(11, 6))
        for emb in sorted(emb_results.keys()):
            res = emb_results[emb]
            xs = [r[0] for r in res]
            ys = [r[2] if r[2] is not None else float("nan") for r in res]
            plt.plot(xs, ys, "o-", linewidth=1.5, markersize=3, label=emb)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Success Rate", fontsize=12)
        plt.title(
            f"EMA success ({cat_label}) — by embodiment, every {MILESTONE_INTERVAL} steps",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_fig_by_emb, dpi=150)
        plt.close()
        print(f"  图（分 embodiment）: {out_fig_by_emb}")
    print()


def plot_series(
    steps: list[int],
    y: list[float],
    title: str,
    out_path: Path,
    label: str | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    if label:
        plt.plot(steps, y, "o-", linewidth=2, markersize=4, label=label)
        plt.legend()
    else:
        plt.plot(steps, y, "o-", linewidth=2, markersize=4)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate X-NavDP training evaluation TXT logs into CSV summaries and plots."
    )
    parser.add_argument("txt_dir", type=Path, help="Directory containing per-scene TXT logs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <repo>/result/<txt_dir_name>.",
    )
    parser.add_argument("--milestone-interval", type=int, default=1250)
    parser.add_argument("--max-step", type=int, default=200000)
    parser.add_argument(
        "--max-milestone-distance",
        type=int,
        default=None,
        help="Maximum distance to the nearest logged step; defaults to milestone interval.",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.01)
    parser.add_argument("--scene-ema-alpha", type=float, default=0.01)
    parser.add_argument("--no-plots", action="store_true", help="Write CSV files only.")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> Path:
    global RESULT_DIR, MILESTONE_INTERVAL, MILESTONES, MAX_MILESTONE_STEP_DIFF
    global EMA_ALPHA_NEW, EMA_ALPHA_OLD, EMA_SCENE_ALPHA_NEW, EMA_SCENE_ALPHA_OLD
    global MAKE_PLOTS

    if args.milestone_interval <= 0 or args.max_step <= 0:
        raise ValueError("milestone interval and max step must be positive")
    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be in (0, 1]")
    if not 0.0 < args.scene_ema_alpha <= 1.0:
        raise ValueError("--scene-ema-alpha must be in (0, 1]")

    txt_dir = args.txt_dir.expanduser().resolve()
    RESULT_DIR = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "result" / txt_dir.name
    )
    MILESTONE_INTERVAL = args.milestone_interval
    MILESTONES = list(range(MILESTONE_INTERVAL, args.max_step + 1, MILESTONE_INTERVAL))
    MAX_MILESTONE_STEP_DIFF = (
        args.max_milestone_distance
        if args.max_milestone_distance is not None
        else MILESTONE_INTERVAL
    )
    if MAX_MILESTONE_STEP_DIFF < 0:
        raise ValueError("--max-milestone-distance must be non-negative")
    EMA_ALPHA_NEW = args.ema_alpha
    EMA_ALPHA_OLD = 1.0 - EMA_ALPHA_NEW
    EMA_SCENE_ALPHA_NEW = args.scene_ema_alpha
    EMA_SCENE_ALPHA_OLD = 1.0 - EMA_SCENE_ALPHA_NEW
    MAKE_PLOTS = not args.no_plots
    return txt_dir


def main() -> int:
    args = parse_args()
    txt_dir = configure(args)
    if not txt_dir.is_dir():
        print(f"Input directory does not exist: {txt_dir}")
        return 2

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        print(f"No TXT files found in: {txt_dir}")
        return 2

    by_cat_classic: dict[str, list[tuple[Path, list[tuple[int, bool]]]]] = {
        cat: [] for cat in SCENE_CATEGORIES
    }
    by_cat_emb_classic: dict[str, dict[str, list[tuple[Path, list[tuple[int, bool]]]]]] = {
        cat: defaultdict(list) for cat in SCENE_CATEGORIES
    }
    by_cat_emb_scene: dict[
        str, dict[tuple[str, str], list[tuple[Path, list[tuple[int, float]]]]]
    ] = {cat: defaultdict(list) for cat in SCENE_CATEGORIES}
    bad_names: list[str] = []

    for fp in txt_files:
        category = get_scene_category(fp.name)
        if category is None:
            bad_names.append(fp.name)
            continue

        emb_c = parse_embodiment_classic(fp.name)
        if emb_c is not None:
            ev_c = parse_file_events_classic(fp)
            by_cat_classic[category].append((fp, ev_c))
            by_cat_emb_classic[category][emb_c].append((fp, ev_c))

        emb, scene = parse_embodiment_and_scene(fp.name)
        if emb is not None and scene is not None:
            events = parse_file_events(fp)
            by_cat_emb_scene[category][(emb, scene)].append((fp, events))

    if bad_names:
        print(
            f"[WARN] 无法识别场景类别（已跳过 {len(bad_names)} 个）: "
            f"{bad_names[:5]}{'...' if len(bad_names) > 5 else ''}"
        )

    print(f"Input directory: {txt_dir}")
    print(f"TXT files: {len(txt_files)}")
    print(f"Milestones: every {MILESTONE_INTERVAL} steps, {len(MILESTONES)} total")
    print(
        f"classic EMA ({EMA_ALPHA_OLD}/{EMA_ALPHA_NEW})；"
        f"场景 EMA ({EMA_SCENE_ALPHA_OLD}/{EMA_SCENE_ALPHA_NEW})"
    )
    print(f"Output directory: {RESULT_DIR}")
    print()

    for category in SCENE_CATEGORIES:
        process_category(
            category,
            by_cat_classic[category],
            by_cat_emb_classic[category],
            by_cat_emb_scene[category],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
