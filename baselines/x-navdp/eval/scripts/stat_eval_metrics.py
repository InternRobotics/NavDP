#!/usr/bin/env python3
"""Print per-USD SR and SPL from X-NavDP evaluation metric.csv files.

Input is one evaluation group directory, for example:

    /cpfs/user/yangtianyu/NavDP/baselines/x-navdp/outputs/evaluation/quadruped_commercial

The script recursively finds metric.csv files under the input directory. Each
metric.csv is expected to contain success and SPL in the first two columns, or
named columns "success" and "spl".
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "outputs" / "evaluation" / "quadruped_commercial"


def usd_name(metric_path: Path) -> str:
    """metric.csv parent directory is usually <scene>_usd."""
    scene_dir = metric_path.parent.name
    if scene_dir.endswith("_usd"):
        return f"{scene_dir[:-4]}.usd"
    if scene_dir.endswith(".usd"):
        return scene_dir
    return scene_dir


def read_success_spl(path: Path) -> tuple[list[float], list[float]]:
    successes: list[float] = []
    spls: list[float] = []

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if {"success", "spl"}.issubset(fieldnames):
            for row in reader:
                try:
                    successes.append(float(row["success"]))
                    spls.append(float(row["spl"]))
                except (TypeError, ValueError):
                    continue
            return successes, spls

    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                successes.append(float(row[0]))
                spls.append(float(row[1]))
            except ValueError:
                continue
    return successes, spls


def summarize(successes: list[float], spls: list[float]) -> tuple[int, int, float, float]:
    episodes = len(successes)
    if episodes == 0:
        return 0, 0, 0.0, 0.0
    success_count = sum(1 for value in successes if value >= 0.5)
    sr = success_count / episodes
    mean_spl = sum(spls) / len(spls) if spls else 0.0
    return episodes, success_count, sr, mean_spl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Evaluation group directory. Default: {DEFAULT_ROOT}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"Input directory does not exist: {root}")
        return 2

    buckets: dict[str, tuple[list[float], list[float], int]] = defaultdict(
        lambda: ([], [], 0)
    )
    for metric_path in sorted(root.rglob("metric.csv")):
        successes, spls = read_success_spl(metric_path)
        usd = usd_name(metric_path)
        bucket_successes, bucket_spls, file_count = buckets[usd]
        bucket_successes.extend(successes)
        bucket_spls.extend(spls)
        buckets[usd] = (bucket_successes, bucket_spls, file_count + 1)

    if not buckets:
        print(f"No metric.csv files found under: {root}")
        return 2

    print(f"Input: {root}")
    print(f"Metric files: {sum(item[2] for item in buckets.values())}")
    print()
    print("USD\tfiles\tepisodes\tsuccess\tSR\tSPL")

    total_successes: list[float] = []
    total_spls: list[float] = []
    for usd in sorted(buckets):
        successes, spls, file_count = buckets[usd]
        episodes, success_count, sr, mean_spl = summarize(successes, spls)
        total_successes.extend(successes)
        total_spls.extend(spls)
        print(
            f"{usd}\t{file_count}\t{episodes}\t{success_count}\t"
            f"{sr:.4%}\t{mean_spl:.6f}"
        )

    episodes, success_count, sr, mean_spl = summarize(total_successes, total_spls)
    print()
    print(
        f"Overall\tfiles={sum(item[2] for item in buckets.values())}\t"
        f"episodes={episodes}\tsuccess={success_count}\tSR={sr:.4%}\tSPL={mean_spl:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
