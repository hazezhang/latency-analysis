#!/usr/bin/env python3
"""Aggregate joint LQ/EXP/LAT outer-fold predictions across formal seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


SEEDS = ("20260718", "20260719", "20260720")
VARIANTS = ("joint", "joint_delay")
TASKS = ("LQ", "EXP", "LAT")


def safe_corr(gold, prediction, method="pearson"):
    gold, prediction = np.asarray(gold, dtype=float), np.asarray(prediction, dtype=float)
    if len(gold) < 3 or np.std(gold) == 0 or np.std(prediction) == 0:
        return None
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(gold, prediction).statistic)


def metrics(rows, task):
    prediction_key = "prediction" if task == "LAT" else f"pred_{task}"
    gold = np.asarray([row[task] for row in rows], dtype=float)
    prediction = np.asarray([row[prediction_key] for row in rows], dtype=float)
    return {
        "n": len(rows), "pearson": safe_corr(gold, prediction),
        "spearman": safe_corr(gold, prediction, "spearman"),
        "mse": float(np.mean((gold - prediction) ** 2)),
        "mae": float(np.mean(np.abs(gold - prediction))),
        "pred_std": float(prediction.std()),
    }


def load_rows(root):
    paths = sorted(root.glob("outer_*/predictions.json"))
    if len(paths) != 16:
        raise ValueError(f"{root}: expected 16 prediction files, found {len(paths)}")
    rows = []
    for path in paths:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) != 622 or len({row["segment_id"] for row in rows}) != 622:
        raise ValueError(f"{root}: expected 622 unique segments")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-template", default="experiments/aaai_multitask_lat_seed_{}")
    parser.add_argument("--output-dir", default="experiments/aaai_multitask_lat_summary_20260722")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": {}, "aggregate": {}}
    table_rows = []
    for seed in SEEDS:
        payload["seeds"][seed] = {}
        for variant in VARIANTS:
            rows = load_rows(Path(args.root_template.format(seed)) / variant)
            result = {task: metrics(rows, task) for task in TASKS}
            payload["seeds"][seed][variant] = result
            for task in TASKS:
                table_rows.append({"seed": seed, "variant": variant, "task": task, **result[task]})
    for variant in VARIANTS:
        payload["aggregate"][variant] = {}
        for task in TASKS:
            payload["aggregate"][variant][task] = {}
            for metric in ("pearson", "spearman", "mse", "mae", "pred_std"):
                values = np.asarray([payload["seeds"][seed][variant][task][metric] for seed in SEEDS])
                payload["aggregate"][variant][task][metric] = {
                    "mean": float(values.mean()), "sd": float(values.std(ddof=1)), "values": values.tolist()
                }
    (output_dir / "multitask_lat_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "multitask_lat_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
