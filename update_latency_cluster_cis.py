#!/usr/bin/env python3
"""Replace segment-iid CIs with speech-group cluster-bootstrap CIs.

Predictions are already out-of-fold. This utility changes only uncertainty
intervals, resampling entire held-out speech groups to match the evaluation
unit used by the LAT experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEED = 20260712
BOOTSTRAPS = 5000


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 3 or not np.std(y_true) or not np.std(y_pred):
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    return pearson(average_rank(y_true), average_rank(y_pred))


def cluster_ci(rows: list[dict], metric) -> list[float | None]:
    groups = sorted({row["speech_group"] for row in rows})
    by_group = {group: [row for row in rows if row["speech_group"] == group] for group in groups}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAPS):
        sample = []
        for group in rng.choice(groups, size=len(groups), replace=True):
            sample.extend(by_group[group])
        y_true = np.asarray([float(row["LAT"]) for row in sample])
        y_pred = np.asarray([float(row["prediction"]) for row in sample])
        value = metric(y_true, y_pred)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return [None, None]
    return [round(float(np.quantile(values, 0.025)), 4), round(float(np.quantile(values, 0.975)), 4)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--result-csv", required=True)
    parser.add_argument("--oof-csv", required=True)
    args = parser.parse_args()

    result_path = Path(args.result_json)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with Path(args.oof_csv).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        by_model: dict[str, list[dict]] = {}
        for row in reader:
            by_model.setdefault(row["model"], []).append(row)

    for name, block in result["models"].items():
        rows = by_model[name]
        block["metrics"]["pearson"]["ci95"] = cluster_ci(rows, pearson)
        block["metrics"]["spearman"]["ci95"] = cluster_ci(rows, spearman)
        block["metrics"]["pearson"]["bootstrap_valid"] = BOOTSTRAPS
        block["metrics"]["spearman"]["bootstrap_valid"] = BOOTSTRAPS
    result.setdefault("metadata", {})["ci_rule"] = "95% CIs use speech-group cluster bootstrap, matching LeaveOneSpeechGroupOut evaluation."
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = Path(args.result_csv)
    fieldnames = [
        "model", "n_segments", "pearson", "pearson_ci_low", "pearson_ci_high",
        "spearman", "spearman_ci_low", "spearman_ci_high", "mse", "mae", "r2", "pred_std",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, block in result["models"].items():
            metrics = block["metrics"]
            writer.writerow({
                "model": name,
                "n_segments": metrics["n_segments"],
                "pearson": metrics["pearson"]["estimate"],
                "pearson_ci_low": metrics["pearson"]["ci95"][0],
                "pearson_ci_high": metrics["pearson"]["ci95"][1],
                "spearman": metrics["spearman"]["estimate"],
                "spearman_ci_low": metrics["spearman"]["ci95"][0],
                "spearman_ci_high": metrics["spearman"]["ci95"][1],
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "pred_std": metrics["pred_std"],
            })
    print(csv_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
