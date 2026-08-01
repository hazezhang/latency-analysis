#!/usr/bin/env python3
"""Aggregate three-seed CORAL ordinal direct-LAT results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = ("20260718", "20260719", "20260720")
VARIANTS = ("ordinal_text", "ordinal_text_delay")
METRICS = ("pearson", "spearman", "mse", "mae", "pred_std", "quadratic_weighted_kappa", "within_0.5_accuracy")


def quadratic_weighted_kappa(gold_classes, prediction_classes, n_classes=11):
    gold_classes = np.asarray(gold_classes, dtype=int)
    prediction_classes = np.asarray(prediction_classes, dtype=int)
    observed = np.zeros((n_classes, n_classes), dtype=float)
    for gold, prediction in zip(gold_classes, prediction_classes):
        observed[gold, prediction] += 1.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(observed.sum(), 1.0)
    grid = np.arange(n_classes)
    weights = (grid[:, None] - grid[None, :]) ** 2 / float((n_classes - 1) ** 2)
    denominator = float(np.sum(weights * expected))
    return 1.0 - float(np.sum(weights * observed)) / denominator if denominator else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-template", default="experiments/aaai_ordinal_lat_seed_{}")
    parser.add_argument("--output-dir", default="experiments/aaai_ordinal_lat_summary")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, table_rows = {"protocol": "CORAL ordinal direct LAT; three fixed outer-speech seeds", "seeds": {}, "aggregate": {}}, []
    for seed in SEEDS:
        payload["seeds"][seed] = {}
        for variant in VARIANTS:
            rows = []
            for path in sorted((Path(args.root_template.format(seed)) / variant).glob("outer_*/predictions.json")):
                rows.extend(json.loads(path.read_text(encoding="utf-8")))
            if len(rows) != 622 or len({str(row["segment_id"]) for row in rows}) != 622:
                raise ValueError(f"{seed}/{variant}: expected 622 unique outer predictions")
            gold = np.asarray([float(row["LAT"]) for row in rows])
            prediction = np.asarray([float(row["prediction"]) for row in rows])
            rounded_gold = np.clip(np.rint((gold - .5) / .25), 0, 10)
            rounded_prediction = np.clip(np.rint((prediction - .5) / .25), 0, 10)
            from scipy.stats import pearsonr, spearmanr
            result = {
                "n": len(rows), "pearson": float(pearsonr(gold, prediction).statistic), "spearman": float(spearmanr(gold, prediction).statistic),
                "mse": float(np.mean((gold - prediction) ** 2)), "mae": float(np.mean(np.abs(gold - prediction))),
                "pred_std": float(prediction.std()), "quadratic_weighted_kappa": quadratic_weighted_kappa(rounded_gold, rounded_prediction),
                "within_0.5_accuracy": float(np.mean(np.abs(gold - prediction) <= .5)),
            }
            payload["seeds"][seed][variant] = result
            table_rows.append({"seed": seed, "variant": variant, **result})
    for variant in VARIANTS:
        payload["aggregate"][variant] = {}
        for metric in METRICS:
            values = np.asarray([payload["seeds"][seed][variant][metric] for seed in SEEDS])
            payload["aggregate"][variant][metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "values": values.tolist()}
    (output_dir / "ordinal_lat_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "ordinal_lat_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader(); writer.writerows(table_rows)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
