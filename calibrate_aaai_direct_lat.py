#!/usr/bin/env python3
"""Apply development-only fold-wise calibration to direct-LAT predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


SEEDS = ("20260718", "20260719", "20260720")
VARIANTS = ("text", "text_delay")


def safe_corr(gold, prediction, method="pearson"):
    gold, prediction = np.asarray(gold, dtype=float), np.asarray(prediction, dtype=float)
    if len(gold) < 3 or np.std(gold) == 0 or np.std(prediction) == 0:
        return None
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(gold, prediction).statistic)


def metrics(rows, key="prediction"):
    gold = np.asarray([row["LAT"] for row in rows], dtype=float)
    prediction = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "n": len(rows),
        "pearson": safe_corr(gold, prediction),
        "spearman": safe_corr(gold, prediction, "spearman"),
        "mse": float(np.mean((gold - prediction) ** 2)),
        "mae": float(np.mean(np.abs(gold - prediction))),
        "gold_mean": float(gold.mean()),
        "gold_std": float(gold.std()),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def fit_affine(rows):
    prediction = np.asarray([row["prediction"] for row in rows], dtype=float)
    gold = np.asarray([row["LAT"] for row in rows], dtype=float)
    design = np.column_stack([prediction, np.ones(len(prediction))])
    slope, intercept = np.linalg.lstsq(design, gold, rcond=None)[0]
    return float(slope), float(intercept)


def fit_mean_variance(rows):
    prediction = np.asarray([row["prediction"] for row in rows], dtype=float)
    gold = np.asarray([row["LAT"] for row in rows], dtype=float)
    slope = float(gold.std() / prediction.std()) if prediction.std() else 1.0
    intercept = float(gold.mean() - slope * prediction.mean())
    return slope, intercept


def segment_ids(rows, path):
    values = [str(row.get("segment_id")) for row in rows]
    if any(value in {"", "None"} for value in values):
        raise ValueError(f"Missing segment_id in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate segment_id in {path}")
    return set(values)


def round_tree(value):
    if isinstance(value, dict):
        return {key: round_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_tree(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root-template", default="experiments/aaai_direct_lat_calibration_inputs_seed_{}")
    parser.add_argument("--direct-root-template", default="experiments/aaai_direct_lat_corrected_seed_{}")
    parser.add_argument("--output-dir", default="experiments/aaai_direct_lat_calibrated_20260722")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": "calibration fitted independently on each outer fold development predictions; outer test gold never used", "seeds": {}}
    csv_rows = []

    for seed in SEEDS:
        payload["seeds"][seed] = {}
        for variant in VARIANTS:
            calibration_root = Path(args.calibration_root_template.format(seed)) / variant
            direct_root = Path(args.direct_root_template.format(seed)) / variant
            predictions = []
            folds = []
            outer_test_ids = set()
            for dev_path in sorted(calibration_root.glob("outer_*/dev_predictions.json")):
                fold = dev_path.parent.name
                dev_rows = json.loads(dev_path.read_text(encoding="utf-8"))
                test_rows = json.loads((direct_root / fold / "predictions.json").read_text(encoding="utf-8"))
                dev_ids = segment_ids(dev_rows, dev_path)
                test_ids = segment_ids(test_rows, direct_root / fold / "predictions.json")
                if dev_ids & test_ids:
                    raise ValueError(f"Calibration leakage: dev/test segment overlap in {seed}/{variant}/{fold}")
                if outer_test_ids & test_ids:
                    raise ValueError(f"Outer-test prediction overlap across folds in {seed}/{variant}/{fold}")
                outer_test_ids.update(test_ids)
                affine_slope, affine_intercept = fit_affine(dev_rows)
                mv_slope, mv_intercept = fit_mean_variance(dev_rows)
                for row in test_rows:
                    item = dict(row)
                    item["raw_prediction"] = float(row["prediction"])
                    item["affine_prediction"] = affine_slope * item["raw_prediction"] + affine_intercept
                    item["mean_variance_prediction"] = mv_slope * item["raw_prediction"] + mv_intercept
                    item["outer_fold"] = fold
                    predictions.append(item)
                folds.append({
                    "outer_fold": fold,
                    "n_dev": len(dev_rows),
                    "affine_slope": affine_slope,
                    "affine_intercept": affine_intercept,
                    "mean_variance_slope": mv_slope,
                    "mean_variance_intercept": mv_intercept,
                    "dev_raw_metrics": metrics(dev_rows),
                })
            if len(folds) != 16:
                raise ValueError(f"{seed}/{variant}: expected 16 calibration folds, found {len(folds)}")
            if len(predictions) != len(outer_test_ids):
                raise ValueError(f"{seed}/{variant}: duplicate outer predictions after calibration")
            result = {
                "raw": metrics(predictions, "raw_prediction"),
                "affine_calibrated": metrics(predictions, "affine_prediction"),
                "mean_variance_calibrated": metrics(predictions, "mean_variance_prediction"),
                "folds": folds,
            }
            payload["seeds"][seed][variant] = result
            for method in ("raw", "affine_calibrated", "mean_variance_calibrated"):
                csv_rows.append({"seed": seed, "variant": variant, "method": method, **result[method]})
            with (output_dir / f"predictions_{seed}_{variant}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
                writer.writeheader()
                writer.writerows(predictions)

    aggregate = defaultdict(dict)
    for variant in VARIANTS:
        for method in ("raw", "affine_calibrated", "mean_variance_calibrated"):
            aggregate[variant][method] = {}
            for metric in ("pearson", "spearman", "mse", "mae", "prediction_mean", "prediction_std"):
                values = np.asarray([payload["seeds"][seed][variant][method][metric] for seed in SEEDS])
                aggregate[variant][method][metric] = {
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "values": values.tolist(),
                }
    payload["aggregate"] = aggregate
    rounded = round_tree(payload)
    (output_dir / "direct_lat_calibration_results.json").write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "direct_lat_calibration_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(round_tree(csv_rows))
    print(json.dumps(rounded["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
