#!/usr/bin/env python3
"""Evaluate one R027 upstream seed under the frozen outer-nested LAT protocol."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).parent
DATA_ROOT = BASE / "data/experiments/r027_shared_outer_quality"
DEFAULT_PRED_ROOT = BASE / "experiments/r027_shared_outer_quality_seed_20260718"
DEFAULT_OUT = BASE / "experiments/latency_r027_seed_20260718"
SEED = 20260718
BOOTSTRAPS = 5000
KNOTS = (2.0, 4.0, 6.0, 10.0)


def corr(y, prediction, method="pearson"):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    value = pearsonr(y, prediction)[0] if method == "pearson" else spearmanr(y, prediction)[0]
    return float(value)


def cluster_ci(y, prediction, groups, method):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        value = corr(y[indices], prediction[indices], method)
        if value is not None and np.isfinite(value):
            values.append(value)
    return [round(float(np.quantile(values, 0.025)), 4), round(float(np.quantile(values, 0.975)), 4)]


def metrics(y, prediction, groups):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mse = float(np.mean((y - prediction) ** 2))
    return {
        "n_segments": int(len(y)),
        "pearson": round(corr(y, prediction), 4),
        "pearson_ci95": cluster_ci(y, prediction, groups, "pearson"),
        "spearman": round(corr(y, prediction, "spearman"), 4),
        "spearman_ci95": cluster_ci(y, prediction, groups, "spearman"),
        "mse": round(mse, 4),
        "mae": round(float(np.mean(np.abs(y - prediction))), 4),
        "r2": round(float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)), 4),
        "pred_std": round(float(np.std(prediction)), 4),
    }


def feature_vector(row, names):
    values = {
        "delay_linear": row["delay_seconds"],
        "delay_hinge_2": max(0.0, row["delay_seconds"] - 2.0),
        "delay_hinge_4": max(0.0, row["delay_seconds"] - 4.0),
        "delay_hinge_6": max(0.0, row["delay_seconds"] - 6.0),
        "delay_hinge_10": max(0.0, row["delay_seconds"] - 10.0),
        "LQ": row["LQ"],
        "EXP": row["EXP"],
        "pred_LQ": row["pred_LQ"],
        "pred_EXP": row["pred_EXP"],
    }
    values["LQ_delay_interaction"] = values["LQ"] * values["delay_linear"]
    values["EXP_delay_interaction"] = values["EXP"] * values["delay_linear"]
    values["pred_LQ_delay_interaction"] = values["pred_LQ"] * values["delay_linear"]
    values["pred_EXP_delay_interaction"] = values["pred_EXP"] * values["delay_linear"]
    return np.asarray([values[name] for name in names], dtype=float)


def ridge_predict(train_rows, test_rows, features):
    x_train = np.vstack([feature_vector(row, features) for row in train_rows])
    y_train = np.asarray([row["perceived_latency"] for row in train_rows], dtype=float)
    x_test = np.vstack([feature_vector(row, features) for row in test_rows])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(x_train.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_test @ beta


def load_predictions(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in rows:
        sid = str(row["segment_id"])
        if sid in result:
            raise ValueError(f"Duplicate segment prediction in {path}: {sid}")
        result[sid] = {"pred_LQ": float(row["pred_LQ"]), "pred_EXP": float(row["pred_EXP"])}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", default=str(DEFAULT_PRED_ROOT.relative_to(BASE)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT.relative_to(BASE)))
    args = parser.parse_args()
    pred_root = BASE / args.prediction_root
    out_dir = BASE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    all_rows = json.loads((DATA_ROOT / "all_lat_segments.json").read_text(encoding="utf-8"))
    by_speech = {fold["outer_test_speech"]: fold for fold in manifest["folds"]}
    model_specs = {
        "delay_linear": ["delay_linear"],
        "delay_piecewise": ["delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"],
        "human_LQ_EXP": ["LQ", "EXP"],
        "human_LQ_EXP_piecewise_delay": ["LQ", "EXP", "delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"],
        "auto_pred_LQ_EXP": ["pred_LQ", "pred_EXP"],
        "auto_pred_LQ_EXP_piecewise_delay": ["pred_LQ", "pred_EXP", "delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"],
        "auto_pred_LQ_EXP_piecewise_delay_interaction": ["pred_LQ", "pred_EXP", "delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10", "pred_LQ_delay_interaction", "pred_EXP_delay_interaction"],
    }
    all_predictions = {name: [] for name in model_specs}
    fold_log = []
    for speech, fold in sorted(by_speech.items()):
        fold_predictions = load_predictions(pred_root / fold["name"] / "predictions_all.json")
        if set(fold_predictions) != {str(row["segment_id"]) for row in all_rows}:
            raise ValueError(f"Prediction/cohort mismatch for {fold['name']}")
        rows = []
        for base in all_rows:
            row = dict(base)
            row.update(fold_predictions[str(row["segment_id"])])
            rows.append(row)
        train = [row for row in rows if row["speech_group"] != speech]
        test = [row for row in rows if row["speech_group"] == speech]
        for name, features in model_specs.items():
            values = ridge_predict(train, test, features)
            for row, value in zip(test, values):
                all_predictions[name].append({
                    "model": name,
                    "segment_id": row["segment_id"],
                    "speech_group": row["speech_group"],
                    "interpreter": row["interpreter"],
                    "LAT": row["perceived_latency"],
                    "prediction": float(value),
                    "quality_fold": fold["name"],
                    "pred_LQ": row["pred_LQ"],
                    "pred_EXP": row["pred_EXP"],
                })
        fold_log.append({"outer_speech": speech, "quality_fold": fold["name"], "n_train": len(train), "n_test": len(test)})

    results = {"metadata": {"manifest": str((DATA_ROOT / 'manifest.json').relative_to(BASE)), "prediction_root": str(pred_root.relative_to(BASE)), "folds": fold_log}, "models": {}}
    table = []
    for name, rows in all_predictions.items():
        rows.sort(key=lambda row: row["segment_id"])
        value = metrics([row["LAT"] for row in rows], [row["prediction"] for row in rows], [row["speech_group"] for row in rows])
        results["models"][name] = {"features": model_specs[name], "metrics": value}
        table.append({"model": name, **value})
    (out_dir / "r027_automatic_bridge_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "r027_automatic_bridge_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (out_dir / "r027_automatic_bridge_oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[next(iter(all_predictions))][0]))
        writer.writeheader()
        for rows in all_predictions.values():
            writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
