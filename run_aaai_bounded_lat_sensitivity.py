#!/usr/bin/env python3
"""Range and calibration sensitivity for the fixed second-stage LAT model.

This audit reuses the existing outer-fold OOF quality predictions.  It never
uses an outer-test LAT label to fit, calibrate, select, or bound a prediction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).resolve().parent
SEEDS = (20260718, 20260719, 20260720)
LOWER, UPPER = 0.5, 3.0
FEATURES = ("pred_LQ", "pred_EXP", "delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10")
PROTOCOLS = {
    "source_speech_group_held_out": {
        "oof_pattern": "experiments/aaai_crossfitted_corrected_lat_seed_{seed}_20260721/crossfitted_lat_oof_predictions.csv",
        "group_key": "speech_group",
    },
    "interpreter_disjoint": {
        "oof_pattern": "experiments/aaai_loio_corrected_lat_seed_{seed}/loio_lat_oof_predictions.csv",
        "group_key": "interpreter",
    },
}


def corr(y: np.ndarray, prediction: np.ndarray, kind: str) -> float | None:
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    fn = pearsonr if kind == "pearson" else spearmanr
    return float(fn(y, prediction)[0])


def calibration(y: np.ndarray, prediction: np.ndarray) -> tuple[float | None, float | None]:
    """OLS calibration convention: gold = intercept + slope * prediction."""
    if np.std(prediction) == 0:
        return float(np.mean(y)), None
    slope, intercept = np.polyfit(prediction, y, deg=1)
    return float(intercept), float(slope)


def evaluate(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    intercept, slope = calibration(y, prediction)
    return {
        "n_segments": int(len(y)),
        "below_0_5_count": int(np.sum(prediction < LOWER)),
        "below_0_5_rate": float(np.mean(prediction < LOWER)),
        "above_3_0_count": int(np.sum(prediction > UPPER)),
        "above_3_0_rate": float(np.mean(prediction > UPPER)),
        "pearson": corr(y, prediction, "pearson"),
        "spearman": corr(y, prediction, "spearman"),
        "mse": float(np.mean((prediction - y) ** 2)),
        "mae": float(np.mean(np.abs(prediction - y))),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "prediction_mean": float(np.mean(prediction)),
        "prediction_sd": float(np.std(prediction)),
    }


def prepare_features(rows: list[dict]) -> np.ndarray:
    values = []
    for row in rows:
        delay = float(row["delay_seconds"])
        values.append([
            float(row["pred_LQ"]), float(row["pred_EXP"]), delay,
            max(0.0, delay - 2.0), max(0.0, delay - 4.0),
            max(0.0, delay - 6.0), max(0.0, delay - 10.0),
        ])
    return np.asarray(values, dtype=float)


def sigmoid_predict(train_rows: list[dict], test_rows: list[dict], alpha: float = 1.0) -> np.ndarray:
    x_train, x_test = prepare_features(train_rows), prepare_features(test_rows)
    y = np.asarray([float(row["perceived_latency"]) for row in train_rows])
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
    x_test = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])

    # Squared error with the same fixed L2 weight as the Ridge sensitivity.
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        p = expit(x_train @ theta)
        prediction = LOWER + (UPPER - LOWER) * p
        residual = prediction - y
        derivative = (UPPER - LOWER) * p * (1.0 - p)
        penalty = 0.5 * alpha * np.dot(theta[1:], theta[1:])
        value = 0.5 * np.dot(residual, residual) + penalty
        gradient = x_train.T @ (residual * derivative)
        gradient[1:] += alpha * theta[1:]
        return float(value), gradient

    initial = np.zeros(x_train.shape[1])
    mean_target = np.clip((y.mean() - LOWER) / (UPPER - LOWER), 1e-4, 1 - 1e-4)
    initial[0] = np.log(mean_target / (1 - mean_target))
    fitted = minimize(lambda theta: objective(theta), initial, jac=True, method="L-BFGS-B")
    if not fitted.success:
        raise RuntimeError(f"Bounded optimizer failed: {fitted.message}")
    return LOWER + (UPPER - LOWER) * expit(x_test @ fitted.x)


def load_seed_rows(oof_path: Path, data_path: Path) -> list[dict]:
    data = {str(row["segment_id"]): row for row in json.loads(data_path.read_text(encoding="utf-8"))}
    rows = []
    with oof_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "auto_pred_LQ_EXP_piecewise_delay":
                continue
            sid = str(row["segment_id"])
            enriched = dict(data[sid])
            enriched.update({
                "outer_fold": row["outer_fold"],
                "pred_LQ": float(row["pred_LQ"]),
                "pred_EXP": float(row["pred_EXP"]),
                "ridge_prediction": float(row["prediction"]),
            })
            rows.append(enriched)
    if len(rows) != len(data) or len({str(row["segment_id"]) for row in rows}) != len(data):
        raise ValueError(f"Unexpected main-model OOF coverage: {oof_path}")
    return rows


def run_protocol(name: str, config: dict, seed: int) -> tuple[list[dict], list[dict]]:
    oof_path = BASE / config["oof_pattern"].format(seed=seed)
    root = "data/experiments/aaai_crossfitted_outer_quality_corrected" if name == "source_speech_group_held_out" else "data/experiments/aaai_loio_outer_quality_corrected"
    rows = load_seed_rows(oof_path, BASE / root / "all_lat_segments.json")
    all_predictions: dict[str, dict[str, float]] = {
        "ridge_raw": {}, "ridge_clipped": {}, "bounded_sigmoid": {}, "training_fold_mean": {},
    }
    for fold in sorted({str(row["outer_fold"]) for row in rows}):
        test_rows = [row for row in rows if str(row["outer_fold"]) == fold]
        train_rows = [row for row in rows if str(row["outer_fold"]) != fold]
        if not train_rows or not test_rows:
            raise ValueError(f"Empty fold {fold}")
        raw = np.asarray([float(row["ridge_prediction"]) for row in test_rows])
        predicted = {
            "ridge_raw": raw,
            "ridge_clipped": np.clip(raw, LOWER, UPPER),
            "bounded_sigmoid": sigmoid_predict(train_rows, test_rows),
            "training_fold_mean": np.repeat(np.mean([float(row["perceived_latency"]) for row in train_rows]), len(test_rows)),
        }
        for method, values in predicted.items():
            for row, value in zip(test_rows, values):
                sid = str(row["segment_id"])
                if sid in all_predictions[method]:
                    raise ValueError(f"Duplicate OOF prediction for {method}: {sid}")
                all_predictions[method][sid] = float(value)

    # Predictions are accumulated fold-by-fold while the OOF CSV is sorted by
    # model/segment; align explicitly by segment ID before corpus metrics.
    ordered = sorted(rows, key=lambda row: str(row["segment_id"]))
    y = np.asarray([float(row["perceived_latency"]) for row in ordered])
    metric_rows, prediction_rows = [], []
    for method, prediction_by_id in all_predictions.items():
        if set(prediction_by_id) != {str(row["segment_id"]) for row in rows}:
            raise ValueError(f"Incomplete OOF coverage for {method}")
        prediction = np.asarray([prediction_by_id[str(row["segment_id"])] for row in ordered])
        values = evaluate(y, prediction)
        metric_rows.append({"protocol": name, "seed": seed, "method": method, **values})
        for row, value in zip(ordered, prediction):
            prediction_rows.append({
                "protocol": name, "seed": seed, "method": method,
                "segment_id": row["segment_id"], "outer_fold": row["outer_fold"],
                "perceived_promptness": row["perceived_latency"], "prediction": float(value),
            })
    return metric_rows, prediction_rows


def main() -> int:
    out_dir = BASE / "experiments/aaai_bounded_lat_sensitivity_20260726_r2"
    out_dir.mkdir(parents=True, exist_ok=False)
    metrics, predictions = [], []
    for name, config in PROTOCOLS.items():
        for seed in SEEDS:
            value, outputs = run_protocol(name, config, seed)
            metrics.extend(value)
            predictions.extend(outputs)
    fields = list(metrics[0])
    with (out_dir / "bounded_lat_metrics_by_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(metrics)
    with (out_dir / "bounded_lat_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    aggregate = []
    numeric = [name for name in fields if name not in {"protocol", "seed", "method", "n_segments", "below_0_5_count", "above_3_0_count"}]
    for protocol in PROTOCOLS:
        for method in ("ridge_raw", "ridge_clipped", "bounded_sigmoid", "training_fold_mean"):
            selected = [row for row in metrics if row["protocol"] == protocol and row["method"] == method]
            row = {"protocol": protocol, "method": method, "n_seeds": len(selected), "n_segments": selected[0]["n_segments"]}
            for key in ("below_0_5_count", "above_3_0_count"):
                row[f"{key}_mean"] = float(np.mean([entry[key] for entry in selected]))
            for key in numeric:
                values = [entry[key] for entry in selected if entry[key] is not None]
                row[f"{key}_mean"] = float(np.mean(values)) if values else None
                row[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            aggregate.append(row)
    with (out_dir / "bounded_lat_metrics_three_seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
    payload = {"metadata": {"target_range": [LOWER, UPPER], "seeds": SEEDS, "features": FEATURES, "calibration_convention": "OLS gold = intercept + slope * prediction", "bounded_model": "fold-trained sigmoid-scaled continuous regression with fixed alpha=1.0", "no_outer_test_labels_used": True}, "by_seed": metrics, "three_seed_summary": aggregate}
    (out_dir / "bounded_lat_sensitivity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
