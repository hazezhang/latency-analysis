#!/usr/bin/env python3
"""Evaluate perceived LAT prediction for completely held-out interpreters."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE / "data/experiments/aaai_loio_outer_quality_corrected"
DELAY_FEATURES = ["delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"]
STRUCTURAL_FEATURES = [
    "source_length",
    "target_length",
    "length_ratio",
    "target_punctuation",
    "target_sentence_endings",
    "target_lexical_diversity",
    "very_short_output",
    "direction_en_zh",
]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE))
    except ValueError:
        return str(path)


def corr(y, prediction, method="pearson"):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float(pearsonr(y, prediction)[0] if method == "pearson" else spearmanr(y, prediction)[0])


def cluster_ci(y, prediction, groups, method, seed=20260722, bootstraps=5000):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstraps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        value = corr(y[indices], prediction[indices], method)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return [None, None]
    return [round(float(np.quantile(values, 0.025)), 4), round(float(np.quantile(values, 0.975)), 4)]


def metrics(y, prediction, groups):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    pearson = corr(y, prediction)
    spearman = corr(y, prediction, "spearman")
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "n_segments": int(len(y)),
        "pearson": round(pearson, 4) if pearson is not None else None,
        "pearson_ci95": cluster_ci(y, prediction, groups, "pearson"),
        "spearman": round(spearman, 4) if spearman is not None else None,
        "spearman_ci95": cluster_ci(y, prediction, groups, "spearman"),
        "mse": round(float(np.mean((y - prediction) ** 2)), 4),
        "mae": round(float(np.mean(np.abs(y - prediction))), 4),
        "r2": round(float(1 - np.sum((y - prediction) ** 2) / denominator), 4) if denominator else None,
        "pred_std": round(float(np.std(prediction)), 4),
    }


def per_interpreter_metrics(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["interpreter"])].append(row)
    output = {}
    for interpreter, items in sorted(grouped.items()):
        y = np.asarray([float(row["LAT"]) for row in items])
        prediction = np.asarray([float(row["prediction"]) for row in items])
        pearson = corr(y, prediction)
        spearman = corr(y, prediction, "spearman")
        output[interpreter] = {
            "n_segments": len(items),
            "directions": sorted({str(row["direction"]) for row in items}),
            "pearson": round(pearson, 4) if pearson is not None else None,
            "spearman": round(spearman, 4) if spearman is not None else None,
            "mse": round(float(np.mean((y - prediction) ** 2)), 4),
            "mae": round(float(np.mean(np.abs(y - prediction))), 4),
            "calibration_bias": round(float(np.mean(prediction - y)), 4),
            "gold_std": round(float(np.std(y)), 4),
            "pred_std": round(float(np.std(prediction)), 4),
        }
    return output


def within_interpreter_centered_metrics(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["interpreter"])].append(row)
    centered_y = []
    centered_prediction = []
    for items in grouped.values():
        y = np.asarray([float(row["LAT"]) for row in items])
        prediction = np.asarray([float(row["prediction"]) for row in items])
        centered_y.extend(y - y.mean())
        centered_prediction.extend(prediction - prediction.mean())
    value = corr(centered_y, centered_prediction)
    return {
        "n_segments": len(centered_y),
        "pearson": round(value, 4) if value is not None else None,
    }


def macro_interpreter_metrics(per_interpreter: dict[str, dict]) -> dict:
    output = {}
    for metric in ("pearson", "spearman", "mse", "mae", "calibration_bias"):
        values = [row[metric] for row in per_interpreter.values() if row[metric] is not None]
        output[metric] = round(float(np.mean(values)), 4) if values else None
        output[f"n_{metric}"] = len(values)
    return output


def direction_metrics(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["direction"])].append(row)
    output = {}
    for direction, items in sorted(grouped.items()):
        output[direction] = metrics(
            [row["LAT"] for row in items],
            [row["prediction"] for row in items],
            [row["interpreter"] for row in items],
        )
    return output


def feature_vector(row, names):
    delay = float(row["delay_seconds"])
    source_tokens = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(row["src"]).lower())
    target_tokens = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(row["mt"]).lower())
    source_length = len(source_tokens)
    target_length = len(target_tokens)
    values = {
        "delay_linear": delay,
        "delay_hinge_2": max(0.0, delay - 2.0),
        "delay_hinge_4": max(0.0, delay - 4.0),
        "delay_hinge_6": max(0.0, delay - 6.0),
        "delay_hinge_10": max(0.0, delay - 10.0),
        "pred_LQ": float(row["pred_LQ"]),
        "pred_EXP": float(row["pred_EXP"]),
        "pred_quality_mean": (float(row["pred_LQ"]) + float(row["pred_EXP"])) / 2.0,
        "source_length": source_length,
        "target_length": target_length,
        "length_ratio": target_length / max(source_length, 1),
        "target_punctuation": len(re.findall(r"[.!?;,:。！？；，：]", str(row["mt"]))),
        "target_sentence_endings": len(re.findall(r"[.!?。！？]", str(row["mt"]))),
        "target_lexical_diversity": len(set(target_tokens)) / max(target_length, 1),
        "very_short_output": float(target_length < 5),
        "direction_en_zh": float(row["direction"] == "en-zh"),
    }
    return np.asarray([values[name] for name in names], dtype=float)


def ridge_predict(train_rows, test_rows, features):
    x_train = np.vstack([feature_vector(row, features) for row in train_rows])
    y_train = np.asarray([row["perceived_latency"] for row in train_rows], dtype=float)
    x_test = np.vstack([feature_vector(row, features) for row in test_rows])
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(x_train.shape[1])
    penalty[0, 0] = 0.0
    return x_test @ np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)


def load_predictions(path: Path) -> dict[str, dict[str, float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        sid = str(row["segment_id"])
        if sid in values:
            raise ValueError(f"Duplicate prediction: {path} / {sid}")
        values[sid] = {"pred_LQ": float(row["pred_LQ"]), "pred_EXP": float(row["pred_EXP"])}
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT.relative_to(BASE)))
    args = parser.parse_args()
    pred_root = BASE / args.prediction_root
    out_dir = BASE / args.output_dir
    data_root = BASE / args.data_root
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    source_rows = json.loads((data_root / "all_lat_segments.json").read_text(encoding="utf-8"))
    all_rows = {str(row["segment_id"]): row for row in source_rows}
    if len(all_rows) != len(source_rows):
        raise ValueError("Duplicate cohort segment IDs")
    specs = {
        "delay_piecewise": DELAY_FEATURES,
        "auto_pred_quality_mean": ["pred_quality_mean"],
        "auto_pred_quality_mean_piecewise_delay": ["pred_quality_mean", *DELAY_FEATURES],
        "auto_pred_LQ_EXP": ["pred_LQ", "pred_EXP"],
        "auto_pred_LQ_EXP_piecewise_delay": ["pred_LQ", "pred_EXP", *DELAY_FEATURES],
        "lexical_structural": STRUCTURAL_FEATURES,
        "lexical_structural_piecewise_delay": [*STRUCTURAL_FEATURES, *DELAY_FEATURES],
        "auto_pred_LQ_EXP_lexical_structural": ["pred_LQ", "pred_EXP", *STRUCTURAL_FEATURES],
        "auto_pred_LQ_EXP_piecewise_delay_lexical_structural": [
            "pred_LQ",
            "pred_EXP",
            *STRUCTURAL_FEATURES,
            *DELAY_FEATURES,
        ],
    }
    all_predictions = {name: [] for name in specs}
    fold_log = []

    for fold in manifest["folds"]:
        outer_name = str(fold["name"])
        outer_interpreter = str(fold["outer_test_interpreter"])
        train_predictions: dict[str, dict[str, float]] = {}
        for inner in fold["inner_folds"]:
            values = load_predictions(pred_root / outer_name / inner["name"] / "predictions.json")
            duplicate = set(train_predictions) & set(values)
            if duplicate:
                raise ValueError(f"OOF duplicate rows in {outer_name}: {sorted(duplicate)[:3]}")
            train_predictions.update(values)
        outer_predictions = load_predictions(pred_root / outer_name / "final_outer" / "predictions.json")
        expected_train = {sid for sid, row in all_rows.items() if str(row["interpreter"]) != outer_interpreter}
        expected_test = set(all_rows) - expected_train
        if set(train_predictions) != expected_train:
            raise ValueError(f"Inner-OOF coverage mismatch in {outer_name}")
        if set(outer_predictions) != expected_test:
            raise ValueError(f"Outer prediction coverage mismatch in {outer_name}")

        train_rows = []
        test_rows = []
        for sid, row in all_rows.items():
            enriched = dict(row)
            if str(row["interpreter"]) == outer_interpreter:
                enriched.update(outer_predictions[sid])
                test_rows.append(enriched)
            else:
                enriched.update(train_predictions[sid])
                train_rows.append(enriched)
        if {str(row["interpreter"]) for row in test_rows} != {outer_interpreter}:
            raise ValueError(f"LAT test interpreter mismatch in {outer_name}")
        if outer_interpreter in {str(row["interpreter"]) for row in train_rows}:
            raise ValueError(f"Outer interpreter leaked into LAT training in {outer_name}")

        for name, features in specs.items():
            values = ridge_predict(train_rows, test_rows, features)
            all_predictions[name].extend({
                "model": name,
                "segment_id": row["segment_id"],
                "speech_group": row["speech_group"],
                "interpreter": row["interpreter"],
                "direction": row["direction"],
                "LAT": row["perceived_latency"],
                "prediction": float(value),
                "outer_fold": outer_name,
                "pred_LQ": row["pred_LQ"],
                "pred_EXP": row["pred_EXP"],
            } for row, value in zip(test_rows, values))
        fold_log.append({
            "outer_fold": outer_name,
            "outer_interpreter": outer_interpreter,
            "n_train_oof": len(train_rows),
            "n_test": len(test_rows),
            "source_speech_overlap_allowed": True,
        })

    results = {
        "metadata": {
            "protocol": manifest["protocol"],
            "scope": manifest["scope"],
            "data_root": display_path(data_root),
            "prediction_root": display_path(pred_root),
            "folds": fold_log,
        },
        "models": {},
    }
    table = []
    for name, rows in all_predictions.items():
        rows.sort(key=lambda row: str(row["segment_id"]))
        if len(rows) != len(all_rows) or len({str(row["segment_id"]) for row in rows}) != len(all_rows):
            raise ValueError(f"Final LOIO coverage mismatch for {name}")
        value = metrics(
            [row["LAT"] for row in rows],
            [row["prediction"] for row in rows],
            [row["interpreter"] for row in rows],
        )
        per_interpreter = per_interpreter_metrics(rows)
        results["models"][name] = {
            "features": specs[name],
            "metrics": value,
            "per_interpreter": per_interpreter,
            "macro_interpreter": macro_interpreter_metrics(per_interpreter),
            "within_interpreter_centered": within_interpreter_centered_metrics(rows),
            "by_direction": direction_metrics(rows),
        }
        table.append({"model": name, **value})
    (out_dir / "loio_lat_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "loio_lat_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (out_dir / "loio_lat_oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[next(iter(all_predictions))][0]))
        writer.writeheader()
        for rows in all_predictions.values():
            writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
