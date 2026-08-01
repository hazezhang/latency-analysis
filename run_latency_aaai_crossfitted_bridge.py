#!/usr/bin/env python3
"""Evaluate LAT using cross-fitted quality predictions under outer speech CV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).parent
DEFAULT_DATA_ROOT = BASE / "data/experiments/aaai_crossfitted_outer_quality_corrected"


def corr(y, prediction, method="pearson"):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float(pearsonr(y, prediction)[0] if method == "pearson" else spearmanr(y, prediction)[0])


def cluster_ci(y, prediction, groups, method, seed=20260720, bootstraps=5000):
    y, prediction, groups = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float), np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstraps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        value = corr(y[indices], prediction[indices], method)
        if value is not None and np.isfinite(value):
            values.append(value)
    return [round(float(np.quantile(values, 0.025)), 4), round(float(np.quantile(values, 0.975)), 4)]


def metrics(y, prediction, groups):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    return {
        "n_segments": int(len(y)), "pearson": round(corr(y, prediction), 4),
        "pearson_ci95": cluster_ci(y, prediction, groups, "pearson"),
        "spearman": round(corr(y, prediction, "spearman"), 4),
        "spearman_ci95": cluster_ci(y, prediction, groups, "spearman"),
        "mse": round(float(np.mean((y - prediction) ** 2)), 4),
        "mae": round(float(np.mean(np.abs(y - prediction))), 4),
        "r2": round(float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)), 4),
        "pred_std": round(float(np.std(prediction)), 4),
    }


def feature_vector(row, names):
    source_tokens = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(row["src"]).lower())
    target_tokens = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(row["mt"]).lower())
    source_length = len(source_tokens)
    target_length = len(target_tokens)
    values = {
        "delay_linear": row["delay_seconds"], "delay_hinge_2": max(0.0, row["delay_seconds"] - 2.0),
        "delay_hinge_4": max(0.0, row["delay_seconds"] - 4.0), "delay_hinge_6": max(0.0, row["delay_seconds"] - 6.0),
        "delay_hinge_10": max(0.0, row["delay_seconds"] - 10.0), "LQ": row["LQ"], "EXP": row["EXP"],
        "pred_LQ": row["pred_LQ"], "pred_EXP": row["pred_EXP"],
        "pred_quality_mean": (row["pred_LQ"] + row["pred_EXP"]) / 2.0,
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


def ridge_predict(train_rows, test_rows, features, ridge_alpha=1.0):
    x_train = np.vstack([feature_vector(row, features) for row in train_rows])
    y_train = np.asarray([row["perceived_latency"] for row in train_rows], dtype=float)
    x_test = np.vstack([feature_vector(row, features) for row in test_rows])
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train, x_test = (x_train - mean) / scale, (x_test - mean) / scale
    x_train, x_test = np.column_stack([np.ones(len(x_train)), x_train]), np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(x_train.shape[1]); penalty[0, 0] = 0.0
    penalty *= ridge_alpha
    if ridge_alpha == 0:
        return x_test @ np.linalg.lstsq(x_train, y_train, rcond=None)[0]
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
    parser.add_argument("--prediction-root", required=True, help="Seed directory produced by run_aaai_crossfitted_outer_quality.sh")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT.relative_to(BASE)))
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Fixed nonnegative Ridge penalty; 0 is OLS.")
    args = parser.parse_args()
    if args.ridge_alpha < 0:
        raise ValueError("--ridge-alpha must be nonnegative")
    pred_root = BASE / args.prediction_root
    out_dir = BASE / args.output_dir
    data_root = BASE / args.data_root
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    all_rows = {str(row["segment_id"]): row for row in json.loads((data_root / "all_lat_segments.json").read_text(encoding="utf-8"))}
    specs = {
        "delay_piecewise": ["delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"],
        "human_LQ_EXP": ["LQ", "EXP"],
        "auto_pred_quality_mean": ["pred_quality_mean"],
        "auto_pred_quality_mean_piecewise_delay": ["pred_quality_mean", "delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"],
        "auto_pred_LQ_EXP": ["pred_LQ", "pred_EXP"],
        "auto_pred_LQ_EXP_piecewise_delay": ["pred_LQ", "pred_EXP", "delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"],
        "lexical_structural": ["source_length", "target_length", "length_ratio", "target_punctuation", "target_sentence_endings", "target_lexical_diversity", "very_short_output", "direction_en_zh"],
        "lexical_structural_piecewise_delay": ["source_length", "target_length", "length_ratio", "target_punctuation", "target_sentence_endings", "target_lexical_diversity", "very_short_output", "direction_en_zh", "delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"],
        "auto_pred_LQ_EXP_lexical_structural": ["pred_LQ", "pred_EXP", "source_length", "target_length", "length_ratio", "target_punctuation", "target_sentence_endings", "target_lexical_diversity", "very_short_output", "direction_en_zh"],
        "auto_pred_LQ_EXP_piecewise_delay_lexical_structural": ["pred_LQ", "pred_EXP", "delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10", "source_length", "target_length", "length_ratio", "target_punctuation", "target_sentence_endings", "target_lexical_diversity", "very_short_output", "direction_en_zh"],
    }
    all_predictions = {name: [] for name in specs}
    fold_log = []

    for fold in manifest["folds"]:
        outer_name = fold["name"]
        outer_speech = fold["outer_test_speech"]
        train_predictions: dict[str, dict[str, float]] = {}
        for inner in fold["inner_folds"]:
            values = load_predictions(pred_root / outer_name / inner["name"] / "predictions.json")
            duplicate = set(train_predictions) & set(values)
            if duplicate:
                raise ValueError(f"OOF duplicate rows in {outer_name}: {sorted(duplicate)[:3]}")
            train_predictions.update(values)
        outer_predictions = load_predictions(pred_root / outer_name / "final_outer" / "predictions.json")
        train_rows = []
        test_rows = []
        for sid, row in all_rows.items():
            enriched = dict(row)
            if row["speech_group"] == outer_speech:
                if sid not in outer_predictions:
                    raise ValueError(f"Missing outer prediction for {outer_name}: {sid}")
                enriched.update(outer_predictions[sid])
                test_rows.append(enriched)
            else:
                if sid not in train_predictions:
                    raise ValueError(f"Missing OOF train prediction for {outer_name}: {sid}")
                enriched.update(train_predictions[sid])
                train_rows.append(enriched)
        if len(train_rows) + len(test_rows) != len(all_rows):
            raise ValueError(f"Cohort mismatch in {outer_name}")
        for name, features in specs.items():
            values = ridge_predict(train_rows, test_rows, features, args.ridge_alpha)
            all_predictions[name].extend({
                "model": name, "segment_id": row["segment_id"], "speech_group": row["speech_group"],
                "interpreter": row["interpreter"], "LAT": row["perceived_latency"], "prediction": float(value),
                "outer_fold": outer_name, "pred_LQ": row["pred_LQ"], "pred_EXP": row["pred_EXP"],
            } for row, value in zip(test_rows, values))
        fold_log.append({"outer_fold": outer_name, "outer_speech": outer_speech, "n_train_oof": len(train_rows), "n_test": len(test_rows)})

    results = {
        "metadata": {
            "protocol": manifest["protocol"],
            "data_root": str(data_root.relative_to(BASE)),
            "prediction_root": str(pred_root.relative_to(BASE)),
            "ridge_alpha": args.ridge_alpha,
            "folds": fold_log,
        },
        "models": {},
    }
    table = []
    for name, rows in all_predictions.items():
        rows.sort(key=lambda row: row["segment_id"])
        value = metrics([row["LAT"] for row in rows], [row["prediction"] for row in rows], [row["speech_group"] for row in rows])
        results["models"][name] = {"features": specs[name], "metrics": value}
        table.append({"model": name, **value})
    (out_dir / "crossfitted_lat_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "crossfitted_lat_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (out_dir / "crossfitted_lat_oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[next(iter(all_predictions))][0]))
        writer.writeheader()
        for rows in all_predictions.values():
            writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
