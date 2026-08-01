#!/usr/bin/env python3
"""Source-speech-held-out length and lexical/structural LAT baselines."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"
OUT = ROOT / "experiments/aaai_p1_simple_confound_baselines_20260722"
BOOTSTRAPS = 5000
SEED = 20260722


def tokens(text):
    return re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text).lower())


def features(row):
    source = tokens(row["src"])
    target = tokens(row["mt"])
    source_n, target_n = len(source), len(target)
    punctuation = re.findall(r"[.!?;,:。！？；，：]", str(row["mt"]))
    sentence_endings = re.findall(r"[.!?。！？]", str(row["mt"]))
    lexical_diversity = len(set(target)) / max(target_n, 1)
    delay = float(row["delay_seconds"])
    return {
        "source_length": float(source_n),
        "target_length": float(target_n),
        "length_ratio": float(target_n / max(source_n, 1)),
        "target_punctuation": float(len(punctuation)),
        "target_sentence_endings": float(len(sentence_endings)),
        "target_lexical_diversity": float(lexical_diversity),
        "very_short_output": float(target_n < 5),
        "direction_en_zh": float(row["direction"] == "en-zh"),
        "delay": delay,
        "hinge_2": max(0.0, delay - 2.0),
        "hinge_4": max(0.0, delay - 4.0),
        "hinge_6": max(0.0, delay - 6.0),
        "hinge_10": max(0.0, delay - 10.0),
    }


SPECS = {
    "source_length_only": ("source_length",),
    "target_length_only": ("target_length",),
    "length_ratio_only": ("length_ratio",),
    "direction_only": ("direction_en_zh",),
    "length_only": ("source_length", "target_length", "length_ratio"),
    "punctuation_completeness_only": ("target_punctuation", "target_sentence_endings", "target_lexical_diversity", "very_short_output"),
    "delay_plus_length": ("delay", "hinge_2", "hinge_4", "hinge_6", "hinge_10", "source_length", "target_length", "length_ratio"),
    "lexical_structural_no_direction": ("source_length", "target_length", "length_ratio", "target_punctuation", "target_sentence_endings", "target_lexical_diversity", "very_short_output"),
    "lexical_structural": ("source_length", "target_length", "length_ratio", "target_punctuation", "target_sentence_endings", "target_lexical_diversity", "very_short_output", "direction_en_zh"),
    "delay_plus_lexical_structural": ("delay", "hinge_2", "hinge_4", "hinge_6", "hinge_10", "source_length", "target_length", "length_ratio", "target_punctuation", "target_sentence_endings", "target_lexical_diversity", "very_short_output", "direction_en_zh"),
}


def safe_corr(gold, prediction, method="pearson"):
    gold, prediction = np.asarray(gold), np.asarray(prediction)
    if np.std(gold) == 0 or np.std(prediction) == 0:
        return None
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(gold, prediction).statistic)


def ridge_predict(train, test, names, alpha=1.0):
    x_train = np.asarray([[row["features"][name] for name in names] for row in train], dtype=float)
    x_test = np.asarray([[row["features"][name] for name in names] for row in test], dtype=float)
    y_train = np.asarray([row["LAT"] for row in train], dtype=float)
    mean, scale = x_train.mean(0), x_train.std(0)
    scale[scale == 0] = 1.0
    x_train = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
    x_test = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_test @ beta


def metrics(rows):
    gold = np.asarray([row["LAT"] for row in rows])
    prediction = np.asarray([row["prediction"] for row in rows])
    return {
        "n": len(rows), "pearson": safe_corr(gold, prediction),
        "spearman": safe_corr(gold, prediction, "spearman"),
        "mse": float(np.mean((gold - prediction) ** 2)),
        "mae": float(np.mean(np.abs(gold - prediction))),
        "pred_std": float(prediction.std()),
    }


def cluster_ci(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[row["speech_group"]].append(row)
    names = sorted(groups)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(names, size=len(names), replace=True)
        value = metrics([row for name in sampled for row in groups[name]])[metric]
        if value is not None and np.isfinite(value):
            values.append(value)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw in json.loads(DATA.read_text(encoding="utf-8")):
        rows.append({
            **raw, "LAT": float(raw["perceived_latency"]), "features": features(raw),
        })
    predictions = {name: [] for name in SPECS}
    for outer in sorted({row["speech_group"] for row in rows}):
        train = [row for row in rows if row["speech_group"] != outer]
        test = [row for row in rows if row["speech_group"] == outer]
        for model, names in SPECS.items():
            values = ridge_predict(train, test, names)
            predictions[model].extend({
                "model": model, "segment_id": row["segment_id"],
                "speech_group": row["speech_group"], "interpreter": row["interpreter"],
                "direction": row["direction"], "LAT": row["LAT"], "prediction": float(value),
            } for row, value in zip(test, values))
    results = {}
    table = []
    for model, items in predictions.items():
        result = metrics(items)
        result["pearson_ci95_speech_cluster"] = cluster_ci(items, "pearson")
        result["mse_ci95_speech_cluster"] = cluster_ci(items, "mse")
        results[model] = result
        table.append({"model": model, **result})
    payload = {
        "protocol": {
            "outer_unit": "source_speech_group", "n_outer_groups": 16,
            "ridge_alpha": 1.0, "bootstrap_unit": "source_speech_group", "bootstrap_samples": BOOTSTRAPS,
            "token_definition": "each CJK character and each alphanumeric word is one lexical unit",
            "comet_score_status": "not evaluated because the shared cohort lacks complete original COMET/COMET-KIWI coverage",
        },
        "models": results,
    }
    (OUT / "simple_confound_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "simple_confound_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (OUT / "simple_confound_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        flat = [row for items in predictions.values() for row in items]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
