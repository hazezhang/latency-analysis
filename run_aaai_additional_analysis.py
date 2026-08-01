#!/usr/bin/env python3
"""Summarize one-factor and interpreter-disjoint diagnostics for the AAAI paper."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


BASE = Path(__file__).resolve().parent
SEEDS = ["20260718", "20260719", "20260720"]
OUT = BASE / "experiments/aaai_additional_analysis_20260722"
LOIO_MODELS = [
    "delay_piecewise",
    "auto_pred_quality_mean",
    "auto_pred_quality_mean_piecewise_delay",
    "auto_pred_LQ_EXP",
    "auto_pred_LQ_EXP_piecewise_delay",
]
SPEECH_MODELS = [
    "delay_piecewise",
    "human_LQ_EXP",
    "auto_pred_quality_mean",
    "auto_pred_quality_mean_piecewise_delay",
    "auto_pred_LQ_EXP",
    "auto_pred_LQ_EXP_piecewise_delay",
]


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": round(float(values.mean()), 4),
        "sd": round(float(values.std(ddof=1)), 4) if len(values) > 1 else 0.0,
    }


def bootstrap_pearson(y, prediction, seed=20260722, samples=10000):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = rng.integers(0, len(y), len(y))
        if np.std(y[index]) == 0 or np.std(prediction[index]) == 0:
            continue
        values.append(pearsonr(y[index], prediction[index])[0])
    return [
        round(float(np.quantile(values, 0.025)), 4),
        round(float(np.quantile(values, 0.975)), 4),
    ]


def load_loio_results():
    output = {}
    raw = {}
    for seed in SEEDS:
        root = BASE / f"experiments/aaai_loio_onefactor_seed_{seed}_20260722"
        raw[seed] = json.loads((root / "loio_lat_results.json").read_text(encoding="utf-8"))
    for model in LOIO_MODELS:
        output[model] = {}
        for metric in ["pearson", "spearman", "mse", "mae", "pred_std"]:
            output[model][metric] = mean_sd([
                raw[seed]["models"][model]["metrics"][metric] for seed in SEEDS
            ])
        output[model]["within_interpreter_centered_pearson"] = mean_sd([
            raw[seed]["models"][model]["within_interpreter_centered"]["pearson"]
            for seed in SEEDS
        ])
        output[model]["macro_interpreter_pearson"] = mean_sd([
            np.mean([
                item["pearson"]
                for item in raw[seed]["models"][model]["per_interpreter"].values()
                if item["pearson"] is not None
            ])
            for seed in SEEDS
        ])
    return output, raw


def load_speech_results():
    raw = {}
    for seed in SEEDS:
        root = BASE / f"experiments/aaai_crossfitted_onefactor_seed_{seed}_20260722"
        raw[seed] = json.loads((root / "crossfitted_lat_results.json").read_text(encoding="utf-8"))
    output = {}
    for model in SPEECH_MODELS:
        output[model] = {}
        for metric in ["pearson", "spearman", "mse", "mae", "pred_std"]:
            output[model][metric] = mean_sd([
                raw[seed]["models"][model]["metrics"][metric] for seed in SEEDS
            ])
    return output


def interpreter_table(raw):
    predictions = defaultdict(lambda: defaultdict(dict))
    metadata = {}
    model = "auto_pred_LQ_EXP_piecewise_delay"
    for seed in SEEDS:
        path = BASE / f"experiments/aaai_loio_onefactor_seed_{seed}_20260722/loio_lat_oof_predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["model"] != model:
                    continue
                sid = row["segment_id"]
                interpreter = row["interpreter"]
                predictions[interpreter][sid][seed] = float(row["prediction"])
                metadata[sid] = {
                    "LAT": float(row["LAT"]),
                    "direction": row["direction"],
                }

    rows = []
    for interpreter, segments in sorted(predictions.items()):
        sids = sorted(segments)
        y = np.asarray([metadata[sid]["LAT"] for sid in sids])
        ensemble = np.asarray([
            np.mean([segments[sid][seed] for seed in SEEDS]) for sid in sids
        ])
        r = float(pearsonr(y, ensemble)[0])
        per_seed = [
            raw[seed]["models"][model]["per_interpreter"][interpreter]
            for seed in SEEDS
        ]
        rows.append({
            "interpreter": interpreter,
            "n_segments": len(sids),
            "direction": "+".join(sorted({metadata[sid]["direction"] for sid in sids})),
            "delay_only_pearson": raw[SEEDS[0]]["models"]["delay_piecewise"]["per_interpreter"][interpreter]["pearson"],
            "predicted_quality_pearson_mean": round(float(np.mean([
                raw[seed]["models"]["auto_pred_LQ_EXP"]["per_interpreter"][interpreter]["pearson"]
                for seed in SEEDS
            ])), 4),
            "predicted_quality_delay_pearson_mean": round(float(np.mean([item["pearson"] for item in per_seed])), 4),
            "ensemble_pearson": round(r, 4),
            "ensemble_pearson_ci95": bootstrap_pearson(y, ensemble),
            "mae_mean": round(float(np.mean([item["mae"] for item in per_seed])), 4),
            "calibration_bias_mean": round(float(np.mean([item["calibration_bias"] for item in per_seed])), 4),
            "target_lat_std": per_seed[0]["gold_std"],
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loio_summary, raw = load_loio_results()
    speech_summary = load_speech_results()
    rows = interpreter_table(raw)
    payload = {
        "seeds": SEEDS,
        "loio_models": loio_summary,
        "speech_models": speech_summary,
        "per_interpreter_combined_model": rows,
        "notes": {
            "ensemble_ci": "segment bootstrap over the mean prediction from three fixed seeds",
            "centered_metric": "gold and prediction are centered within interpreter before pooled Pearson",
        },
    }
    (OUT / "additional_analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "loio_per_interpreter.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
