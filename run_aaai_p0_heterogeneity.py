#!/usr/bin/env python3
"""Direction, interpreter, and source-speech heterogeneity tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
COHORT_PATH = ROOT / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"
OUT = ROOT / "experiments/aaai_p0_heterogeneity_20260722"
SEEDS = ("20260718", "20260719", "20260720")
SPEECH_MODELS = ("delay_piecewise", "auto_pred_LQ_EXP", "auto_pred_LQ_EXP_piecewise_delay")
LOIO_MODELS = SPEECH_MODELS


def safe_corr(gold, prediction, method="pearson"):
    gold, prediction = np.asarray(gold, dtype=float), np.asarray(prediction, dtype=float)
    if len(gold) < 3 or np.std(gold) == 0 or np.std(prediction) == 0:
        return None
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(gold, prediction).statistic)


def metrics(rows):
    gold = np.asarray([row["LAT"] for row in rows], dtype=float)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=float)
    return {
        "n": len(rows),
        "pearson": safe_corr(gold, prediction),
        "spearman": safe_corr(gold, prediction, "spearman"),
        "mse": float(np.mean((gold - prediction) ** 2)),
        "mae": float(np.mean(np.abs(gold - prediction))),
        "bias": float(np.mean(prediction - gold)),
        "pred_std": float(prediction.std()),
    }


def load_prediction_csv(path, models):
    output = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in models:
                continue
            output.append({
                "model": row["model"], "segment_id": row["segment_id"],
                "speech_group": str(row["speech_group"]), "interpreter": row["interpreter"],
                "LAT": float(row["LAT"]), "prediction": float(row["prediction"]),
            })
    return output


def ensemble_rows(seed_rows, model):
    predictions = defaultdict(list)
    metadata = {}
    for rows in seed_rows.values():
        for row in rows:
            if row["model"] != model:
                continue
            predictions[row["segment_id"]].append(row["prediction"])
            metadata[row["segment_id"]] = row
    output = []
    for sid, values in predictions.items():
        if len(values) != len(SEEDS):
            raise ValueError(f"{model}/{sid}: expected {len(SEEDS)} predictions")
        output.append({**metadata[sid], "prediction": float(np.mean(values))})
    return output


def direct_rows(seed, variant):
    rows = []
    root = ROOT / f"experiments/aaai_direct_lat_corrected_seed_{seed}" / variant
    for path in sorted(root.glob("outer_*/predictions.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows.append({
                "model": f"direct_{variant}", "segment_id": str(row["segment_id"]),
                "speech_group": str(row["speech_group"]), "interpreter": row["interpreter"],
                "LAT": float(row["LAT"]), "prediction": float(row["prediction"]),
            })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def round_value(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = {str(row["segment_id"]): row for row in json.loads(COHORT_PATH.read_text(encoding="utf-8"))}
    speech_seed_rows = {}
    loio_seed_rows = {}
    for seed in SEEDS:
        speech_seed_rows[seed] = load_prediction_csv(
            ROOT / f"experiments/aaai_crossfitted_corrected_lat_seed_{seed}_20260721/crossfitted_lat_oof_predictions.csv",
            SPEECH_MODELS,
        ) + direct_rows(seed, "text") + direct_rows(seed, "text_delay")
        loio_seed_rows[seed] = load_prediction_csv(
            ROOT / f"experiments/aaai_loio_corrected_lat_seed_{seed}/loio_lat_oof_predictions.csv",
            LOIO_MODELS,
        )

    speech_models = (*SPEECH_MODELS, "direct_text", "direct_text_delay")
    direction_rows = []
    speech_group_rows = []
    group_distribution = {}
    for model in speech_models:
        ensemble = ensemble_rows(speech_seed_rows, model)
        for row in ensemble:
            row["direction"] = cohort[row["segment_id"]]["direction"]
        direction_metrics = {}
        for direction in sorted({row["direction"] for row in ensemble}):
            item = metrics([row for row in ensemble if row["direction"] == direction])
            direction_metrics[direction] = item
            direction_rows.append({"model": model, "direction": direction, **item})
        valid_direction_r = [item["pearson"] for item in direction_metrics.values() if item["pearson"] is not None]
        direction_rows.append({
            "model": model, "direction": "macro_direction",
            "n": len(ensemble), "pearson": float(np.mean(valid_direction_r)),
            "spearman": None, "mse": float(np.mean([item["mse"] for item in direction_metrics.values()])),
            "mae": float(np.mean([item["mae"] for item in direction_metrics.values()])),
            "bias": float(np.mean([item["bias"] for item in direction_metrics.values()])),
            "pred_std": None,
        })
        per_group = []
        for group in sorted({row["speech_group"] for row in ensemble}):
            items = [row for row in ensemble if row["speech_group"] == group]
            item = metrics(items)
            meta = [cohort[row["segment_id"]] for row in items]
            record = {
                "model": model, "speech_group": group,
                "direction": "+".join(sorted({row["direction"] for row in meta})),
                "n_interpreters": len({row["interpreter"] for row in meta}),
                **item,
            }
            speech_group_rows.append(record)
            if item["pearson"] is not None:
                per_group.append(item["pearson"])
        array = np.asarray(per_group, dtype=float)
        group_distribution[model] = {
            "n_valid_groups": len(array), "macro_pearson": float(array.mean()),
            "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max()),
            "q25": float(np.quantile(array, 0.25)), "q75": float(np.quantile(array, 0.75)),
            "n_negative": int(np.sum(array < 0)),
        }

    interpreter_rows = []
    for model in LOIO_MODELS:
        ensemble = ensemble_rows(loio_seed_rows, model)
        for interpreter in sorted({row["interpreter"] for row in ensemble}):
            items = [row for row in ensemble if row["interpreter"] == interpreter]
            item = metrics(items)
            meta = [cohort[row["segment_id"]] for row in items]
            gold = np.asarray([row["LAT"] for row in items], dtype=float)
            delay = np.asarray([row["delay_seconds"] for row in meta], dtype=float)
            interpreter_rows.append({
                "model": model, "interpreter": interpreter, "n": len(items),
                "n_speech_groups": len({row["speech_group"] for row in meta}),
                "direction": "+".join(sorted({row["direction"] for row in meta})),
                "gold_LAT_mean": float(gold.mean()), "gold_LAT_std": float(gold.std()),
                "gold_LAT_min": float(gold.min()), "gold_LAT_max": float(gold.max()),
                "delay_mean": float(delay.mean()), "delay_std": float(delay.std()),
                "pearson": item["pearson"],
                "centered_pearson": item["pearson"],
                "mse": item["mse"], "mae": item["mae"], "mean_bias": item["bias"],
            })

    direction_rows = [{key: round_value(value) for key, value in row.items()} for row in direction_rows]
    speech_group_rows = [{key: round_value(value) for key, value in row.items()} for row in speech_group_rows]
    interpreter_rows = [{key: round_value(value) for key, value in row.items()} for row in interpreter_rows]
    write_csv(OUT / "direction_results.csv", direction_rows)
    write_csv(OUT / "speech_group_results.csv", speech_group_rows)
    write_csv(OUT / "interpreter_disjoint_results.csv", interpreter_rows)
    payload = {
        "protocol": {
            "seed_aggregation": "per-segment mean prediction across the three fixed seeds",
            "macro_direction": "unweighted mean of the two direction-specific metrics",
            "interpreter_centered_note": "within one held-out interpreter, subtracting a constant leaves Pearson unchanged",
            "inference_note": "tables are descriptive; cluster-aware inferential comparisons remain in reviewer_priority_audits.json",
        },
        "group_result_distribution": {model: {key: round_value(value) for key, value in item.items()} for model, item in group_distribution.items()},
    }
    (OUT / "p0_heterogeneity_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
