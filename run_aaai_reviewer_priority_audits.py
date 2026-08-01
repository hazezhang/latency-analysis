#!/usr/bin/env python3
"""Reviewer-priority audits over frozen cross-fitted AAAI predictions.

This script performs no neural training and does not select a best seed. It
uses the three formal source-speech-group-held-out OOF prediction files to
report direction, group-macro, and incremental quality/delay evidence, then
exports the existing evaluator-transfer audit as a paper-ready table.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
SEEDS = ("20260718", "20260719", "20260720")
PREDICTION_ROOT = "experiments/aaai_crossfitted_corrected_lat_seed_{}_20260721"
COHORT = ROOT / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"
RATER_AUDIT = ROOT / "experiments/aaai_reviewer_cpu_corrected_20260721/aaai_reviewer_cpu_results.json"
OUT = ROOT / "experiments/aaai_reviewer_priority_audits_20260722"
MODELS = (
    "delay_piecewise",
    "auto_pred_LQ_EXP",
    "auto_pred_LQ_EXP_piecewise_delay",
)
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20260722


def safe_corr(y, pred, method="pearson"):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return None
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(y, pred).statistic)


def metrics(rows):
    y = np.asarray([row["LAT"] for row in rows], dtype=float)
    pred = np.asarray([row["prediction"] for row in rows], dtype=float)
    return {
        "n": len(rows),
        "pearson": safe_corr(y, pred),
        "spearman": safe_corr(y, pred, "spearman"),
        "mse": float(np.mean((y - pred) ** 2)),
        "mae": float(np.mean(np.abs(y - pred))),
    }


def rounded(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def load_rows():
    metadata = {
        row["segment_id"]: row
        for row in json.loads(COHORT.read_text(encoding="utf-8"))
    }
    by_seed = {}
    for seed in SEEDS:
        path = ROOT / PREDICTION_ROOT.format(seed) / "crossfitted_lat_oof_predictions.csv"
        rows = []
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw["model"] not in MODELS:
                    continue
                meta = metadata[raw["segment_id"]]
                rows.append({
                    "seed": seed,
                    "model": raw["model"],
                    "segment_id": raw["segment_id"],
                    "speech_group": str(raw["speech_group"]),
                    "interpreter": raw["interpreter"],
                    "direction": meta["direction"],
                    "LAT": float(raw["LAT"]),
                    "prediction": float(raw["prediction"]),
                })
        expected = len(metadata) * len(MODELS)
        if len(rows) != expected:
            raise ValueError(f"{seed}: expected {expected} rows, found {len(rows)}")
        by_seed[seed] = rows
    return by_seed


def direction_results(by_seed):
    output = {}
    for seed, rows in by_seed.items():
        output[seed] = {}
        for model in MODELS:
            output[seed][model] = {}
            model_rows = [row for row in rows if row["model"] == model]
            for direction in sorted({row["direction"] for row in model_rows}):
                output[seed][model][direction] = metrics([
                    row for row in model_rows if row["direction"] == direction
                ])
    return output


def aggregate_seed_metrics(seed_results):
    values = defaultdict(list)
    for seed_items in seed_results.values():
        for model, subgroup_items in seed_items.items():
            for subgroup, item in subgroup_items.items():
                for metric in ("pearson", "spearman", "mse", "mae"):
                    values[(model, subgroup, metric)].append(item[metric])
    output = defaultdict(dict)
    for (model, subgroup, metric), items in values.items():
        array = np.asarray(items, dtype=float)
        output[model].setdefault(subgroup, {})[metric] = {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }
    return {model: dict(subgroups) for model, subgroups in output.items()}


def group_macro_results(by_seed):
    output = {}
    for seed, rows in by_seed.items():
        output[seed] = {}
        for model in MODELS:
            model_rows = [row for row in rows if row["model"] == model]
            per_group = {}
            for group in sorted({row["speech_group"] for row in model_rows}):
                group_metrics = metrics([row for row in model_rows if row["speech_group"] == group])
                per_group[group] = group_metrics
            valid = [item["pearson"] for item in per_group.values() if item["pearson"] is not None]
            output[seed][model] = {
                "n_valid_groups": len(valid),
                "macro_pearson": float(np.mean(valid)),
                "median_group_pearson": float(np.median(valid)),
                "per_group": per_group,
            }
    return output


def aggregate_group_macro(seed_results):
    output = {}
    for model in MODELS:
        result = {}
        for metric in ("macro_pearson", "median_group_pearson"):
            values = np.asarray([seed_results[seed][model][metric] for seed in SEEDS], dtype=float)
            result[metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
            }
        output[model] = result
    return output


def aligned_model_rows(rows):
    aligned = defaultdict(dict)
    metadata = {}
    for row in rows:
        aligned[row["segment_id"]][row["model"]] = row["prediction"]
        metadata[row["segment_id"]] = row
    for segment_id, values in aligned.items():
        missing = set(MODELS) - set(values)
        if missing:
            raise ValueError(f"{segment_id}: missing predictions {sorted(missing)}")
    return aligned, metadata


def cluster_bootstrap_delta(rows, added_model, base_model):
    aligned, metadata = aligned_model_rows(rows)
    groups = defaultdict(list)
    for segment_id, row in metadata.items():
        groups[row["speech_group"]].append(segment_id)
    group_names = sorted(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAPS):
        sampled_groups = rng.choice(group_names, size=len(group_names), replace=True)
        sampled_ids = [segment_id for group in sampled_groups for segment_id in groups[group]]
        y = np.asarray([metadata[segment_id]["LAT"] for segment_id in sampled_ids])
        added = np.asarray([aligned[segment_id][added_model] for segment_id in sampled_ids])
        base = np.asarray([aligned[segment_id][base_model] for segment_id in sampled_ids])
        added_r = safe_corr(y, added)
        base_r = safe_corr(y, base)
        if added_r is None or base_r is None:
            continue
        values.append((
            added_r - base_r,
            float(np.mean((y - added) ** 2) - np.mean((y - base) ** 2)),
        ))
    array = np.asarray(values)
    return {
        "definition": "added model minus base model; positive delta_r and negative delta_mse favor added model",
        "delta_pearson": {
            "estimate": float(np.mean(array[:, 0])),
            "ci95": np.quantile(array[:, 0], [0.025, 0.975]).tolist(),
            "probability_positive": float(np.mean(array[:, 0] > 0)),
        },
        "delta_mse": {
            "estimate": float(np.mean(array[:, 1])),
            "ci95": np.quantile(array[:, 1], [0.025, 0.975]).tolist(),
            "probability_negative": float(np.mean(array[:, 1] < 0)),
        },
        "bootstrap_speech_groups": BOOTSTRAPS,
    }


def incremental_results(by_seed):
    output = {}
    for seed, rows in by_seed.items():
        output[seed] = {
            "delay_plus_predicted_quality_vs_delay": cluster_bootstrap_delta(
                rows, "auto_pred_LQ_EXP_piecewise_delay", "delay_piecewise"
            ),
            "predicted_quality_plus_delay_vs_predicted_quality": cluster_bootstrap_delta(
                rows, "auto_pred_LQ_EXP_piecewise_delay", "auto_pred_LQ_EXP"
            ),
        }
    return output


def cross_rater_table():
    audit = json.loads(RATER_AUDIT.read_text(encoding="utf-8"))
    per_rater = audit["per_rater_targets"]
    transfer = audit["cross_rater_transfer"]
    mean_ref = audit["speech_held_out_baselines"]["LQ_EXP_piecewise"]
    rows = [
        {"quality_input": "Evaluator A LQ/EXP + delay", "LAT_target": "Evaluator A LAT", **per_rater["R05"]["LQ_EXP_piecewise"]},
        {"quality_input": "Evaluator A LQ/EXP + delay", "LAT_target": "Evaluator B LAT", **transfer["R05_to_R06_quality_delay"]},
        {"quality_input": "Evaluator B LQ/EXP + delay", "LAT_target": "Evaluator A LAT", **transfer["R06_to_R05_quality_delay"]},
        {"quality_input": "Evaluator B LQ/EXP + delay", "LAT_target": "Evaluator B LAT", **per_rater["R06"]["LQ_EXP_piecewise"]},
        {"quality_input": "Mean LQ/EXP + delay", "LAT_target": "Mean LAT", **mean_ref},
    ]
    return rows


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_seed = load_rows()
    directions = direction_results(by_seed)
    group_macro = group_macro_results(by_seed)
    incremental = incremental_results(by_seed)
    rater_rows = cross_rater_table()
    payload = rounded({
        "protocol": "Frozen three-seed source-speech-group-held-out OOF predictions; no best-seed selection",
        "seeds": list(SEEDS),
        "direction_results": directions,
        "direction_three_seed_summary": aggregate_seed_metrics(directions),
        "speech_group_macro_results": group_macro,
        "speech_group_macro_three_seed_summary": aggregate_group_macro(group_macro),
        "incremental_cluster_bootstrap": incremental,
        "cross_rater_table": rater_rows,
    })
    (OUT / "reviewer_priority_audits.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    direction_rows = []
    for seed, models in payload["direction_results"].items():
        for model, direction_items in models.items():
            for direction, item in direction_items.items():
                direction_rows.append({"seed": seed, "model": model, "direction": direction, **item})
    write_csv(
        OUT / "automatic_direction_results.csv",
        direction_rows,
        ["seed", "model", "direction", "n", "pearson", "spearman", "mse", "mae"],
    )
    write_csv(
        OUT / "cross_rater_quality_lat.csv",
        rater_rows,
        ["quality_input", "LAT_target", "n", "pearson", "spearman", "mse", "mae", "pred_std"],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
