#!/usr/bin/env python3
"""Compute speech-group-clustered CIs for the corrected AAAI LAT bridge.

For every bootstrap sample, metrics are computed separately for each fixed
quality-model seed and then averaged. This preserves the paper's reported
point-estimate convention (mean of seed-level metrics), while the bootstrap
quantifies source-speech sampling uncertainty. The seed standard deviation
remains a separate optimization-variation summary.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parent
FORMAL_SEEDS = ("20260718", "20260719", "20260720")
MODELS = ("delay_piecewise", "auto_pred_LQ_EXP", "auto_pred_LQ_EXP_piecewise_delay")


def pearson(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    return pearson(rankdata(y), rankdata(pred))


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - y
    return {
        "pearson": pearson(y, pred),
        "spearman": spearman(y, pred),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
    }


def q95(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    return [round(float(np.quantile(array, q)), 4) for q in (0.025, 0.975)]


def read_rows(path: Path) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = {model: {} for model in MODELS}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            if model in rows:
                rows[model][row["segment_id"]] = row
    return rows


def read_seed(seed: str) -> dict[str, dict[str, dict]]:
    path = ROOT / "experiments" / f"aaai_crossfitted_corrected_lat_seed_{seed}_20260721" / "crossfitted_lat_oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    result = read_rows(path)
    missing = [model for model, records in result.items() if not records]
    if missing:
        raise ValueError(f"{path}: missing models {missing}")
    return result


def build_seed_predictions() -> tuple[list[str], np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    per_seed = {seed: read_seed(seed) for seed in FORMAL_SEEDS}
    ids = sorted(per_seed[FORMAL_SEEDS[0]][MODELS[0]])
    for seed, by_model in per_seed.items():
        for model in MODELS:
            if sorted(by_model[model]) != ids:
                raise ValueError(f"{seed}/{model}: segment IDs do not match")

    base = per_seed[FORMAL_SEEDS[0]][MODELS[0]]
    groups = np.asarray([base[sid]["speech_group"] for sid in ids])
    gold = np.asarray([float(base[sid]["LAT"]) for sid in ids])
    predictions: dict[str, dict[str, np.ndarray]] = {model: {} for model in MODELS}
    for model in MODELS:
        for seed in FORMAL_SEEDS:
            seed_rows = per_seed[seed][model]
            if any(seed_rows[sid]["speech_group"] != base[sid]["speech_group"] or float(seed_rows[sid]["LAT"]) != gold[i] for i, sid in enumerate(ids)):
                raise ValueError(f"{seed}/{model}: group or gold mismatch")
            predictions[model][seed] = np.asarray([float(seed_rows[sid]["prediction"]) for sid in ids])
    return ids, groups, gold, predictions


def mean_seed_metrics(gold: np.ndarray, prediction_by_seed: dict[str, np.ndarray]) -> dict[str, float]:
    seed_metrics = [metrics(gold, prediction_by_seed[seed]) for seed in FORMAL_SEEDS]
    return {metric: float(np.mean([values[metric] for values in seed_metrics])) for metric in seed_metrics[0]}


def cluster_bootstrap(groups: np.ndarray, gold: np.ndarray, predictions: dict[str, dict[str, np.ndarray]], samples: int, rng_seed: int) -> tuple[dict, dict]:
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    point = {model: mean_seed_metrics(gold, pred_by_seed) for model, pred_by_seed in predictions.items()}
    comparisons = {
        "predicted_quality_plus_delay_minus_delay_only": ("auto_pred_LQ_EXP_piecewise_delay", "delay_piecewise"),
        "predicted_quality_plus_delay_minus_predicted_quality": ("auto_pred_LQ_EXP_piecewise_delay", "auto_pred_LQ_EXP"),
    }
    draws = {model: defaultdict(list) for model in MODELS}
    delta_draws = {name: defaultdict(list) for name in comparisons}
    rng = np.random.default_rng(rng_seed)

    for _ in range(samples):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected = np.concatenate([indices[group] for group in chosen])
        sampled = {
            model: mean_seed_metrics(gold[selected], {seed: pred[selected] for seed, pred in pred_by_seed.items()})
            for model, pred_by_seed in predictions.items()
        }
        for model, values in sampled.items():
            for name, value in values.items():
                if np.isfinite(value):
                    draws[model][name].append(value)
        for name, (left, right) in comparisons.items():
            for metric in ("pearson", "spearman", "mse", "mae"):
                value = sampled[left][metric] - sampled[right][metric]
                if np.isfinite(value):
                    delta_draws[name][metric].append(value)

    model_summary = {
        model: {
            "point": {name: round(value, 4) for name, value in values.items()},
            "ci95_speech_group_cluster": {name: q95(draws[model][name]) for name in values},
        }
        for model, values in point.items()
    }
    delta_summary = {}
    for name, (left, right) in comparisons.items():
        point_delta = {metric: point[left][metric] - point[right][metric] for metric in point[left]}
        delta_summary[name] = {
            "definition": f"{left} minus {right}; positive correlation differences and negative error differences favor predicted quality plus delay",
            "point": {metric: round(value, 4) for metric, value in point_delta.items()},
            "ci95_speech_group_cluster": {metric: q95(delta_draws[name][metric]) for metric in point_delta},
        }
    return model_summary, delta_summary


def primary_exact_group_permutation(groups: np.ndarray, gold: np.ndarray, predictions: dict[str, dict[str, np.ndarray]]) -> dict:
    """One sole two-sided test using seed-averaged predictions and group swaps."""
    quality_delay = np.mean(
        np.stack([predictions["auto_pred_LQ_EXP_piecewise_delay"][seed] for seed in FORMAL_SEEDS]), axis=0
    )
    timing_only = np.mean(
        np.stack([predictions["delay_piecewise"][seed] for seed in FORMAL_SEEDS]), axis=0
    )
    observed = pearson(gold, quality_delay) - pearson(gold, timing_only)
    unique_groups = np.unique(groups)
    group_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    null = []
    for swaps in itertools.product((False, True), repeat=len(unique_groups)):
        left, right = quality_delay.copy(), timing_only.copy()
        for swap, indices in zip(swaps, group_indices):
            if swap:
                left[indices], right[indices] = right[indices].copy(), left[indices].copy()
        null.append(pearson(gold, left) - pearson(gold, right))
    extreme = sum(abs(value) >= abs(observed) for value in null)
    total = len(null)
    return {
        "comparison": "three-seed mean predicted quality plus delay versus timing-only",
        "randomization_unit": "entire source-speech group",
        "n_source_speech_groups": len(unique_groups),
        "unequal_group_size_handling": "Each sampled group retains all of its observed segments; the corpus-level Pearson statistic therefore preserves observed group-size contributions.",
        "test_statistic": "Pearson(gold, quality_plus_delay_prediction) minus Pearson(gold, timing_only_prediction)",
        "alternative": "two-sided difference from zero",
        "observed_statistic": round(float(observed), 4),
        "permutations": total,
        "extreme_permutations": extreme,
        "plus_one_corrected_p_value": round(float((extreme + 1) / (total + 1)), 6),
        "plus_one_formula": "(number of permutations with |T| >= |T_observed| + 1) / (total permutations + 1)",
        "seed_aggregation": "Predictions are averaged over the three fixed training seeds before the test.",
    }


def write_csv(path: Path, model_summary: dict, delta_summary: dict) -> None:
    rows = []
    for model, block in model_summary.items():
        rows.append({
            "kind": "model", "name": model,
            **{f"{metric}_point": block["point"][metric] for metric in block["point"]},
            **{f"{metric}_ci95": json.dumps(block["ci95_speech_group_cluster"][metric]) for metric in block["point"]},
        })
    for name, block in delta_summary.items():
        rows.append({
            "kind": "paired_difference", "name": name,
            **{f"{metric}_point": block["point"][metric] for metric in block["point"]},
            **{f"{metric}_ci95": json.dumps(block["ci95_speech_group_cluster"][metric]) for metric in block["point"]},
        })
    fields = ["kind", "name", "pearson_point", "pearson_ci95", "spearman_point", "spearman_ci95", "mse_point", "mse_ci95", "mae_point", "mae_ci95"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/aaai_speech_group_clustered_ci_20260726")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--rng-seed", type=int, default=20260726)
    args = parser.parse_args()
    _, groups, gold, predictions = build_seed_predictions()
    models, deltas = cluster_bootstrap(groups, gold, predictions, args.samples, args.rng_seed)
    primary_test = primary_exact_group_permutation(groups, gold, predictions)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "Each draw resamples the 16 source-speech groups with replacement and retains all segments from every selected group. Metrics are computed separately for the three fixed seeds within each draw and then averaged.",
        "n_segments": len(gold), "n_source_speech_groups": len(np.unique(groups)),
        "formal_seeds": list(FORMAL_SEEDS), "bootstrap_samples": args.samples, "rng_seed": args.rng_seed,
        "seed_sd_interpretation": "Separate variation across the three fixed quality-model training seeds; not a sampling confidence interval.",
        "models": models, "paired_differences": deltas, "primary_permutation_test": primary_test,
    }
    (output / "speech_group_clustered_ci.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output / "speech_group_clustered_ci.csv", models, deltas)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
