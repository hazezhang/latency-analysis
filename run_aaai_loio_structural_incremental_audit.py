#!/usr/bin/env python3
"""Summarize interpreter-disjoint structural LAT models with cluster-aware tests."""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments/aaai_loio_structural_audit_20260722"
SEEDS = ("20260718", "20260719", "20260720")
BOOTSTRAPS = 10_000
RNG_SEED = 20260722
COMPARISONS = (
    ("auto_pred_LQ_EXP_piecewise_delay_lexical_structural", "lexical_structural_piecewise_delay"),
    ("auto_pred_LQ_EXP_piecewise_delay_lexical_structural", "auto_pred_LQ_EXP_piecewise_delay"),
    ("lexical_structural_piecewise_delay", "delay_piecewise"),
)


def corr(y, prediction):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float(pearsonr(y, prediction).statistic)


def load_rows(seed):
    path = ROOT / f"experiments/aaai_loio_structural_seed_{seed}_20260722/loio_lat_oof_predictions.csv"
    rows = defaultdict(dict)
    metadata = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            segment_id = str(row["segment_id"])
            rows[segment_id][str(row["model"])] = float(row["prediction"])
            metadata[segment_id] = {
                "LAT": float(row["LAT"]),
                "interpreter": str(row["interpreter"]),
                "speech_group": str(row["speech_group"]),
            }
    return rows, metadata


def effect(rows, metadata, ids, added, base):
    gold = np.asarray([metadata[segment_id]["LAT"] for segment_id in ids])
    added_prediction = np.asarray([rows[segment_id][added] for segment_id in ids])
    base_prediction = np.asarray([rows[segment_id][base] for segment_id in ids])
    return (
        corr(gold, added_prediction) - corr(gold, base_prediction),
        float(np.mean((gold - added_prediction) ** 2) - np.mean((gold - base_prediction) ** 2)),
    )


def comparison(rows, metadata, added, base):
    ids = sorted(metadata)
    interpreters = sorted({metadata[segment_id]["interpreter"] for segment_id in ids})
    grouped = {
        interpreter: [segment_id for segment_id in ids if metadata[segment_id]["interpreter"] == interpreter]
        for interpreter in interpreters
    }
    observed = effect(rows, metadata, ids, added, base)

    rng = np.random.default_rng(RNG_SEED)
    bootstrap = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(interpreters, size=len(interpreters), replace=True)
        sample_ids = [segment_id for interpreter in sampled for segment_id in grouped[interpreter]]
        bootstrap.append(effect(rows, metadata, sample_ids, added, base))
    bootstrap = np.asarray(bootstrap)

    gold = np.asarray([metadata[segment_id]["LAT"] for segment_id in ids])
    added_prediction = np.asarray([rows[segment_id][added] for segment_id in ids])
    base_prediction = np.asarray([rows[segment_id][base] for segment_id in ids])
    interpreter_indices = [
        np.asarray([index for index, segment_id in enumerate(ids) if metadata[segment_id]["interpreter"] == interpreter])
        for interpreter in interpreters
    ]
    null = []
    for swaps in itertools.product((False, True), repeat=len(interpreters)):
        left = added_prediction.copy()
        right = base_prediction.copy()
        for swap, indices in zip(swaps, interpreter_indices):
            if swap:
                left[indices], right[indices] = right[indices].copy(), left[indices].copy()
        null.append(corr(gold, left) - corr(gold, right))

    return {
        "added_model": added,
        "base_model": base,
        "delta_pearson": observed[0],
        "delta_pearson_ci95": np.quantile(bootstrap[:, 0], [0.025, 0.975]).tolist(),
        "delta_mse": observed[1],
        "delta_mse_ci95": np.quantile(bootstrap[:, 1], [0.025, 0.975]).tolist(),
        "exact_two_sided_interpreter_swap_p": float(np.mean(np.abs(null) >= abs(observed[0]))),
        "exact_assignments": len(null),
        "bootstrap_samples": BOOTSTRAPS,
    }


def aggregate_seed_metrics(per_seed):
    models = sorted(next(iter(per_seed.values()))["models"])
    output = {}
    for model in models:
        output[model] = {}
        for metric in ("pearson", "spearman", "mse", "mae", "pred_std"):
            values = np.asarray([per_seed[seed]["models"][model]["metrics"][metric] for seed in SEEDS], dtype=float)
            output[model][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "values": values.tolist(),
            }
        for section, metric in (("within_interpreter_centered", "pearson"), ("macro_interpreter", "pearson")):
            values = np.asarray([per_seed[seed]["models"][model][section][metric] for seed in SEEDS], dtype=float)
            output[model][f"{section}_{metric}"] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "values": values.tolist(),
            }
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = {
        seed: json.loads((ROOT / f"experiments/aaai_loio_structural_seed_{seed}_20260722/loio_lat_results.json").read_text(encoding="utf-8"))
        for seed in SEEDS
    }
    payload = {
        "protocol": (
            "Interpreter-disjoint quality predictions with fold-local LAT Ridge. "
            "Paired uncertainty uses 7-interpreter cluster bootstrap and exact 2^7 prediction swaps."
        ),
        "seeds": list(SEEDS),
        "seed_metrics": per_seed,
        "aggregate": aggregate_seed_metrics(per_seed),
        "comparisons": {},
    }
    comparison_rows = []
    for seed in SEEDS:
        rows, metadata = load_rows(seed)
        seed_comparisons = []
        for added, base in COMPARISONS:
            result = comparison(rows, metadata, added, base)
            seed_comparisons.append(result)
            comparison_rows.append({"seed": seed, **result})
        payload["comparisons"][seed] = seed_comparisons

    summary_rows = []
    for model, values in payload["aggregate"].items():
        summary_rows.append({
            "model": model,
            **{f"{metric}_mean": values[metric]["mean"] for metric in ("pearson", "spearman", "mse", "mae", "pred_std")},
            **{f"{metric}_sd": values[metric]["sd"] for metric in ("pearson", "spearman", "mse", "mae", "pred_std")},
            "within_interpreter_centered_pearson_mean": values["within_interpreter_centered_pearson"]["mean"],
            "macro_interpreter_pearson_mean": values["macro_interpreter_pearson"]["mean"],
        })
    (OUT / "loio_structural_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "loio_structural_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUT / "loio_structural_incremental_tests.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(json.dumps({"aggregate": payload["aggregate"], "comparisons": payload["comparisons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
