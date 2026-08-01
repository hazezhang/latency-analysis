#!/usr/bin/env python3
"""Audit the descriptive full-versus-structural incremental comparison.

The paper has one confirmatory test: predicted quality plus delay versus
timing-only. This script reports the stricter full-model increment over
structural-plus-delay with a speech-group cluster bootstrap, without creating
an additional p-value claim.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent
SEEDS = ("20260718", "20260719", "20260720")
FULL_MODEL = "auto_pred_LQ_EXP_piecewise_delay_lexical_structural"
STRUCTURAL_MODEL = "lexical_structural_piecewise_delay"
BOOTSTRAPS = 10_000
RNG_SEED = 20260726
OUT = ROOT / "experiments" / "aaai_key_incremental_comparison_20260726"
SUPPLEMENT = ROOT / "aaai26_paper_staging" / "anonymous_supplement"


def pearson(gold: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(gold) < 2 or np.std(gold) == 0 or np.std(prediction) == 0:
        return None
    return float(pearsonr(gold, prediction).statistic)


def mse(gold: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((gold - prediction) ** 2))


def load_seed(seed: str) -> tuple[dict[str, dict[str, float | str]], dict[str, dict[str, float]]]:
    path = ROOT / f"experiments/aaai_crossfitted_structural_seed_{seed}_20260722/crossfitted_lat_oof_predictions.csv"
    metadata: dict[str, dict[str, float | str]] = {}
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            segment_id = row["segment_id"]
            metadata[segment_id] = {
                "speech_group": row["speech_group"],
                "LAT": float(row["LAT"]),
            }
            if row["model"] in {FULL_MODEL, STRUCTURAL_MODEL}:
                predictions[segment_id][row["model"]] = float(row["prediction"])
    missing = [sid for sid in metadata if set(predictions[sid]) != {FULL_MODEL, STRUCTURAL_MODEL}]
    if missing:
        raise ValueError(f"{seed}: missing comparison predictions for {len(missing)} segments")
    return metadata, predictions


def effect(ids: list[str], metadata: dict[str, dict[str, float | str]], predictions: dict[str, dict[str, float]]) -> tuple[float | None, float]:
    gold = np.asarray([metadata[sid]["LAT"] for sid in ids], dtype=float)
    full = np.asarray([predictions[sid][FULL_MODEL] for sid in ids], dtype=float)
    structural = np.asarray([predictions[sid][STRUCTURAL_MODEL] for sid in ids], dtype=float)
    full_r = pearson(gold, full)
    structural_r = pearson(gold, structural)
    delta_r = None if full_r is None or structural_r is None else full_r - structural_r
    return delta_r, mse(gold, full) - mse(gold, structural)


def quantile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive_count": int(np.sum(array > 0)),
        "n": int(len(array)),
    }


def main() -> int:
    seed_data = {seed: load_seed(seed) for seed in SEEDS}
    reference_metadata = seed_data[SEEDS[0]][0]
    ids = sorted(reference_metadata)
    groups = sorted({str(reference_metadata[sid]["speech_group"]) for sid in ids})
    anonymous_groups = {group: f"speech_{index:02d}" for index, group in enumerate(groups, start=1)}

    for seed, (metadata, predictions) in seed_data.items():
        if sorted(metadata) != ids:
            raise ValueError(f"{seed}: segment membership differs across seeds")
        for sid in ids:
            if metadata[sid] != reference_metadata[sid]:
                raise ValueError(f"{seed}: metadata differs for {sid}")

    by_group = {group: [sid for sid in ids if reference_metadata[sid]["speech_group"] == group] for group in groups}
    per_seed = {}
    group_rows = []
    for seed, (metadata, predictions) in seed_data.items():
        pooled_delta_r, pooled_delta_mse = effect(ids, metadata, predictions)
        group_delta_r, group_delta_mse = [], []
        for group in groups:
            delta_r, delta_mse = effect(by_group[group], metadata, predictions)
            if delta_r is not None:
                group_delta_r.append(delta_r)
            group_delta_mse.append(delta_mse)
            group_rows.append(
                {
                    "seed": seed,
                    "speech_group": anonymous_groups[group],
                    "n_segments": len(by_group[group]),
                    "delta_pearson_full_minus_structural": delta_r,
                    "delta_mse_full_minus_structural": delta_mse,
                }
            )
        per_seed[seed] = {
            "pooled_delta_pearson": pooled_delta_r,
            "pooled_delta_mse": pooled_delta_mse,
            "speech_group_delta_pearson_distribution": quantile_summary(group_delta_r),
            "speech_group_delta_mse_distribution": quantile_summary(group_delta_mse),
        }

    averaged_predictions: dict[str, dict[str, float]] = {}
    for sid in ids:
        averaged_predictions[sid] = {
            model: float(np.mean([seed_data[seed][1][sid][model] for seed in SEEDS]))
            for model in (FULL_MODEL, STRUCTURAL_MODEL)
        }
    observed_delta_r, observed_delta_mse = effect(ids, reference_metadata, averaged_predictions)
    rng = np.random.default_rng(RNG_SEED)
    bootstrap_r, bootstrap_mse = [], []
    while len(bootstrap_r) < BOOTSTRAPS:
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_ids = [sid for group in sampled_groups for sid in by_group[group]]
        delta_r, delta_mse = effect(sampled_ids, reference_metadata, averaged_predictions)
        if delta_r is not None:
            bootstrap_r.append(delta_r)
            bootstrap_mse.append(delta_mse)

    aggregate = {
        "comparison": "full predicted-quality+structural+delay minus structural+delay",
        "inference_status": "descriptive secondary comparison; no additional confirmatory p-value is reported",
        "aggregation": "predictions averaged within segment over three fixed seeds before cluster resampling",
        "cluster_unit": "16 source-speech groups, resampled with replacement while retaining all group segments",
        "bootstrap_samples": BOOTSTRAPS,
        "rng_seed": RNG_SEED,
        "delta_pearson": observed_delta_r,
        "delta_pearson_ci95": [float(np.quantile(bootstrap_r, 0.025)), float(np.quantile(bootstrap_r, 0.975))],
        "delta_mse": observed_delta_mse,
        "delta_mse_ci95": [float(np.quantile(bootstrap_mse, 0.025)), float(np.quantile(bootstrap_mse, 0.975))],
    }
    payload = {
        "formal_seeds": list(SEEDS),
        "n_segments": len(ids),
        "n_speech_groups": len(groups),
        "aggregate": aggregate,
        "per_seed": per_seed,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    (OUT / "key_incremental_comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (OUT / "key_incremental_group_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    (SUPPLEMENT / "key_incremental_comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (SUPPLEMENT / "key_incremental_group_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    lines = [
        "# Key Incremental Comparison: Predicted Quality Beyond Structure",
        "",
        "The primary construct comparison tests whether predicted professional quality plus delay adds information beyond timing alone. This separate key incremental comparison tests whether adding predicted quality to deterministic lexical/structural features plus delay improves the full model. It is a descriptive secondary analysis: no second confirmatory p-value is reported.",
        "",
        "Predictions are averaged per segment across the three fixed training seeds before 10,000 whole-source-speech-group bootstrap draws. Every draw resamples the 16 groups with replacement and retains all rows within a selected group.",
        "",
        "| Comparison | Delta Pearson r (95% speech-group CI) | Delta MSE (95% speech-group CI) |",
        "| --- | ---: | ---: |",
        f"| Full model minus structural+delay | {aggregate['delta_pearson']:.3f} [{aggregate['delta_pearson_ci95'][0]:.3f}, {aggregate['delta_pearson_ci95'][1]:.3f}] | {aggregate['delta_mse']:.3f} [{aggregate['delta_mse_ci95'][0]:.3f}, {aggregate['delta_mse_ci95'][1]:.3f}] |",
        "",
        "Negative delta MSE favors the full model. The interval describes source-speech sampling uncertainty; it does not create an additional confirmatory significance claim.",
        "",
        "## Fixed-Seed and Speech-Group Distribution",
        "",
        "| Seed | Pooled Delta r | Pooled Delta MSE | Speech-group Delta r: median [IQR], range, positive groups | Speech-group Delta MSE: median [IQR], range |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for seed in SEEDS:
        values = per_seed[seed]
        r_summary = values["speech_group_delta_pearson_distribution"]
        mse_summary = values["speech_group_delta_mse_distribution"]
        lines.append(
            f"| {seed} | {values['pooled_delta_pearson']:.3f} | {values['pooled_delta_mse']:.3f} | "
            f"{r_summary['median']:.3f} [{r_summary['q1']:.3f}, {r_summary['q3']:.3f}], "
            f"[{r_summary['minimum']:.3f}, {r_summary['maximum']:.3f}], {r_summary['positive_count']}/{r_summary['n']} | "
            f"{mse_summary['median']:.3f} [{mse_summary['q1']:.3f}, {mse_summary['q3']:.3f}], "
            f"[{mse_summary['minimum']:.3f}, {mse_summary['maximum']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "The anonymous per-group values are in `key_incremental_group_distribution.csv`. They show the distribution of held-out speech-group increments rather than treating the three random seeds as independent studies.",
        ]
    )
    (SUPPLEMENT / "key_incremental_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
