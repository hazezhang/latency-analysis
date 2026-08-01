#!/usr/bin/env python3
"""Paired source-speech-group inference for the structural-feature baselines."""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments/aaai_structural_incremental_audit_20260722"
SEEDS = ("20260718", "20260719", "20260720")
BOOTSTRAPS = 10000
RNG_SEED = 20260722
COMPARISONS = (
    ("auto_pred_LQ_EXP_piecewise_delay_lexical_structural", "lexical_structural_piecewise_delay"),
    ("auto_pred_LQ_EXP_piecewise_delay_lexical_structural", "auto_pred_LQ_EXP_piecewise_delay"),
    ("lexical_structural_piecewise_delay", "delay_piecewise"),
)


def load_rows(seed):
    path = ROOT / f"experiments/aaai_crossfitted_structural_seed_{seed}_20260722/crossfitted_lat_oof_predictions.csv"
    rows = defaultdict(dict)
    meta = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = row["segment_id"]
            rows[sid][row["model"]] = float(row["prediction"])
            meta[sid] = {"LAT": float(row["LAT"]), "speech_group": str(row["speech_group"])}
    return rows, meta


def corr(y, pred):
    if np.std(y) == 0 or np.std(pred) == 0:
        return None
    return float(pearsonr(y, pred).statistic)


def comparison(rows, meta, added, base):
    ids = sorted(meta)
    groups = sorted({meta[sid]["speech_group"] for sid in ids})
    grouped = {group: [sid for sid in ids if meta[sid]["speech_group"] == group] for group in groups}

    def effect(sample_ids):
        y = np.asarray([meta[sid]["LAT"] for sid in sample_ids])
        a = np.asarray([rows[sid][added] for sid in sample_ids])
        b = np.asarray([rows[sid][base] for sid in sample_ids])
        return corr(y, a) - corr(y, b), float(np.mean((y - a) ** 2) - np.mean((y - b) ** 2))

    observed = effect(ids)
    rng = np.random.default_rng(RNG_SEED)
    boot = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        boot.append(effect([sid for group in sampled for sid in grouped[group]]))
    boot = np.asarray(boot)

    null = []
    y = np.asarray([meta[sid]["LAT"] for sid in ids])
    added_pred = np.asarray([rows[sid][added] for sid in ids])
    base_pred = np.asarray([rows[sid][base] for sid in ids])
    group_indices = [np.asarray([i for i, sid in enumerate(ids) if meta[sid]["speech_group"] == group]) for group in groups]
    for swaps in itertools.product((False, True), repeat=len(groups)):
        left, right = added_pred.copy(), base_pred.copy()
        for swap, indices in zip(swaps, group_indices):
            if swap:
                left[indices], right[indices] = right[indices].copy(), left[indices].copy()
        null.append(corr(y, left) - corr(y, right))
    p_value = float(np.mean(np.abs(null) >= abs(observed[0])))
    return {
        "added_model": added, "base_model": base,
        "delta_pearson": observed[0], "delta_pearson_ci95": np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
        "delta_mse": observed[1], "delta_mse_ci95": np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
        "exact_two_sided_group_swap_p": p_value, "exact_assignments": len(null),
        "bootstrap_samples": BOOTSTRAPS,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": "paired complete-source-speech-group bootstrap and exact prediction swap", "seeds": {}}
    table = []
    for seed in SEEDS:
        rows, meta = load_rows(seed)
        payload["seeds"][seed] = []
        for added, base in COMPARISONS:
            result = comparison(rows, meta, added, base)
            payload["seeds"][seed].append(result)
            table.append({"seed": seed, **result})
    (OUT / "structural_incremental_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "structural_incremental_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
