#!/usr/bin/env python3
"""Exact speech-group paired comparison of R027 automatic and delay models."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path(__file__).parent
AUTO = "auto_pred_LQ_EXP_piecewise_delay"
DELAY = "delay_linear"


def pearson(y, prediction):
    return float(np.corrcoef(y, prediction)[0, 1])


def one_table(path: Path) -> dict:
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in {AUTO, DELAY}:
                rows[row["model"]][row["segment_id"]] = row
    if set(rows) != {AUTO, DELAY} or set(rows[AUTO]) != set(rows[DELAY]):
        raise ValueError(f"Missing or mismatched model rows in {path}")
    ids = sorted(rows[AUTO])
    groups = sorted({rows[AUTO][sid]["speech_group"] for sid in ids})
    y = np.asarray([float(rows[AUTO][sid]["LAT"]) for sid in ids])
    auto = np.asarray([float(rows[AUTO][sid]["prediction"]) for sid in ids])
    delay = np.asarray([float(rows[DELAY][sid]["prediction"]) for sid in ids])
    group_indices = [np.asarray([i for i, sid in enumerate(ids) if rows[AUTO][sid]["speech_group"] == group]) for group in groups]
    observed = pearson(y, auto) - pearson(y, delay)
    null = []
    for swaps in itertools.product((False, True), repeat=len(groups)):
        left, right = auto.copy(), delay.copy()
        for swap, indices in zip(swaps, group_indices):
            if swap:
                left[indices], right[indices] = right[indices].copy(), left[indices].copy()
        null.append(pearson(y, left) - pearson(y, right))
    p_value = sum(abs(value) >= abs(observed) for value in null) / len(null)
    return {
        "table": str(path.relative_to(BASE)),
        "n_segments": len(ids),
        "n_speech_groups": len(groups),
        "automatic_pearson": round(pearson(y, auto), 4),
        "delay_pearson": round(pearson(y, delay), 4),
        "delta_pearson": round(observed, 4),
        "test": {
            "method": "exact paired speech-group prediction swap",
            "num_permutations": len(null),
            "two_sided_p": round(float(p_value), 6),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-csvs", nargs="+", required=True)
    parser.add_argument("--output", default="experiments/r027_seed_summary_20260718/r027_cluster_comparison.json")
    args = parser.parse_args()
    result = {"automatic_model": AUTO, "delay_model": DELAY, "seeds": [one_table(BASE / value) for value in args.oof_csvs]}
    (BASE / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
