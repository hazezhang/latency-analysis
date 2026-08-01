#!/usr/bin/env python3
"""Speech-group robustness analysis for R020 automatic quality prediction.

Reports group-level correlations and an exact group-swap permutation test for
the difference between R020 predicted LQ/EXP and the delay-only baseline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path(__file__).parent
DEFAULT_OOF = BASE / "experiments/latency_r020_nested_quality_20260712/latency_r020_nested_quality_oof_predictions.csv"
DEFAULT_OUT = BASE / "experiments/latency_r020_nested_quality_20260712"
AUTO = "R020_pred_LQ_EXP"
DELAY = "R014_delay_only"


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(y_true) < 3 or not np.std(y_true) or not np.std(y_pred):
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def load_rows(path: Path) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] not in {AUTO, DELAY}:
                continue
            grouped[row["model"]][row["segment_id"]] = row
    if set(grouped) != {AUTO, DELAY}:
        raise ValueError("Missing automatic or delay-only predictions")
    if set(grouped[AUTO]) != set(grouped[DELAY]):
        raise ValueError("Automatic and delay-only segment IDs do not match")
    return grouped


def overall_corr(rows: list[dict]) -> float:
    return pearson(
        np.asarray([float(row["LAT"]) for row in rows]),
        np.asarray([float(row["prediction"]) for row in rows]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-csv", default=str(DEFAULT_OOF.relative_to(BASE)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT.relative_to(BASE)))
    args = parser.parse_args()

    oof_path = BASE / args.oof_csv
    out_dir = BASE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(oof_path)
    auto_rows = rows[AUTO]
    delay_rows = rows[DELAY]

    by_group: dict[str, list[str]] = defaultdict(list)
    for segment_id, row in auto_rows.items():
        by_group[row["speech_group"]].append(segment_id)
    groups = sorted(by_group)

    group_table = []
    for group in groups:
        ids = sorted(by_group[group])
        auto = [auto_rows[sid] for sid in ids]
        delay = [delay_rows[sid] for sid in ids]
        auto_r = overall_corr(auto)
        delay_r = overall_corr(delay)
        group_table.append({
            "speech_group": group,
            "n_segments": len(ids),
            "auto_pearson": None if auto_r is None else round(auto_r, 4),
            "delay_pearson": None if delay_r is None else round(delay_r, 4),
            "difference": None if auto_r is None or delay_r is None else round(auto_r - delay_r, 4),
        })

    all_auto = [auto_rows[sid] for sid in sorted(auto_rows)]
    all_delay = [delay_rows[sid] for sid in sorted(delay_rows)]
    observed_auto = overall_corr(all_auto)
    observed_delay = overall_corr(all_delay)
    observed_difference = observed_auto - observed_delay

    # Exact cluster-level paired randomization: swap the two model predictions
    # as a whole within each speech group under the null of equal performance.
    null_differences = []
    for choices in itertools.product((0, 1), repeat=len(groups)):
        perm_auto, perm_delay = [], []
        for group, swap in zip(groups, choices):
            for sid in by_group[group]:
                left, right = auto_rows[sid], delay_rows[sid]
                if swap:
                    left, right = right, left
                perm_auto.append(left)
                perm_delay.append(right)
        null_differences.append(overall_corr(perm_auto) - overall_corr(perm_delay))
    p_two_sided = sum(abs(value) >= abs(observed_difference) for value in null_differences) / len(null_differences)

    result = {
        "oof_csv": str(oof_path.relative_to(BASE)),
        "models": {"automatic": AUTO, "delay": DELAY},
        "n_segments": len(all_auto),
        "n_speech_groups": len(groups),
        "overall": {
            "automatic_pearson": round(observed_auto, 4),
            "delay_pearson": round(observed_delay, 4),
            "difference": round(observed_difference, 4),
        },
        "group_swap_permutation": {
            "method": "exact paired speech-group prediction swap",
            "num_permutations": len(null_differences),
            "two_sided_p": round(float(p_two_sided), 4),
        },
        "per_speech_group": group_table,
        "interpretation": "Tests whether automatic and delay-only predictions are exchangeable at the speech-group level; small-cluster inference remains exploratory.",
    }
    json_path = out_dir / "r020_cluster_robustness.json"
    csv_path = out_dir / "r020_per_speech_robustness.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_table[0]))
        writer.writeheader()
        writer.writerows(group_table)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
