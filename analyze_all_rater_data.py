#!/usr/bin/env python3
"""Audit professional and student rater data before rater-aware experiments.

This is intentionally descriptive: it establishes overlap, missingness, and
score-scale compatibility before any student labels are allowed into training.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments/rater_design_audit_20260720"


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalized_id(row: dict) -> str:
    return f"{str(row.get('file_id')).zfill(3)}:{int(row.get('original_segment_id'))}"


def pearson(left: list[float], right: list[float]):
    if len(left) < 3:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    cov = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    left_ss = sum((x - mean_left) ** 2 for x in left)
    right_ss = sum((y - mean_right) ** 2 for y in right)
    if not left_ss or not right_ss:
        return None
    return round(cov / math.sqrt(left_ss * right_ss), 4)


def load_rows(path: Path, source: str):
    rows = json.loads(path.read_text(encoding="utf-8"))
    normalized = []
    for row in rows:
        normalized.append(
            {
                "segment_id": normalized_id(row),
                "rater": row["evaluator_id"],
                "source": source,
                "direction": row.get("direction") or ("en-zh" if row.get("source_english") else "zh-en"),
                "LQ": number(row.get("LQ", row.get("language_quality"))),
                "EXP": number(row.get("EXP", row.get("expressiveness"))),
                "LAT": number(row.get("perceived_latency")),
                "delay_seconds": number(row.get("delay_seconds")),
            }
        )
    return normalized


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(ROOT / "data/evaluation/profess_eval_delay_enriched.json", "professional")
    rows += load_rows(ROOT / "data/evaluation/student_eval.json", "student")
    by_rater = defaultdict(list)
    by_segment = defaultdict(list)
    for row in rows:
        by_rater[row["rater"]].append(row)
        by_segment[row["segment_id"]].append(row)

    rater_rows = []
    for rater, items in sorted(by_rater.items()):
        complete = [row for row in items if all(row[key] is not None for key in ("LQ", "EXP", "LAT"))]
        with_delay = [row for row in complete if row["delay_seconds"] is not None]
        rater_rows.append(
            {
                "rater": rater,
                "source": items[0]["source"],
                "n_rows": len(items),
                "n_complete_LQ_EXP_LAT": len(complete),
                "n_complete_with_delay": len(with_delay),
                "directions": ";".join(f"{key}:{value}" for key, value in sorted(Counter(row["direction"] for row in items).items())),
                "LQ_mean": round(sum(row["LQ"] for row in complete) / len(complete), 4) if complete else None,
                "EXP_mean": round(sum(row["EXP"] for row in complete) / len(complete), 4) if complete else None,
                "LAT_mean": round(sum(row["LAT"] for row in complete) / len(complete), 4) if complete else None,
            }
        )

    pair_rows = []
    raters = sorted(by_rater)
    index = {(row["segment_id"], row["rater"]): row for row in rows}
    for left_index, left_rater in enumerate(raters):
        for right_rater in raters[left_index + 1 :]:
            paired = [
                (index[(segment_id, left_rater)], index[(segment_id, right_rater)])
                for segment_id in {row["segment_id"] for row in by_rater[left_rater]}
                if (segment_id, left_rater) in index and (segment_id, right_rater) in index
            ]
            result = {"rater_left": left_rater, "rater_right": right_rater, "n_overlap": len(paired)}
            for outcome in ("LQ", "EXP", "LAT"):
                valid = [(left[outcome], right[outcome]) for left, right in paired if left[outcome] is not None and right[outcome] is not None]
                result[f"n_{outcome}"] = len(valid)
                result[f"pearson_{outcome}"] = pearson([left for left, _ in valid], [right for _, right in valid])
            pair_rows.append(result)

    def write(name, fields, values):
        with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)

    write("rater_inventory.csv", list(rater_rows[0]), rater_rows)
    write("rater_overlap.csv", list(pair_rows[0]), pair_rows)
    summary = {
        "purpose": "pre-registration audit for rater-aware experiments; no score pooling decision is made here",
        "n_rows": len(rows),
        "n_unique_segments": len(by_segment),
        "rater_inventory": rater_rows,
        "student_policy": "student labels require rater-aware modeling or pretrain-then-professional-finetune; do not pool raw student and professional scores",
        "recommended_primary": "shared professional R05/R06 mean only",
        "recommended_sensitivity": [
            "professional individual-rater models (R05 and R06 separate)",
            "professional rater-aware multi-task model",
            "student-only out-of-domain model",
            "student pretraining followed by professional-only fine-tuning",
        ],
    }
    (OUT / "rater_design_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
