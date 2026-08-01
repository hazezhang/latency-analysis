#!/usr/bin/env python3
"""Prepare auditable qualitative examples for the perceived-latency paper.

The script uses professional comments only after model fitting. It never feeds
comments into the quality or latency models. Syntactic-complexity flags are
screening proxies for manual review, not automatically inferred explanations.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments/paper_qualitative_cases_20260720"
MODEL = "auto_pred_LQ_EXP_piecewise_delay"
SEEDS = ("20260718", "20260719", "20260720")


def normalize_key(file_id: str, original_segment_id: object) -> str:
    return f"{str(file_id).zfill(3)}:{int(original_segment_id)}"


def complexity_proxy(text: str, direction: str) -> tuple[int, str]:
    if direction == "en-zh":
        markers = ("which", "that", "if", "when", "while", "because", "although", "after", "before", "but", "rather than", "not only")
    else:
        markers = ("因为", "如果", "虽然", "但是", "当", "为了", "之后", "之前", "同时", "不仅", "而且", "尽管")
    found = [marker for marker in markers if marker in text.lower()]
    return len(found), "; ".join(found)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[float]] = defaultdict(list)
    for seed in SEEDS:
        path = ROOT / f"experiments/latency_r027_seed_{seed}/r027_automatic_bridge_oof_predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["model"] == MODEL:
                    predictions[row["segment_id"]].append(float(row["prediction"]))

    comments: dict[str, list[str]] = defaultdict(list)
    raw_rows = json.loads((ROOT / "data/evaluation/profess_eval_delay_enriched.json").read_text(encoding="utf-8"))
    for row in raw_rows:
        comment = (row.get("comments") or "").strip()
        if comment and not comment.startswith("R06 review:"):
            comments[normalize_key(row["file_id"], row["original_segment_id"])].append(comment)

    cohort = json.loads((ROOT / "data/experiments/r027_shared_outer_quality/all_lat_segments.json").read_text(encoding="utf-8"))
    cases = []
    for row in cohort:
        scores = predictions[row["segment_id"]]
        assert len(scores) == 3, row["segment_id"]
        mean_prediction = statistics.mean(scores)
        complexity_count, markers = complexity_proxy(row["src"], row["direction"])
        cases.append(
            {
                "segment_id": row["segment_id"],
                "speech_group": row["speech_group"],
                "direction": row["direction"],
                "delay_seconds": row["delay_seconds"],
                "LQ": row["LQ"],
                "EXP": row["EXP"],
                "human_LAT": row["perceived_latency"],
                "predicted_LAT_mean": round(mean_prediction, 4),
                "absolute_error": round(abs(mean_prediction - row["perceived_latency"]), 4),
                "prediction_sd": round(statistics.pstdev(scores), 4),
                "source_complexity_proxy_count": complexity_count,
                "source_complexity_proxy_markers": markers,
                "source": row["src"],
                "interpretation": row["mt"],
                "professional_comments": " | ".join(comments[row["segment_id"]]),
                "manual_structural_reordering": "TO_REVIEW",
                "paper_use_decision": "TO_REVIEW",
            }
        )

    fields = list(cases[0])
    selections = {
        "good_predictions.csv": sorted(
            (case for case in cases if case["professional_comments"] and case["absolute_error"] <= 0.05),
            key=lambda case: (case["direction"], case["absolute_error"], -case["human_LAT"]),
        ),
        "difficult_predictions.csv": sorted(
            (case for case in cases if case["professional_comments"] and case["absolute_error"] >= 1.0),
            key=lambda case: (-case["absolute_error"], case["prediction_sd"]),
        ),
        "long_delay_structural_candidates.csv": sorted(
            (case for case in cases if case["professional_comments"] and case["delay_seconds"] >= 10),
            key=lambda case: (-case["source_complexity_proxy_count"], -case["delay_seconds"], -case["absolute_error"]),
        ),
    }
    for filename, rows in selections.items():
        with (OUTPUT / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "model": MODEL,
        "seeds": list(SEEDS),
        "comments_usage": "qualitative post-hoc analysis only; not model features",
        "structural_proxy": "screening proxy only; all structural-reordering claims require manual confirmation",
        "counts": {filename: len(rows) for filename, rows in selections.items()},
    }
    (OUTPUT / "README.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
