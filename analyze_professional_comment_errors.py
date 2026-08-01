#!/usr/bin/env python3
"""Build a blindable qualitative review set from professional comments.

This script does not use comments as model inputs. It joins the existing
professional comments to per-segment R020 LAT prediction errors so a human can
code error types after inspecting the source, interpretation, ratings, and
comment text.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path(__file__).parent
DEFAULT_RAW = "data/evaluation/profess_eval_delay_enriched.json"
DEFAULT_RUNS = [
    "experiments/latency_r020_nested_quality_20260712/latency_r020_nested_quality_oof_predictions.csv",
    "experiments/latency_r020_seed_20260713/latency_r020_nested_quality_oof_predictions.csv",
    "experiments/latency_r020_seed_20260714/latency_r020_nested_quality_oof_predictions.csv",
]
MODEL = "R020_pred_LQ_EXP"


def speech_group(row: dict) -> str:
    return str(row.get("speech") or row.get("source_file") or row.get("file_id") or "")


def mean_or_blank(rows: list[dict], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return "" if not values else f"{np.mean(values):.4f}"


def load_predictions(paths: list[Path]) -> dict[str, list[float]]:
    by_segment: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["model"] == MODEL]
        if not rows:
            raise ValueError(f"No {MODEL} rows in {path}")
        for row in rows:
            by_segment[row["segment_id"]].append(float(row["prediction"]))
    expected_runs = len(paths)
    incomplete = [sid for sid, values in by_segment.items() if len(values) != expected_runs]
    if incomplete:
        raise ValueError(f"Missing seed predictions for {len(incomplete)} segments")
    return by_segment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", default=DEFAULT_RAW)
    parser.add_argument("--prediction-csvs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", default="experiments/qualitative_comment_error_analysis_20260713")
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    raw_path = BASE / args.raw_data
    pred_paths = [BASE / path for path in args.prediction_csvs]
    output_dir = BASE / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
    predictions = load_predictions(pred_paths)
    by_segment: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        sid = str(row.get("segment_id") or "")
        if sid in predictions:
            by_segment[sid].append(row)

    cases = []
    for sid, predicted in predictions.items():
        rows = by_segment[sid]
        commented = [row for row in rows if row.get("comments")]
        if not commented:
            continue
        lat_values = [float(row["perceived_latency"]) for row in rows if row.get("perceived_latency") not in (None, "")]
        if not lat_values:
            continue
        lat = float(np.mean(lat_values))
        mean_prediction = float(np.mean(predicted))
        comments = " | ".join(
            f"{row.get('evaluator_id', 'unknown')}: {row['comments']}" for row in commented
        )
        first = rows[0]
        cases.append({
            "segment_id": sid,
            "speech_group": speech_group(first),
            "direction": first.get("direction") or "",
            "source": first.get("src") or "",
            "interpretation": first.get("mt") or "",
            "LQ_mean": mean_or_blank(rows, "LQ"),
            "EXP_mean": mean_or_blank(rows, "EXP"),
            "perceived_latency_mean": f"{lat:.4f}",
            "predicted_LAT_mean": f"{mean_prediction:.4f}",
            "mean_absolute_error": f"{abs(mean_prediction - lat):.4f}",
            "prediction_sd_across_runs": f"{np.std(predicted, ddof=1):.4f}",
            "n_comment_rows": len(commented),
            "professional_comments": comments,
            "human_error_code": "",
            "reviewer_notes": "",
        })

    cases.sort(key=lambda row: (float(row["mean_absolute_error"]), float(row["prediction_sd_across_runs"])), reverse=True)
    fieldnames = list(cases[0]) if cases else []
    all_path = output_dir / "all_commented_r020_cases.csv"
    review_path = output_dir / "priority_review_cases.csv"
    for path, rows in [(all_path, cases), (review_path, cases[:args.top_k])]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "purpose": "Qualitative error analysis only; professional comments are not model features.",
        "model": MODEL,
        "n_prediction_runs": len(pred_paths),
        "n_r020_segments": len(predictions),
        "n_segments_with_professional_comments": len(cases),
        "n_priority_cases": min(args.top_k, len(cases)),
        "selection": "Sorted by mean absolute LAT prediction error, then prediction standard deviation across runs.",
        "review_protocol": [
            "Assign human_error_code without changing the numerical experiment outputs.",
            "Use a fixed codebook before counting categories.",
            "Report counts descriptively; do not treat codes as independent statistical observations.",
        ],
        "files": {"all_cases": str(all_path.relative_to(BASE)), "priority_cases": str(review_path.relative_to(BASE))},
    }
    (output_dir / "comment_error_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
