#!/usr/bin/env python3
"""Rebuild the four manuscript cases from corrected cross-fitted predictions."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


BASE = Path(__file__).resolve().parent
SEEDS = ("20260718", "20260719", "20260720")
MODEL = "auto_pred_LQ_EXP_piecewise_delay"
CASES = {
    "accurate_high_score": "080:6",
    "short_delay_failure": "024:1",
    "long_delay_counterexample": "009:6",
    "structural_difficult_case": "025:4",
}
LEGACY_FILES = (
    "experiments/paper_qualitative_cases_20260720/good_predictions.csv",
    "experiments/paper_qualitative_cases_20260720/difficult_predictions.csv",
    "experiments/paper_qualitative_cases_20260720/long_delay_structural_candidates.csv",
)


def main() -> int:
    gold_rows = json.loads(
        (BASE / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json").read_text(
            encoding="utf-8"
        )
    )
    gold = {row["segment_id"]: row for row in gold_rows}

    metadata = {}
    for relative in LEGACY_FILES:
        with (BASE / relative).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["segment_id"] in CASES.values():
                    metadata[row["segment_id"]] = row

    predictions = {segment_id: [] for segment_id in CASES.values()}
    for seed in SEEDS:
        path = BASE / f"experiments/aaai_crossfitted_corrected_lat_seed_{seed}_20260721/crossfitted_lat_oof_predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["model"] == MODEL and row["segment_id"] in predictions:
                    predictions[row["segment_id"]].append(float(row["prediction"]))

    output_rows = []
    for case_name, segment_id in CASES.items():
        if segment_id not in gold or segment_id not in metadata:
            raise ValueError(f"Missing corrected gold or case metadata for {segment_id}")
        values = predictions[segment_id]
        if len(values) != len(SEEDS):
            raise ValueError(f"Expected {len(SEEDS)} predictions for {segment_id}, found {len(values)}")
        target = float(gold[segment_id]["perceived_latency"])
        prediction = statistics.mean(values)
        source = metadata[segment_id]
        output_rows.append(
            {
                "case": case_name,
                "segment_id": segment_id,
                "speech_group": gold[segment_id]["speech_group"],
                "direction": gold[segment_id]["direction"],
                "delay_seconds": gold[segment_id]["delay_seconds"],
                "LQ": gold[segment_id]["LQ"],
                "EXP": gold[segment_id]["EXP"],
                "human_LAT": target,
                "seed_20260718": values[0],
                "seed_20260719": values[1],
                "seed_20260720": values[2],
                "predicted_LAT_mean": prediction,
                "prediction_sd": statistics.stdev(values),
                "absolute_error": abs(prediction - target),
                "source": source["source"],
                "interpretation": source["interpretation"],
                "professional_comments": source["professional_comments"],
            }
        )

    output_dir = BASE / "experiments/paper_qualitative_cases_corrected_20260722"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paper_cases_corrected.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        "model": MODEL,
        "seeds": list(SEEDS),
        "prediction_scope": "corrected outer-speech-held-out LAT predictions",
        "selection_note": "Fixed manuscript cases; comments are post-hoc evidence and not model inputs.",
        "output": str(csv_path.relative_to(BASE)),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(csv_path.relative_to(BASE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
