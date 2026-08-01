#!/usr/bin/env python3
"""Create a blinded, stratified manual sheet for validating reordering features."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COHORT = ROOT / "data/experiments/r027_shared_outer_quality/all_lat_segments.json"
COMMENTS = ROOT / "data/evaluation/profess_eval_delay_enriched.json"
OUT = ROOT / "experiments/aaai_reordering_annotation_20260720"
BINS = ((0.0, 2.0, "0-2"), (2.0, 6.0, "2-6"), (6.0, 10.0, "6-10"), (10.0, 20.0, "10-20"))


def comment_index(rows):
    result = defaultdict(list)
    for row in rows:
        comment = (row.get("comments") or "").strip()
        if comment and not comment.startswith("R06 review:"):
            result[f"{str(row['file_id']).zfill(3)}:{int(row['original_segment_id'])}"].append(comment)
    return result


def complexity_proxy(text, direction):
    markers = ("which", "that", "if", "when", "while", "because", "although", "after", "before", "but") if direction == "en-zh" else ("因为", "如果", "虽然", "但是", "当", "为了", "之后", "之前", "同时", "不仅", "而且")
    return sum(marker in text.lower() for marker in markers) + text.count(",")


def stable_hash(segment_id):
    return hashlib.sha256(segment_id.encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comments = comment_index(json.loads(COMMENTS.read_text(encoding="utf-8")))
    rows = json.loads(COHORT.read_text(encoding="utf-8"))
    selected = []
    for direction in ("zh-en", "en-zh"):
        for lower, upper, label in BINS:
            candidates = [
                row for row in rows
                if row["direction"] == direction and lower <= float(row["delay_seconds"]) <= upper and row["segment_id"] in comments
            ]
            candidates.sort(key=lambda row: (-complexity_proxy(row["src"], direction), stable_hash(row["segment_id"])))
            # Two structure-dense and two comparison candidates per stratum.
            dense = candidates[:2]
            remaining = sorted(candidates[2:], key=lambda row: stable_hash(row["segment_id"]))[:2]
            selected.extend(dense + remaining)

    fields = [
        "blind_case_id", "segment_id", "direction", "delay_bin", "delay_seconds", "source", "interpretation",
        "human_LQ", "human_EXP", "human_LAT", "professional_comment", "source_clause_complexity_0_2",
        "target_reorders_main_information_none_partial_substantial_unclear", "anticipation_none_partial_clear_unclear",
        "omission_none_minor_major_unclear", "notes",
    ]
    output = []
    for index, row in enumerate(selected, 1):
        delay = float(row["delay_seconds"])
        label = next(label for lower, upper, label in BINS if lower <= delay <= upper)
        output.append({
            "blind_case_id": f"case_{index:02d}", "segment_id": row["segment_id"], "direction": row["direction"],
            "delay_bin": label, "delay_seconds": delay, "source": row["src"], "interpretation": row["mt"],
            "human_LQ": row["LQ"], "human_EXP": row["EXP"], "human_LAT": row["perceived_latency"],
            "professional_comment": " | ".join(comments[row["segment_id"]]),
            "source_clause_complexity_0_2": "", "target_reorders_main_information_none_partial_substantial_unclear": "",
            "anticipation_none_partial_clear_unclear": "", "omission_none_minor_major_unclear": "", "notes": "",
        })
    with (OUT / "reordering_manual_annotation_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    (OUT / "README.md").write_text(
        "# Manual Reordering Annotation\n\n"
        "This 32-case sheet is stratified by direction and measured-delay range. Annotators should inspect source and interpretation before consulting scores/comments. "
        "`source_clause_complexity`: 0=simple, 1=one embedded/dependent relation, 2=multiple embedded or deferred relations. "
        "Reordering concerns main information order, not literal word order. Omission must be coded separately from reordering. Two independent annotators should code the sheet, resolve disagreements, and record the protocol before the alignment-feature analysis is interpreted.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(output)} stratified cases to {OUT}")


if __name__ == "__main__":
    main()
