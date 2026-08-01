#!/usr/bin/env python3
"""Add only verified delay values to professional ratings.

The original files are preserved. Delay belongs to the source/interpretation
segment, so matching deliberately uses normalized source text + translation
text, not evaluator-specific segment IDs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import re


BASE = Path(__file__).parent
STUDENT = BASE / "data/evaluation/student_eval.json"
PROFESSIONAL = BASE / "data/evaluation/profess_eval.json"
OUT = BASE / "data/evaluation/profess_eval_delay_enriched.json"
REPORT = BASE / "data/evaluation/profess_eval_delay_enrichment_report.json"
EXTRA_DELAY_SOURCES = [
    BASE / "merged_dataset.json",
    BASE / "dev_set.json",
    BASE / "test_set.json",
]


def norm(value):
    return " ".join(str(value or "").strip().split()).casefold()


def norm_aggressive(value):
    value = str(value or "").casefold()
    value = value.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def text_pair(row):
    # student_eval stores source_chinese/target_english; profess_eval also
    # carries src/mt. Prefer the explicit pair when it is available.
    source = row.get("src") or row.get("source_chinese") or row.get("source_english") or ""
    target = (
        row.get("mt")
        or row.get("target_english")
        or row.get("target_chinese")
        or row.get("offline_mt_en")
        or ""
    )
    return norm(source), norm(target)


def aggressive_text_pair(row):
    source, target = text_pair(row)
    return norm_aggressive(source), norm_aggressive(target)


def has_delay(row):
    return row.get("delay_seconds") not in (None, "")


def add_delay(index, key, delay, source_name):
    if not key or not all(key):
        return
    index[key].append({"delay": float(delay), "source": source_name})


def unique_values(matches):
    return sorted({round(x["delay"], 6) for x in matches})


def source_names(matches):
    return sorted({x["source"] for x in matches})


def is_prefix_match(left, right, min_chars=30):
    left = norm_aggressive(left)
    right = norm_aggressive(right)
    if len(left) < min_chars or len(right) < min_chars:
        return False
    return left.startswith(right) or right.startswith(left)


def main():
    student = json.loads(STUDENT.read_text(encoding="utf-8"))
    professional = json.loads(PROFESSIONAL.read_text(encoding="utf-8"))
    delay_sources = [("student_eval.json", student)]
    for path in EXTRA_DELAY_SOURCES:
        if path.exists():
            delay_sources.append((path.name, json.loads(path.read_text(encoding="utf-8"))))

    exact_index = defaultdict(list)
    aggressive_index = defaultdict(list)
    source_index = defaultdict(list)
    for source_name, rows in delay_sources:
        for row in rows:
            if not has_delay(row):
                continue
            add_delay(exact_index, text_pair(row), row["delay_seconds"], source_name)
            add_delay(aggressive_index, aggressive_text_pair(row), row["delay_seconds"], source_name)
            source_key, _ = aggressive_text_pair(row)
            if source_key:
                source_index[source_key].append({
                    "delay": float(row["delay_seconds"]),
                    "source": source_name,
                    "target": text_pair(row)[1],
                })

    enriched = []
    stats = {
        "professional_rows": len(professional),
        "professional_rows_already_with_delay": 0,
        "indexed_delay_sources": {name: sum(has_delay(x) for x in rows) for name, rows in delay_sources},
        "copied_rows": 0,
        "copied_rows_exact_src_mt": 0,
        "copied_rows_aggressive_src_mt": 0,
        "copied_rows_prefix_src_mt": 0,
        "copied_unique_segments": 0,
        "ambiguous_matches": 0,
        "unresolved_rows": 0,
    }
    copied_segments = set()
    examples = []
    ambiguous_examples = []

    for original in professional:
        row = dict(original)
        if has_delay(row):
            row["delay_source"] = row.get("delay_source") or "professional_original"
            stats["professional_rows_already_with_delay"] += 1
            enriched.append(row)
            continue

        key = text_pair(row)
        matches = exact_index.get(key, [])
        match_confidence = "exact_src_and_mt_text"
        if not matches:
            matches = aggressive_index.get(aggressive_text_pair(row), [])
            match_confidence = "aggressive_exact_src_and_mt_text"
        if not matches:
            source_key, target_key = aggressive_text_pair(row)
            matches = [
                match
                for match in source_index.get(source_key, [])
                if is_prefix_match(target_key, match["target"])
            ]
            match_confidence = "same_src_and_prefix_mt_text"

        values = unique_values(matches)
        if len(values) == 1:
            row["delay_seconds"] = values[0]
            row["delay_source"] = "copied_from_verified_delay_source"
            row["delay_match_confidence"] = match_confidence
            row["delay_match_sources"] = source_names(matches)
            stats["copied_rows"] += 1
            if match_confidence == "exact_src_and_mt_text":
                stats["copied_rows_exact_src_mt"] += 1
            elif match_confidence == "aggressive_exact_src_and_mt_text":
                stats["copied_rows_aggressive_src_mt"] += 1
            else:
                stats["copied_rows_prefix_src_mt"] += 1
            copied_segments.add(str(row.get("segment_id")))
            if len(examples) < 10:
                examples.append({
                    "segment_id": row.get("segment_id"),
                    "delay_seconds": row["delay_seconds"],
                    "source_file": row.get("source_file"),
                    "match_confidence": match_confidence,
                    "match_sources": row["delay_match_sources"],
                })
        elif len(values) > 1:
            row["delay_source"] = "unresolved_delay_conflict"
            row["delay_conflict_values"] = values
            row["delay_match_confidence"] = match_confidence
            stats["ambiguous_matches"] += 1
            if len(ambiguous_examples) < 10:
                ambiguous_examples.append({
                    "segment_id": row.get("segment_id"),
                    "source_file": row.get("source_file"),
                    "values": values,
                    "match_confidence": match_confidence,
                    "match_sources": source_names(matches),
                })
        else:
            row["delay_source"] = "missing"
            stats["unresolved_rows"] += 1
        enriched.append(row)

    stats["copied_unique_segments"] = len(copied_segments)
    stats["examples"] = examples
    stats["ambiguous_examples"] = ambiguous_examples
    stats["output_delay_rows"] = sum(has_delay(x) for x in enriched)
    stats["output_delay_unique_segments"] = len({str(x.get("segment_id")) for x in enriched if x.get("delay_seconds") not in (None, "")})

    OUT.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
