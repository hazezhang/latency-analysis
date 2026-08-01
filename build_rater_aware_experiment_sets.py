#!/usr/bin/env python3
"""Build leakage-audited data sets for AAAI rater robustness experiments."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
PROFESSIONAL = BASE / "data/evaluation/profess_eval_delay_enriched.json"
STUDENTS = BASE / "data/evaluation/student_eval.json"
SHARED = BASE / "data/experiments/lqexp_shared_20260718"
OUT = BASE / "data/experiments/aaai_rater_aware_20260720"
SPLITS = ("train", "dev", "test")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_id(row):
    return f"{str(row['file_id']).zfill(3)}{int(row['original_segment_id']):02d}"


def text_key(row):
    src = row.get("src") or row.get("source_chinese") or row.get("source_english")
    mt = row.get("mt") or row.get("target_english") or row.get("target_chinese")
    return (" ".join(str(src).split()), " ".join(str(mt).split()))


def complete(row, professional):
    lq = row.get("LQ") if professional else row.get("language_quality")
    exp = row.get("EXP") if professional else row.get("expressiveness")
    return lq is not None and exp is not None


def to_training_row(row, professional, rater, split):
    src = row.get("src") or row.get("source_chinese") or row.get("source_english")
    mt = row.get("mt") or row.get("target_english") or row.get("target_chinese")
    return {
        "src": src,
        "mt": mt,
        "ref": row.get("offline_mt_en") or "",
        "LQ": float(row.get("LQ") if professional else row.get("language_quality")),
        "EXP": float(row.get("EXP") if professional else row.get("expressiveness")),
        "segment_id": stable_id(row),
        "evaluator_id": rater,
        "supervision_domain": "professional" if professional else "student",
        "source_split": split,
    }


def main():
    shared = {split: load(SHARED / f"professional_shared_{split}.json") for split in SPLITS}
    split_by_id = {row["segment_id"]: split for split, rows in shared.items() for row in rows}
    professional_rows = [row for row in load(PROFESSIONAL) if complete(row, professional=True)]
    student_rows = [row for row in load(STUDENTS) if complete(row, professional=False)]

    # E1: identical segment split, separate professional supervision.
    e1 = {}
    for rater in ("R05", "R06"):
        by_split = defaultdict(list)
        for row in professional_rows:
            if row["evaluator_id"] != rater:
                continue
            split = split_by_id.get(stable_id(row))
            if split:
                by_split[split].append(to_training_row(row, True, rater, split))
        for split in SPLITS:
            rows = sorted(by_split[split], key=lambda row: row["segment_id"])
            expected = len(shared[split])
            if len(rows) != expected:
                raise ValueError(f"{rater} {split}: expected {expected}, found {len(rows)}")
            write(OUT / "professional_individual" / rater / f"{split}.json", rows)
        e1[rater] = {split: len(by_split[split]) for split in SPLITS}

    # E3: no student text may overlap professional dev/test during pretraining.
    protected_keys = {text_key(row) for split in ("dev", "test") for row in shared[split]}
    student_train, excluded = [], []
    for row in student_rows:
        record = to_training_row(row, False, row["evaluator_id"], "student_pretrain")
        if text_key(row) in protected_keys:
            excluded.append(record)
        else:
            student_train.append(record)
    write(OUT / "student_pretraining" / "train.json", student_train)
    write(OUT / "student_pretraining" / "excluded_professional_dev_test_overlap.json", excluded)

    # E2 input: professional row-level supervision with the same leakage-safe split.
    e2 = defaultdict(list)
    for row in professional_rows:
        split = split_by_id.get(stable_id(row))
        if split and row["evaluator_id"] in {"R05", "R06"}:
            e2[split].append(to_training_row(row, True, row["evaluator_id"], split))
    for split in SPLITS:
        rows = sorted(e2[split], key=lambda row: (row["segment_id"], row["evaluator_id"]))
        write(OUT / "professional_rater_rows" / f"{split}.json", rows)

    report = {
        "purpose": "AAAI rater robustness; no raw student-professional score pooling",
        "shared_reference_split": str(SHARED.relative_to(BASE)),
        "E1_professional_individual_raters": e1,
        "E2_professional_rater_rows": {split: len(e2[split]) for split in SPLITS},
        "E3_student_pretraining": {
            "usable_rows": len(student_train),
            "excluded_text_overlap_with_professional_dev_test": len(excluded),
            "rater_counts": dict(sorted(Counter(row["evaluator_id"] for row in student_train).items())),
        },
        "leakage_rule": "student pretraining excludes any source-interpretation pair in professional dev or test",
    }
    write(OUT / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
