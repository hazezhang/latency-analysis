#!/usr/bin/env python3
"""Build fold-safe student weak-supervision inputs and audit M0 invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
DEFAULT_PROFESSIONAL = BASE / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"
DEFAULT_STUDENTS = BASE / "data/evaluation/student_eval.json"
DEFAULT_OUTPUT = BASE / "data/experiments/student_weak_supervision_m0_20260804"


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_pair(row: dict) -> tuple[str, str] | None:
    source = row.get("src") or row.get("source_chinese") or row.get("source_english")
    target = row.get("mt") or row.get("target_english") or row.get("target_chinese")
    if not source or not target:
        return None
    return (" ".join(str(source).split()), " ".join(str(target).split()))


def complete_student(row: dict) -> bool:
    return (
        row.get("language_quality") is not None
        and row.get("expressiveness") is not None
        and normalized_pair(row) is not None
    )


def student_record(row: dict, outer_speech: str) -> dict:
    pair = normalized_pair(row)
    assert pair is not None
    return {
        "segment_id": str(row.get("segment_id")),
        "file_id": str(row.get("file_id")).zfill(3),
        "original_segment_id": int(row.get("original_segment_id")),
        "src": pair[0],
        "mt": pair[1],
        "ref": row.get("offline_mt_en") or "",
        "LQ": float(row["language_quality"]),
        "EXP": float(row["expressiveness"]),
        "evaluator_id": str(row.get("evaluator_id")),
        "supervision_domain": "student_raw",
        "excluded_professional_outer_speech": outer_speech,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sequence_audit(rows: list[dict]) -> dict:
    by_file: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        file_id, original_segment_id = str(row["segment_id"]).split(":", maxsplit=1)
        by_file[file_id].append(int(original_segment_id))

    files = []
    for file_id, positions in sorted(by_file.items()):
        duplicate_positions = sorted(position for position, count in Counter(positions).items() if count > 1)
        files.append(
            {
                "file_id": file_id,
                "n_segments": len(positions),
                "first_original_segment_id": min(positions),
                "last_original_segment_id": max(positions),
                "duplicate_original_segment_ids": duplicate_positions,
            }
        )
    return {
        "n_files": len(files),
        "files": files,
        "online_feature_rule": (
            "Use only previous observed delay/text features ordered by file_id and "
            "original_segment_id. No full-speech aggregate, future segment, or future-aware normalization is allowed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--professional", type=Path, default=DEFAULT_PROFESSIONAL)
    parser.add_argument("--students", type=Path, default=DEFAULT_STUDENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    professional = load_json(args.professional)
    students = [row for row in load_json(args.students) if complete_student(row)]
    ids = [str(row["segment_id"]) for row in professional]
    speeches = sorted({str(row["speech_group"]) for row in professional})

    if len(professional) != 622 or len(speeches) != 16:
        raise ValueError(f"Expected 622 professional segments across 16 speech groups, found {len(professional)} / {len(speeches)}")
    if len(ids) != len(set(ids)):
        raise ValueError("Professional main cohort has duplicate segment IDs")

    audit = sequence_audit(professional)
    duplicate_positions = [item for item in audit["files"] if item["duplicate_original_segment_ids"]]
    if duplicate_positions:
        raise ValueError(f"Duplicate within-file positions: {duplicate_positions[:3]}")

    manifest: dict[str, object] = {
        "purpose": "M0 audit and per-outer-fold raw-student weak-supervision inputs",
        "professional_main_cohort": {
            "definition": "shared professional ratings of interpreter performance",
            "n_segments": len(professional),
            "n_speech_groups": len(speeches),
            "input": str(args.professional.relative_to(BASE)),
            "sha256": sha256(args.professional),
        },
        "student_raw_supervision": {
            "definition": "complete student LQ/EXP rating rows before each outer-test text-pair exclusion",
            "n_rows": len(students),
            "rater_counts": dict(sorted(Counter(str(row.get("evaluator_id")) for row in students).items())),
            "input": str(args.students.relative_to(BASE)),
            "sha256": sha256(args.students),
        },
        "outer_fold_rule": "For every professional outer test speech, exclude all student rows whose normalized source-target pair appears in that professional outer test speech.",
        "sequence_audit": audit,
        "folds": [],
    }

    for index, speech in enumerate(speeches, start=1):
        outer_rows = [row for row in professional if str(row["speech_group"]) == speech]
        protected_pairs = {normalized_pair(row) for row in outer_rows}
        allowed, excluded = [], []
        for row in students:
            record = student_record(row, speech)
            (excluded if normalized_pair(row) in protected_pairs else allowed).append(record)
        overlap = {normalized_pair(row) for row in allowed} & protected_pairs
        if overlap:
            raise ValueError(f"Student/professional outer-test leakage for speech {speech}")

        fold_name = f"outer_{index:02d}_speech_{speech}"
        fold_dir = args.output_dir / fold_name
        write_json(fold_dir / "student_raw_train.json", allowed)
        write_json(fold_dir / "student_excluded_outer_test_pairs.json", excluded)
        manifest["folds"].append(
            {
                "name": fold_name,
                "outer_test_speech": speech,
                "n_professional_outer_test_segments": len(outer_rows),
                "n_protected_text_pairs": len(protected_pairs),
                "n_student_raw_train": len(allowed),
                "n_student_excluded_outer_test_pairs": len(excluded),
                "student_train_overlap_with_outer_test_pairs": len(overlap),
            }
        )

    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
