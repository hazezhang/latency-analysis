#!/usr/bin/env python3
"""Prepare R020 outer-nested quality-model folds for LAT prediction.

For each outer LAT speech group, the quality train/dev split excludes that
outer speech from all upstream model training and checkpoint selection. The
trained quality model then predicts every post-R021 LAT segment; the LAT
second-stage script uses the prediction file that corresponds to its current
outer speech fold.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


BASE = Path(__file__).parent
SAFE_DIR = BASE / "data/experiments/lqexp_leakage_safe"
RAW_PATH = BASE / "data/evaluation/profess_eval_delay_enriched.json"
OUT_DIR = BASE / "data/experiments/r020_nested_quality"
SPLIT_FILES = (
    "professional_shared_train.json",
    "professional_shared_dev.json",
    "professional_shared_test.json",
)
LAT_REQUIRED = ("LQ", "EXP", "perceived_latency", "delay_seconds")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def text_key(row: dict) -> tuple[str, str] | None:
    src = row.get("src")
    mt = row.get("mt")
    if not src or not mt:
        direction = row.get("direction")
        source = row.get("source_chinese") or row.get("source_english")
        target = row.get("target_english") or row.get("target_chinese")
        if direction == "en-zh":
            src, mt = target, source
        else:
            src, mt = source, target
    if not src or not mt:
        return None
    return (" ".join(str(src).split()), " ".join(str(mt).split()))


def speech_of_safe(row: dict) -> str:
    return str(row["speech_group"]).rsplit("|", 1)[-1]


def aggregate_lat_segments(raw_rows: list[dict]) -> list[dict]:
    by_segment: dict[str, list[dict]] = defaultdict(list)
    for row in raw_rows:
        if row.get("segment_id") not in (None, ""):
            by_segment[str(row["segment_id"])].append(row)

    output = []
    for segment_id, rows in sorted(by_segment.items()):
        if not all(any(number(row.get(key)) is not None for row in rows) for key in LAT_REQUIRED):
            continue
        first = rows[0]
        key = text_key(first)
        if key is None:
            continue
        output.append({
            "segment_id": segment_id,
            "src": key[0],
            "mt": key[1],
            "ref": first.get("offline_mt_en") or "",
            "LQ": round(mean(number(row.get("LQ")) for row in rows if number(row.get("LQ")) is not None), 3),
            "EXP": round(mean(number(row.get("EXP")) for row in rows if number(row.get("EXP")) is not None), 3),
            "perceived_latency": round(mean(number(row.get("perceived_latency")) for row in rows if number(row.get("perceived_latency")) is not None), 3),
            "delay_seconds": round(mean(number(row.get("delay_seconds")) for row in rows if number(row.get("delay_seconds")) is not None), 3),
            "direction": first.get("direction"),
            "speech": str(first.get("speech")),
            "round": first.get("round"),
            "source_file": first.get("source_file"),
        })
    return output


def choose_inner_dev(outer_speech: str, safe_speeches: list[str]) -> str:
    candidates = [speech for speech in safe_speeches if speech != outer_speech]
    if not candidates:
        raise ValueError(f"No inner-dev speech available for outer={outer_speech}")
    if outer_speech in safe_speeches:
        return candidates[(safe_speeches.index(outer_speech) + 1) % len(candidates)]
    return candidates[0]


def main() -> int:
    safe_rows = []
    for filename in SPLIT_FILES:
        safe_rows.extend(load_json(SAFE_DIR / filename))
    safe_ids = {str(row["segment_id"]) for row in safe_rows}

    raw_rows = load_json(RAW_PATH)
    lat_rows = aggregate_lat_segments(raw_rows)
    if len(lat_rows) != 150:
        raise ValueError(f"Expected 150 post-R021 LAT segments, found {len(lat_rows)}")

    safe_speeches = sorted({speech_of_safe(row) for row in safe_rows})
    by_outer: dict[str, list[dict]] = defaultdict(list)
    for row in lat_rows:
        by_outer[row["speech"]].append(row)

    manifest = {
        "purpose": "R020 outer-nested two-stage quality predictions for LAT",
        "quality_training_source": str(SAFE_DIR.relative_to(BASE)),
        "lat_target_source": str(RAW_PATH.relative_to(BASE)),
        "lat_target_segments": len(lat_rows),
        "lat_target_speeches": dict(sorted(Counter(row["speech"] for row in lat_rows).items())),
        "lat_target_directions": dict(sorted(Counter(row["direction"] for row in lat_rows).items())),
        "safe_quality_segments": len(safe_rows),
        "safe_quality_speeches": dict(sorted(Counter(speech_of_safe(row) for row in safe_rows).items())),
        "nesting_rule": "For each LAT outer speech, quality train/dev exclude that speech; LAT train and test features in that fold use predictions from the same excluded-speech quality model. This is outer-nested two-stage evaluation, not inner-OOF second-stage feature generation.",
        "folds": [],
    }

    write_json(OUT_DIR / "all_lat_segments.json", sorted(lat_rows, key=lambda row: row["segment_id"]))

    for fold_index, outer_speech in enumerate(sorted(by_outer), start=1):
        inner_dev = choose_inner_dev(outer_speech, safe_speeches)
        train_rows = [row for row in safe_rows if speech_of_safe(row) not in {outer_speech, inner_dev}]
        dev_rows = [row for row in safe_rows if speech_of_safe(row) == inner_dev]
        outer_test_rows = sorted(by_outer[outer_speech], key=lambda row: row["segment_id"])

        train_keys = {text_key(row) for row in train_rows}
        dev_keys = {text_key(row) for row in dev_rows}
        outer_keys = {text_key(row) for row in outer_test_rows}
        overlaps = {
            "train_outer_test_text": len(train_keys & outer_keys),
            "dev_outer_test_text": len(dev_keys & outer_keys),
            "train_dev_text": len(train_keys & dev_keys),
        }
        if any(overlaps.values()):
            raise ValueError(f"Text overlap in R020 fold {outer_speech}: {overlaps}")
        if any(speech_of_safe(row) == outer_speech for row in train_rows + dev_rows):
            raise ValueError(f"Outer speech leaked into quality train/dev for {outer_speech}")

        fold_name = f"fold_{fold_index:02d}_speech_{outer_speech}"
        fold_dir = OUT_DIR / fold_name
        write_json(fold_dir / "train.json", train_rows)
        write_json(fold_dir / "dev.json", dev_rows)
        write_json(fold_dir / "predict_all.json", sorted(lat_rows, key=lambda row: row["segment_id"]))
        write_json(fold_dir / "outer_test_segments.json", outer_test_rows)

        manifest["folds"].append({
            "fold": fold_index,
            "name": fold_name,
            "outer_test_speech": outer_speech,
            "inner_dev_speech": inner_dev,
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "predict_rows": len(lat_rows),
            "outer_test_rows": len(outer_test_rows),
            "outer_test_rows_present_in_safe_quality_pool": sum(str(row["segment_id"]) in safe_ids for row in outer_test_rows),
            "text_overlap_audit": overlaps,
        })

    write_json(OUT_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
