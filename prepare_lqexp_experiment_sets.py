#!/usr/bin/env python3
"""Prepare LQ/EXP experiment JSON files for local, SSH, or Colab training.

Inputs:
  train_set.json
  dev_set.json
  data/evaluation/profess_eval.json

Outputs under data/experiments/lqexp/:
  professional_shared_train.json
  professional_shared_dev.json
  professional_shared_test.json
  train_original_plus_professional.json
  dev_original_plus_professional.json
  lqexp_experiment_sets_report.json

The professional split is segment-level: R05/R06 ratings are averaged by
segment_id, and only shared two-rater segments are used by default.

Use --exclude-original-eval-overlap for pooled training experiments. It removes
professional segments whose exact source/translation pair occurs in the
original dev or test set before the professional split is created.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_SEED = 20260708


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def speech_group(item: dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("direction") or "unknown"),
            str(item.get("round") or "unknown"),
            str(item.get("speech") or item.get("topic") or "unknown"),
        ]
    )


def infer_src_mt(item: dict[str, Any]) -> tuple[str | None, str | None]:
    src = item.get("src")
    mt = item.get("mt")
    if src and mt:
        return str(src), str(mt)

    direction = item.get("direction")
    source_chinese = item.get("source_chinese")
    target_english = item.get("target_english")
    if direction == "zh-en":
        return source_chinese, target_english
    if direction == "en-zh":
        return target_english, source_chinese
    return source_chinese, target_english


def text_key(item: dict[str, Any]) -> tuple[str, str] | None:
    src, mt = infer_src_mt(item)
    if not src or not mt:
        return None
    return (" ".join(str(src).split()), " ".join(str(mt).split()))


def to_lqexp_item(item: dict[str, Any]) -> dict[str, Any] | None:
    src, mt = infer_src_mt(item)
    if not src or not mt or item.get("LQ") is None or item.get("EXP") is None:
        return None
    return {
        "src": src,
        "mt": mt,
        "ref": item.get("offline_mt_en") or "",
        "LQ": float(item["LQ"]),
        "EXP": float(item["EXP"]),
        "segment_id": item.get("segment_id"),
        "direction": item.get("direction"),
        "speech_group": item.get("speech_group") or speech_group(item),
        "num_raters": item.get("num_raters"),
        "raters": item.get("raters"),
    }


def aggregate_professional_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        segment_id = row.get("segment_id")
        if segment_id:
            by_id[str(segment_id)].append(row)

    segment_rows = []
    for segment_id, items in by_id.items():
        lqs = [float(item["LQ"]) for item in items if item.get("LQ") is not None]
        exps = [float(item["EXP"]) for item in items if item.get("EXP") is not None]
        if not lqs or not exps:
            continue
        base = dict(items[0])
        raters = sorted({str(item.get("evaluator_id")) for item in items if item.get("evaluator_id")})
        base["segment_id"] = segment_id
        base["LQ"] = round(mean(lqs), 3)
        base["EXP"] = round(mean(exps), 3)
        base["num_raters"] = len(raters)
        base["raters"] = raters
        base["speech_group"] = speech_group(base)
        segment_rows.append(base)

    return sorted(segment_rows, key=lambda item: str(item["segment_id"]))


def split_by_group(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["speech_group"]].append(row)

    by_direction: dict[str, list[str]] = defaultdict(list)
    for group_name in groups:
        direction = group_name.split("|", 1)[0]
        by_direction[direction].append(group_name)

    rng = random.Random(seed)
    split_groups = {"train": set(), "dev": set(), "test": set()}
    for group_names in by_direction.values():
        rng.shuffle(group_names)
        n = len(group_names)
        n_test = max(1, round(n * 0.15))
        n_dev = max(1, round(n * 0.15)) if n >= 3 else 0
        split_groups["test"].update(group_names[:n_test])
        split_groups["dev"].update(group_names[n_test : n_test + n_dev])
        split_groups["train"].update(group_names[n_test + n_dev :])

    splits = {"train": [], "dev": [], "test": []}
    for split, names in split_groups.items():
        for name in sorted(names):
            splits[split].extend(groups[name])
        splits[split].sort(key=lambda item: str(item["segment_id"]))
    return splits


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_segments": len({row.get("segment_id") for row in rows}),
        "directions": dict(Counter(row.get("direction") for row in rows)),
        "num_raters": dict(Counter(row.get("num_raters") for row in rows)),
        "speech_groups": len({row.get("speech_group") for row in rows}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LQ/EXP experiment data.")
    parser.add_argument("--train", default="train_set.json")
    parser.add_argument("--dev", default="dev_set.json")
    parser.add_argument("--test", default="test_set.json")
    parser.add_argument("--profess", default="data/evaluation/profess_eval.json")
    parser.add_argument("--out-dir", default="data/experiments/lqexp")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--include-single-rater",
        action="store_true",
        help="Use all professional segments instead of only R05/R06 shared segments.",
    )
    parser.add_argument(
        "--exclude-original-eval-overlap",
        action="store_true",
        help="Remove professional segments overlapping original dev/test text pairs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)

    original_train = load_json(Path(args.train))
    original_dev = load_json(Path(args.dev))
    original_test = load_json(Path(args.test))
    professional_rows = load_json(Path(args.profess))

    professional_segments = aggregate_professional_segments(professional_rows)
    if args.include_single_rater:
        professional_for_split = professional_segments
        professional_scope = "all_segments"
    else:
        professional_for_split = [row for row in professional_segments if row.get("num_raters", 0) >= 2]
        professional_scope = "shared_two_rater_segments"

    original_eval_keys = {
        key
        for item in original_dev + original_test
        if (key := text_key(item)) is not None
    }
    overlap_before = sum(
        text_key(item) in original_eval_keys
        for item in professional_for_split
        if text_key(item) is not None
    )
    removed_overlap_segments = 0
    if args.exclude_original_eval_overlap:
        retained = []
        for item in professional_for_split:
            if text_key(item) in original_eval_keys:
                removed_overlap_segments += 1
            else:
                retained.append(item)
        professional_for_split = retained

    splits = split_by_group(professional_for_split, args.seed)
    prof_train = [item for item in (to_lqexp_item(row) for row in splits["train"]) if item]
    prof_dev = [item for item in (to_lqexp_item(row) for row in splits["dev"]) if item]
    prof_test = [item for item in (to_lqexp_item(row) for row in splits["test"]) if item]

    original_plus_prof_train = original_train + prof_train
    original_plus_prof_dev = original_dev + prof_dev

    outputs = {
        "professional_shared_train": out_dir / "professional_shared_train.json",
        "professional_shared_dev": out_dir / "professional_shared_dev.json",
        "professional_shared_test": out_dir / "professional_shared_test.json",
        "train_original_plus_professional": out_dir / "train_original_plus_professional.json",
        "dev_original_plus_professional": out_dir / "dev_original_plus_professional.json",
    }
    write_json(outputs["professional_shared_train"], prof_train)
    write_json(outputs["professional_shared_dev"], prof_dev)
    write_json(outputs["professional_shared_test"], prof_test)
    write_json(outputs["train_original_plus_professional"], original_plus_prof_train)
    write_json(outputs["dev_original_plus_professional"], original_plus_prof_dev)

    report = {
        "seed": args.seed,
        "professional_scope": professional_scope,
        "inputs": {
            "original_train_rows": len(original_train),
            "original_dev_rows": len(original_dev),
            "professional_rater_rows": len(professional_rows),
            "professional_segment_rows": len(professional_segments),
            "professional_used_segment_rows": len(professional_for_split),
        },
        "original_evaluation_overlap_audit": {
            "enabled": args.exclude_original_eval_overlap,
            "original_dev_test_text_pairs": len(original_eval_keys),
            "professional_segments_overlapping_before_exclusion": overlap_before,
            "professional_segments_removed": removed_overlap_segments,
            "professional_segments_overlapping_after_exclusion": sum(
                text_key(item) in original_eval_keys
                for item in professional_for_split
                if text_key(item) is not None
            ),
        },
        "professional_splits": {
            "train": summarize(splits["train"]),
            "dev": summarize(splits["dev"]),
            "test": summarize(splits["test"]),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "output_counts": {
            "professional_train_lqexp_rows": len(prof_train),
            "professional_dev_lqexp_rows": len(prof_dev),
            "professional_test_lqexp_rows": len(prof_test),
            "train_original_plus_professional_rows": len(original_plus_prof_train),
            "dev_original_plus_professional_rows": len(original_plus_prof_dev),
        },
    }
    report_path = out_dir / "lqexp_experiment_sets_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
