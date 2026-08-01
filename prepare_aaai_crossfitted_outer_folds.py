#!/usr/bin/env python3
"""Create outer-speech folds with cross-fitted quality-prediction splits.

For an outer LAT test speech, every remaining segment receives LQ/EXP features
from a quality model that did not train on that segment's speech group. The
outer test speech is excluded from every upstream model in its outer fold.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).parent
RAW_PATH = BASE / "data/evaluation/profess_eval_delay_enriched_namespace_corrected.json"
OUT_DIR = BASE / "data/experiments/aaai_crossfitted_outer_quality_corrected"
N_INNER_PARTITIONS = 4


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def number(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def stable_id(row: dict) -> str:
    return f"{row.get('file_id') or 'unknown'}:{row.get('original_segment_id') or row.get('segment_id') or 'unknown'}"


def text_pair(row: dict) -> tuple[str, str] | None:
    src = row.get("src") or row.get("source_chinese") or row.get("source_english")
    target = row.get("mt") or row.get("target_english") or row.get("target_chinese")
    if not src or not target:
        return None
    return (" ".join(str(src).split()), " ".join(str(target).split()))


def mode(values: list[str]) -> str:
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def aggregate_shared(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        values = {key: number(row.get(key)) for key in ("LQ", "EXP", "perceived_latency", "delay_seconds")}
        if any(value is None for value in values.values()):
            continue
        item = dict(row)
        item.update(values)
        grouped[stable_id(item)].append(item)

    output, metadata_conflicts = [], []
    for sample_id, items in sorted(grouped.items()):
        raters = {str(item.get("evaluator_id")) for item in items}
        if not {"R05", "R06"}.issubset(raters):
            continue
        by_rater: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            by_rater[str(item.get("evaluator_id"))].append(item)
        rater_means = {
            rater: {
                key: sum(item[key] for item in rater_rows) / len(rater_rows)
                for key in ("LQ", "EXP", "perceived_latency", "delay_seconds")
            }
            for rater, rater_rows in by_rater.items()
        }
        delay = sum(rater_means[rater]["delay_seconds"] for rater in ("R05", "R06")) / 2
        if not 0.0 <= delay <= 20.0:
            continue
        speeches = [str(item.get("speech") or item.get("source_file") or item.get("file_id")) for item in items]
        interpreters = [str(item.get("interpreter") or "unknown").strip().casefold() for item in items]
        base = items[0]
        pair = text_pair(base)
        if pair is None:
            continue
        speech, interpreter = mode(speeches), mode(interpreters)
        if len(set(speeches)) > 1 or len(set(interpreters)) > 1:
            metadata_conflicts.append({"segment_id": sample_id, "speech_values": sorted(set(speeches)), "interpreter_values": sorted(set(interpreters)), "resolved_speech": speech, "resolved_interpreter": interpreter})
        output.append({
            "segment_id": sample_id, "src": pair[0], "mt": pair[1], "ref": base.get("offline_mt_en") or "",
            "LQ": round(sum(rater_means[rater]["LQ"] for rater in ("R05", "R06")) / 2, 3),
            "EXP": round(sum(rater_means[rater]["EXP"] for rater in ("R05", "R06")) / 2, 3),
            "perceived_latency": round(sum(rater_means[rater]["perceived_latency"] for rater in ("R05", "R06")) / 2, 3),
            "delay_seconds": round(delay, 3),
            "direction": mode([str(item.get("direction") or "unknown") for item in items]),
            "speech_group": speech, "interpreter": interpreter, "rater_ids": sorted(raters),
        })
    return output, metadata_conflicts


def choose_inner_dev(outer: str, speech_counts: Counter) -> str:
    candidates = [speech for speech in speech_counts if speech != outer]
    return sorted(candidates, key=lambda speech: (-speech_counts[speech], speech))[0]


def balanced_partitions(speeches: list[str], counts: Counter) -> list[list[str]]:
    """Greedily balance segment counts, with a stable tie-break by speech name."""
    partitions: list[list[str]] = [[] for _ in range(N_INNER_PARTITIONS)]
    totals = [0] * N_INNER_PARTITIONS
    for speech in sorted(speeches, key=lambda item: (-counts[item], item)):
        index = min(range(N_INNER_PARTITIONS), key=lambda item: (totals[item], item))
        partitions[index].append(speech)
        totals[index] += counts[speech]
    return [sorted(partition) for partition in partitions]


def without_outer_duplicates(rows: list[dict], outer_text: set[tuple[str, str]]) -> list[dict]:
    return [row for row in rows if text_pair(row) not in outer_text]


def main() -> int:
    raw_rows = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    segments, metadata_conflicts = aggregate_shared(raw_rows)
    by_speech: dict[str, list[dict]] = defaultdict(list)
    for row in segments:
        by_speech[row["speech_group"]].append(row)
    speeches = sorted(by_speech)
    if len(segments) != 622 or len(speeches) != 16:
        raise ValueError(f"Expected 622 shared segments across 16 speeches; got {len(segments)} / {len(speeches)}")

    counts = Counter(row["speech_group"] for row in segments)
    manifest: dict[str, object] = {
        "protocol": "outer-speech-held-out two-stage LAT evaluation with cross-fitted upstream quality features",
        "input": str(RAW_PATH.relative_to(BASE)),
        "n_segments": len(segments),
        "n_speech_groups": len(speeches),
        "n_inner_partitions": N_INNER_PARTITIONS,
        "cohort_rule": "R05 and R06 both rate a segment; repeated judgments are averaged within evaluator and evaluator means receive equal weight; delay is within [0, 20] seconds.",
        "cross_fitting_rule": "For every outer fold, every LAT-training segment is predicted by an upstream quality model that excludes its speech group. Every upstream model also excludes the outer test speech and its exact source-target duplicates.",
        "metadata_conflicts_resolved_by_mode": metadata_conflicts,
        "folds": [],
    }
    write_json(OUT_DIR / "all_lat_segments.json", sorted(segments, key=lambda row: row["segment_id"]))

    for fold_index, outer in enumerate(speeches, start=1):
        fold_name = f"outer_{fold_index:02d}_speech_{outer}"
        fold_dir = OUT_DIR / fold_name
        outer_rows = sorted(by_speech[outer], key=lambda row: row["segment_id"])
        outer_text = {text_pair(row) for row in outer_rows}
        available = [speech for speech in speeches if speech != outer]
        partitions = balanced_partitions(available, counts)
        inner_manifest = []

        for inner_index, held_out_speeches in enumerate(partitions, start=1):
            name = f"inner_{inner_index:02d}"
            candidates = [speech for speech in available if speech not in held_out_speeches]
            dev_speech = choose_inner_dev(outer, Counter({speech: counts[speech] for speech in candidates}))
            train = [row for row in segments if row["speech_group"] in candidates and row["speech_group"] != dev_speech]
            dev = [row for row in segments if row["speech_group"] == dev_speech]
            predict = [row for row in segments if row["speech_group"] in held_out_speeches]
            train = without_outer_duplicates(train, outer_text)
            dev = without_outer_duplicates(dev, outer_text)
            predict = without_outer_duplicates(predict, outer_text)
            if not predict:
                raise ValueError(f"No prediction rows for {fold_name}/{name}")
            if {row["speech_group"] for row in predict} != set(held_out_speeches):
                raise ValueError(f"Unexpected duplicate removal from held-out inner speech in {fold_name}/{name}")
            write_json(fold_dir / name / "train.json", train)
            write_json(fold_dir / name / "dev.json", dev)
            write_json(fold_dir / name / "predict.json", sorted(predict, key=lambda row: row["segment_id"]))
            inner_manifest.append({
                "name": name,
                "held_out_speeches": held_out_speeches,
                "dev_speech": dev_speech,
                "n_train": len(train),
                "n_dev": len(dev),
                "n_predict": len(predict),
            })

        final_dev = choose_inner_dev(outer, counts)
        final_train = [row for row in segments if row["speech_group"] not in {outer, final_dev}]
        final_dev_rows = [row for row in segments if row["speech_group"] == final_dev]
        final_train = without_outer_duplicates(final_train, outer_text)
        final_dev_rows = without_outer_duplicates(final_dev_rows, outer_text)
        write_json(fold_dir / "final_outer" / "train.json", final_train)
        write_json(fold_dir / "final_outer" / "dev.json", final_dev_rows)
        write_json(fold_dir / "final_outer" / "predict.json", outer_rows)
        manifest["folds"].append({
            "name": fold_name,
            "outer_test_speech": outer,
            "outer_test_rows": len(outer_rows),
            "inner_folds": inner_manifest,
            "final_outer": {"dev_speech": final_dev, "n_train": len(final_train), "n_dev": len(final_dev_rows), "n_predict": len(outer_rows)},
        })

    write_json(OUT_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
