#!/usr/bin/env python3
"""Prepare leave-one-interpreter-out folds with inner speech-group OOF quality features."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"
DEFAULT_OUTPUT = BASE / "data/experiments/aaai_loio_outer_quality_corrected"
N_INNER_PARTITIONS = 4


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalized_pair(row: dict) -> tuple[str, str]:
    return (" ".join(str(row["src"]).split()), " ".join(str(row["mt"]).split()))


def normalized_source(row: dict) -> str:
    return " ".join(str(row["src"]).split())


def balanced_partitions(groups: list[str], counts: Counter) -> list[list[str]]:
    partitions: list[list[str]] = [[] for _ in range(N_INNER_PARTITIONS)]
    totals = [0] * N_INNER_PARTITIONS
    for group in sorted(groups, key=lambda item: (-counts[item], item)):
        index = min(range(N_INNER_PARTITIONS), key=lambda item: (totals[item], item))
        partitions[index].append(group)
        totals[index] += counts[group]
    return [sorted(partition) for partition in partitions]


def choose_dev_group(candidate_groups: list[str], counts: Counter) -> str:
    if not candidate_groups:
        raise ValueError("No candidate speech group is available for development")
    return sorted(candidate_groups, key=lambda group: (-counts[group], group))[0]


def assert_no_outer(rows: list[dict], outer_interpreter: str, label: str) -> None:
    offenders = [row["segment_id"] for row in rows if row["interpreter"] == outer_interpreter]
    if offenders:
        raise ValueError(f"Outer interpreter leaked into {label}: {offenders[:3]}")


def assert_disjoint(*row_sets: list[dict]) -> None:
    seen: set[str] = set()
    for rows in row_sets:
        ids = {row["segment_id"] for row in rows}
        overlap = seen & ids
        if overlap:
            raise ValueError(f"Split overlap: {sorted(overlap)[:3]}")
        seen.update(ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT.relative_to(BASE)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.relative_to(BASE)))
    args = parser.parse_args()

    input_path = BASE / args.input
    output_dir = BASE / args.output_dir
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if len(rows) != 622:
        raise ValueError(f"Expected 622 corrected shared-label rows, found {len(rows)}")

    ids = [str(row["segment_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Primary cohort contains duplicate segment IDs")
    interpreters = sorted({str(row["interpreter"]) for row in rows})
    if len(interpreters) != 7:
        raise ValueError(f"Expected seven interpreters, found {len(interpreters)}")

    manifest: dict[str, object] = {
        "protocol": "automatic leave-one-interpreter-out two-stage LAT evaluation with inner speech-group OOF quality features",
        "scope": "unseen interpreter; source speeches may appear through other interpreters",
        "input": str(input_path.relative_to(BASE)),
        "n_segments": len(rows),
        "n_interpreters": len(interpreters),
        "n_speech_groups": len({row["speech_group"] for row in rows}),
        "n_inner_partitions": N_INNER_PARTITIONS,
        "outer_exclusion_rule": "All rows from the held-out interpreter are excluded from quality train, quality development, checkpoint selection, inner OOF training, and LAT Ridge training.",
        "inner_cross_fitting_rule": "Remaining-interpreter LAT-training rows receive predicted LQ/EXP from models holding out their complete speech group.",
        "folds": [],
    }
    write_json(output_dir / "all_lat_segments.json", sorted(rows, key=lambda row: row["segment_id"]))

    for fold_index, outer_interpreter in enumerate(interpreters, start=1):
        fold_name = f"outer_{fold_index:02d}_interpreter_{outer_interpreter}"
        fold_dir = output_dir / fold_name
        outer_rows = sorted(
            [row for row in rows if row["interpreter"] == outer_interpreter],
            key=lambda row: row["segment_id"],
        )
        available_rows = [row for row in rows if row["interpreter"] != outer_interpreter]
        assert_no_outer(available_rows, outer_interpreter, f"{fold_name}/available")

        outer_pairs = {normalized_pair(row) for row in outer_rows}
        outer_sources = {normalized_source(row) for row in outer_rows}
        available_by_speech: dict[str, list[dict]] = defaultdict(list)
        for row in available_rows:
            available_by_speech[str(row["speech_group"])].append(row)
        speech_counts = Counter({speech: len(items) for speech, items in available_by_speech.items()})
        speeches = sorted(available_by_speech)
        partitions = balanced_partitions(speeches, speech_counts)
        inner_manifest = []
        predicted_training_ids: set[str] = set()

        for inner_index, held_out_speeches in enumerate(partitions, start=1):
            inner_name = f"inner_{inner_index:02d}"
            candidate_groups = [speech for speech in speeches if speech not in held_out_speeches]
            dev_speech = choose_dev_group(candidate_groups, speech_counts)
            train = [
                row for row in available_rows
                if row["speech_group"] in candidate_groups
                and row["speech_group"] != dev_speech
                and normalized_pair(row) not in outer_pairs
            ]
            dev = [
                row for row in available_rows
                if row["speech_group"] == dev_speech and normalized_pair(row) not in outer_pairs
            ]
            predict = [row for row in available_rows if row["speech_group"] in held_out_speeches]
            assert_no_outer(train, outer_interpreter, f"{fold_name}/{inner_name}/train")
            assert_no_outer(dev, outer_interpreter, f"{fold_name}/{inner_name}/dev")
            assert_no_outer(predict, outer_interpreter, f"{fold_name}/{inner_name}/predict")
            assert_disjoint(train, dev, predict)
            if not train or not dev or not predict:
                raise ValueError(f"Empty split in {fold_name}/{inner_name}")
            predict_ids = {row["segment_id"] for row in predict}
            if predicted_training_ids & predict_ids:
                raise ValueError(f"OOF prediction overlap in {fold_name}/{inner_name}")
            predicted_training_ids.update(predict_ids)
            write_json(fold_dir / inner_name / "train.json", train)
            write_json(fold_dir / inner_name / "dev.json", dev)
            write_json(fold_dir / inner_name / "predict.json", sorted(predict, key=lambda row: row["segment_id"]))
            inner_manifest.append({
                "name": inner_name,
                "held_out_speeches": held_out_speeches,
                "dev_speech": dev_speech,
                "n_train": len(train),
                "n_dev": len(dev),
                "n_predict": len(predict),
            })

        available_ids = {row["segment_id"] for row in available_rows}
        if predicted_training_ids != available_ids:
            missing = sorted(available_ids - predicted_training_ids)
            extra = sorted(predicted_training_ids - available_ids)
            raise ValueError(f"Incomplete OOF coverage in {fold_name}: missing={missing[:3]} extra={extra[:3]}")

        final_dev_speech = choose_dev_group(speeches, speech_counts)
        final_train = [
            row for row in available_rows
            if row["speech_group"] != final_dev_speech and normalized_pair(row) not in outer_pairs
        ]
        final_dev = [
            row for row in available_rows
            if row["speech_group"] == final_dev_speech and normalized_pair(row) not in outer_pairs
        ]
        assert_no_outer(final_train, outer_interpreter, f"{fold_name}/final_outer/train")
        assert_no_outer(final_dev, outer_interpreter, f"{fold_name}/final_outer/dev")
        assert_disjoint(final_train, final_dev, outer_rows)
        if not final_train or not final_dev or not outer_rows:
            raise ValueError(f"Empty final split in {fold_name}")
        write_json(fold_dir / "final_outer" / "train.json", final_train)
        write_json(fold_dir / "final_outer" / "dev.json", final_dev)
        write_json(fold_dir / "final_outer" / "predict.json", outer_rows)

        manifest["folds"].append({
            "name": fold_name,
            "outer_test_interpreter": outer_interpreter,
            "outer_test_rows": len(outer_rows),
            "outer_test_speeches": sorted({row["speech_group"] for row in outer_rows}),
            "n_outer_source_texts_seen_via_other_interpreters": len(
                outer_sources & {normalized_source(row) for row in available_rows}
            ),
            "n_exact_outer_pairs_removed_from_upstream_train_dev": sum(
                normalized_pair(row) in outer_pairs for row in available_rows
            ),
            "n_latency_train_rows": len(available_rows),
            "inner_folds": inner_manifest,
            "final_outer": {
                "dev_speech": final_dev_speech,
                "n_train": len(final_train),
                "n_dev": len(final_dev),
                "n_predict": len(outer_rows),
            },
        })

    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
