#!/usr/bin/env python3
"""Leakage-safe, speech-group-held-out reordering mechanism study.

Requires a GPU/ML environment with: torch, transformers, simalign, numpy.
This study is exploratory until its automatic alignment features are manually
validated. It reports associations and held-out prediction only, never causes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE / "data/experiments/r027_shared_outer_quality/all_lat_segments.json"


def pearson(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def mse(left, right):
    return float(np.mean((np.asarray(left, dtype=float) - np.asarray(right, dtype=float)) ** 2))


def source_length(text, direction):
    return len([char for char in text if not char.isspace()]) if direction == "zh-en" else len(text.split())


def target_length(text, direction):
    return len(text.split()) if direction == "zh-en" else len([char for char in text if not char.isspace()])


def alignment_metrics(pairs, source_tokens, target_tokens):
    """Compute alignment-based reordering features from word index pairs."""
    pairs = sorted({(int(source), int(target)) for source, target in pairs})
    n_source, n_target = max(1, len(source_tokens)), max(1, len(target_tokens))
    crossings = 0
    comparable = 0
    for left, (source_a, target_a) in enumerate(pairs):
        for source_b, target_b in pairs[left + 1 :]:
            if source_a != source_b and target_a != target_b:
                comparable += 1
                crossings += int((source_a - source_b) * (target_a - target_b) < 0)
    displacement = [abs(source / max(1, n_source - 1) - target / max(1, n_target - 1)) for source, target in pairs]
    aligned_source = {source for source, _ in pairs}
    return {
        "n_source_tokens": n_source,
        "n_target_tokens": n_target,
        "target_source_length_ratio": n_target / n_source,
        "n_alignment_pairs": len(pairs),
        "reordering_crossing_rate": crossings / comparable if comparable else 0.0,
        "aligned_displacement": float(np.mean(displacement)) if displacement else 1.0,
        "unaligned_source_rate": 1.0 - len(aligned_source) / n_source,
    }


def one_hot(values, vocabulary):
    return np.asarray([[1.0 if value == item else 0.0 for item in vocabulary[:-1]] for value in values], dtype=float)


def logo_ridge(rows, target, numeric_features, categorical_features):
    """Fixed-alpha Ridge with held-out speech groups; unseen categories are zero."""
    groups = sorted({row["speech_group"] for row in rows})
    predictions = np.full(len(rows), np.nan)
    y = np.asarray([row[target] for row in rows], dtype=float)
    for held_out in groups:
        train_indices = [index for index, row in enumerate(rows) if row["speech_group"] != held_out]
        test_indices = [index for index, row in enumerate(rows) if row["speech_group"] == held_out]
        blocks_train = [np.asarray([[rows[index][feature] for feature in numeric_features] for index in train_indices], dtype=float)]
        blocks_test = [np.asarray([[rows[index][feature] for feature in numeric_features] for index in test_indices], dtype=float)]
        for feature in categorical_features:
            vocabulary = sorted({rows[index][feature] for index in train_indices})
            blocks_train.append(one_hot([rows[index][feature] for index in train_indices], vocabulary))
            blocks_test.append(one_hot([rows[index][feature] for index in test_indices], vocabulary))
        x_train, x_test = np.hstack(blocks_train), np.hstack(blocks_test)
        mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
        scale[scale == 0] = 1.0
        x_train, x_test = (x_train - mean) / scale, (x_test - mean) / scale
        x_train = np.column_stack([np.ones(len(x_train)), x_train])
        x_test = np.column_stack([np.ones(len(x_test)), x_test])
        penalty = np.eye(x_train.shape[1])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y[train_indices])
        predictions[test_indices] = x_test @ beta
    return {"pearson": pearson(y, predictions), "mse": mse(y, predictions), "predictions": predictions.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aligner-model", default="bert")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    try:
        from simalign import SentenceAligner
    except ImportError as error:
        raise SystemExit("Missing simalign. Install with: pip install simalign") from error

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.max_samples:
        rows = rows[: args.max_samples]
    aligner = SentenceAligner(model=args.aligner_model, token_type="bpe", matching_methods="mai")
    features = []
    failures = []
    for index, row in enumerate(rows, 1):
        try:
            result = aligner.get_word_aligns(row["src"], row["mt"])
            pairs = result["itermax"]
            # SimAlign's word indices are based on its own word tokenization;
            # use whitespace/character counts only as language-aware length controls.
            metrics = alignment_metrics(pairs, row["src"].split() or list(row["src"]), row["mt"].split() or list(row["mt"]))
            features.append(
                {
                    "segment_id": row["segment_id"],
                    "speech_group": row["speech_group"],
                    "interpreter": row["interpreter"],
                    "direction": row["direction"],
                    "delay_seconds": float(row["delay_seconds"]),
                    "LAT": float(row["perceived_latency"]),
                    "LQ": float(row["LQ"]),
                    "EXP": float(row["EXP"]),
                    "source_length": source_length(row["src"], row["direction"]),
                    **metrics,
                }
            )
        except Exception as error:  # Keep failures explicit rather than silently dropping them.
            failures.append({"segment_id": row["segment_id"], "error": repr(error)})
        if index % 25 == 0:
            print(f"aligned {index}/{len(rows)}; usable={len(features)}; failures={len(failures)}", flush=True)

    if len(features) < 50:
        raise SystemExit(f"Only {len(features)} samples aligned; inspect failures before modeling.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control = ["source_length", "target_source_length_ratio", "delay_seconds"]
    reorder = ["reordering_crossing_rate", "aligned_displacement", "unaligned_source_rate"]
    runs = {
        "delay_control": logo_ridge(features, "delay_seconds", ["source_length", "target_source_length_ratio"], ["direction", "interpreter"]),
        "delay_plus_reordering": logo_ridge(features, "delay_seconds", ["source_length", "target_source_length_ratio", *reorder], ["direction", "interpreter"]),
        "LAT_control": logo_ridge(features, "LAT", control + ["LQ", "EXP"], ["direction", "interpreter"]),
        "LAT_plus_reordering": logo_ridge(features, "LAT", control + ["LQ", "EXP", *reorder], ["direction", "interpreter"]),
    }
    for result in runs.values():
        del result["predictions"]
    with (args.output_dir / "reordering_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    with (args.output_dir / "alignment_failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2)
    summary = {
        "input": str(args.input),
        "n_input": len(rows),
        "n_aligned": len(features),
        "n_failures": len(failures),
        "aligner": {"package": "simalign", "model": args.aligner_model, "matching_method": "itermax"},
        "claim_scope": "exploratory alignment-based association; not a causal reordering claim",
        "runs": runs,
    }
    (args.output_dir / "reordering_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
