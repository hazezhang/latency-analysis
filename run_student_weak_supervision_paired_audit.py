#!/usr/bin/env python3
"""Paired source-speech-group audit for strict S0/S3/S4a predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = ("S0", "S3", "S4a")
TARGETS = ("LQ", "EXP")
COMPARISONS = (("S3", "S0"), ("S4a", "S0"), ("S4a", "S3"))


def pearson(gold: np.ndarray, prediction: np.ndarray) -> float:
    if len(gold) < 3 or np.std(gold) == 0 or np.std(prediction) == 0:
        return float("nan")
    return float(np.corrcoef(gold, prediction)[0, 1])


def metrics(gold: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "pearson": pearson(gold, prediction),
        "mae": float(np.mean(np.abs(prediction - gold))),
        "prediction_sd": float(np.std(prediction)),
    }


def interval(values: list[float]) -> list[float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def load_method(path: Path) -> dict[str, dict]:
    files = sorted(path.glob("outer_*/predictions_outer_test.json"))
    if len(files) != 16:
        raise ValueError(f"{path}: expected 16 outer prediction files, found {len(files)}")
    rows: dict[str, dict] = {}
    for prediction_file in files:
        fold = prediction_file.parent.name
        payload = json.loads(prediction_file.read_text(encoding="utf-8"))
        for row in payload:
            segment_id = str(row["segment_id"])
            if segment_id in rows:
                raise ValueError(f"{path}: duplicate segment_id {segment_id}")
            rows[segment_id] = {**row, "outer_fold": fold}
    if len(rows) != 622:
        raise ValueError(f"{path}: expected 622 unique segments, found {len(rows)}")
    return rows


def load_metadata(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["segment_id"]): row for row in payload}
    if len(rows) != len(payload):
        raise ValueError(f"{path}: duplicate segment IDs")
    return rows


def build_arrays(method_roots: dict[str, Path], metadata_path: Path) -> dict:
    method_rows = {method: load_method(path) for method, path in method_roots.items()}
    metadata = load_metadata(metadata_path)
    ids = sorted(method_rows["S0"])
    if set(ids) != set(metadata):
        raise ValueError("S0 IDs do not match strict professional metadata")
    reference = method_rows["S0"]
    for method, rows in method_rows.items():
        if sorted(rows) != ids:
            raise ValueError(f"{method}: segment IDs do not match S0")
        for segment_id in ids:
            for target in TARGETS:
                if float(rows[segment_id][f"human_{target}"]) != float(reference[segment_id][f"human_{target}"]):
                    raise ValueError(f"{method}/{segment_id}: {target} gold mismatch")
            if rows[segment_id]["src"] != reference[segment_id]["src"] or rows[segment_id]["mt"] != reference[segment_id]["mt"]:
                raise ValueError(f"{method}/{segment_id}: text mismatch")
            if rows[segment_id]["outer_fold"] != reference[segment_id]["outer_fold"]:
                raise ValueError(f"{method}/{segment_id}: outer fold mismatch")

    groups = np.asarray([str(metadata[segment_id]["speech_group"]) for segment_id in ids])
    directions = np.asarray([str(metadata[segment_id]["direction"]) for segment_id in ids])
    if len(np.unique(groups)) != 16:
        raise ValueError(f"Expected 16 source-speech groups, found {len(np.unique(groups))}")
    for index, segment_id in enumerate(ids):
        fold_group = reference[segment_id]["outer_fold"].split("_speech_", 1)[1]
        if fold_group != groups[index]:
            raise ValueError(f"{segment_id}: fold/group mismatch ({fold_group} != {groups[index]})")

    gold = {
        target: np.asarray([float(reference[segment_id][f"human_{target}"]) for segment_id in ids])
        for target in TARGETS
    }
    predictions = {
        method: {
            target: np.asarray([float(method_rows[method][segment_id][f"pred_{target}"]) for segment_id in ids])
            for target in TARGETS
        }
        for method in METHODS
    }
    return {
        "ids": ids,
        "groups": groups,
        "directions": directions,
        "gold": gold,
        "predictions": predictions,
    }


def evaluate_subset(indices: np.ndarray, gold: dict, predictions: dict) -> dict:
    return {
        method: {
            target: metrics(gold[target][indices], predictions[method][target][indices])
            for target in TARGETS
        }
        for method in METHODS
    }


def deltas(values: dict) -> dict:
    result = {}
    for candidate, baseline in COMPARISONS:
        name = f"{candidate}_minus_{baseline}"
        result[name] = {}
        for target in TARGETS:
            result[name][target] = {
                metric: values[candidate][target][metric] - values[baseline][target][metric]
                for metric in ("pearson", "mae", "prediction_sd")
            }
    return result


def decision(
    comparison: dict,
    direction_deltas: dict,
    candidate_metrics: dict,
    thresholds: dict,
    n_training_seeds: int,
) -> dict:
    per_target = {}
    for target in TARGETS:
        point = comparison[target]["point"]
        ci = comparison[target]["ci95_source_speech_group"]
        direction_values = [block[target]["pearson"] for block in direction_deltas.values()]
        checks = {
            "minimum_pearson_gain": point["pearson"] >= thresholds["minimum_pearson_gain"],
            "pearson_ci_not_strongly_negative": ci["pearson"][0] >= thresholds["pearson_ci_lower_tolerance"],
            "mae_not_materially_worse": point["mae"] <= thresholds["mae_worsening_tolerance"],
            "prediction_sd_not_collapsed": candidate_metrics[target]["prediction_sd"] >= thresholds["prediction_sd_floor"],
            "no_direction_clear_reversal": all(
                value >= thresholds["direction_pearson_reversal_tolerance"] for value in direction_values if np.isfinite(value)
            ),
        }
        provisional = all(checks.values())
        per_target[target] = {
            "checks": checks,
            "provisional_pass_excluding_seed_stability": provisional,
            "confirmatory_pass": provisional and n_training_seeds >= 3,
        }
    return {
        "seed_stability_status": (
            "not_evaluable: fewer than three matched training seeds"
            if n_training_seeds < 3
            else "requires at least two of three positive seed-level Pearson deltas"
        ),
        "per_target": per_target,
        "candidate_advances_provisionally": any(block["provisional_pass_excluding_seed_stability"] for block in per_target.values()),
        "candidate_advances_confirmatorily": any(block["confirmatory_pass"] for block in per_target.values()),
    }


def write_csv(path: Path, payload: dict) -> None:
    rows = []
    for method, targets in payload["methods"].items():
        for target, block in targets.items():
            rows.append({
                "kind": "method",
                "name": method,
                "target": target,
                "pearson": block["point"]["pearson"],
                "pearson_ci95_low": block["ci95_source_speech_group"]["pearson"][0],
                "pearson_ci95_high": block["ci95_source_speech_group"]["pearson"][1],
                "mae": block["point"]["mae"],
                "mae_ci95_low": block["ci95_source_speech_group"]["mae"][0],
                "mae_ci95_high": block["ci95_source_speech_group"]["mae"][1],
                "prediction_sd": block["point"]["prediction_sd"],
                "prediction_sd_ci95_low": block["ci95_source_speech_group"]["prediction_sd"][0],
                "prediction_sd_ci95_high": block["ci95_source_speech_group"]["prediction_sd"][1],
            })
    for name, targets in payload["comparisons"].items():
        for target, block in targets.items():
            rows.append({
                "kind": "paired_delta",
                "name": name,
                "target": target,
                "pearson": block["point"]["pearson"],
                "pearson_ci95_low": block["ci95_source_speech_group"]["pearson"][0],
                "pearson_ci95_high": block["ci95_source_speech_group"]["pearson"][1],
                "mae": block["point"]["mae"],
                "mae_ci95_low": block["ci95_source_speech_group"]["mae"][0],
                "mae_ci95_high": block["ci95_source_speech_group"]["mae"][1],
                "prediction_sd": block["point"]["prediction_sd"],
                "prediction_sd_ci95_low": block["ci95_source_speech_group"]["prediction_sd"][0],
                "prediction_sd_ci95_high": block["ci95_source_speech_group"]["prediction_sd"][1],
            })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Strict Student Weak-Supervision Paired Audit",
        "",
        f"This audit uses {payload['bootstrap_samples']:,} paired whole-source-speech-group bootstrap draws over "
        f"{payload['n_source_speech_groups']} groups and {payload['n_segments']} professional segments.",
        "",
        "| System | Target | Pearson (95% group CI) | MAE (95% group CI) | Prediction SD |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for method, targets in payload["methods"].items():
        for target, block in targets.items():
            point, ci = block["point"], block["ci95_source_speech_group"]
            lines.append(
                f"| {method} | {target} | {point['pearson']:.4f} [{ci['pearson'][0]:.4f}, {ci['pearson'][1]:.4f}] | "
                f"{point['mae']:.4f} [{ci['mae'][0]:.4f}, {ci['mae'][1]:.4f}] | {point['prediction_sd']:.4f} |"
            )
    lines.extend(["", "| Comparison | Target | Delta Pearson (95% group CI) | Delta MAE (95% group CI) | Provisional gate |", "| --- | --- | ---: | ---: | --- |"])
    for name, targets in payload["comparisons"].items():
        for target, block in targets.items():
            point, ci = block["point"], block["ci95_source_speech_group"]
            gate = payload["decisions"][name]["per_target"][target]["provisional_pass_excluding_seed_stability"]
            lines.append(
                f"| {name.replace('_minus_', ' - ')} | {target} | {point['pearson']:+.4f} [{ci['pearson'][0]:+.4f}, {ci['pearson'][1]:+.4f}] | "
                f"{point['mae']:+.4f} [{ci['mae'][0]:+.4f}, {ci['mae'][1]:+.4f}] | {'PASS' if gate else 'FAIL'} |"
            )
    lines.extend([
        "",
        "Positive Delta Pearson favors the first system; negative Delta MAE favors the first system.",
        "The confirmatory gate remains unavailable until three matched training seeds exist; outer folds are not treated as seeds.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0-root", type=Path, default=Path("experiments/student_weak_supervision_s0_20260805"))
    parser.add_argument("--s3-root", type=Path, default=Path("experiments/student_weak_supervision_s3_strict_20260806"))
    parser.add_argument("--s4a-root", type=Path, default=Path("experiments/student_weak_supervision_s4a_strict_20260806"))
    parser.add_argument("--metadata", type=Path, default=Path("data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/student_weak_supervision_paired_audit_20260810"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=20260810)
    parser.add_argument("--n-training-seeds", type=int, default=1)
    args = parser.parse_args()

    thresholds = {
        "minimum_pearson_gain": 0.01,
        "pearson_ci_lower_tolerance": -0.02,
        "mae_worsening_tolerance": 0.02,
        "prediction_sd_floor": 0.10,
        "direction_pearson_reversal_tolerance": -0.05,
    }
    arrays = build_arrays({"S0": args.s0_root, "S3": args.s3_root, "S4a": args.s4a_root}, args.metadata)
    all_indices = np.arange(len(arrays["ids"]))
    point_metrics = evaluate_subset(all_indices, arrays["gold"], arrays["predictions"])
    point_deltas = deltas(point_metrics)

    unique_groups = sorted(np.unique(arrays["groups"]))
    group_indices = {group: np.flatnonzero(arrays["groups"] == group) for group in unique_groups}
    method_draws = {method: {target: defaultdict(list) for target in TARGETS} for method in METHODS}
    delta_draws = {
        f"{candidate}_minus_{baseline}": {target: defaultdict(list) for target in TARGETS}
        for candidate, baseline in COMPARISONS
    }
    rng = np.random.default_rng(args.rng_seed)
    for _ in range(args.bootstrap_samples):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sampled_metrics = evaluate_subset(sampled_indices, arrays["gold"], arrays["predictions"])
        sampled_deltas = deltas(sampled_metrics)
        for method in METHODS:
            for target in TARGETS:
                for metric, value in sampled_metrics[method][target].items():
                    method_draws[method][target][metric].append(value)
        for name, targets in sampled_deltas.items():
            for target, values in targets.items():
                for metric, value in values.items():
                    delta_draws[name][target][metric].append(value)

    methods = {
        method: {
            target: {
                "point": point_metrics[method][target],
                "ci95_source_speech_group": {
                    metric: interval(method_draws[method][target][metric]) for metric in point_metrics[method][target]
                },
            }
            for target in TARGETS
        }
        for method in METHODS
    }
    comparisons = {
        name: {
            target: {
                "point": point_deltas[name][target],
                "ci95_source_speech_group": {
                    metric: interval(delta_draws[name][target][metric]) for metric in point_deltas[name][target]
                },
            }
            for target in TARGETS
        }
        for name in point_deltas
    }
    direction_metrics = {}
    direction_deltas = {}
    for direction in sorted(np.unique(arrays["directions"])):
        indices = np.flatnonzero(arrays["directions"] == direction)
        direction_metrics[direction] = evaluate_subset(indices, arrays["gold"], arrays["predictions"])
        direction_deltas[direction] = deltas(direction_metrics[direction])

    decisions = {}
    for candidate, baseline in COMPARISONS:
        name = f"{candidate}_minus_{baseline}"
        decisions[name] = decision(
            comparisons[name],
            {direction: values[name] for direction, values in direction_deltas.items()},
            point_metrics[candidate],
            thresholds,
            args.n_training_seeds,
        )

    payload = {
        "protocol": "Paired bootstrap of all 16 source-speech groups with replacement; every selected group retains all observed professional outer-test segments.",
        "n_segments": len(arrays["ids"]),
        "n_source_speech_groups": len(unique_groups),
        "directions": {direction: int(np.sum(arrays["directions"] == direction)) for direction in sorted(np.unique(arrays["directions"]))},
        "bootstrap_samples": args.bootstrap_samples,
        "rng_seed": args.rng_seed,
        "matched_training_seeds": args.n_training_seeds,
        "thresholds": thresholds,
        "methods": methods,
        "comparisons": comparisons,
        "direction_metrics": direction_metrics,
        "direction_deltas": direction_deltas,
        "decisions": decisions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_group_bootstrap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "paired_group_bootstrap.csv", payload)
    write_markdown(args.output_dir / "paired_group_bootstrap.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
