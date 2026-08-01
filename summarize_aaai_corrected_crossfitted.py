#!/usr/bin/env python3
"""Summarize corrected AAAI cross-fitted LAT bridge outputs.

Inputs are per-seed directories produced by run_latency_aaai_crossfitted_bridge.py.
The script writes paper-ready aggregate tables and cluster-aware paired checks.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
AUTO_MODELS = ("auto_pred_LQ_EXP", "auto_pred_LQ_EXP_piecewise_delay")
REFERENCE_MODELS = ("delay_piecewise", "human_LQ_EXP")
DELAY_MODEL = "delay_piecewise"
BOOTSTRAPS = 5000
RNG_SEED = 20260721


def as_base_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else BASE / value


def pearson(y: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float(np.corrcoef(y, prediction)[0, 1])


def mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((y - prediction) ** 2))


def metric_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def read_result_dir(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    result_path = path / "crossfitted_lat_results.json"
    oof_path = path / "crossfitted_lat_oof_predictions.csv"
    if not result_path.exists() or not oof_path.exists():
        raise FileNotFoundError(f"Missing LAT outputs in {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows: dict[str, list[dict]] = defaultdict(list)
    with oof_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["model"]].append(row)
    return result, rows


def seed_from_dir(path: Path) -> str:
    for part in reversed(path.parts):
        for token in part.split("_"):
            if token.isdigit() and len(token) == 8:
                return token
    return path.name


def values_for(rows: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    ids = [row["segment_id"] for row in rows]
    groups = np.asarray([row["speech_group"] for row in rows])
    y = np.asarray([float(row["LAT"]) for row in rows])
    pred = np.asarray([float(row["prediction"]) for row in rows])
    return ids, groups, y, pred


def cluster_bootstrap_delta(rows_a: list[dict], rows_b: list[dict]) -> dict:
    by_a = {row["segment_id"]: row for row in rows_a}
    by_b = {row["segment_id"]: row for row in rows_b}
    if set(by_a) != set(by_b):
        raise ValueError("Model rows are not paired by segment_id")
    ids = sorted(by_a)
    groups = np.asarray([by_a[sid]["speech_group"] for sid in ids])
    y = np.asarray([float(by_a[sid]["LAT"]) for sid in ids])
    pred_a = np.asarray([float(by_a[sid]["prediction"]) for sid in ids])
    pred_b = np.asarray([float(by_b[sid]["prediction"]) for sid in ids])
    unique = np.unique(groups)
    rng = np.random.default_rng(RNG_SEED)
    delta_r: list[float] = []
    delta_mse: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        r_a = pearson(y[idx], pred_a[idx])
        r_b = pearson(y[idx], pred_b[idx])
        if r_a is not None and r_b is not None:
            delta_r.append(r_a - r_b)
        delta_mse.append(mse(y[idx], pred_a[idx]) - mse(y[idx], pred_b[idx]))
    return {
        "definition": "model_a minus model_b; positive delta_pearson and negative delta_mse favor model_a",
        "delta_pearson": round((pearson(y, pred_a) or 0.0) - (pearson(y, pred_b) or 0.0), 4),
        "delta_pearson_ci95": quantiles(delta_r),
        "delta_mse": round(mse(y, pred_a) - mse(y, pred_b), 4),
        "delta_mse_ci95": quantiles(delta_mse),
    }


def exact_group_swap(rows_a: list[dict], rows_b: list[dict]) -> dict:
    by_a = {row["segment_id"]: row for row in rows_a}
    by_b = {row["segment_id"]: row for row in rows_b}
    ids = sorted(by_a)
    groups = sorted({by_a[sid]["speech_group"] for sid in ids})
    y = np.asarray([float(by_a[sid]["LAT"]) for sid in ids])
    pred_a = np.asarray([float(by_a[sid]["prediction"]) for sid in ids])
    pred_b = np.asarray([float(by_b[sid]["prediction"]) for sid in ids])
    group_indices = [np.asarray([i for i, sid in enumerate(ids) if by_a[sid]["speech_group"] == group]) for group in groups]
    observed = (pearson(y, pred_a) or 0.0) - (pearson(y, pred_b) or 0.0)
    null = []
    for swaps in itertools.product((False, True), repeat=len(groups)):
        left, right = pred_a.copy(), pred_b.copy()
        for swap, idx in zip(swaps, group_indices):
            if swap:
                left[idx], right[idx] = right[idx].copy(), left[idx].copy()
        null.append((pearson(y, left) or 0.0) - (pearson(y, right) or 0.0))
    p_value = sum(abs(value) >= abs(observed) for value in null) / len(null)
    return {
        "method": "exact paired speech-group prediction swap",
        "num_permutations": len(null),
        "two_sided_p": round(float(p_value), 6),
    }


def quantiles(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    arr = np.asarray(values, dtype=float)
    return [round(float(np.quantile(arr, 0.025)), 4), round(float(np.quantile(arr, 0.975)), 4)]


def macro_speech(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["speech_group"]].append(row)
    per_group = {}
    for group, items in sorted(grouped.items()):
        _, _, y, pred = values_for(items)
        value = pearson(y, pred)
        per_group[group] = round(value, 4) if value is not None else None
    valid = [value for value in per_group.values() if value is not None]
    return {
        "n_groups_total": len(grouped),
        "n_groups_valid": len(valid),
        "macro_pearson": round(float(statistics.mean(valid)), 4) if valid else None,
        "per_group_pearson": per_group,
    }


def largest_group_sensitivity(rows: list[dict]) -> dict:
    counts = Counter(row["speech_group"] for row in rows)
    group, n = counts.most_common(1)[0]
    kept = [row for row in rows if row["speech_group"] != group]
    _, _, y_all, pred_all = values_for(rows)
    _, _, y_kept, pred_kept = values_for(kept)
    return {
        "excluded_group": group,
        "excluded_n": n,
        "pearson_all": round(pearson(y_all, pred_all) or 0.0, 4),
        "pearson_without_largest_group": round(pearson(y_kept, pred_kept) or 0.0, 4),
    }


def upstream_quality_metrics(rows: list[dict], gold_by_id: dict[str, dict]) -> dict:
    ids = [row["segment_id"] for row in rows]
    missing = sorted(set(ids) - set(gold_by_id))
    if missing:
        raise ValueError(f"Missing gold quality labels for {len(missing)} segments")
    gold_lq = np.asarray([float(gold_by_id[sid]["LQ"]) for sid in ids])
    gold_exp = np.asarray([float(gold_by_id[sid]["EXP"]) for sid in ids])
    pred_lq = np.asarray([float(row["pred_LQ"]) for row in rows])
    pred_exp = np.asarray([float(row["pred_EXP"]) for row in rows])
    return {
        "n_segments": len(rows),
        "lq_pearson": round(pearson(gold_lq, pred_lq) or 0.0, 4),
        "exp_pearson": round(pearson(gold_exp, pred_exp) or 0.0, 4),
        "lq_mse": round(mse(gold_lq, pred_lq), 4),
        "exp_mse": round(mse(gold_exp, pred_exp), 4),
        "pred_lq_std": round(float(np.std(pred_lq)), 4),
        "pred_exp_std": round(float(np.std(pred_exp)), 4),
        "pred_head_pearson": round(pearson(pred_lq, pred_exp) or 0.0, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="experiments/aaai_crossfitted_corrected_summary_20260721")
    parser.add_argument(
        "--gold-quality-json",
        default="data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json",
    )
    args = parser.parse_args()

    result_dirs = [as_base_path(value) for value in args.result_dirs]
    output_dir = as_base_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_rows = json.loads(as_base_path(args.gold_quality_json).read_text(encoding="utf-8"))
    gold_by_id = {row["segment_id"]: row for row in gold_rows}

    seed_tables = []
    all_rows_by_seed: dict[str, dict[str, list[dict]]] = {}
    metadata = {}
    for path in result_dirs:
        seed = seed_from_dir(path)
        result, rows_by_model = read_result_dir(path)
        all_rows_by_seed[seed] = rows_by_model
        metadata[seed] = result.get("metadata", {})
        for model, block in result["models"].items():
            metrics = block["metrics"]
            seed_tables.append({"model": model, "seed": seed, **metrics})

    summary: dict = {
        "protocol": "corrected cross-fitted outer-speech-held-out two-stage evaluation",
        "seeds": sorted(all_rows_by_seed),
        "metadata_by_seed": metadata,
        "upstream_quality": {},
        "models": {},
        "paired_against_delay_piecewise": {},
    }

    for model in REFERENCE_MODELS:
        rows = [row for row in seed_tables if row["model"] == model]
        if rows:
            summary["models"][model] = rows[0]

    for model in AUTO_MODELS:
        rows = [row for row in seed_tables if row["model"] == model]
        summary["models"][model] = {
            "runs": rows,
            "aggregate": {
                key + "_mean": round(float(statistics.mean(float(row[key]) for row in rows)), 4)
                for key in ("pearson", "spearman", "mse", "mae", "r2", "pred_std")
            }
            | {
                key + "_sd": round(metric_sd([float(row[key]) for row in rows]), 4)
                for key in ("pearson", "spearman", "mse", "mae", "r2", "pred_std")
            },
        }

    quality_runs = []
    for seed, rows_by_model in sorted(all_rows_by_seed.items()):
        quality_runs.append({
            "seed": seed,
            **upstream_quality_metrics(rows_by_model["auto_pred_LQ_EXP"], gold_by_id),
        })
    quality_fields = (
        "lq_pearson",
        "exp_pearson",
        "lq_mse",
        "exp_mse",
        "pred_lq_std",
        "pred_exp_std",
        "pred_head_pearson",
    )
    summary["upstream_quality"] = {
        "prediction_scope": "concatenated outer-speech-held-out predictions used by the LAT bridge",
        "runs": quality_runs,
        "aggregate": {
            key + "_mean": round(float(statistics.mean(float(row[key]) for row in quality_runs)), 4)
            for key in quality_fields
        }
        | {
            key + "_sd": round(metric_sd([float(row[key]) for row in quality_runs]), 4)
            for key in quality_fields
        },
    }

    paired = {}
    for seed, rows_by_model in all_rows_by_seed.items():
        delay_rows = rows_by_model[DELAY_MODEL]
        for model in AUTO_MODELS:
            key = f"{model}_vs_{DELAY_MODEL}"
            paired.setdefault(key, []).append({
                "seed": seed,
                "cluster_bootstrap": cluster_bootstrap_delta(rows_by_model[model], delay_rows),
                "exact_group_swap": exact_group_swap(rows_by_model[model], delay_rows),
                "macro_speech": macro_speech(rows_by_model[model]),
                "largest_group_sensitivity": largest_group_sensitivity(rows_by_model[model]),
            })
    summary["paired_against_delay_piecewise"] = paired

    summary_path = output_dir / "aaai_crossfitted_corrected_seed_summary.json"
    csv_path = output_dir / "aaai_crossfitted_corrected_main_results.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["model", "seed", "n_segments", "pearson", "pearson_ci95", "spearman", "spearman_ci95", "mse", "mae", "r2", "pred_std"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_tables)
    print(json.dumps({"summary": str(summary_path.relative_to(BASE)), "csv": str(csv_path.relative_to(BASE))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
