#!/usr/bin/env python3
"""Summarize three-seed leave-one-interpreter-out quality-to-LAT evaluation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
AUTO_MODELS = ("auto_pred_LQ_EXP", "auto_pred_LQ_EXP_piecewise_delay")
ALL_MODELS = ("delay_piecewise", *AUTO_MODELS)
DELAY_MODEL = "delay_piecewise"
BOOTSTRAPS = 5000
RNG_SEED = 20260722


def as_base_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else BASE / value


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE))
    except ValueError:
        return str(path)


def pearson(y: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float(np.corrcoef(y, prediction)[0, 1])


def mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((y - prediction) ** 2))


def quantiles(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray(values, dtype=float)
    return [round(float(np.quantile(array, 0.025)), 4), round(float(np.quantile(array, 0.975)), 4)]


def metric_sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def read_result_dir(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    result_path = path / "loio_lat_results.json"
    oof_path = path / "loio_lat_oof_predictions.csv"
    if not result_path.exists() or not oof_path.exists():
        raise FileNotFoundError(f"Missing LOIO LAT outputs in {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows: dict[str, list[dict]] = defaultdict(list)
    with oof_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["model"]].append(row)
    if set(rows) != set(ALL_MODELS):
        raise ValueError(f"Unexpected model set in {path}: {sorted(rows)}")
    return result, rows


def seed_from_dir(path: Path) -> str:
    for part in reversed(path.parts):
        for token in part.split("_"):
            if token.isdigit() and len(token) == 8:
                return token
    return path.name


def paired_arrays(rows_a: list[dict], rows_b: list[dict]):
    by_a = {row["segment_id"]: row for row in rows_a}
    by_b = {row["segment_id"]: row for row in rows_b}
    if set(by_a) != set(by_b):
        raise ValueError("Model rows are not paired by segment_id")
    ids = sorted(by_a)
    interpreters = np.asarray([by_a[sid]["interpreter"] for sid in ids])
    y = np.asarray([float(by_a[sid]["LAT"]) for sid in ids])
    pred_a = np.asarray([float(by_a[sid]["prediction"]) for sid in ids])
    pred_b = np.asarray([float(by_b[sid]["prediction"]) for sid in ids])
    return ids, interpreters, y, pred_a, pred_b


def cluster_bootstrap_delta(rows_a: list[dict], rows_b: list[dict]) -> dict:
    _, interpreters, y, pred_a, pred_b = paired_arrays(rows_a, rows_b)
    unique = np.unique(interpreters)
    rng = np.random.default_rng(RNG_SEED)
    delta_r: list[float] = []
    delta_mse: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(interpreters == interpreter) for interpreter in sampled])
        r_a = pearson(y[idx], pred_a[idx])
        r_b = pearson(y[idx], pred_b[idx])
        if r_a is not None and r_b is not None:
            delta_r.append(r_a - r_b)
        delta_mse.append(mse(y[idx], pred_a[idx]) - mse(y[idx], pred_b[idx]))
    r_a = pearson(y, pred_a)
    r_b = pearson(y, pred_b)
    return {
        "definition": "model_a minus delay-only; positive delta_pearson and negative delta_mse favor model_a",
        "delta_pearson": round((r_a or 0.0) - (r_b or 0.0), 4),
        "delta_pearson_ci95": quantiles(delta_r),
        "delta_mse": round(mse(y, pred_a) - mse(y, pred_b), 4),
        "delta_mse_ci95": quantiles(delta_mse),
    }


def exact_interpreter_swap(rows_a: list[dict], rows_b: list[dict]) -> dict:
    _, interpreters, y, pred_a, pred_b = paired_arrays(rows_a, rows_b)
    groups = sorted(np.unique(interpreters))
    group_indices = [np.flatnonzero(interpreters == group) for group in groups]
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
        "method": "exact paired held-out-interpreter prediction swap",
        "num_permutations": len(null),
        "two_sided_p": round(float(p_value), 6),
    }


def per_interpreter(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["interpreter"]].append(row)
    values = {}
    for interpreter, items in sorted(grouped.items()):
        y = np.asarray([float(row["LAT"]) for row in items])
        prediction = np.asarray([float(row["prediction"]) for row in items])
        value = pearson(y, prediction)
        values[interpreter] = {
            "n_segments": len(items),
            "pearson": round(value, 4) if value is not None else None,
            "mse": round(mse(y, prediction), 4),
            "pred_std": round(float(np.std(prediction)), 4),
        }
    valid_r = [row["pearson"] for row in values.values() if row["pearson"] is not None]
    return {
        "per_interpreter": values,
        "macro_interpreter_pearson": round(float(statistics.mean(valid_r)), 4) if valid_r else None,
        "macro_interpreter_mse": round(float(statistics.mean(row["mse"] for row in values.values())), 4),
        "n_valid_interpreter_correlations": len(valid_r),
    }


def upstream_quality_metrics(rows: list[dict], gold_by_id: dict[str, dict]) -> dict:
    ids = [row["segment_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(gold_by_id):
        raise ValueError("Upstream LOIO prediction coverage mismatch")
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
        "lq_collapsed_below_0_10": bool(np.std(pred_lq) < 0.10),
        "exp_collapsed_below_0_10": bool(np.std(pred_exp) < 0.10),
    }


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {
        key + "_mean": round(float(statistics.mean(float(row[key]) for row in rows)), 4)
        for key in keys
    } | {
        key + "_sd": round(metric_sd([float(row[key]) for row in rows]), 4)
        for key in keys
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="experiments/aaai_loio_corrected_summary_20260722")
    parser.add_argument("--gold-quality-json", default="data/experiments/aaai_loio_outer_quality_corrected/all_lat_segments.json")
    args = parser.parse_args()

    result_dirs = [as_base_path(value) for value in args.result_dirs]
    if len(result_dirs) != 3:
        raise ValueError(f"Expected exactly three formal seed result directories, found {len(result_dirs)}")
    output_dir = as_base_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_rows = json.loads(as_base_path(args.gold_quality_json).read_text(encoding="utf-8"))
    gold_by_id = {str(row["segment_id"]): row for row in gold_rows}

    seed_tables = []
    all_rows_by_seed: dict[str, dict[str, list[dict]]] = {}
    metadata = {}
    for path in result_dirs:
        seed = seed_from_dir(path)
        if seed in all_rows_by_seed:
            raise ValueError(f"Duplicate seed: {seed}")
        result, rows_by_model = read_result_dir(path)
        all_rows_by_seed[seed] = rows_by_model
        metadata[seed] = result.get("metadata", {})
        for model, block in result["models"].items():
            seed_tables.append({"model": model, "seed": seed, **block["metrics"]})

    summary: dict = {
        "protocol": "automatic leave-one-interpreter-out two-stage evaluation with inner speech-group OOF quality features",
        "scope": "unseen interpreter; source speeches may appear via other interpreters",
        "seeds": sorted(all_rows_by_seed),
        "metadata_by_seed": metadata,
        "upstream_quality": {},
        "models": {},
        "paired_against_delay_piecewise": {},
    }
    metric_keys = ("pearson", "spearman", "mse", "mae", "r2", "pred_std")
    for model in ALL_MODELS:
        rows = [row for row in seed_tables if row["model"] == model]
        summary["models"][model] = {
            "runs": rows,
            "aggregate": aggregate(rows, metric_keys),
            "interpreter_robustness": [
                {"seed": seed, **per_interpreter(all_rows_by_seed[seed][model])}
                for seed in sorted(all_rows_by_seed)
            ],
        }

    quality_runs = [
        {"seed": seed, **upstream_quality_metrics(rows_by_model["auto_pred_LQ_EXP"], gold_by_id)}
        for seed, rows_by_model in sorted(all_rows_by_seed.items())
    ]
    quality_fields = ("lq_pearson", "exp_pearson", "lq_mse", "exp_mse", "pred_lq_std", "pred_exp_std", "pred_head_pearson")
    summary["upstream_quality"] = {
        "prediction_scope": "concatenated final outer-interpreter predictions used on each held-out interpreter",
        "runs": quality_runs,
        "aggregate": aggregate(quality_runs, quality_fields),
    }

    for model in AUTO_MODELS:
        key = f"{model}_vs_{DELAY_MODEL}"
        summary["paired_against_delay_piecewise"][key] = []
        for seed, rows_by_model in sorted(all_rows_by_seed.items()):
            summary["paired_against_delay_piecewise"][key].append({
                "seed": seed,
                "cluster_bootstrap": cluster_bootstrap_delta(rows_by_model[model], rows_by_model[DELAY_MODEL]),
                "exact_interpreter_swap": exact_interpreter_swap(rows_by_model[model], rows_by_model[DELAY_MODEL]),
            })

    summary_path = output_dir / "aaai_loio_corrected_seed_summary.json"
    csv_path = output_dir / "aaai_loio_corrected_main_results.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["model", "seed", "n_segments", "pearson", "pearson_ci95", "spearman", "spearman_ci95", "mse", "mae", "r2", "pred_std"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_tables)
    print(json.dumps({"summary": display_path(summary_path), "csv": display_path(csv_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
