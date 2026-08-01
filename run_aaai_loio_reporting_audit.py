#!/usr/bin/env python3
"""Produce anonymous per-interpreter and direction LOIO reporting tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).resolve().parent
SEEDS = (20260718, 20260719, 20260720)
MODELS = ("auto_pred_LQ_EXP", "auto_pred_LQ_EXP_piecewise_delay")


def correlation(y: np.ndarray, prediction: np.ndarray, kind: str) -> float | None:
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float((pearsonr if kind == "pearson" else spearmanr)(y, prediction)[0])


def metrics(rows: list[dict]) -> dict[str, float | int | None]:
    y = np.asarray([float(row["LAT"]) for row in rows])
    prediction = np.asarray([float(row["prediction"]) for row in rows])
    if np.std(prediction) == 0:
        intercept, slope = float(y.mean()), None
    else:
        slope, intercept = np.polyfit(prediction, y, deg=1)
    return {
        "n_segments": len(rows),
        "pearson": correlation(y, prediction, "pearson"),
        "spearman": correlation(y, prediction, "spearman"),
        "mse": float(np.mean((prediction - y) ** 2)),
        "mae": float(np.mean(np.abs(prediction - y))),
        "calibration_bias": float(np.mean(prediction - y)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope) if slope is not None else None,
    }


def aggregate(per_seed: list[dict], keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        values = [row[key] for row in per_seed if row[key] is not None]
        result[f"{key}_mean"] = float(np.mean(values)) if values else None
        result[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return result


def main() -> int:
    out_dir = BASE / "experiments/aaai_loio_reporting_audit_20260726_r2"
    out_dir.mkdir(parents=True, exist_ok=False)
    cohort = {
        str(row["segment_id"]): row
        for row in json.loads((BASE / "data/experiments/aaai_loio_outer_quality_corrected/all_lat_segments.json").read_text(encoding="utf-8"))
    }
    interpreter_alias = {
        value: f"interpreter_{index:02d}"
        for index, value in enumerate(sorted({str(row["interpreter"]) for row in cohort.values()}), start=1)
    }
    per_group: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    raw = []
    for seed in SEEDS:
        path = BASE / f"experiments/aaai_loio_corrected_lat_seed_{seed}/loio_lat_oof_predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["model"] not in MODELS:
                    continue
                row["direction"] = str(cohort[str(row["segment_id"])]["direction"])
                row["seed"] = str(seed)
                raw.append(row)
                per_group[(row["model"], "interpreter", interpreter_alias[row["interpreter"]])].append(row)
                per_group[(row["model"], "direction", row["direction"])].append(row)

    metric_keys = ("pearson", "spearman", "mse", "mae", "calibration_bias", "calibration_intercept", "calibration_slope")
    tables: dict[str, list[dict]] = {"interpreter": [], "direction": []}
    for (model, group_type, group), rows in sorted(per_group.items()):
        by_seed: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)
        seed_values = [{"seed": seed, **metrics(items)} for seed, items in sorted(by_seed.items())]
        directions = sorted({row["direction"] for row in rows})
        tables[group_type].append({
            "model": model,
            group_type: group,
            "directions": "+".join(directions),
            "n_segments": seed_values[0]["n_segments"],
            **aggregate(seed_values, metric_keys),
        })

    for group_type, rows in tables.items():
        output = out_dir / f"loio_{group_type}_metrics_three_seed.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    macro = []
    for model in MODELS:
        for seed in SEEDS:
            entries = []
            for row in tables["interpreter"]:
                if row["model"] != model:
                    continue
                subset = [
                    item for item in per_group[(model, "interpreter", row["interpreter"])]
                    if item["seed"] == str(seed)
                ]
                entries.append(metrics(subset))
            macro.append({"model": model, "seed": seed, **{f"macro_{key}": float(np.mean([entry[key] for entry in entries if entry[key] is not None])) for key in metric_keys}})
    summary = []
    for model in MODELS:
        selected = [row for row in macro if row["model"] == model]
        row = {"model": model, "n_interpreters": 7}
        for key in metric_keys:
            values = [entry[f"macro_{key}"] for entry in selected]
            row[f"macro_{key}_mean"] = float(np.mean(values))
            row[f"macro_{key}_sd"] = float(np.std(values, ddof=1))
        summary.append(row)
    with (out_dir / "loio_macro_interpreter_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    payload = {
        "metadata": {
            "protocol": "leave-one-interpreter-out; source speeches may recur through other interpreters",
            "seeds": SEEDS,
            "models": MODELS,
            "calibration": "OLS gold = intercept + slope * prediction",
            "aggregation": "each listed metric is computed per seed within a group, then mean and sample SD are taken across seeds",
        },
        "per_interpreter": tables["interpreter"],
        "per_direction": tables["direction"],
        "macro_interpreter": summary,
    }
    (out_dir / "loio_reporting_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
