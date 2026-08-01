#!/usr/bin/env python3
"""Aggregate direct-LAT outer-fold predictions and compare official systems."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


OFFICIAL_MODELS = (
    "delay_piecewise",
    "auto_pred_LQ_EXP",
    "auto_pred_LQ_EXP_piecewise_delay",
)


def corr(y, prediction, method="pearson"):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(y, prediction).statistic)


def metrics(rows):
    y = np.asarray([row["LAT"] for row in rows], dtype=float)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=float)
    return {
        "n": len(rows),
        "pearson": corr(y, prediction),
        "spearman": corr(y, prediction, "spearman"),
        "mse": float(np.mean((y - prediction) ** 2)),
        "mae": float(np.mean(np.abs(y - prediction))),
        "pred_std": float(np.std(prediction)),
    }


def round_tree(value):
    if isinstance(value, dict):
        return {key: round_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_tree(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def load_direct(root, variant):
    paths = sorted((root / variant).glob("outer_*/predictions.json"))
    if len(paths) != 16:
        raise ValueError(f"{root}/{variant}: expected 16 folds, found {len(paths)}")
    rows = []
    seen = set()
    for path in paths:
        for row in json.loads(path.read_text(encoding="utf-8")):
            segment_id = row["segment_id"]
            if segment_id in seen:
                raise ValueError(f"Duplicate direct prediction: {segment_id}")
            seen.add(segment_id)
            item = dict(row)
            item["LAT"] = float(item["LAT"])
            item["prediction"] = float(item["prediction"])
            item["model"] = "direct_text_delay" if variant == "text_delay" else "direct_text"
            rows.append(item)
    if len(rows) != 622:
        raise ValueError(f"{root}/{variant}: expected 622 predictions, found {len(rows)}")
    return rows


def load_official(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["model"] not in OFFICIAL_MODELS:
                continue
            rows.append({
                "model": raw["model"],
                "segment_id": raw["segment_id"],
                "speech_group": raw["speech_group"],
                "interpreter": raw["interpreter"],
                "LAT": float(raw["LAT"]),
                "prediction": float(raw["prediction"]),
            })
    return rows


def macro_group(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["speech_group"])].append(row)
    values = [metrics(items)["pearson"] for items in grouped.values()]
    values = [value for value in values if value is not None]
    return {
        "n_valid_groups": len(values),
        "macro_pearson": float(np.mean(values)),
        "median_group_pearson": float(np.median(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    parser.add_argument("--direct-root", required=True)
    parser.add_argument("--official-oof", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    direct_root = Path(args.direct_root)
    all_rows = load_direct(direct_root, "text") + load_direct(direct_root, "text_delay")
    all_rows += load_official(Path(args.official_oof))
    models = sorted({row["model"] for row in all_rows})
    summary = {}
    for model in models:
        rows = [row for row in all_rows if row["model"] == model]
        summary[model] = {
            "metrics": metrics(rows),
            "speech_group_macro": macro_group(rows),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = round_tree({"seed": args.seed, "models": summary})
    (output_dir / "direct_lat_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "direct_lat_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "n", "pearson", "spearman", "mse", "mae", "pred_std", "macro_group_pearson"])
        for model in models:
            item = payload["models"][model]
            writer.writerow([
                model,
                item["metrics"]["n"],
                item["metrics"]["pearson"],
                item["metrics"]["spearman"],
                item["metrics"]["mse"],
                item["metrics"]["mae"],
                item["metrics"]["pred_std"],
                item["speech_group_macro"]["macro_pearson"],
            ])
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
