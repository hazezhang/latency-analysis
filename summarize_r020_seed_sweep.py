#!/usr/bin/env python3
"""Summarize R020 automatic-bridge metrics across independently seeded runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


BASE = Path(__file__).parent
MODEL = "R020_pred_LQ_EXP"


def read_model_row(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == MODEL:
                return row
    raise ValueError(f"{MODEL} not found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--output-dir", default="experiments/r020_seed_summary_20260713")
    args = parser.parse_args()

    records = []
    for table in args.tables:
        path = BASE / table
        row = read_model_row(path)
        records.append({
            "table": str(path.relative_to(BASE)),
            "pearson": float(row["pearson"]),
            "spearman": float(row["spearman"]),
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
            "r2": float(row["r2"]),
        })

    summary = {"n_runs": len(records), "model": MODEL, "runs": records, "aggregate": {}}
    for metric in ("pearson", "spearman", "mse", "mae", "r2"):
        values = [record[metric] for record in records]
        summary["aggregate"][metric] = {
            "mean": round(mean(values), 4),
            "std": round(stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    out_dir = BASE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r020_seed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "r020_seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
