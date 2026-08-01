#!/usr/bin/env python3
"""Aggregate frozen R027 LAT tables across upstream quality-model seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


BASE = Path(__file__).parent
METRICS = ("pearson", "spearman", "mse", "mae", "r2", "pred_std")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--output-dir", default="experiments/r027_seed_summary_20260718")
    args = parser.parse_args()
    models: dict[str, list[dict]] = {}
    for table_name in args.tables:
        table = BASE / table_name
        with table.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                record = {"table": str(table.relative_to(BASE)), **{metric: float(row[metric]) for metric in METRICS}}
                models.setdefault(row["model"], []).append(record)
    summary = {"n_seeds": len(args.tables), "models": {}}
    for model, records in sorted(models.items()):
        aggregate = {}
        for metric in METRICS:
            values = [record[metric] for record in records]
            aggregate[metric] = {
                "mean": round(mean(values), 4),
                "std": round(stdev(values), 4) if len(values) > 1 else 0.0,
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
        summary["models"][model] = {"runs": records, "aggregate": aggregate}
    out_dir = BASE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r027_automatic_bridge_seed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
