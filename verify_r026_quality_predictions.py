#!/usr/bin/env python3
"""Derive the paper's upstream-quality summary directly from prediction files."""

from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path


BASE = Path(__file__).resolve().parent
PATTERN = str(BASE / "experiments/r026q_shared_seed_*/predictions_test.json")
OUT = BASE / "experiments/aaai_quality_prediction_audit_20260720"


def pearson(left, right):
    mean_left, mean_right = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - mean_left) ** 2 for x in left) * sum((y - mean_right) ** 2 for y in right))
    return numerator / denominator


def summary(values):
    return {"mean": round(statistics.mean(values), 4), "sample_sd": round(statistics.stdev(values), 4), "values": [round(value, 4) for value in values]}


def main():
    runs = []
    for path_string in sorted(glob.glob(PATTERN)):
        path = Path(path_string)
        rows = json.loads(path.read_text(encoding="utf-8"))
        record = {"file": str(path.relative_to(BASE)), "n": len(rows)}
        for dimension in ("LQ", "EXP"):
            human = [float(row[f"human_{dimension}"]) for row in rows]
            prediction = [float(row[f"pred_{dimension}"]) for row in rows]
            record[dimension] = {"pearson": round(pearson(human, prediction), 4), "prediction_population_sd": round(statistics.pstdev(prediction), 4)}
        runs.append(record)
    result = {
        "source": "raw held-out predictions",
        "n_seeds": len(runs),
        "runs": runs,
        "aggregate": {
            dimension: {
                "pearson": summary([run[dimension]["pearson"] for run in runs]),
                "prediction_population_sd": summary([run[dimension]["prediction_population_sd"] for run in runs]),
            }
            for dimension in ("LQ", "EXP")
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r026_quality_prediction_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
