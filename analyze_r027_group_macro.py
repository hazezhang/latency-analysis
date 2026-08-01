#!/usr/bin/env python3
"""Report macro speech-group correlations from the R027 OOF prediction files."""

from __future__ import annotations

import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
OUT = BASE / "experiments/aaai_group_macro_audit_20260720"
MODELS = ("delay_linear", "auto_pred_LQ_EXP_piecewise_delay")


def corr(pairs):
    y, prediction = zip(*pairs)
    mean_y, mean_prediction = statistics.mean(y), statistics.mean(prediction)
    y_ss = sum((value - mean_y) ** 2 for value in y)
    prediction_ss = sum((value - mean_prediction) ** 2 for value in prediction)
    if not y_ss or not prediction_ss:
        return None
    return sum((left - mean_y) * (right - mean_prediction) for left, right in pairs) / math.sqrt(y_ss * prediction_ss)


def main():
    output = {"models": {}}
    for model in MODELS:
        seed_rows = []
        for path_string in sorted(glob.glob(str(BASE / "experiments/latency_r027_seed_*/r027_automatic_bridge_oof_predictions.csv"))):
            grouped = defaultdict(list)
            with Path(path_string).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row["model"] == model:
                        grouped[row["speech_group"]].append((float(row["LAT"]), float(row["prediction"])))
            per_group = {group: corr(pairs) for group, pairs in grouped.items()}
            valid = [value for value in per_group.values() if value is not None]
            seed_rows.append({
                "file": str(Path(path_string).relative_to(BASE)),
                "n_groups_total": len(per_group),
                "n_groups_valid": len(valid),
                "macro_pearson": round(statistics.mean(valid), 4),
                "per_group_pearson": {group: round(value, 4) if value is not None else None for group, value in sorted(per_group.items())},
            })
        values = [row["macro_pearson"] for row in seed_rows]
        output["models"][model] = {"runs": seed_rows, "macro_pearson_mean": round(statistics.mean(values), 4), "macro_pearson_sample_sd": round(statistics.stdev(values), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r027_group_macro_summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
