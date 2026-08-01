#!/usr/bin/env python3
"""Aggregate fixed-alpha sensitivity without selecting on outer test folds."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments/aaai_ridge_sensitivity_20260723"
SEEDS = ("20260718", "20260719", "20260720")
ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
MODELS = (
    "lexical_structural_piecewise_delay",
    "auto_pred_LQ_EXP_piecewise_delay",
    "auto_pred_LQ_EXP_piecewise_delay_lexical_structural",
)
METRICS = ("pearson", "spearman", "mse", "mae", "pred_std")


def alpha_name(alpha: float) -> str:
    return str(int(alpha)) if alpha.is_integer() else str(alpha).replace(".", "_")


def load(alpha: float, seed: str) -> dict:
    path = ROOT / f"experiments/aaai_crossfitted_alpha_{alpha_name(alpha)}_seed_{seed}_20260722/crossfitted_lat_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if float(payload["metadata"]["ridge_alpha"]) != alpha:
        raise ValueError(f"Alpha metadata mismatch: {path}")
    return payload


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol": (
            "Fixed-alpha sensitivity on pre-existing source-speech-group-held-out folds. "
            "All alpha values are evaluated independently; no alpha is selected using outer-test labels."
        ),
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "models": {},
    }
    rows = []
    for alpha in ALPHAS:
        payloads = {seed: load(alpha, seed) for seed in SEEDS}
        output["models"][str(alpha)] = {}
        for model in MODELS:
            per_seed = {seed: payloads[seed]["models"][model]["metrics"] for seed in SEEDS}
            aggregate = {}
            for metric in METRICS:
                values = np.asarray([per_seed[seed][metric] for seed in SEEDS], dtype=float)
                aggregate[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "values": values.tolist()}
            output["models"][str(alpha)][model] = {"per_seed": per_seed, "aggregate": aggregate}
            rows.append({
                "ridge_alpha": alpha,
                "model": model,
                **{f"{metric}_mean": aggregate[metric]["mean"] for metric in METRICS},
                **{f"{metric}_sd": aggregate[metric]["sd"] for metric in METRICS},
            })

    # The alpha=1 path is an implementation reproducibility check, not a model comparison.
    reproducibility = []
    for seed in SEEDS:
        frozen_path = ROOT / f"experiments/aaai_crossfitted_structural_seed_{seed}_20260722/crossfitted_lat_results.json"
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        current = load(1.0, seed)
        for model in MODELS:
            for metric in METRICS:
                delta = current["models"][model]["metrics"][metric] - frozen["models"][model]["metrics"][metric]
                reproducibility.append({"seed": seed, "model": model, "metric": metric, "delta": delta})
    if any(abs(row["delta"]) > 1e-12 for row in reproducibility):
        raise AssertionError("Alpha=1 sensitivity outputs differ from frozen structural results")
    output["alpha_one_reproducibility"] = reproducibility

    (OUT / "ridge_sensitivity_results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "ridge_sensitivity_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output["models"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
