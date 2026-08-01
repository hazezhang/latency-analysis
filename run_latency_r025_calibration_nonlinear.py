#!/usr/bin/env python3
"""R025: rater-by-interpreter calibration and nonlinear delay analysis.

The primary analysis keeps physically plausible segment delays in [0, 20]
seconds.  Rows outside that range are retained in a data-quality sensitivity
summary, but cannot create a spurious range threshold in the main analysis.
Comments are deliberately not model features.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


BASE = Path(__file__).parent
RAW_PATH = BASE / "data/evaluation/profess_eval_delay_enriched.json"
DEFAULT_OUT = BASE / "experiments/latency_r025_calibration_nonlinear_20260718"
SEED = 20260718
BOOTSTRAPS = 5000
PRIMARY_MIN_DELAY = 0.0
PRIMARY_MAX_DELAY = 20.0
BIN_EDGES = (0.0, 2.0, 4.0, 6.0, 10.0, 20.0)


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def stable_segment_id(row: dict) -> str:
    file_id = str(row.get("file_id") or "unknown")
    original = str(row.get("original_segment_id") or row.get("segment_id") or "unknown")
    return f"{file_id}:{original}"


def normalized_interpreter(row: dict) -> str:
    return str(row.get("interpreter") or "unknown").strip().casefold()


def speech_group(row: dict) -> str:
    return str(row.get("speech") or row.get("source_file") or row.get("file_id") or "unknown")


def mode(values: list[str]) -> str:
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def corr(x, y, method: str = "pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = pearsonr(x, y)[0] if method == "pearson" else spearmanr(x, y)[0]
    return round(float(value), 4)


def bootstrap_corr(df: pd.DataFrame, left: str, right: str, seed: int) -> list[float | None]:
    if len(df) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        sample = df.iloc[rng.integers(0, len(df), size=len(df))]
        value = corr(sample[left], sample[right])
        if value is not None:
            values.append(value)
    if not values:
        return [None, None]
    return [
        round(float(np.quantile(values, 0.025)), 3),
        round(float(np.quantile(values, 0.975)), 3),
    ]


def residualized_corr(df: pd.DataFrame, left: str, right: str) -> float | None:
    left_resid = df[left] - df.groupby("interpreter")[left].transform("mean")
    right_resid = df[right] - df.groupby("interpreter")[right].transform("mean")
    return corr(left_resid, right_resid)


def f_statistic(values: np.ndarray, groups: np.ndarray) -> float:
    overall = float(np.mean(values))
    unique = np.unique(groups)
    between = sum(np.sum(groups == group) * (float(np.mean(values[groups == group])) - overall) ** 2 for group in unique)
    within = sum(np.sum((values[groups == group] - float(np.mean(values[groups == group]))) ** 2) for group in unique)
    df_between = len(unique) - 1
    df_within = len(values) - len(unique)
    if df_between <= 0 or df_within <= 0 or within == 0:
        return float("nan")
    return (between / df_between) / (within / df_within)


def permutation_anova(values: list[float], groups: list[str], n_permutations: int = 5000) -> dict:
    values_array = np.asarray(values, dtype=float)
    groups_array = np.asarray(groups)
    observed = f_statistic(values_array, groups_array)
    rng = np.random.default_rng(SEED)
    null = [f_statistic(values_array, rng.permutation(groups_array)) for _ in range(n_permutations)]
    p_value = (1 + sum(value >= observed for value in null if np.isfinite(value))) / (1 + n_permutations)
    return {"f_statistic": round(float(observed), 4), "permutations": n_permutations, "p_value": round(float(p_value), 4)}


def canonicalize(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    complete = []
    for raw in rows:
        values = {key: number(raw.get(key)) for key in ("LQ", "EXP", "perceived_latency", "delay_seconds")}
        if any(value is None for value in values.values()):
            continue
        row = dict(raw)
        row.update(values)
        row["sample_id"] = stable_segment_id(row)
        row["interpreter_norm"] = normalized_interpreter(row)
        row["speech_group_norm"] = speech_group(row)
        complete.append(row)

    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in complete:
        by_sample[row["sample_id"]].append(row)

    segment_rows, metadata_conflicts = [], []
    for sample_id, items in sorted(by_sample.items()):
        speeches = [item["speech_group_norm"] for item in items]
        interpreters = [item["interpreter_norm"] for item in items]
        selected_speech = mode(speeches)
        selected_interpreter = mode(interpreters)
        if len(set(speeches)) > 1 or len(set(interpreters)) > 1:
            metadata_conflicts.append({
                "sample_id": sample_id,
                "speech_values": sorted(set(speeches)),
                "interpreter_values": sorted(set(interpreters)),
                "resolved_speech": selected_speech,
                "resolved_interpreter": selected_interpreter,
            })
        segment_rows.append({
            "sample_id": sample_id,
            "speech_group": selected_speech,
            "interpreter": selected_interpreter,
            "direction": mode([str(item.get("direction") or "unknown") for item in items]),
            "n_rater_rows": len(items),
            "LQ": float(np.mean([item["LQ"] for item in items])),
            "EXP": float(np.mean([item["EXP"] for item in items])),
            "LAT": float(np.mean([item["perceived_latency"] for item in items])),
            "delay_seconds": float(np.mean([item["delay_seconds"] for item in items])),
        })
    return pd.DataFrame(complete), pd.DataFrame(segment_rows), metadata_conflicts


def build_piecewise_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    delay = out["delay_seconds"].to_numpy(dtype=float)
    out["delay_linear"] = delay
    for knot in BIN_EDGES[1:-1]:
        out[f"delay_hinge_{int(knot)}"] = np.maximum(0.0, delay - knot)
    out["LQ_delay_interaction"] = out["LQ"] * delay
    out["EXP_delay_interaction"] = out["EXP"] * delay
    return out


def logo_metrics(df: pd.DataFrame, features: list[str]) -> dict:
    y = df["LAT"].to_numpy(dtype=float)
    groups = df["speech_group"].to_numpy()
    prediction = np.full(len(df), np.nan)
    for speech in sorted(set(groups)):
        train_idx = np.flatnonzero(groups != speech)
        test_idx = np.flatnonzero(groups == speech)
        x_train = df.iloc[train_idx][features].to_numpy(dtype=float)
        x_test = df.iloc[test_idx][features].to_numpy(dtype=float)
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0)
        scale[scale == 0] = 1.0
        x_train = (x_train - mean) / scale
        x_test = (x_test - mean) / scale
        # Match a standard scaled Ridge fit while leaving the intercept unpenalized.
        x_train = np.column_stack([np.ones(len(x_train)), x_train])
        x_test = np.column_stack([np.ones(len(x_test)), x_test])
        penalty = np.eye(x_train.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y[train_idx])
        prediction[test_idx] = x_test @ coefficients
    return {
        "n_segments": int(len(df)),
        "n_speech_groups": int(df["speech_group"].nunique()),
        "pearson": corr(y, prediction),
        "spearman": corr(y, prediction, "spearman"),
        "mse": round(float(np.mean((y - prediction) ** 2)), 4),
        "mae": round(float(np.mean(np.abs(y - prediction))), 4),
    }


def calibration_analysis(rater_df: pd.DataFrame) -> tuple[list[dict], dict]:
    by_sample_rater: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rater_df.to_dict("records"):
        by_sample_rater[(row["sample_id"], str(row.get("evaluator_id")))].append(row)

    paired = []
    sample_ids = sorted({sample_id for sample_id, _ in by_sample_rater})
    for sample_id in sample_ids:
        if (sample_id, "R05") not in by_sample_rater or (sample_id, "R06") not in by_sample_rater:
            continue
        r05 = by_sample_rater[(sample_id, "R05")]
        r06 = by_sample_rater[(sample_id, "R06")]
        base = r05[0]
        for outcome, source in (("LQ", "LQ"), ("EXP", "EXP"), ("LAT", "perceived_latency")):
            paired.append({
                "sample_id": sample_id,
                "interpreter": base["interpreter_norm"],
                "outcome": outcome,
                "r05": float(np.mean([row[source] for row in r05])),
                "r06": float(np.mean([row[source] for row in r06])),
                "r06_minus_r05": float(np.mean([row[source] for row in r06]) - np.mean([row[source] for row in r05])),
            })
    paired_df = pd.DataFrame(paired)
    table, omnibus = [], {}
    for outcome, block in paired_df.groupby("outcome"):
        valid = block.groupby("interpreter").filter(lambda x: len(x) >= 3)
        omnibus[outcome] = {
            "n_paired_segments": int(len(valid)),
            "rater_by_interpreter_gap_test": permutation_anova(valid["r06_minus_r05"].tolist(), valid["interpreter"].tolist()),
        }
        for interpreter, rows in valid.groupby("interpreter"):
            gaps = rows["r06_minus_r05"].to_numpy(dtype=float)
            table.append({
                "outcome": outcome,
                "interpreter": interpreter,
                "n_paired_segments": int(len(rows)),
                "R05_mean": round(float(rows["r05"].mean()), 4),
                "R06_mean": round(float(rows["r06"].mean()), 4),
                "R06_minus_R05_mean": round(float(gaps.mean()), 4),
                "R06_minus_R05_sd": round(float(gaps.std(ddof=1)), 4),
            })
    return table, omnibus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_PATH.relative_to(BASE)))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT.relative_to(BASE)))
    args = parser.parse_args()
    raw_path = BASE / args.input
    out_dir = BASE / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
    rater_df, segment_df, metadata_conflicts = canonicalize(raw_rows)
    primary_df = segment_df[(segment_df["delay_seconds"] >= PRIMARY_MIN_DELAY) & (segment_df["delay_seconds"] <= PRIMARY_MAX_DELAY)].copy()
    primary_df = build_piecewise_features(primary_df)

    calibration_table, calibration_omnibus = calibration_analysis(rater_df)
    bin_rows = []
    bootstrap_rows = []
    for bin_index, (low, high) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        block = primary_df[(primary_df["delay_seconds"] >= low) & (primary_df["delay_seconds"] < high if high < PRIMARY_MAX_DELAY else primary_df["delay_seconds"] <= high)]
        row = {
            "delay_bin_seconds": f"{low:g}-{high:g}",
            "n_segments": int(len(block)),
            "delay_LAT_pearson": corr(block["delay_seconds"], block["LAT"]),
            "LQ_LAT_pearson": corr(block["LQ"], block["LAT"]),
            "EXP_LAT_pearson": corr(block["EXP"], block["LAT"]),
            "LQ_EXP_pearson": corr(block["LQ"], block["EXP"]),
        }
        bin_rows.append(row)
        bootstrap_rows.append({
            "bin": row["delay_bin_seconds"],
            "n": row["n_segments"],
            "delay": {
                "r": row["delay_LAT_pearson"],
                "ci": bootstrap_corr(block, "delay_seconds", "LAT", SEED + 10 * bin_index),
            },
            "LQ": {
                "r": row["LQ_LAT_pearson"],
                "ci": bootstrap_corr(block, "LQ", "LAT", SEED + 10 * bin_index + 1),
            },
            "EXP": {
                "r": row["EXP_LAT_pearson"],
                "ci": bootstrap_corr(block, "EXP", "LAT", SEED + 10 * bin_index + 2),
            },
        })

    hinge_features = ["delay_linear", "delay_hinge_2", "delay_hinge_4", "delay_hinge_6", "delay_hinge_10"]
    model_specs = {
        "delay_linear": ["delay_linear"],
        "delay_piecewise": hinge_features,
        "human_quality": ["LQ", "EXP"],
        "quality_plus_piecewise_delay": ["LQ", "EXP", *hinge_features],
        "quality_delay_interaction": ["LQ", "EXP", *hinge_features, "LQ_delay_interaction", "EXP_delay_interaction"],
    }
    model_results = {name: {"features": features, "metrics": logo_metrics(primary_df, features)} for name, features in model_specs.items()}

    associations = {}
    for left, right in (("delay_seconds", "LAT"), ("LQ", "LAT"), ("EXP", "LAT")):
        associations[f"{left}_vs_{right}"] = {
            "overall_pearson": corr(primary_df[left], primary_df[right]),
            "within_interpreter_pearson": residualized_corr(primary_df, left, right),
        }

    summary = {
        "protocol": {
            "purpose": "Exploratory construct analysis; comments are not model inputs and no causal threshold is claimed.",
            "outcome_direction": "Higher perceived_latency score denotes better perceived latency; a negative delay-LAT association is therefore the expected direction for longer delay.",
            "primary_delay_window_seconds": [PRIMARY_MIN_DELAY, PRIMARY_MAX_DELAY],
            "delay_bins_seconds": list(BIN_EDGES),
            "cv": "LeaveOneSpeechGroupOut for predictive comparisons.",
        },
        "data_audit": {
            "input_rows": len(raw_rows),
            "complete_rater_rows": int(len(rater_df)),
            "complete_segments": int(len(segment_df)),
            "primary_segments": int(len(primary_df)),
            "segments_outside_primary_delay_window": int(len(segment_df) - len(primary_df)),
            "speech_groups_primary": int(primary_df["speech_group"].nunique()),
            "interpreters_primary": int(primary_df["interpreter"].nunique()),
            "metadata_conflicts_resolved_by_mode": metadata_conflicts,
        },
        "rater_interpreter_calibration": calibration_omnibus,
        "associations": associations,
        "model_results": model_results,
        "interpretation_guardrail": "Bin correlations and interaction-model comparisons test heterogeneity, not a confirmed causal threshold. Confirmatory threshold claims require an independently specified cutoff and new held-out data.",
    }
    (out_dir / "r025_calibration_nonlinear_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "r025_delay_bin_bootstrap.json").write_text(
        json.dumps(bootstrap_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, rows in (("r025_delay_bin_associations.csv", bin_rows), ("r025_rater_interpreter_calibration.csv", calibration_table)):
        with (out_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
