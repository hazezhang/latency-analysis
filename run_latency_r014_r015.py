#!/usr/bin/env python3
"""R014/R015 same-cohort LAT prediction baselines.

Builds the post-R021 professional feature-complete cohort and compares:
- R014 delay-only acoustic baseline.
- R015 human LQ+EXP quality-associated predictors.
- R015 rater-controlled row-level quality model, aggregated back to segments.

All reported correlations use out-of-fold predictions with speech-group
cross-validation and identical segment rows.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).parent
INPUT = BASE / "data/evaluation/profess_eval_delay_enriched.json"
OUT_DIR = BASE / "experiments/latency_r014_r015_post_r021_20260712"
RESULT_JSON = OUT_DIR / "latency_r014_r015_results.json"
RESULT_CSV = OUT_DIR / "latency_r014_r015_table.csv"
OOF_CSV = OUT_DIR / "latency_r014_r015_oof_predictions.csv"
SEED = 20260712
BOOTSTRAPS = 5000


def number(value):
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def safe_corr(y_true, y_pred, method):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return None
    if method == "pearson":
        return float(pearsonr(y_true, y_pred)[0])
    return float(spearmanr(y_true, y_pred)[0])


def bootstrap_ci(y_true, y_pred, method, groups=None, n=BOOTSTRAPS, seed=SEED):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    estimate = safe_corr(y_true, y_pred, method)
    rng = np.random.default_rng(seed)
    groups = None if groups is None else np.asarray(groups)
    unique_groups = None if groups is None else np.unique(groups)
    values = []
    for _ in range(n):
        if unique_groups is None:
            idx = rng.integers(0, len(y_true), size=len(y_true))
        else:
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        value = safe_corr(y_true[idx], y_pred[idx], method)
        if value is not None and np.isfinite(value):
            values.append(value)
    ci = [None, None] if not values else [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {
        "estimate": None if estimate is None else round(estimate, 4),
        "ci95": [None if v is None else round(v, 4) for v in ci],
        "bootstrap_valid": len(values),
    }


def model_metrics(y_true, y_pred, groups=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n_segments": int(len(y_true)),
        "pearson": bootstrap_ci(y_true, y_pred, "pearson", groups=groups),
        "spearman": bootstrap_ci(y_true, y_pred, "spearman", groups=groups),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "pred_std": round(float(np.std(y_pred)), 4),
    }


def speech_group(row):
    return str(row.get("speech") or row.get("source_file") or row.get("file_id") or row.get("segment_id"))


def build_cohort(rows):
    complete = []
    required = ("segment_id", "LQ", "EXP", "perceived_latency", "delay_seconds")
    for row in rows:
        values = {key: number(row.get(key)) for key in required if key != "segment_id"}
        if row.get("segment_id") in (None, "") or any(values[key] is None for key in values):
            continue
        item = dict(row)
        item.update(values)
        item["segment_id"] = str(row["segment_id"])
        item["speech_group"] = speech_group(row)
        complete.append(item)

    grouped = defaultdict(list)
    for row in complete:
        grouped[row["segment_id"]].append(row)

    segment_rows = []
    for sid, items in sorted(grouped.items()):
        segment_rows.append({
            "segment_id": sid,
            "speech_group": items[0]["speech_group"],
            "direction": next((x.get("direction") for x in items if x.get("direction")), None),
            "source_file": next((x.get("source_file") for x in items if x.get("source_file")), None),
            "n_rater_rows": len(items),
            "LQ": float(np.mean([x["LQ"] for x in items])),
            "EXP": float(np.mean([x["EXP"] for x in items])),
            "LAT": float(np.mean([x["perceived_latency"] for x in items])),
            "delay_seconds": float(np.mean([x["delay_seconds"] for x in items])),
        })

    return complete, segment_rows


def make_pipeline(numeric_features, categorical_features=()):
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), list(numeric_features)))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_features)))
    pre = ColumnTransformer(transformers)
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])


def group_oof_segment_model(df, features, categorical=()):
    groups = df["speech_group"].to_numpy()
    y = df["LAT"].to_numpy(dtype=float)
    pred = np.full(len(df), np.nan)
    logo = LeaveOneGroupOut()
    folds = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(df, y, groups), start=1):
        model = make_pipeline(features, categorical)
        model.fit(df.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(df.iloc[test_idx])
        folds.append({
            "fold": fold,
            "heldout_group": str(groups[test_idx][0]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        })
    out = df[["segment_id", "speech_group", "LAT"]].copy()
    out["prediction"] = pred
    return out, folds


def group_oof_rater_model(rater_df):
    groups = rater_df["speech_group"].to_numpy()
    y = rater_df["perceived_latency"].to_numpy(dtype=float)
    pred = np.full(len(rater_df), np.nan)
    logo = LeaveOneGroupOut()
    folds = []
    for fold, (train_idx, test_idx) in enumerate(logo.split(rater_df, y, groups), start=1):
        model = make_pipeline(["LQ", "EXP"], ["evaluator_id"])
        model.fit(rater_df.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(rater_df.iloc[test_idx])
        folds.append({
            "fold": fold,
            "heldout_group": str(groups[test_idx][0]),
            "n_train_rows": int(len(train_idx)),
            "n_test_rows": int(len(test_idx)),
        })
    out = rater_df[["segment_id", "speech_group", "perceived_latency"]].copy()
    out["prediction"] = pred
    segment = out.groupby(["segment_id"], as_index=False).agg(
        speech_group=("speech_group", "first"),
        LAT=("perceived_latency", "mean"),
        prediction=("prediction", "mean"),
        n_rater_rows=("prediction", "size"),
    )
    return segment, folds


def main():
    rows = json.loads(INPUT.read_text(encoding="utf-8"))
    rater_rows, segment_rows = build_cohort(rows)
    seg_df = pd.DataFrame(segment_rows)
    rater_df = pd.DataFrame(rater_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = {}
    predictions = []

    model_specs = [
        ("R014_delay_only", ["delay_seconds"], []),
        ("R015_human_LQ_EXP", ["LQ", "EXP"], []),
        ("R015_human_LQ_EXP_delay", ["LQ", "EXP", "delay_seconds"], []),
    ]
    for name, numeric, categorical in model_specs:
        oof, folds = group_oof_segment_model(seg_df, numeric, categorical)
        metrics = model_metrics(oof["LAT"], oof["prediction"], oof["speech_group"])
        runs[name] = {
            "features": numeric + categorical,
            "unit": "segment",
            "cv": "LeaveOneSpeechGroupOut",
            "folds": folds,
            "metrics": metrics,
        }
        for row in oof.to_dict("records"):
            predictions.append({"model": name, **row})

    rater_oof, rater_folds = group_oof_rater_model(rater_df)
    runs["R015_human_LQ_EXP_rater_control"] = {
        "features": ["LQ", "EXP", "evaluator_id"],
        "unit": "rater_row_model_segment_averaged_predictions",
        "cv": "LeaveOneSpeechGroupOut",
        "folds": rater_folds,
        "metrics": model_metrics(rater_oof["LAT"], rater_oof["prediction"], rater_oof["speech_group"]),
    }
    for row in rater_oof.to_dict("records"):
        predictions.append({"model": "R015_human_LQ_EXP_rater_control", **row})

    metadata = {
        "input": str(INPUT.relative_to(BASE)),
        "seed": SEED,
        "bootstraps": BOOTSTRAPS,
        "feature_complete_rater_rows": len(rater_rows),
        "feature_complete_segments": len(segment_rows),
        "speech_group_key": "speech if present else source_file/file_id",
        "speech_group_counts": dict(Counter(seg_df["speech_group"])),
        "direction_counts": dict(Counter(seg_df["direction"])),
        "direction_control_note": "Direction is reported descriptively. Direction-specific coefficients require enough held-out groups per direction.",
        "rater_counts": dict(Counter(rater_df["evaluator_id"])),
        "comparison_rule": f"All models are evaluated on the same {len(segment_rows)} post-R021 feature-complete professional segments.",
        "ci_rule": "95% CIs use speech-group cluster bootstrap, matching LeaveOneSpeechGroupOut evaluation.",
    }

    result = {"metadata": metadata, "models": runs}
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["model", "n_segments", "pearson", "pearson_ci_low", "pearson_ci_high", "spearman", "spearman_ci_low", "spearman_ci_high", "mse", "mae", "r2", "pred_std"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, block in runs.items():
            m = block["metrics"]
            writer.writerow({
                "model": name,
                "n_segments": m["n_segments"],
                "pearson": m["pearson"]["estimate"],
                "pearson_ci_low": m["pearson"]["ci95"][0],
                "pearson_ci_high": m["pearson"]["ci95"][1],
                "spearman": m["spearman"]["estimate"],
                "spearman_ci_low": m["spearman"]["ci95"][0],
                "spearman_ci_high": m["spearman"]["ci95"][1],
                "mse": m["mse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "pred_std": m["pred_std"],
            })

    pd.DataFrame(predictions).to_csv(OOF_CSV, index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {RESULT_JSON}")
    print(f"Wrote {RESULT_CSV}")
    print(f"Wrote {OOF_CSV}")


if __name__ == "__main__":
    main()
