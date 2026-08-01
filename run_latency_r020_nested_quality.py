#!/usr/bin/env python3
"""R020 LAT prediction from outer-nested two-stage quality predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_latency_r014_r015 import (
    BASE,
    INPUT,
    build_cohort,
    group_oof_rater_model,
    group_oof_segment_model,
    make_pipeline,
    model_metrics,
)


MANIFEST = BASE / "data/experiments/r020_nested_quality/manifest.json"
PRED_ROOT = BASE / "experiments/r020_nested_quality_20260712"
OUT_DIR = BASE / "experiments/latency_r020_nested_quality_20260712"
RESULT_JSON = OUT_DIR / "latency_r020_nested_quality_results.json"
RESULT_CSV = OUT_DIR / "latency_r020_nested_quality_table.csv"
OOF_CSV = OUT_DIR / "latency_r020_nested_quality_oof_predictions.csv"

TEXT_FEATURES = [
    "src_char_len",
    "mt_char_len",
    "src_token_len",
    "mt_token_len",
    "char_len_ratio",
    "token_len_ratio",
]


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def first_present(items, key):
    for item in items:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def ratio(num, den):
    if num is None or den in (None, 0):
        return None
    return num / den


def add_text_features(segment_rows, raw_rows):
    by_segment = defaultdict(list)
    for row in raw_rows:
        if row.get("segment_id") not in (None, ""):
            by_segment[str(row["segment_id"])].append(row)

    output = []
    for row in segment_rows:
        item = dict(row)
        raw_items = by_segment[item["segment_id"]]
        src = first_present(raw_items, "src")
        mt = first_present(raw_items, "mt")
        src_char = len(src) if isinstance(src, str) else None
        mt_char = len(mt) if isinstance(mt, str) else None
        src_tok = len(src.split()) if isinstance(src, str) else None
        mt_tok = len(mt.split()) if isinstance(mt, str) else None
        item.update({
            "src_char_len": src_char,
            "mt_char_len": mt_char,
            "src_token_len": src_tok,
            "mt_token_len": mt_tok,
            "char_len_ratio": ratio(mt_char, src_char),
            "token_len_ratio": ratio(mt_tok, src_tok),
        })
        output.append(item)
    return output


def load_fold_predictions(fold_name):
    path = PRED_ROOT / fold_name / "predictions_all.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    pred = {}
    for row in rows:
        sid = str(row["segment_id"])
        if sid in pred:
            raise ValueError(f"Duplicate prediction in {path}: segment_id={sid}")
        pred[sid] = {
            "pred_LQ": float(row["pred_LQ"]),
            "pred_EXP": float(row["pred_EXP"]),
            "quality_fold": fold_name,
        }
    return pred


def nested_oof_model(base_df, manifest, name, features):
    groups = base_df["speech_group"].to_numpy()
    y = base_df["LAT"].to_numpy(dtype=float)
    pred = np.full(len(base_df), np.nan)
    folds = []
    predictions = []

    fold_by_speech = {
        str(fold["outer_test_speech"]): fold
        for fold in manifest["folds"]
    }
    for fold_index, speech in enumerate(sorted(fold_by_speech), start=1):
        fold = fold_by_speech[speech]
        fold_name = fold["name"]
        fold_pred = load_fold_predictions(fold_name)
        missing = sorted(set(base_df["segment_id"]) - set(fold_pred))
        extra = sorted(set(fold_pred) - set(base_df["segment_id"]))
        if missing or extra:
            raise ValueError(f"R020 prediction/cohort mismatch for {fold_name}: missing={missing[:10]} extra={extra[:10]}")

        fold_df = base_df.copy()
        fold_df["pred_LQ"] = [fold_pred[sid]["pred_LQ"] for sid in fold_df["segment_id"]]
        fold_df["pred_EXP"] = [fold_pred[sid]["pred_EXP"] for sid in fold_df["segment_id"]]
        train_idx = np.flatnonzero(groups != speech)
        test_idx = np.flatnonzero(groups == speech)
        model = make_pipeline(features)
        model.fit(fold_df.iloc[train_idx], y[train_idx])
        fold_pred_y = model.predict(fold_df.iloc[test_idx])
        pred[test_idx] = fold_pred_y
        folds.append({
            "fold": fold_index,
            "heldout_group": speech,
            "quality_fold": fold_name,
            "inner_dev_speech": fold["inner_dev_speech"],
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        })
        for idx, value in zip(test_idx, fold_pred_y):
            row = fold_df.iloc[idx]
            predictions.append({
                "model": name,
                "segment_id": row["segment_id"],
                "speech_group": row["speech_group"],
                "direction": row["direction"],
                "LAT": row["LAT"],
                "prediction": float(value),
                "quality_fold": fold_name,
                "pred_LQ": row["pred_LQ"],
                "pred_EXP": row["pred_EXP"],
            })

    out = base_df[["segment_id", "speech_group", "LAT"]].copy()
    out["prediction"] = pred
    return {
        "features": features,
        "unit": "segment",
        "cv": "LeaveOneSpeechGroupOut",
        "quality_prediction_protocol": "outer-nested two-stage; LAT train and test features in a fold use predictions from the same quality model whose train/dev excludes that fold's held-out speech; this is not inner-OOF second-stage feature generation",
        "folds": folds,
        "metrics": model_metrics(out["LAT"], out["prediction"], out["speech_group"]),
    }, predictions


def main():
    global PRED_ROOT, OUT_DIR, RESULT_JSON, RESULT_CSV, OOF_CSV
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", default=str(PRED_ROOT.relative_to(BASE)))
    parser.add_argument("--output-dir", default=str(OUT_DIR.relative_to(BASE)))
    args = parser.parse_args()

    PRED_ROOT = BASE / args.prediction_root
    OUT_DIR = BASE / args.output_dir
    RESULT_JSON = OUT_DIR / "latency_r020_nested_quality_results.json"
    RESULT_CSV = OUT_DIR / "latency_r020_nested_quality_table.csv"
    OOF_CSV = OUT_DIR / "latency_r020_nested_quality_oof_predictions.csv"

    raw_rows = json.loads(INPUT.read_text(encoding="utf-8"))
    rater_rows, segment_rows = build_cohort(raw_rows)
    segment_rows = add_text_features(segment_rows, raw_rows)
    seg_df = pd.DataFrame(segment_rows)
    if len(seg_df) != 150:
        raise ValueError(f"Expected 150 post-R021 segments, found {len(seg_df)}")
    for feature in TEXT_FEATURES:
        if seg_df[feature].isna().any():
            raise ValueError(f"Text feature has missing values: {feature}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = {}
    predictions = []

    for name, features in [
        ("R014_delay_only", ["delay_seconds"]),
        ("R015_human_LQ_EXP", ["LQ", "EXP"]),
        ("R015_human_LQ_EXP_delay", ["LQ", "EXP", "delay_seconds"]),
        ("R020_text_only", TEXT_FEATURES),
    ]:
        oof, folds = group_oof_segment_model(seg_df, features)
        runs[name] = {
            "features": features,
            "unit": "segment",
            "cv": "LeaveOneSpeechGroupOut",
            "folds": folds,
            "metrics": model_metrics(oof["LAT"], oof["prediction"], oof["speech_group"]),
        }
        for row in oof.to_dict("records"):
            predictions.append({"model": name, **row})

    for name, features in [
        ("R020_pred_LQ_EXP", ["pred_LQ", "pred_EXP"]),
        ("R020_pred_LQ_EXP_delay", ["pred_LQ", "pred_EXP", "delay_seconds"]),
        ("R020_pred_LQ_EXP_text", ["pred_LQ", "pred_EXP", *TEXT_FEATURES]),
        ("R020_pred_LQ_EXP_delay_text", ["pred_LQ", "pred_EXP", "delay_seconds", *TEXT_FEATURES]),
    ]:
        block, rows = nested_oof_model(seg_df, manifest, name, features)
        runs[name] = block
        predictions.extend(rows)

    rater_oof, rater_folds = group_oof_rater_model(pd.DataFrame(rater_rows))
    runs["R015_human_LQ_EXP_rater_control"] = {
        "features": ["LQ", "EXP", "evaluator_id"],
        "unit": "rater_row_model_segment_averaged_predictions",
        "cv": "LeaveOneSpeechGroupOut",
        "folds": rater_folds,
        "metrics": model_metrics(rater_oof["LAT"], rater_oof["prediction"], rater_oof["speech_group"]),
    }
    for row in rater_oof.to_dict("records"):
        predictions.append({"model": "R015_human_LQ_EXP_rater_control", **row})

    result = {
        "metadata": {
            "input": str(INPUT.relative_to(BASE)),
            "quality_manifest": str(MANIFEST.relative_to(BASE)),
            "quality_prediction_root": str(PRED_ROOT.relative_to(BASE)),
            "feature_complete_rater_rows": len(rater_rows),
            "feature_complete_segments": len(seg_df),
            "speech_group_counts": dict(Counter(seg_df["speech_group"])),
            "direction_counts": dict(Counter(seg_df["direction"])),
            "comparison_rule": "All LAT models use the same post-R021 150-segment professional cohort.",
            "nesting_rule": manifest["nesting_rule"],
            "ci_rule": "95% CIs use speech-group cluster bootstrap, matching LeaveOneSpeechGroupOut evaluation.",
        },
        "models": runs,
    }
    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "model",
        "n_segments",
        "pearson",
        "pearson_ci_low",
        "pearson_ci_high",
        "spearman",
        "spearman_ci_low",
        "spearman_ci_high",
        "mse",
        "mae",
        "r2",
        "pred_std",
    ]
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
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
    print(RESULT_JSON)
    print(RESULT_CSV)
    print(OOF_CSV)
    print(RESULT_CSV.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
