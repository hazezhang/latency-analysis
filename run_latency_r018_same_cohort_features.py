#!/usr/bin/env python3
"""R018b same-cohort automatic LAT feature expansion.

Uses only features complete on the R014/R015/R017 134-segment cohort.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from run_latency_r014_r015 import (
    BASE,
    group_oof_segment_model,
    model_metrics,
)


FEATURE_MATRIX = BASE / "experiments/r018_feature_audit_20260712/r018_segment_feature_matrix.csv"
QUALITY_MANIFEST = BASE / "data/experiments/r017_oof_quality/manifest.json"
OUT_DIR = BASE / "experiments/latency_r018_same_cohort_20260712"
RESULT_JSON = OUT_DIR / "latency_r018_same_cohort_results.json"
RESULT_CSV = OUT_DIR / "latency_r018_same_cohort_table.csv"
OOF_CSV = OUT_DIR / "latency_r018_same_cohort_oof_predictions.csv"


MODEL_SPECS = [
    ("R014_delay_only", ["delay_seconds"]),
    ("R017_pred_LQ_EXP", ["pred_LQ", "pred_EXP"]),
    ("R018_text_only", ["src_char_len", "mt_char_len", "src_token_len", "mt_token_len", "char_len_ratio", "token_len_ratio"]),
    ("R018_pred_LQ_EXP_text", ["pred_LQ", "pred_EXP", "src_char_len", "mt_char_len", "src_token_len", "mt_token_len", "char_len_ratio", "token_len_ratio"]),
    ("R018_pred_LQ_EXP_delay_text", ["pred_LQ", "pred_EXP", "delay_seconds", "src_char_len", "mt_char_len", "src_token_len", "mt_token_len", "char_len_ratio", "token_len_ratio"]),
    ("R015_human_LQ_EXP", ["LQ", "EXP"]),
]


def validate_quality_fold_alignment(df):
    manifest = json.loads(QUALITY_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        str(fold["outer_test_speech"]): str(fold["name"])
        for fold in manifest["folds"]
    }
    if "r017_fold" not in df.columns:
        raise ValueError("Feature matrix is missing r017_fold; cannot audit quality fold alignment.")
    mismatches = []
    for row in df[["segment_id", "speech_group", "r017_fold"]].to_dict("records"):
        expected_fold = expected.get(str(row["speech_group"]))
        if row["r017_fold"] != expected_fold:
            mismatches.append({**row, "expected_fold": expected_fold})
    if mismatches:
        raise ValueError(f"R018 quality fold/LAT speech mismatch: {mismatches[:10]}")


def run_model(df, name, features):
    oof, folds = group_oof_segment_model(df, features)
    return {
        "name": name,
        "features": features,
        "oof": oof,
        "folds": folds,
        "metrics": model_metrics(oof["LAT"], oof["prediction"], oof["speech_group"]),
    }


def main():
    df = pd.read_csv(FEATURE_MATRIX)
    required = {"segment_id", "speech_group", "LAT"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {FEATURE_MATRIX}: {missing}")
    if len(df) != 134:
        raise ValueError(f"Expected 134 segments, found {len(df)}")
    validate_quality_fold_alignment(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {}
    predictions = []
    for name, features in MODEL_SPECS:
        block = run_model(df, name, features)
        runs[name] = {
            "features": features,
            "unit": "segment",
            "cv": "LeaveOneSpeechGroupOut",
            "folds": block["folds"],
            "metrics": block["metrics"],
        }
        for row in block["oof"].to_dict("records"):
            predictions.append({"model": name, **row})

    result = {
        "metadata": {
            "feature_matrix": str(FEATURE_MATRIX.relative_to(BASE)),
            "feature_rule": "Only features complete on all 134 professional zh-en segments are used.",
            "excluded_features": ["COMET_score", "COMETkiwi_score", "source_audio_start", "translation_audio_start", "direction"],
            "excluded_reason": "Missingness or no variation would break the same-cohort comparison.",
            "speech_group_counts": dict(Counter(df["speech_group"])),
            "direction_counts": dict(Counter(df["direction"])),
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
