#!/usr/bin/env python3
"""R017 LAT prediction with out-of-fold predicted LQ/EXP features."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from run_latency_r014_r015 import (
    BASE,
    INPUT,
    build_cohort,
    group_oof_rater_model,
    group_oof_segment_model,
    model_metrics,
)


QUALITY_PRED = BASE / "experiments/r017_oof_quality_20260712/r017_oof_quality_predictions.json"
QUALITY_MANIFEST = BASE / "data/experiments/r017_oof_quality/manifest.json"
OUT_DIR = BASE / "experiments/latency_r017_pred_quality_20260712"
RESULT_JSON = OUT_DIR / "latency_r017_pred_quality_results.json"
RESULT_CSV = OUT_DIR / "latency_r017_pred_quality_table.csv"
OOF_CSV = OUT_DIR / "latency_r017_pred_quality_oof_predictions.csv"


def load_quality_predictions():
    rows = json.loads(QUALITY_PRED.read_text(encoding="utf-8"))
    pred_by_segment = {}
    for row in rows:
        sid = str(row["segment_id"])
        if sid in pred_by_segment:
            raise ValueError(f"Duplicate quality prediction for segment_id={sid}")
        pred_by_segment[sid] = {
            "segment_id": sid,
            "pred_LQ": float(row["pred_LQ"]),
            "pred_EXP": float(row["pred_EXP"]),
            "quality_fold": row.get("r017_fold"),
        }
    return pred_by_segment


def expected_quality_folds():
    manifest = json.loads(QUALITY_MANIFEST.read_text(encoding="utf-8"))
    return {
        str(fold["outer_test_speech"]): str(fold["name"])
        for fold in manifest["folds"]
    }


def validate_quality_fold_alignment(rows):
    expected = expected_quality_folds()
    mismatches = []
    for row in rows:
        expected_fold = expected.get(str(row["speech_group"]))
        if row.get("quality_fold") != expected_fold:
            mismatches.append({
                "segment_id": row["segment_id"],
                "speech_group": row["speech_group"],
                "quality_fold": row.get("quality_fold"),
                "expected_fold": expected_fold,
            })
    if mismatches:
        raise ValueError(f"R017 quality fold/LAT speech mismatch: {mismatches[:10]}")


def add_run(runs, predictions, name, seg_df, numeric, categorical=()):
    oof, folds = group_oof_segment_model(seg_df, numeric, categorical)
    runs[name] = {
        "features": list(numeric) + list(categorical),
        "unit": "segment",
        "cv": "LeaveOneSpeechGroupOut",
        "folds": folds,
        "metrics": model_metrics(oof["LAT"], oof["prediction"], oof["speech_group"]),
    }
    for row in oof.to_dict("records"):
        predictions.append({"model": name, **row})


def main():
    raw_rows = json.loads(INPUT.read_text(encoding="utf-8"))
    rater_rows, segment_rows = build_cohort(raw_rows)
    pred_by_segment = load_quality_predictions()

    missing = sorted(str(row["segment_id"]) for row in segment_rows if str(row["segment_id"]) not in pred_by_segment)
    extra = sorted(set(pred_by_segment) - {str(row["segment_id"]) for row in segment_rows})
    if missing or extra:
        raise ValueError(f"R017 prediction/cohort mismatch: missing={missing[:10]} extra={extra[:10]}")

    merged_rows = []
    for row in segment_rows:
        item = dict(row)
        item.update(pred_by_segment[str(row["segment_id"])])
        merged_rows.append(item)
    validate_quality_fold_alignment(merged_rows)

    seg_df = pd.DataFrame(merged_rows)
    rater_df = pd.DataFrame(rater_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = {}
    predictions = []
    add_run(runs, predictions, "R014_delay_only", seg_df, ["delay_seconds"])
    add_run(runs, predictions, "R015_human_LQ_EXP", seg_df, ["LQ", "EXP"])
    add_run(runs, predictions, "R015_human_LQ_EXP_delay", seg_df, ["LQ", "EXP", "delay_seconds"])
    add_run(runs, predictions, "R017_pred_LQ_EXP", seg_df, ["pred_LQ", "pred_EXP"])
    add_run(runs, predictions, "R017_pred_LQ_EXP_delay", seg_df, ["pred_LQ", "pred_EXP", "delay_seconds"])

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

    result = {
        "metadata": {
            "input": str(INPUT.relative_to(BASE)),
            "quality_predictions": str(QUALITY_PRED.relative_to(BASE)),
            "feature_complete_rater_rows": len(rater_rows),
            "feature_complete_segments": len(segment_rows),
            "speech_group_counts": dict(Counter(seg_df["speech_group"])),
            "direction_counts": dict(Counter(seg_df["direction"])),
            "comparison_rule": "All LAT models use the same 134 feature-complete professional zh-en segments.",
            "quality_prediction_rule": "pred_LQ/pred_EXP are out-of-fold with the same held-out speech groups; no local LAT test data was uploaded to the remote GPU.",
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
