#!/usr/bin/env python3
"""Audit feature completeness for R018 LAT model expansion.

The audit starts from the same 134 professional zh-en segments used by
R014/R015/R017 and checks which candidate features can be added without
changing the cohort.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).parent
RAW_PATH = BASE / "data/evaluation/profess_eval_delay_enriched.json"
R017_PRED_PATH = BASE / "experiments/r017_oof_quality_20260712/r017_oof_quality_predictions.json"
OUT_DIR = BASE / "experiments/r018_feature_audit_20260712"
SUMMARY_JSON = OUT_DIR / "r018_feature_completeness_summary.json"
FEATURE_CSV = OUT_DIR / "r018_feature_completeness_table.csv"
SEGMENT_CSV = OUT_DIR / "r018_segment_feature_matrix.csv"


BASE_REQUIRED = ("segment_id", "LQ", "EXP", "perceived_latency", "delay_seconds")
RAW_CANDIDATES = (
    "COMET_score",
    "COMETkiwi_score",
    "source_audio_start",
    "translation_audio_start",
)
DERIVED_TEXT_FEATURES = (
    "src_char_len",
    "mt_char_len",
    "src_token_len",
    "mt_token_len",
    "char_len_ratio",
    "token_len_ratio",
)
R017_FEATURES = ("pred_LQ", "pred_EXP")


def number(value):
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def speech_group(row):
    return str(row.get("speech") or row.get("source_file") or row.get("file_id") or row.get("segment_id"))


def is_complete_base(row):
    if row.get("segment_id") in (None, ""):
        return False
    return all(number(row.get(key)) is not None for key in BASE_REQUIRED if key != "segment_id")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


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


def build_base_segments(raw_rows):
    complete = [dict(row) for row in raw_rows if is_complete_base(row)]
    grouped = defaultdict(list)
    for row in complete:
        row["segment_id"] = str(row["segment_id"])
        grouped[row["segment_id"]].append(row)

    segments = []
    for sid, items in sorted(grouped.items()):
        src = first_present(items, "src")
        mt = first_present(items, "mt")
        src_char = len(src) if isinstance(src, str) else None
        mt_char = len(mt) if isinstance(mt, str) else None
        src_tok = len(src.split()) if isinstance(src, str) else None
        mt_tok = len(mt.split()) if isinstance(mt, str) else None
        item = {
            "segment_id": sid,
            "speech_group": speech_group(items[0]),
            "direction": first_present(items, "direction"),
            "source_file": first_present(items, "source_file"),
            "src": src,
            "mt": mt,
            "n_rater_rows": len(items),
            "LQ": mean(number(x.get("LQ")) for x in items),
            "EXP": mean(number(x.get("EXP")) for x in items),
            "LAT": mean(number(x.get("perceived_latency")) for x in items),
            "delay_seconds": mean(number(x.get("delay_seconds")) for x in items),
            "COMET_score": mean(number(x.get("COMET_score")) for x in items),
            "COMETkiwi_score": mean(number(x.get("COMETkiwi_score")) for x in items),
            "source_audio_start": mean(number(x.get("source_audio_start")) for x in items),
            "translation_audio_start": mean(number(x.get("translation_audio_start")) for x in items),
            "src_char_len": src_char,
            "mt_char_len": mt_char,
            "src_token_len": src_tok,
            "mt_token_len": mt_tok,
            "char_len_ratio": ratio(mt_char, src_char),
            "token_len_ratio": ratio(mt_tok, src_tok),
        }
        segments.append(item)
    return complete, segments


def load_r017_predictions():
    rows = json.loads(R017_PRED_PATH.read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        sid = str(row["segment_id"])
        if sid in out:
            raise ValueError(f"Duplicate R017 prediction for segment_id={sid}")
        out[sid] = {
            "pred_LQ": number(row.get("pred_LQ")),
            "pred_EXP": number(row.get("pred_EXP")),
            "r017_fold": row.get("r017_fold"),
        }
    return out


def completeness(values):
    n = len(values)
    present = sum(value is not None for value in values)
    return {
        "n_segments": n,
        "present": present,
        "missing": n - present,
        "coverage": round(present / n, 6) if n else None,
    }


def main():
    raw_rows = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    rater_rows, segments = build_base_segments(raw_rows)
    r017 = load_r017_predictions()

    missing_pred = sorted(set(row["segment_id"] for row in segments) - set(r017))
    extra_pred = sorted(set(r017) - set(row["segment_id"] for row in segments))
    if missing_pred or extra_pred:
        raise ValueError(f"R017 mismatch: missing={missing_pred[:10]} extra={extra_pred[:10]}")

    for row in segments:
        row.update(r017[row["segment_id"]])

    candidates = (
        "delay_seconds",
        "LQ",
        "EXP",
        *R017_FEATURES,
        *RAW_CANDIDATES,
        *DERIVED_TEXT_FEATURES,
    )
    feature_rows = []
    for feature in candidates:
        values = [row.get(feature) for row in segments]
        block = {
            "feature": feature,
            **completeness(values),
            "usable_same_134": all(value is not None for value in values),
            "unique_values": len(set(value for value in values if value is not None)),
        }
        for group in sorted(set(row["speech_group"] for row in segments)):
            group_values = [row.get(feature) for row in segments if row["speech_group"] == group]
            group_block = completeness(group_values)
            block[f"{group}_present"] = group_block["present"]
            block[f"{group}_coverage"] = group_block["coverage"]
        feature_rows.append(block)

    same_134_features = [row["feature"] for row in feature_rows if row["usable_same_134"]]
    recommended_r018 = [
        feature
        for feature in same_134_features
        if feature not in {"LQ", "EXP", "source_audio_start", "translation_audio_start"}
    ]

    summary = {
        "input": str(RAW_PATH.relative_to(BASE)),
        "r017_predictions": str(R017_PRED_PATH.relative_to(BASE)),
        "base_rule": "same 134 feature-complete professional zh-en segments from R014/R015/R017",
        "n_raw_rater_rows": len(raw_rows),
        "n_feature_complete_rater_rows": len(rater_rows),
        "n_segments": len(segments),
        "speech_group_counts": dict(Counter(row["speech_group"] for row in segments)),
        "direction_counts": dict(Counter(row["direction"] for row in segments)),
        "same_134_features": same_134_features,
        "recommended_automatic_r018_features": recommended_r018,
        "excluded_due_to_missingness": [
            row["feature"] for row in feature_rows if not row["usable_same_134"]
        ],
        "notes": [
            "Do not include COMET_score or COMETkiwi_score unless they are recomputed for all 134 segments.",
            "Direction is constant in this cohort and should not be modeled as a coefficient.",
            "Human LQ/EXP remain oracle features; automatic deployment should prioritize pred_LQ/pred_EXP plus complete text/delay features.",
        ],
        "features": feature_rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with FEATURE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    segment_fields = [
        "segment_id",
        "speech_group",
        "direction",
        "LQ",
        "EXP",
        "LAT",
        "delay_seconds",
        "pred_LQ",
        "pred_EXP",
        "COMET_score",
        "COMETkiwi_score",
        *DERIVED_TEXT_FEATURES,
        "r017_fold",
    ]
    with SEGMENT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=segment_fields)
        writer.writeheader()
        for row in segments:
            writer.writerow({key: row.get(key) for key in segment_fields})

    print(json.dumps({
        "n_segments": len(segments),
        "same_134_features": same_134_features,
        "recommended_automatic_r018_features": recommended_r018,
        "excluded_due_to_missingness": summary["excluded_due_to_missingness"],
        "outputs": [str(SUMMARY_JSON), str(FEATURE_CSV), str(SEGMENT_CSV)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
