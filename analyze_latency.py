#!/usr/bin/env python3
"""Exploratory latency analysis for the follow-up paper.

This script intentionally avoids COMET fine-tuning. It reports rater
agreement, severity differences, and simple Pearson/Ridge-style analyses
that can be discussed before committing to a larger modeling experiment.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


BASE = Path(__file__).parent
OUT = BASE / "latency_analysis_results.json"


def load(path: str) -> list[dict]:
    return json.loads((BASE / path).read_text(encoding="utf-8"))


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return round(float(np.sum(x * y) / den), 4) if den else None


def spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x = np.asarray([p[0] for p in pairs])
    y = np.asarray([p[1] for p in pairs])
    return pearson(x.argsort().argsort(), y.argsort().argsort())


def corr_table(rows: list[dict], pairs: list[tuple[str, str]]) -> dict:
    out = {}
    for a, b in pairs:
        out[f"{a}__{b}"] = {
            "pearson": pearson([r.get(a) for r in rows], [r.get(b) for r in rows]),
            "spearman": spearman([r.get(a) for r in rows], [r.get(b) for r in rows]),
            "n": sum(r.get(a) is not None and r.get(b) is not None for r in rows),
        }
    return out


def aggregate(rows: list[dict], value_key: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if row.get("segment_id") is not None and value is not None:
            grouped[str(row["segment_id"])].append(float(value))
    return {key: float(mean(values)) for key, values in grouped.items()}


def ridge_predict(train_x, train_y, test_x, alpha=1.0):
    """Standardized ridge with an intercept, implemented with NumPy only."""
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x = (train_x - mu) / sd
    z = (test_x - mu) / sd
    x1 = np.column_stack([np.ones(len(x)), x])
    z1 = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(x1.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x1.T @ x1 + penalty, x1.T @ train_y)
    return z1 @ beta


def safe_float(value):
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def main():
    student = load("data/evaluation/student_eval.json")
    professional = load("data/evaluation/profess_eval.json")
    dev = load("dev_set.json")
    test = load("test_set.json")

    # Normalize the evaluator-specific files to the names used in the paper.
    for row in student:
        row["LQ"] = safe_float(row.get("language_quality"))
        row["EXP"] = safe_float(row.get("expressiveness"))
        row["LAT"] = safe_float(row.get("perceived_latency"))
    for row in professional:
        row["LAT"] = safe_float(row.get("perceived_latency"))

    prof_by_id = defaultdict(list)
    for row in professional:
        prof_by_id[str(row["segment_id"])].append(row)
    prof_seg = []
    for sid, items in prof_by_id.items():
        first = items[0]
        entry = {"segment_id": sid, "direction": first.get("direction")}
        for key in ("LQ", "EXP", "LAT", "delay_seconds"):
            vals = [safe_float(x.get(key)) for x in items]
            vals = [x for x in vals if x is not None]
            entry[key] = float(mean(vals)) if vals else None
        prof_seg.append(entry)

    # Student aggregates are used only for agreement analysis, not as a new split.
    student_by_id = defaultdict(list)
    for row in student:
        student_by_id[str(row["segment_id"])].append(row)
    student_seg = []
    for sid, items in student_by_id.items():
        entry = {"segment_id": sid}
        for key in ("LQ", "EXP", "LAT"):
            vals = [x.get(key) for x in items if x.get(key) is not None]
            entry[key] = float(mean(vals)) if vals else None
        student_seg.append(entry)

    prof_by_sid = {x["segment_id"]: x for x in prof_seg}
    stud_by_sid = {x["segment_id"]: x for x in student_seg}
    common = []
    for sid in sorted(set(prof_by_sid) & set(stud_by_sid)):
        common.append({
            "segment_id": sid,
            "student_LQ": stud_by_sid[sid].get("LQ"),
            "professional_LQ": prof_by_sid[sid].get("LQ"),
            "student_EXP": stud_by_sid[sid].get("EXP"),
            "professional_EXP": prof_by_sid[sid].get("EXP"),
            "student_LAT": stud_by_sid[sid].get("LAT"),
            "professional_LAT": prof_by_sid[sid].get("LAT"),
        })

    # The original split contains the cleanest comparable latency labels.
    split_rows = [r for r in dev + test if r.get("perceived_latency_mean") is not None]
    for row in split_rows:
        row["LAT"] = safe_float(row.get("perceived_latency_mean"))
        row["LQ"] = safe_float(row.get("LQ"))
        row["EXP"] = safe_float(row.get("EXP"))
        row["delay"] = safe_float(row.get("delay_seconds"))
        row["COMET"] = safe_float(row.get("COMET_score"))
        row["KIWI"] = safe_float(row.get("COMETkiwi_score"))

    result = {
        "data_counts": {
            "student_rating_rows": len(student),
            "student_segments": len(student_seg),
            "professional_rating_rows": len(professional),
            "professional_segments": len(prof_seg),
            "student_professional_common_segments": len(common),
            "original_dev_test_latency_rows": len(split_rows),
        },
        "student_professional_agreement": {
            "LQ": {"pearson": pearson([x["student_LQ"] for x in common], [x["professional_LQ"] for x in common]), "spearman": spearman([x["student_LQ"] for x in common], [x["professional_LQ"] for x in common])},
            "EXP": {"pearson": pearson([x["student_EXP"] for x in common], [x["professional_EXP"] for x in common]), "spearman": spearman([x["student_EXP"] for x in common], [x["professional_EXP"] for x in common])},
            "LAT": {"pearson": pearson([x["student_LAT"] for x in common], [x["professional_LAT"] for x in common]), "spearman": spearman([x["student_LAT"] for x in common], [x["professional_LAT"] for x in common])},
        },
        "professional_correlations": corr_table(prof_seg, [("LQ", "LAT"), ("EXP", "LAT"), ("LQ", "EXP"), ("delay_seconds", "LAT")]),
        "original_split_correlations": corr_table(split_rows, [("delay", "LAT"), ("LQ", "LAT"), ("EXP", "LAT"), ("COMET", "LAT"), ("KIWI", "LAT")]),
        "rater_severity": {},
    }

    for evaluator in sorted({str(x.get("evaluator_id")) for x in professional}):
        rows = [x for x in professional if str(x.get("evaluator_id")) == evaluator]
        result["rater_severity"][evaluator] = {
            "n": len(rows),
            "LQ_mean": round(mean(float(x["LQ"]) for x in rows if x.get("LQ") is not None), 4),
            "EXP_mean": round(mean(float(x["EXP"]) for x in rows if x.get("EXP") is not None), 4),
            "LAT_mean": round(mean(float(x["LAT"]) for x in rows if x.get("LAT") is not None), 4),
        }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
