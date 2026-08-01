#!/usr/bin/env python3
"""Stable, segment-level latency analysis for the follow-up paper.

Outputs bootstrap confidence intervals for:
  1. professional-only LQ/EXP/LAT relationships;
  2. student/professional agreement on shared segments;
  3. R05/R06 raw and rater-scale-calibrated comparisons.

All bootstrap resampling is performed at segment level, never at rating-row
level, so repeated ratings from one segment cannot artificially narrow CIs.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


BASE = Path(__file__).parent
SEED = 20260711
BOOTSTRAPS = 5000
RESULT_JSON = BASE / "latency_stability_results.json"
RESULT_CSV = BASE / "latency_stability_table.csv"


def load(relpath: str) -> list[dict]:
    return json.loads((BASE / relpath).read_text(encoding="utf-8"))


def number(value):
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3:
        return None
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / den) if den else None


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3:
        return None
    # Ties are common on the 0-3 rubric; average ranks are important here.
    def average_ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and values[order[j]] == values[order[i]]:
                j += 1
            ranks[order[i:j]] = (i + j - 1) / 2.0
            i = j
        return ranks
    return pearson(average_ranks(x), average_ranks(y))


def metric(x, y, method):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    a, b = zip(*pairs)
    return pearson(a, b) if method == "pearson" else spearman(a, b)


def bootstrap(rows, x_key, y_key, method="pearson", seed=SEED, n=BOOTSTRAPS):
    usable = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    if len(usable) < 3:
        return {"n": len(usable), "estimate": None, "ci95": [None, None], "bootstrap_valid": 0}
    x = np.asarray([r[x_key] for r in usable], dtype=float)
    y = np.asarray([r[y_key] for r in usable], dtype=float)
    estimate = metric(x, y, method)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(usable), size=len(usable))
        value = metric(x[idx], y[idx], method)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        ci = [None, None]
    else:
        ci = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {
        "n": len(usable),
        "estimate": round(estimate, 4) if estimate is not None else None,
        "ci95": [round(v, 4) if v is not None else None for v in ci],
        "bootstrap_valid": len(values),
    }


def aggregate_rows(rows, key_map, group_key="segment_id"):
    grouped = defaultdict(list)
    for row in rows:
        sid = row.get(group_key)
        if sid is not None:
            grouped[str(sid)].append(row)
    output = []
    for sid, items in sorted(grouped.items()):
        item = {"segment_id": sid}
        for out_key, source_key in key_map.items():
            values = [number(x.get(source_key)) for x in items]
            values = [x for x in values if x is not None]
            item[out_key] = float(mean(values)) if values else None
        for key in ("direction", "evaluator_id", "round", "speech"):
            values = [x.get(key) for x in items if x.get(key) not in (None, "")]
            if values:
                item[key] = values[0]
        output.append(item)
    return output


def text_key(row):
    if row.get("_source_text_for_match") is not None and row.get("_target_text_for_match") is not None:
        return row["_source_text_for_match"], row["_target_text_for_match"]
    source = row.get("src") or row.get("source_chinese") or row.get("source_english") or ""
    target = row.get("mt") or row.get("target_english") or row.get("target_chinese") or ""
    return " ".join(str(source).split()).casefold(), " ".join(str(target).split()).casefold()


def count_multitext_segment_ids(rows):
    grouped = defaultdict(set)
    for row in rows:
        sid = row.get("segment_id")
        if sid is not None:
            grouped[str(sid)].add(text_key(row))
    return sum(1 for values in grouped.values() if len(values) > 1)


def aggregate_student_for_professional_text(student, prof_seg):
    grouped = defaultdict(list)
    for row in student:
        sid = row.get("segment_id")
        if sid is not None:
            grouped[(str(sid), text_key(row))].append(row)

    output = []
    for prof in prof_seg:
        sid = prof["segment_id"]
        items = grouped.get((sid, text_key(prof)), [])
        if not items:
            continue
        item = {"segment_id": sid}
        for out_key, source_key in {
            "LQ": "language_quality",
            "EXP": "expressiveness",
            "LAT": "perceived_latency",
        }.items():
            values = [number(x.get(source_key)) for x in items]
            values = [x for x in values if x is not None]
            item[out_key] = float(mean(values)) if values else None
        output.append(item)
    return output


def attach_professional_text(professional, prof_seg):
    text_by_id = {}
    for row in professional:
        sid = row.get("segment_id")
        if sid is not None and str(sid) not in text_by_id:
            text_by_id[str(sid)] = {
                "_source_text_for_match": text_key(row)[0],
                "_target_text_for_match": text_key(row)[1],
            }
    for row in prof_seg:
        row.update(text_by_id.get(row["segment_id"], {}))


def paired_rater_rows(professional):
    grouped = defaultdict(dict)
    for row in professional:
        sid = str(row["segment_id"])
        evaluator = str(row.get("evaluator_id"))
        grouped[sid][evaluator] = row
    output = []
    for sid, items in sorted(grouped.items()):
        if "R05" not in items or "R06" not in items:
            continue
        output.append({
            "segment_id": sid,
            "R05_LQ": number(items["R05"].get("LQ")),
            "R06_LQ": number(items["R06"].get("LQ")),
            "R05_EXP": number(items["R05"].get("EXP")),
            "R06_EXP": number(items["R06"].get("EXP")),
            "R05_LAT": number(items["R05"].get("perceived_latency")),
            "R06_LAT": number(items["R06"].get("perceived_latency")),
        })
    return output


def calibration(rows, a_key, b_key):
    usable = [r for r in rows if r.get(a_key) is not None and r.get(b_key) is not None]
    if not usable:
        return {"n": 0}
    a = np.asarray([r[a_key] for r in usable], dtype=float)
    b = np.asarray([r[b_key] for r in usable], dtype=float)
    result = {"n": len(usable)}
    mean_difference = float(np.mean(b - a))
    a_std = float(a.std()) or 1.0
    b_std = float(b.std()) or 1.0
    for label, x, y in [
        ("raw", a, b),
        ("centered", a - a.mean(), b - b.mean()),
        ("zscore", (a - a.mean()) / (a.std() or 1.0), (b - b.mean()) / (b.std() or 1.0)),
    ]:
        if label == "centered":
            comparison_mae = float(np.mean(np.abs((b - mean_difference) - a)))
        elif label == "zscore":
            comparison_mae = float(np.mean(np.abs((b - b.mean()) / b_std - (a - a.mean()) / a_std)))
        else:
            comparison_mae = float(np.mean(np.abs(a - b)))
        result[label] = {
            "pearson": round(pearson(x, y), 4),
            "spearman": round(spearman(x, y), 4),
            "mean_a": round(float(x.mean()), 4),
            "mean_b": round(float(y.mean()), 4),
            "mae_after_scale_alignment": round(comparison_mae, 4),
        }
    result["mean_difference_b_minus_a"] = round(mean_difference, 4)
    result["mae_raw"] = round(float(np.mean(np.abs(a - b))), 4)
    return result


def calibration_stability(rows, a_key, b_key, seed=SEED, n=BOOTSTRAPS):
    usable = [r for r in rows if r.get(a_key) is not None and r.get(b_key) is not None]
    if len(usable) < 3:
        return {"n": len(usable)}
    a = np.asarray([r[a_key] for r in usable], dtype=float)
    b = np.asarray([r[b_key] for r in usable], dtype=float)
    rng = np.random.default_rng(seed)
    pearsons, spearmans, mean_diffs, maes = [], [], [], []
    for _ in range(n):
        idx = rng.integers(0, len(usable), size=len(usable))
        x, y = a[idx], b[idx]
        p, s = pearson(x, y), spearman(x, y)
        if p is not None:
            pearsons.append(p)
        if s is not None:
            spearmans.append(s)
        mean_diffs.append(float(np.mean(y - x)))
        maes.append(float(np.mean(np.abs(y - x))))
    point = calibration(usable, a_key, b_key)
    q = lambda values: [round(float(np.quantile(values, 0.025)), 4), round(float(np.quantile(values, 0.975)), 4)]
    point["bootstrap_ci95"] = {
        "pearson": q(pearsons),
        "spearman": q(spearmans),
        "mean_difference_b_minus_a": q(mean_diffs),
        "mae_raw": q(maes),
    }
    return point


def group_latency(rows, group_name, group_value):
    selected = [r for r in rows if r.get(group_name) == group_value]
    return {
        "n_segments": len(selected),
        "LQ_LAT": bootstrap(selected, "LQ", "LAT"),
        "EXP_LAT": bootstrap(selected, "EXP", "LAT"),
        "LQ_EXP": bootstrap(selected, "LQ", "EXP"),
    }


def main():
    student = load("data/evaluation/student_eval.json")
    professional = load("data/evaluation/profess_eval_delay_enriched.json")

    student_seg = aggregate_rows(student, {
        "LQ": "language_quality",
        "EXP": "expressiveness",
        "LAT": "perceived_latency",
    })
    prof_seg = aggregate_rows(professional, {
        "LQ": "LQ",
        "EXP": "EXP",
        "LAT": "perceived_latency",
        "delay": "delay_seconds",
    })
    attach_professional_text(professional, prof_seg)

    # Shared-segment agreement uses independently aggregated student ratings
    # and professional ratings, avoiding rating-row duplication.
    student_by_id = {r["segment_id"]: r for r in student_seg}
    prof_by_id = {r["segment_id"]: r for r in prof_seg}
    agreement = []
    for sid in sorted(set(student_by_id) & set(prof_by_id)):
        s, p = student_by_id[sid], prof_by_id[sid]
        agreement.append({
            "segment_id": sid,
            "student_LQ": s.get("LQ"), "professional_LQ": p.get("LQ"),
            "student_EXP": s.get("EXP"), "professional_EXP": p.get("EXP"),
            "student_LAT": s.get("LAT"), "professional_LAT": p.get("LAT"),
        })

    student_text_seg = aggregate_student_for_professional_text(student, prof_seg)
    student_text_by_id = {r["segment_id"]: r for r in student_text_seg}
    agreement_text_matched = []
    for sid in sorted(set(student_text_by_id) & set(prof_by_id)):
        s, p = student_text_by_id[sid], prof_by_id[sid]
        agreement_text_matched.append({
            "segment_id": sid,
            "student_LQ": s.get("LQ"), "professional_LQ": p.get("LQ"),
            "student_EXP": s.get("EXP"), "professional_EXP": p.get("EXP"),
            "student_LAT": s.get("LAT"), "professional_LAT": p.get("LAT"),
        })

    paired = paired_rater_rows(professional)
    result = {
        "metadata": {
            "seed": SEED,
            "bootstraps": BOOTSTRAPS,
            "unit": "segment",
            "student_rating_rows": len(student),
            "student_segments": len(student_seg),
            "professional_rating_rows": len(professional),
            "professional_segments": len(prof_seg),
            "student_professional_shared_segments": len(agreement),
            "student_professional_text_matched_segments": len(agreement_text_matched),
            "student_multitext_segment_ids": count_multitext_segment_ids(student),
            "professional_multitext_segment_ids": count_multitext_segment_ids(professional),
            "R05_R06_shared_segments": len(paired),
        },
        "professional_only_overall": {
            "n_segments": len(prof_seg),
            "LQ_LAT": bootstrap(prof_seg, "LQ", "LAT"),
            "EXP_LAT": bootstrap(prof_seg, "EXP", "LAT"),
            "LQ_EXP": bootstrap(prof_seg, "LQ", "EXP"),
            "delay_LAT": bootstrap(prof_seg, "delay", "LAT"),
        },
        "professional_by_direction": {},
        "student_professional_agreement": {
            "LQ": bootstrap(agreement, "student_LQ", "professional_LQ"),
            "EXP": bootstrap(agreement, "student_EXP", "professional_EXP"),
            "LAT": bootstrap(agreement, "student_LAT", "professional_LAT"),
        },
        "student_professional_text_matched_agreement": {
            "LQ": bootstrap(agreement_text_matched, "student_LQ", "professional_LQ"),
            "EXP": bootstrap(agreement_text_matched, "student_EXP", "professional_EXP"),
            "LAT": bootstrap(agreement_text_matched, "student_LAT", "professional_LAT"),
        },
        "R05_R06_calibration": {
            "LQ": calibration_stability(paired, "R05_LQ", "R06_LQ"),
            "EXP": calibration_stability(paired, "R05_EXP", "R06_EXP"),
            "LAT": calibration_stability(paired, "R05_LAT", "R06_LAT"),
        },
    }

    for direction in sorted({r.get("direction") for r in prof_seg if r.get("direction")}):
        result["professional_by_direction"][direction] = group_latency(prof_seg, "direction", direction)

    # Compact CSV for direct inclusion in a meeting slide or a paper table.
    table_rows = []
    def add(section, metric_name, stats):
        table_rows.append({
            "section": section,
            "metric": metric_name,
            "n": stats.get("n", stats.get("n_segments")),
            "estimate": stats.get("estimate"),
            "ci_low": (stats.get("ci95") or [None, None])[0],
            "ci_high": (stats.get("ci95") or [None, None])[1],
        })
    for section, block in [("professional_overall", result["professional_only_overall"])]:
        for name in ("LQ_LAT", "EXP_LAT", "LQ_EXP", "delay_LAT"):
            add(section, name, block[name])
    for name, stats in result["student_professional_agreement"].items():
        add("student_professional_agreement", name, stats)
    for name, stats in result["student_professional_text_matched_agreement"].items():
        add("student_professional_text_matched_agreement", name, stats)
    for direction, block in result["professional_by_direction"].items():
        for name in ("LQ_LAT", "EXP_LAT", "LQ_EXP"):
            add(f"professional_direction_{direction}", name, block[name])
    RESULT_CSV.write_text("", encoding="utf-8")
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nWrote {RESULT_JSON}")
    print(f"Wrote {RESULT_CSV}")


if __name__ == "__main__":
    main()
