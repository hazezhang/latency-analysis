#!/usr/bin/env python3
"""Prepare professional SI evaluation data for paper experiments.

This script creates segment-level datasets from the R05/R06 rater-level files,
computes inter-rater agreement on shared segments, and writes deterministic
talk-level splits for follow-up modeling.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - handled at runtime
    pearsonr = None
    spearmanr = None


BASE = Path(__file__).parent
OUT_DIR = BASE / "experiments" / "professional"
R05_PATH = BASE / "data" / "evaluation" / "intermediate" / "professional_r05_set.json"
R06_PATH = BASE / "data" / "evaluation" / "intermediate" / "professional_r06_set.json"

RANDOM_SEED = 20260708


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_corr(xs: list[float], ys: list[float], method: str) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    if max(xs) == min(xs) or max(ys) == min(ys):
        return None
    if method == "pearson" and pearsonr:
        return float(pearsonr(xs, ys)[0])
    if method == "spearman" and spearmanr:
        return float(spearmanr(xs, ys)[0])
    return None


def quadratic_weighted_kappa(a: list[float], b: list[float], min_rating: int = 0, max_rating: int = 3) -> float | None:
    if len(a) != len(b) or not a:
        return None
    n_ratings = max_rating - min_rating + 1
    observed = [[0.0 for _ in range(n_ratings)] for _ in range(n_ratings)]
    hist_a = [0.0 for _ in range(n_ratings)]
    hist_b = [0.0 for _ in range(n_ratings)]

    for x, y in zip(a, b):
        xi = int(round(x)) - min_rating
        yi = int(round(y)) - min_rating
        if 0 <= xi < n_ratings and 0 <= yi < n_ratings:
            observed[xi][yi] += 1.0
            hist_a[xi] += 1.0
            hist_b[yi] += 1.0

    total = sum(hist_a)
    if total == 0:
        return None

    numerator = 0.0
    denominator = 0.0
    for i in range(n_ratings):
        for j in range(n_ratings):
            weight = ((i - j) ** 2) / ((n_ratings - 1) ** 2)
            expected = hist_a[i] * hist_b[j] / total
            numerator += weight * observed[i][j]
            denominator += weight * expected
    if abs(denominator) < 1e-12:
        return None
    return 1.0 - numerator / denominator


def score_metrics(r05: list[float], r06: list[float]) -> dict[str, Any]:
    diffs = [abs(x - y) for x, y in zip(r05, r06)]
    return {
        "n": len(r05),
        "pearson": safe_corr(r05, r06, "pearson"),
        "spearman": safe_corr(r05, r06, "spearman"),
        "mae": mean(diffs) if diffs else None,
        "exact_match_rate": mean([1.0 if x == y else 0.0 for x, y in zip(r05, r06)]) if r05 else None,
        "within_1_rate": mean([1.0 if abs(x - y) <= 1 else 0.0 for x, y in zip(r05, r06)]) if r05 else None,
        "quadratic_weighted_kappa": quadratic_weighted_kappa(r05, r06),
    }


def speech_group(item: dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("direction") or "unknown"),
            str(item.get("round") or "unknown"),
            str(item.get("speech") or "unknown"),
        ]
    )


def merge_comments(items: list[dict[str, Any]]) -> str | None:
    comments = []
    for item in items:
        comment = item.get("comments")
        if comment:
            comments.append(f"{item['evaluator_id']}: {comment}")
    return " | ".join(comments) if comments else None


def aggregate_segment(items: list[dict[str, Any]]) -> dict[str, Any]:
    base = items[0]
    lqs = [float(item["LQ"]) for item in items if item.get("LQ") is not None]
    exps = [float(item["EXP"]) for item in items if item.get("EXP") is not None]
    lats = [
        float(item["perceived_latency_mean"])
        for item in items
        if item.get("perceived_latency_mean") is not None
    ]
    raters = [item["evaluator_id"] for item in items]

    entry = {
        "segment_id": base["segment_id"],
        "evaluator_id": "_".join(raters),
        "src": base["src"],
        "mt": base["mt"],
        "delay_seconds": base.get("delay_seconds"),
        "comments": merge_comments(items),
        "flag_uncertain": None,
        "COMETkiwi_score": next((item.get("COMETkiwi_score") for item in items if item.get("COMETkiwi_score") is not None), None),
        "COMET_score": next((item.get("COMET_score") for item in items if item.get("COMET_score") is not None), None),
        "offline_mt_en": base.get("offline_mt_en"),
        "perceived_latency_mean": round(mean(lats), 3) if lats else None,
        "LQ_gap": round(max(lqs) - min(lqs), 3) if len(lqs) > 1 else 0.0,
        "EXP_gap": round(max(exps) - min(exps), 3) if len(exps) > 1 else 0.0,
        "latency_gap": round(max(lats) - min(lats), 3) if len(lats) > 1 else 0.0,
        "num_raters": len(raters),
        "raters": raters,
        "LQ": round(mean(lqs), 3) if lqs else None,
        "EXP": round(mean(exps), 3) if exps else None,
        "direction": base.get("direction"),
        "interpreter": base.get("interpreter"),
        "speech": base.get("speech"),
        "round": base.get("round"),
        "speech_group": speech_group(base),
    }
    return entry


def split_by_group(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        groups[item["speech_group"]].append(item)

    by_direction: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        direction = group.split("|", 1)[0]
        by_direction[direction].append(group)

    rng = random.Random(RANDOM_SEED)
    split_groups = {"train": set(), "dev": set(), "test": set()}
    for direction, group_names in by_direction.items():
        rng.shuffle(group_names)
        n = len(group_names)
        n_test = max(1, round(n * 0.15))
        n_dev = max(1, round(n * 0.15)) if n >= 3 else 0
        split_groups["test"].update(group_names[:n_test])
        split_groups["dev"].update(group_names[n_test : n_test + n_dev])
        split_groups["train"].update(group_names[n_test + n_dev :])

    splits = {"train": [], "dev": [], "test": []}
    for split, names in split_groups.items():
        for name in sorted(names):
            splits[split].extend(groups[name])
        splits[split].sort(key=lambda item: item["segment_id"])
    return splits


def summarize(data: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(data),
        "unique_segments": len({item["segment_id"] for item in data}),
        "directions": dict(Counter(item.get("direction") for item in data)),
        "num_raters": dict(Counter(item.get("num_raters") for item in data)),
        "speech_groups": len({item.get("speech_group") for item in data}),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    r05 = load_json(R05_PATH)
    r06 = load_json(R06_PATH)
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in r05 + r06:
        by_id[item["segment_id"]].append(item)

    shared_ids = sorted(segment_id for segment_id, items in by_id.items() if len(items) == 2)
    all_ids = sorted(by_id)
    shared_segment = [aggregate_segment(by_id[segment_id]) for segment_id in shared_ids]
    all_segment = [aggregate_segment(by_id[segment_id]) for segment_id in all_ids]

    r05_by_id = {item["segment_id"]: item for item in r05}
    r06_by_id = {item["segment_id"]: item for item in r06}
    agreement = {}
    for field, key in [
        ("LQ", "LQ"),
        ("EXP", "EXP"),
        ("LAT", "perceived_latency_mean"),
    ]:
        xs = [float(r05_by_id[segment_id][key]) for segment_id in shared_ids if r05_by_id[segment_id].get(key) is not None and r06_by_id[segment_id].get(key) is not None]
        ys = [float(r06_by_id[segment_id][key]) for segment_id in shared_ids if r05_by_id[segment_id].get(key) is not None and r06_by_id[segment_id].get(key) is not None]
        agreement[field] = score_metrics(xs, ys)

    write_json(OUT_DIR / "professional_shared_segment_set.json", shared_segment)
    write_json(OUT_DIR / "professional_all_segment_set.json", all_segment)

    split_report = {}
    for label, dataset in [("shared", shared_segment), ("all", all_segment)]:
        splits = split_by_group(dataset)
        split_report[label] = {split: summarize(rows) for split, rows in splits.items()}
        for split, rows in splits.items():
            write_json(OUT_DIR / f"professional_{label}_{split}.json", rows)

    report = {
        "seed": RANDOM_SEED,
        "rater_level": {
            "R05_rows": len(r05),
            "R06_rows": len(r06),
            "shared_segments": len(shared_ids),
            "R06_only_segments": len(set(r06_by_id) - set(r05_by_id)),
        },
        "segment_level": {
            "shared": summarize(shared_segment),
            "all": summarize(all_segment),
        },
        "rater_agreement": agreement,
        "splits": split_report,
    }
    write_json(OUT_DIR / "professional_experiment_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
