#!/usr/bin/env python3
"""Evaluate professional promptness as a shared but evaluator-calibrated target.

All predictive analyses leave one source-speech group out. This audit uses
human LQ/EXP labels to examine construct validity; it is not a substitute for
the automatic quality-estimator experiments.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/evaluation/profess_eval_delay_enriched_namespace_corrected.json"
COHORT = ROOT / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"
OUT = ROOT / "experiments/eacl_rater_calibration_audit_20260728"
RATERS = ("R05", "R06")
KNOTS = (2.0, 4.0, 6.0, 10.0)
BOOTSTRAPS = 1000
SEED = 20260728


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def stable_id(row):
    return f"{str(row.get('file_id')).zfill(3)}:{row.get('original_segment_id') or row.get('segment_id')}"


def token_features(row):
    source = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(row["src"]).lower())
    target = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(row["mt"]).lower())
    delay = float(row["delay_seconds"])
    return {
        "delay": delay,
        "hinge_2": max(0.0, delay - 2.0),
        "hinge_4": max(0.0, delay - 4.0),
        "hinge_6": max(0.0, delay - 6.0),
        "hinge_10": max(0.0, delay - 10.0),
        "source_length": float(len(source)),
        "target_length": float(len(target)),
        "length_ratio": float(len(target) / max(len(source), 1)),
        "target_punctuation": float(len(re.findall(r"[.!?;,:。！？；，：]", str(row["mt"]))),),
        "target_sentence_endings": float(len(re.findall(r"[.!?。！？]", str(row["mt"]))),),
        "target_lexical_diversity": float(len(set(target)) / max(len(target), 1)),
        "very_short_output": float(len(target) < 5),
        "direction_en_zh": float(row["direction"] == "en-zh"),
    }


def load_rows():
    cohort = {str(row["segment_id"]): dict(row) for row in json.loads(COHORT.read_text(encoding="utf-8"))}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in json.loads(RAW.read_text(encoding="utf-8")):
        rater = str(row.get("evaluator_id"))
        values = [number(row.get(key)) for key in ("LQ", "EXP", "perceived_latency", "delay_seconds")]
        if rater not in RATERS or any(value is None for value in values):
            continue
        if not 0.0 <= values[3] <= 20.0:
            continue
        grouped[stable_id(row)][rater].append(values)

    rows = []
    for sid, by_rater in sorted(grouped.items()):
        if sid not in cohort or any(rater not in by_rater for rater in RATERS):
            continue
        base = dict(cohort[sid])
        for rater in RATERS:
            lq, exp, lat, _ = np.mean(np.asarray(by_rater[rater], dtype=float), axis=0)
            base[f"{rater}_LQ"] = float(lq)
            base[f"{rater}_EXP"] = float(exp)
            base[f"{rater}_LAT"] = float(lat)
        base.update(token_features(base))
        rows.append(base)
    if len(rows) != len(cohort):
        raise ValueError(f"Expected {len(cohort)} jointly rated cohort rows, found {len(rows)}")
    return rows


STRUCTURE = (
    "source_length", "target_length", "length_ratio", "target_punctuation",
    "target_sentence_endings", "target_lexical_diversity", "very_short_output", "direction_en_zh",
)
DELAY = ("delay", "hinge_2", "hinge_4", "hinge_6", "hinge_10")


def pearson(y, prediction):
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float(pearsonr(y, prediction).statistic)


def spearman(y, prediction):
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float(spearmanr(y, prediction).statistic)


def ridge_predict(x_train, y_train, x_test, alpha=1.0):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    train = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
    test = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])
    penalty = np.eye(train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return test @ np.linalg.solve(train.T @ train + penalty, train.T @ y_train)


def matrix(items, names):
    return np.asarray([[row[name] for name in names] for row in items], dtype=float)


def grouped_cv(rows, features, target):
    prediction = np.full(len(rows), np.nan)
    for group in sorted({row["speech_group"] for row in rows}):
        train_i = [i for i, row in enumerate(rows) if row["speech_group"] != group]
        test_i = [i for i, row in enumerate(rows) if row["speech_group"] == group]
        prediction[test_i] = ridge_predict(
            matrix([rows[i] for i in train_i], features),
            np.asarray([rows[i][target] for i in train_i], dtype=float),
            matrix([rows[i] for i in test_i], features),
        )
    return prediction


def cluster_metrics(rows, prediction, target):
    y = np.asarray([row[target] for row in rows], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    groups = sorted({row["speech_group"] for row in rows})
    by_group = {group: np.asarray([i for i, row in enumerate(rows) if row["speech_group"] == group]) for group in groups}
    rng = np.random.default_rng(SEED)
    samples = {"pearson": [], "spearman": [], "mse": [], "mae": []}
    for _ in range(BOOTSTRAPS):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[group] for group in chosen])
        samples["pearson"].append(pearson(y[indices], prediction[indices]))
        samples["spearman"].append(spearman(y[indices], prediction[indices]))
        samples["mse"].append(float(np.mean((y[indices] - prediction[indices]) ** 2)))
        samples["mae"].append(float(np.mean(np.abs(y[indices] - prediction[indices]))))
    calibration = np.linalg.lstsq(np.column_stack([np.ones(len(y)), prediction]), y, rcond=None)[0]
    result = {
        "n": int(len(y)),
        "pearson": pearson(y, prediction),
        "spearman": spearman(y, prediction),
        "mse": float(np.mean((y - prediction) ** 2)),
        "mae": float(np.mean(np.abs(y - prediction))),
        "pred_std": float(np.std(prediction)),
        "calibration_intercept": float(calibration[0]),
        "calibration_slope": float(calibration[1]),
        "ci95_speech_cluster": {
            name: [float(np.nanquantile(values, .025)), float(np.nanquantile(values, .975))]
            for name, values in samples.items()
        },
    }
    return result


def quadratic_weighted_kappa(left, right):
    values = (0.0, 1.0, 2.0, 3.0)
    counts = np.zeros((4, 4), dtype=int)
    for a, b in zip(left, right):
        counts[values.index(float(a)), values.index(float(b))] += 1
    observed = counts / counts.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.fromfunction(lambda i, j: ((i - j) / 3.0) ** 2, observed.shape)
    return 1.0 - float(np.sum(weights * observed)) / float(np.sum(weights * expected)), counts


def icc_2_1(left, right):
    values = np.column_stack([left, right])
    n, k = values.shape
    grand = values.mean()
    row_means, col_means = values.mean(axis=1), values.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = values - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    return float((ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n))


def interrater(rows):
    left = np.asarray([row["R05_LAT"] for row in rows])
    right = np.asarray([row["R06_LAT"] for row in rows])
    kappa, confusion = quadratic_weighted_kappa(left, right)
    return {
        "n": int(len(rows)),
        "r05_distribution": dict(sorted(Counter(map(float, left)).items())),
        "r06_distribution": dict(sorted(Counter(map(float, right)).items())),
        "exact_agreement": float(np.mean(left == right)),
        "within_one_point_agreement": float(np.mean(np.abs(left - right) <= 1.0)),
        "pearson": pearson(left, right),
        "spearman": spearman(left, right),
        "icc_2_1": icc_2_1(left, right),
        "quadratic_weighted_kappa": float(kappa),
        "confusion_rows_r05_columns_r06": confusion.tolist(),
        "score_order": [0, 1, 2, 3],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows = load_rows()
    specs = {
        "delay_only": DELAY,
        "same_rater_human_quality": None,
        "same_rater_human_quality_delay": None,
        "structural_delay": (*STRUCTURE, *DELAY),
        "full_human_quality_structural_delay": None,
    }
    result = {
        "protocol": {
            "outer_unit": "source_speech_group",
            "clusters": len({row["speech_group"] for row in rows}),
            "bootstrap_samples": BOOTSTRAPS,
            "cohort_n": len(rows),
            "leakage_rule": "Every target-fitting and feature-scaling operation uses only outer-training speech groups.",
        },
        "interrater_promptness": interrater(rows),
        "within_rater_human_label_models": {},
        "cross_rater_human_label_models": {},
        "direction_results": {},
    }
    exported = []
    for rater in RATERS:
        features_by_spec = dict(specs)
        features_by_spec["same_rater_human_quality"] = (f"{rater}_LQ", f"{rater}_EXP")
        features_by_spec["same_rater_human_quality_delay"] = (*features_by_spec["same_rater_human_quality"], *DELAY)
        features_by_spec["full_human_quality_structural_delay"] = (*features_by_spec["same_rater_human_quality"], *STRUCTURE, *DELAY)
        target = f"{rater}_LAT"
        result["within_rater_human_label_models"][rater] = {}
        for name, features in features_by_spec.items():
            prediction = grouped_cv(rows, features, target)
            result["within_rater_human_label_models"][rater][name] = cluster_metrics(rows, prediction, target)
            exported.extend({"analysis": "within_rater", "quality_rater": rater, "target_rater": rater, "model": name, "segment_id": row["segment_id"], "speech_group": row["speech_group"], "direction": row["direction"], "gold": row[target], "prediction": float(value)} for row, value in zip(rows, prediction))
            if name in {"delay_only", "same_rater_human_quality_delay", "structural_delay", "full_human_quality_structural_delay"}:
                result["direction_results"].setdefault(rater, {})[name] = {}
                for direction in sorted({row["direction"] for row in rows}):
                    indices = [i for i, row in enumerate(rows) if row["direction"] == direction]
                    subset = [rows[i] for i in indices]
                    result["direction_results"][rater][name][direction] = cluster_metrics(subset, prediction[indices], target)

    for quality_rater in RATERS:
        for target_rater in RATERS:
            name = f"{quality_rater}_quality_to_{target_rater}_promptness"
            features = (f"{quality_rater}_LQ", f"{quality_rater}_EXP", *DELAY)
            target = f"{target_rater}_LAT"
            prediction = grouped_cv(rows, features, target)
            result["cross_rater_human_label_models"][name] = cluster_metrics(rows, prediction, target)
            exported.extend({"analysis": "cross_rater", "quality_rater": quality_rater, "target_rater": target_rater, "model": "quality_delay", "segment_id": row["segment_id"], "speech_group": row["speech_group"], "direction": row["direction"], "gold": row[target], "prediction": float(value)} for row, value in zip(rows, prediction))

    (OUT / "rater_calibration_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "rater_calibration_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
