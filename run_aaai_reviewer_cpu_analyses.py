#!/usr/bin/env python3
"""CPU analyses requested by an AAAI-style review of the LAT paper.

This script does not replace the three-seed automatic bridge. It audits the
human-label construct and runs lightweight, speech-held-out baselines that do
not require neural-model retraining.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/evaluation/profess_eval_delay_enriched_namespace_corrected.json"
FROZEN = ROOT / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"
OUT = ROOT / "experiments/aaai_reviewer_cpu_corrected_20260721"
RATERS = ("R05", "R06")
FIXED_KNOTS = (2.0, 4.0, 6.0, 10.0)
ALT_KNOTS = (1.0, 3.0, 5.0, 8.0, 12.0)
BOOTSTRAPS = 5000
SEED = 20260721


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def stable_id(row):
    return f"{row.get('file_id') or 'unknown'}:{row.get('original_segment_id') or row.get('segment_id') or 'unknown'}"


def mode(values):
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], str(value)))[0]


def corr(y, pred, method="pearson"):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) < 3 or np.std(y) == 0 or np.std(pred) == 0:
        return None
    value = pearsonr(y, pred).statistic if method == "pearson" else spearmanr(y, pred).statistic
    return float(value)


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "n": int(len(y)),
        "pearson": round(corr(y, pred), 4),
        "spearman": round(corr(y, pred, "spearman"), 4),
        "mse": round(float(np.mean((y - pred) ** 2)), 4),
        "mae": round(float(np.mean(np.abs(y - pred))), 4),
        "pred_std": round(float(np.std(pred)), 4),
    }


def aggregate_balanced(rows):
    grouped = defaultdict(list)
    for raw in rows:
        values = {key: number(raw.get(key)) for key in ("LQ", "EXP", "perceived_latency", "delay_seconds")}
        if any(value is None for value in values.values()):
            continue
        if str(raw.get("evaluator_id")) not in RATERS:
            continue
        item = dict(raw)
        item.update(values)
        grouped[stable_id(item)].append(item)

    output = []
    repeated = []
    conflicts = []
    for sample_id, items in sorted(grouped.items()):
        by_rater = defaultdict(list)
        for item in items:
            by_rater[str(item.get("evaluator_id"))].append(item)
        if any(rater not in by_rater for rater in RATERS):
            continue
        if any(len(by_rater[rater]) > 1 for rater in RATERS):
            repeated.append({"segment_id": sample_id, "counts": {r: len(by_rater[r]) for r in RATERS}})

        rater_means = {}
        for rater in RATERS:
            rater_means[rater] = {
                key: float(np.mean([row[key] for row in by_rater[rater]]))
                for key in ("LQ", "EXP", "perceived_latency", "delay_seconds")
            }
        speeches = [str(item.get("speech") or item.get("source_file") or item.get("file_id")) for item in items]
        interpreters = [str(item.get("interpreter") or "unknown").strip().casefold() for item in items]
        if len(set(speeches)) > 1 or len(set(interpreters)) > 1:
            conflicts.append({
                "segment_id": sample_id,
                "speech_values": sorted(set(speeches)),
                "interpreter_values": sorted(set(interpreters)),
            })
        delay = np.mean([rater_means[r]["delay_seconds"] for r in RATERS])
        if not 0.0 <= delay <= 20.0:
            continue
        output.append({
            "segment_id": sample_id,
            "speech_group": mode(speeches),
            "interpreter": mode(interpreters),
            "direction": mode([str(item.get("direction") or "unknown") for item in items]),
            "delay_seconds": float(delay),
            "LQ": float(np.mean([rater_means[r]["LQ"] for r in RATERS])),
            "EXP": float(np.mean([rater_means[r]["EXP"] for r in RATERS])),
            "LAT": float(np.mean([rater_means[r]["perceived_latency"] for r in RATERS])),
            "R05_LQ": rater_means["R05"]["LQ"],
            "R05_EXP": rater_means["R05"]["EXP"],
            "R05_LAT": rater_means["R05"]["perceived_latency"],
            "R06_LQ": rater_means["R06"]["LQ"],
            "R06_EXP": rater_means["R06"]["EXP"],
            "R06_LAT": rater_means["R06"]["perceived_latency"],
        })
    return output, repeated, conflicts


def weighted_kappa(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    values = sorted(set(left) | set(right))
    index = {value: i for i, value in enumerate(values)}
    observed = np.zeros((len(values), len(values)), dtype=float)
    for a, b in zip(left, right):
        observed[index[a], index[b]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if len(values) == 1:
        return 1.0
    weights = np.fromfunction(lambda i, j: ((i - j) / (len(values) - 1)) ** 2, observed.shape)
    denominator = float(np.sum(weights * expected))
    return 1.0 - float(np.sum(weights * observed)) / denominator if denominator else None


def icc_two_way_absolute(values):
    x = np.asarray(values, dtype=float)
    n, k = x.shape
    grand = x.mean()
    row_mean = x.mean(axis=1)
    col_mean = x.mean(axis=0)
    ms_rows = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_mean - grand) ** 2) / (k - 1)
    residual = x - row_mean[:, None] - col_mean[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    icc_single = (ms_rows - ms_error) / (
        ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    )
    icc_average = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    return float(icc_single), float(icc_average)


def interrater(rows):
    result = {}
    for outcome in ("LQ", "EXP", "LAT"):
        left = np.asarray([row[f"R05_{outcome}"] for row in rows])
        right = np.asarray([row[f"R06_{outcome}"] for row in rows])
        icc_single, icc_average = icc_two_way_absolute(np.column_stack([left, right]))
        result[outcome] = {
            "n": len(rows),
            "pearson": round(corr(left, right), 4),
            "spearman": round(corr(left, right, "spearman"), 4),
            "mae": round(float(np.mean(np.abs(left - right))), 4),
            "mean_gap_R06_minus_R05": round(float(np.mean(right - left)), 4),
            "quadratic_weighted_kappa": round(weighted_kappa(left, right), 4),
            "ICC_2_1": round(icc_single, 4),
            "ICC_2_2": round(icc_average, 4),
        }
    return result


def base_feature_matrix(rows, keys):
    return np.asarray([[row[key] for key in keys] for row in rows], dtype=float)


def piecewise(delay, knots):
    delay = np.asarray(delay, dtype=float)
    return np.column_stack([delay] + [np.maximum(0.0, delay - knot) for knot in knots])


def fixed_bins(delay):
    delay = np.asarray(delay, dtype=float)
    edges = (0.0, 2.0, 4.0, 6.0, 10.0, 20.000001)
    indices = np.digitize(delay, edges[1:-1], right=False)
    return np.eye(5)[indices]


def build_features(train, test, spec):
    d_train = np.asarray([row["delay_seconds"] for row in train])
    d_test = np.asarray([row["delay_seconds"] for row in test])
    if spec == "delay_linear":
        return d_train[:, None], d_test[:, None]
    if spec == "delay_quadratic":
        return np.column_stack([d_train, d_train ** 2]), np.column_stack([d_test, d_test ** 2])
    if spec == "delay_log":
        return np.log1p(d_train)[:, None], np.log1p(d_test)[:, None]
    if spec == "delay_piecewise_fixed":
        return piecewise(d_train, FIXED_KNOTS), piecewise(d_test, FIXED_KNOTS)
    if spec == "delay_piecewise_alt":
        return piecewise(d_train, ALT_KNOTS), piecewise(d_test, ALT_KNOTS)
    if spec == "delay_piecewise_quantile":
        knots = np.quantile(d_train, [0.2, 0.4, 0.6, 0.8])
        return piecewise(d_train, knots), piecewise(d_test, knots)
    if spec == "delay_bins":
        return fixed_bins(d_train), fixed_bins(d_test)
    if spec == "LQ_only":
        return base_feature_matrix(train, ["LQ"]), base_feature_matrix(test, ["LQ"])
    if spec == "EXP_only":
        return base_feature_matrix(train, ["EXP"]), base_feature_matrix(test, ["EXP"])
    if spec == "quality_mean":
        return np.asarray([[(r["LQ"] + r["EXP"]) / 2] for r in train]), np.asarray([[(r["LQ"] + r["EXP"]) / 2] for r in test])
    if spec == "quality_PCA1":
        x_train = base_feature_matrix(train, ["LQ", "EXP"])
        x_test = base_feature_matrix(test, ["LQ", "EXP"])
        mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
        scale[scale == 0] = 1.0
        z_train, z_test = (x_train - mean) / scale, (x_test - mean) / scale
        _, _, vt = np.linalg.svd(z_train, full_matrices=False)
        component = vt[0]
        if component.sum() < 0:
            component = -component
        return (z_train @ component)[:, None], (z_test @ component)[:, None]
    if spec == "LQ_EXP":
        return base_feature_matrix(train, ["LQ", "EXP"]), base_feature_matrix(test, ["LQ", "EXP"])
    if spec == "quality_mean_piecewise":
        q_train, q_test = build_features(train, test, "quality_mean")
        d_train_x, d_test_x = build_features(train, test, "delay_piecewise_fixed")
        return np.column_stack([q_train, d_train_x]), np.column_stack([q_test, d_test_x])
    if spec == "LQ_EXP_piecewise":
        q_train, q_test = build_features(train, test, "LQ_EXP")
        d_train_x, d_test_x = build_features(train, test, "delay_piecewise_fixed")
        return np.column_stack([q_train, d_train_x]), np.column_stack([q_test, d_test_x])
    if spec == "interpreter_identity":
        labels = sorted({row["interpreter"] for row in train})
        index = {label: i for i, label in enumerate(labels)}
        def encode(items):
            x = np.zeros((len(items), len(labels)))
            for i, row in enumerate(items):
                if row["interpreter"] in index:
                    x[i, index[row["interpreter"]]] = 1.0
            return x
        return encode(train), encode(test)
    raise ValueError(spec)


def ridge_predict(x_train, y_train, x_test, alpha=1.0):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_test @ coef


def grouped_cv(rows, spec, group_key="speech_group", target="LAT"):
    groups = sorted({row[group_key] for row in rows})
    predictions = np.full(len(rows), np.nan)
    for group in groups:
        train_idx = [i for i, row in enumerate(rows) if row[group_key] != group]
        test_idx = [i for i, row in enumerate(rows) if row[group_key] == group]
        train = [rows[i] for i in train_idx]
        test = [rows[i] for i in test_idx]
        x_train, x_test = build_features(train, test, spec)
        y_train = np.asarray([row[target] for row in train])
        predictions[test_idx] = ridge_predict(x_train, y_train, x_test)
    y = np.asarray([row[target] for row in rows])
    return predictions, metrics(y, predictions)


def rater_features(items, rater, include_delay):
    quality = np.asarray([[row[f"{rater}_LQ"], row[f"{rater}_EXP"]] for row in items])
    if not include_delay:
        return quality
    delay = piecewise([row["delay_seconds"] for row in items], FIXED_KNOTS)
    return np.column_stack([quality, delay])


def cross_rater_cv(rows, train_rater, test_rater, include_delay=True):
    predictions = np.full(len(rows), np.nan)
    for group in sorted({row["speech_group"] for row in rows}):
        train_idx = [i for i, row in enumerate(rows) if row["speech_group"] != group]
        test_idx = [i for i, row in enumerate(rows) if row["speech_group"] == group]
        train = [rows[i] for i in train_idx]
        test = [rows[i] for i in test_idx]
        x_train = rater_features(train, train_rater, include_delay)
        x_test = rater_features(test, test_rater, include_delay)
        y_train = np.asarray([row[f"{train_rater}_LAT"] for row in train])
        predictions[test_idx] = ridge_predict(x_train, y_train, x_test)
    y = np.asarray([row[f"{test_rater}_LAT"] for row in rows])
    return metrics(y, predictions)


def subgroup_metrics(rows, predictions, key):
    result = {}
    for value in sorted({row[key] for row in rows}):
        idx = [i for i, row in enumerate(rows) if row[key] == value]
        y = [rows[i]["LAT"] for i in idx]
        pred = [predictions[i] for i in idx]
        result[value] = metrics(y, pred) if len(idx) >= 3 else {"n": len(idx)}
    return result


def macro_speech_pearson(rows, predictions):
    values = []
    details = {}
    for group in sorted({row["speech_group"] for row in rows}):
        idx = [i for i, row in enumerate(rows) if row["speech_group"] == group]
        value = corr([rows[i]["LAT"] for i in idx], [predictions[i] for i in idx])
        details[group] = round(value, 4) if value is not None else None
        if value is not None:
            values.append(value)
    return {"n_valid_groups": len(values), "macro_pearson": round(float(np.mean(values)), 4), "per_group": details}


def cluster_bootstrap_delta(rows, pred_a, pred_b, n=BOOTSTRAPS):
    groups = sorted({row["speech_group"] for row in rows})
    by_group = {group: [i for i, row in enumerate(rows) if row["speech_group"] == group] for group in groups}
    rng = np.random.default_rng(SEED)
    delta_r, delta_mse = [], []
    for _ in range(n):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = [i for group in sampled for i in by_group[group]]
        y = np.asarray([rows[i]["LAT"] for i in idx])
        a = np.asarray([pred_a[i] for i in idx])
        b = np.asarray([pred_b[i] for i in idx])
        ra, rb = corr(y, a), corr(y, b)
        if ra is not None and rb is not None:
            delta_r.append(ra - rb)
        delta_mse.append(float(np.mean((y - a) ** 2) - np.mean((y - b) ** 2)))
    quantiles = lambda values: [round(float(np.quantile(values, q)), 4) for q in (0.025, 0.5, 0.975)]
    return {
        "definition": "model_a minus model_b; positive delta_r and negative delta_mse favor model_a",
        "delta_pearson_ci95_median": quantiles(delta_r),
        "delta_mse_ci95_median": quantiles(delta_mse),
        "bootstrap_speech_groups": n,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    rows, repeated, conflicts = aggregate_balanced(raw)
    frozen = {row["segment_id"]: row for row in json.loads(FROZEN.read_text(encoding="utf-8"))}
    if len(rows) != 622:
        raise ValueError(f"Expected 622 balanced segments, got {len(rows)}")

    deltas = []
    for row in rows:
        old = frozen[row["segment_id"]]
        deltas.append({
            "segment_id": row["segment_id"],
            "LQ_delta_balanced_minus_frozen": row["LQ"] - old["LQ"],
            "EXP_delta_balanced_minus_frozen": row["EXP"] - old["EXP"],
            "LAT_delta_balanced_minus_frozen": row["LAT"] - old["perceived_latency"],
        })

    specs = [
        "delay_linear", "delay_quadratic", "delay_log", "delay_piecewise_fixed",
        "delay_piecewise_alt", "delay_piecewise_quantile", "delay_bins",
        "interpreter_identity", "LQ_only", "EXP_only", "quality_mean",
        "quality_PCA1", "LQ_EXP", "quality_mean_piecewise", "LQ_EXP_piecewise",
    ]
    cv_results, predictions = {}, {}
    for spec in specs:
        pred, result = grouped_cv(rows, spec)
        predictions[spec] = pred
        cv_results[spec] = result

    loio_results = {}
    for spec in ("delay_piecewise_fixed", "LQ_EXP", "LQ_EXP_piecewise"):
        _, loio_results[spec] = grouped_cv(rows, spec, group_key="interpreter")

    per_rater = {}
    for rater in RATERS:
        target = f"{rater}_LAT"
        own_quality = []
        for row in rows:
            copy = dict(row)
            copy["LQ"], copy["EXP"] = row[f"{rater}_LQ"], row[f"{rater}_EXP"]
            own_quality.append(copy)
        per_rater[rater] = {}
        for spec in ("delay_piecewise_fixed", "LQ_EXP", "LQ_EXP_piecewise"):
            _, per_rater[rater][spec] = grouped_cv(own_quality, spec, target=target)

    cross_rater = {
        "R05_to_R06_quality_delay": cross_rater_cv(rows, "R05", "R06", True),
        "R06_to_R05_quality_delay": cross_rater_cv(rows, "R06", "R05", True),
        "R05_to_R06_quality_only": cross_rater_cv(rows, "R05", "R06", False),
        "R06_to_R05_quality_only": cross_rater_cv(rows, "R06", "R05", False),
    }

    primary = "LQ_EXP_piecewise"
    largest_group = Counter(row["speech_group"] for row in rows).most_common(1)[0]
    keep = [i for i, row in enumerate(rows) if row["speech_group"] != largest_group[0]]
    largest_sensitivity = {
        "excluded_group": largest_group[0],
        "excluded_n": largest_group[1],
        "delay_piecewise": metrics([rows[i]["LAT"] for i in keep], predictions["delay_piecewise_fixed"][keep]),
        "quality_delay": metrics([rows[i]["LAT"] for i in keep], predictions[primary][keep]),
    }

    result = {
        "status": "CPU reviewer-requested analyses; automatic neural bridge significance still requires current raw cross-fitted predictions",
        "cohort": {
            "n_segments": len(rows),
            "n_speech_groups": len({row["speech_group"] for row in rows}),
            "n_interpreters": len({row["interpreter"] for row in rows}),
            "direction_counts": Counter(row["direction"] for row in rows),
            "repeated_judgment_segments": repeated,
            "metadata_conflicts": conflicts,
        },
        "balanced_label_impact": {
            "n_changed_LQ": sum(abs(row["LQ_delta_balanced_minus_frozen"]) > 1e-9 for row in deltas),
            "n_changed_EXP": sum(abs(row["EXP_delta_balanced_minus_frozen"]) > 1e-9 for row in deltas),
            "n_changed_LAT": sum(abs(row["LAT_delta_balanced_minus_frozen"]) > 1e-9 for row in deltas),
            "max_abs_LQ_delta": round(max(abs(row["LQ_delta_balanced_minus_frozen"]) for row in deltas), 4),
            "max_abs_EXP_delta": round(max(abs(row["EXP_delta_balanced_minus_frozen"]) for row in deltas), 4),
            "max_abs_LAT_delta": round(max(abs(row["LAT_delta_balanced_minus_frozen"]) for row in deltas), 4),
        },
        "interrater_construct": interrater(rows),
        "speech_held_out_baselines": cv_results,
        "unseen_interpreter_sensitivity": loio_results,
        "per_rater_targets": per_rater,
        "cross_rater_transfer": cross_rater,
        "direction_results": {
            spec: subgroup_metrics(rows, predictions[spec], "direction")
            for spec in ("delay_piecewise_fixed", "LQ_EXP", "LQ_EXP_piecewise")
        },
        "speech_macro_results": {
            spec: macro_speech_pearson(rows, predictions[spec])
            for spec in ("delay_piecewise_fixed", "quality_mean", "LQ_EXP", "LQ_EXP_piecewise")
        },
        "largest_group_sensitivity": largest_sensitivity,
        "cluster_bootstrap_human_reference": {
            "LQ_EXP_piecewise_vs_delay_piecewise": cluster_bootstrap_delta(
                rows, predictions["LQ_EXP_piecewise"], predictions["delay_piecewise_fixed"]
            ),
            "LQ_EXP_vs_quality_mean": cluster_bootstrap_delta(
                rows, predictions["LQ_EXP"], predictions["quality_mean"]
            ),
            "LQ_EXP_vs_quality_PCA1": cluster_bootstrap_delta(
                rows, predictions["LQ_EXP"], predictions["quality_PCA1"]
            ),
        },
    }

    (OUT / "aaai_reviewer_cpu_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT / "balanced_label_deltas.csv", deltas)
    write_csv(OUT / "speech_held_out_baselines.csv", [
        {"model": model, **values} for model, values in cv_results.items()
    ])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
