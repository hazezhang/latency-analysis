#!/usr/bin/env python3
"""P0 construct-validity, residual, and cohort audits for the AAAI paper.

All predictive analyses are source-speech-group held out. Bootstrap intervals
resample complete source-speech groups rather than individual segments.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data/experiments/aaai_crossfitted_outer_quality_corrected"
RAW_PATH = ROOT / "data/evaluation/profess_eval_delay_enriched_namespace_corrected.json"
OUT = ROOT / "experiments/aaai_p0_construct_validity_20260722"
SEEDS = ("20260718", "20260719", "20260720")
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20260722
FEATURES = (
    "LAT",
    "LQ",
    "EXP",
    "quality_mean",
    "delay_seconds",
    "source_length",
    "target_length",
    "length_ratio",
)
DELAY_FEATURES = ("delay", "hinge_2", "hinge_4", "hinge_6", "hinge_10")


def number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def stable_id(row):
    return f"{row.get('file_id') or 'unknown'}:{row.get('original_segment_id') or row.get('segment_id') or 'unknown'}"


def lexical_length(text):
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text)))


def safe_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(pearsonr(x, y).statistic)


def center_values(rows, key, group_key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    if group_key is None:
        return values
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row[group_key]].append(index)
    centered = values.copy()
    for indices in groups.values():
        centered[indices] -= values[indices].mean()
    return centered


def clustered_ci(rows, statistic, cluster_key="speech_group", samples=BOOTSTRAPS):
    groups = defaultdict(list)
    for row in rows:
        groups[row[cluster_key]].append(row)
    names = sorted(groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(samples):
        sampled = rng.choice(names, size=len(names), replace=True)
        sample_rows = [row for name in sampled for row in groups[name]]
        value = statistic(sample_rows)
        if value is not None and np.isfinite(value):
            values.append(value)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def correlation_result(rows, left, right, center_by=None):
    def statistic(items):
        return safe_corr(center_values(items, left, center_by), center_values(items, right, center_by))

    return {
        "pearson": statistic(rows),
        "ci95_speech_cluster": clustered_ci(rows, statistic),
        "centered_by": center_by,
    }


def correlation_matrix(rows, center_by=None):
    output = {}
    for i, left in enumerate(FEATURES):
        for right in FEATURES[i + 1 :]:
            output[f"{left}__{right}"] = correlation_result(rows, left, right, center_by)
    return output


def residualize(y, controls):
    y = np.asarray(y, dtype=float)
    x = np.asarray(controls, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    design = np.column_stack([np.ones(len(x)), x])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def partial_corr(rows, left, right, controls):
    def statistic(items):
        x = [row[left] for row in items]
        y = [row[right] for row in items]
        z = [[row[key] for key in controls] for row in items]
        return safe_corr(residualize(x, z), residualize(y, z))

    return {
        "left": left,
        "right": right,
        "controls": list(controls),
        "pearson": statistic(rows),
        "ci95_speech_cluster": clustered_ci(rows, statistic),
    }


def vector(row, features):
    delay = row["delay_seconds"]
    values = {
        "LQ": row["LQ"],
        "EXP": row["EXP"],
        "pred_LQ": row.get("pred_LQ"),
        "pred_EXP": row.get("pred_EXP"),
        "delay": delay,
        "hinge_2": max(0.0, delay - 2.0),
        "hinge_4": max(0.0, delay - 4.0),
        "hinge_6": max(0.0, delay - 6.0),
        "hinge_10": max(0.0, delay - 10.0),
    }
    return np.asarray([values[name] for name in features], dtype=float)


def ridge_fit_predict(train_rows, test_rows, features, target_key="LAT", alpha=1.0):
    x_train = np.vstack([vector(row, features) for row in train_rows])
    x_test = np.vstack([vector(row, features) for row in test_rows])
    y_train = np.asarray([row[target_key] for row in train_rows], dtype=float)
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
    x_test = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)
    return x_test @ beta


def metric_dict(gold, prediction):
    gold, prediction = np.asarray(gold, dtype=float), np.asarray(prediction, dtype=float)
    denominator = np.sum((gold - gold.mean()) ** 2)
    return {
        "n": len(gold),
        "pearson": safe_corr(gold, prediction),
        "mse": float(np.mean((gold - prediction) ** 2)),
        "mae": float(np.mean(np.abs(gold - prediction))),
        "r2": float(1.0 - np.sum((gold - prediction) ** 2) / denominator),
        "pred_mean": float(prediction.mean()),
        "pred_std": float(prediction.std()),
    }


def load_shared_rows():
    rows = json.loads((DATA_ROOT / "all_lat_segments.json").read_text(encoding="utf-8"))
    output = []
    for row in rows:
        source_length = lexical_length(row["src"])
        target_length = lexical_length(row["mt"])
        item = dict(row)
        item.update({
            "LAT": float(row["perceived_latency"]),
            "LQ": float(row["LQ"]),
            "EXP": float(row["EXP"]),
            "quality_mean": (float(row["LQ"]) + float(row["EXP"])) / 2.0,
            "delay_seconds": float(row["delay_seconds"]),
            "source_length": float(source_length),
            "target_length": float(target_length),
            "length_ratio": float(target_length / max(source_length, 1)),
        })
        output.append(item)
    return output


def load_evaluator_rows(shared_rows):
    shared = {row["segment_id"]: row for row in shared_rows}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in json.loads(RAW_PATH.read_text(encoding="utf-8")):
        sid = stable_id(row)
        rater = str(row.get("evaluator_id"))
        if sid not in shared or rater not in {"R05", "R06"}:
            continue
        values = [number(row.get(key)) for key in ("LQ", "EXP", "perceived_latency")]
        if any(value is None for value in values):
            continue
        grouped[sid][rater].append(values)
    output = {"R05": [], "R06": []}
    for sid, base in shared.items():
        for rater in output:
            values = grouped[sid][rater]
            if not values:
                raise ValueError(f"Missing evaluator row: {sid}/{rater}")
            means = np.mean(np.asarray(values, dtype=float), axis=0)
            item = dict(base)
            item.update({
                "LQ": float(means[0]),
                "EXP": float(means[1]),
                "LAT": float(means[2]),
                "quality_mean": float((means[0] + means[1]) / 2.0),
            })
            output[rater].append(item)
    return output


def cross_validated_human_models(rows):
    predictions = {"LQ_EXP": [], "LQ_EXP_delay": []}
    for outer in sorted({row["speech_group"] for row in rows}):
        train = [row for row in rows if row["speech_group"] != outer]
        test = [row for row in rows if row["speech_group"] == outer]
        for name, features in {
            "LQ_EXP": ("LQ", "EXP"),
            "LQ_EXP_delay": ("LQ", "EXP", *DELAY_FEATURES),
        }.items():
            pred = ridge_fit_predict(train, test, features)
            predictions[name].extend({
                "segment_id": row["segment_id"],
                "speech_group": row["speech_group"],
                "gold": row["LAT"],
                "prediction": float(value),
            } for row, value in zip(test, pred))
    return predictions


def incremental_bootstrap(base_rows, added_rows):
    base = {row["segment_id"]: row for row in base_rows}
    added = {row["segment_id"]: row for row in added_rows}
    groups = defaultdict(list)
    for sid, row in base.items():
        groups[row["speech_group"]].append(sid)
    names = sorted(groups)

    def delta(segment_ids):
        y = np.asarray([base[sid]["gold"] for sid in segment_ids])
        b = np.asarray([base[sid]["prediction"] for sid in segment_ids])
        a = np.asarray([added[sid]["prediction"] for sid in segment_ids])
        total = np.sum((y - y.mean()) ** 2)
        return (
            safe_corr(y, a) - safe_corr(y, b),
            float(np.mean((y - a) ** 2) - np.mean((y - b) ** 2)),
            float((1 - np.sum((y - a) ** 2) / total) - (1 - np.sum((y - b) ** 2) / total)),
        )

    point = delta(sorted(base))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAPS):
        selected = rng.choice(names, size=len(names), replace=True)
        samples.append(delta([sid for group in selected for sid in groups[group]]))
    array = np.asarray(samples)
    return {
        "definition": "LQ+EXP+delay minus LQ+EXP",
        "delta_pearson": {"estimate": point[0], "ci95": np.quantile(array[:, 0], [0.025, 0.975]).tolist()},
        "delta_mse": {"estimate": point[1], "ci95": np.quantile(array[:, 1], [0.025, 0.975]).tolist()},
        "delta_r2": {"estimate": point[2], "ci95": np.quantile(array[:, 2], [0.025, 0.975]).tolist()},
        "resampling_unit": "source_speech_group",
        "bootstrap_samples": BOOTSTRAPS,
    }


def residual_delay_analysis(rows, quality_features):
    records = []
    for outer in sorted({row["speech_group"] for row in rows}):
        train = [row for row in rows if row["speech_group"] != outer]
        test = [row for row in rows if row["speech_group"] == outer]
        oof_residual_rows = []
        for held_out in sorted({row["speech_group"] for row in train}):
            inner_train = [row for row in train if row["speech_group"] != held_out]
            inner_test = [row for row in train if row["speech_group"] == held_out]
            quality_prediction = ridge_fit_predict(inner_train, inner_test, quality_features)
            for row, value in zip(inner_test, quality_prediction):
                item = dict(row)
                item["residual"] = row["LAT"] - float(value)
                oof_residual_rows.append(item)
        residual_prediction = ridge_fit_predict(
            oof_residual_rows, test, DELAY_FEATURES, target_key="residual"
        )
        outer_quality = ridge_fit_predict(train, test, quality_features)
        for row, quality_pred, residual_pred in zip(test, outer_quality, residual_prediction):
            records.append({
                "segment_id": row["segment_id"],
                "speech_group": row["speech_group"],
                "gold_residual": row["LAT"] - float(quality_pred),
                "predicted_residual_from_delay": float(residual_pred),
            })
    gold = [row["gold_residual"] for row in records]
    pred = [row["predicted_residual_from_delay"] for row in records]
    metrics = metric_dict(gold, pred)
    metrics["zero_baseline_mse"] = float(np.mean(np.asarray(gold) ** 2))
    metrics["delta_mse_vs_zero"] = metrics["mse"] - metrics["zero_baseline_mse"]
    return records, metrics


def load_quality_features_for_outer(seed, fold, all_rows):
    root = ROOT / f"experiments/aaai_crossfitted_corrected_seed_{seed}" / fold["name"]
    train_predictions = {}
    for inner in fold["inner_folds"]:
        path = root / inner["name"] / "predictions.json"
        for row in json.loads(path.read_text(encoding="utf-8")):
            sid = str(row["segment_id"])
            if sid in train_predictions:
                raise ValueError(f"Duplicate inner prediction: {seed}/{fold['name']}/{sid}")
            train_predictions[sid] = (float(row["pred_LQ"]), float(row["pred_EXP"]))
    outer_predictions = {
        str(row["segment_id"]): (float(row["pred_LQ"]), float(row["pred_EXP"]))
        for row in json.loads((root / "final_outer" / "predictions.json").read_text(encoding="utf-8"))
    }
    train, test = [], []
    for row in all_rows:
        item = dict(row)
        source = outer_predictions if row["speech_group"] == fold["outer_test_speech"] else train_predictions
        pred_lq, pred_exp = source[row["segment_id"]]
        item.update({"pred_LQ": pred_lq, "pred_EXP": pred_exp})
        (test if row["speech_group"] == fold["outer_test_speech"] else train).append(item)
    return train, test


def automatic_residual_delay_analysis(rows, manifest, seed):
    records = []
    for fold in manifest["folds"]:
        train, test = load_quality_features_for_outer(seed, fold, rows)
        oof_residual_rows = []
        for held_out in sorted({row["speech_group"] for row in train}):
            inner_train = [row for row in train if row["speech_group"] != held_out]
            inner_test = [row for row in train if row["speech_group"] == held_out]
            pred = ridge_fit_predict(inner_train, inner_test, ("pred_LQ", "pred_EXP"))
            for row, value in zip(inner_test, pred):
                item = dict(row)
                item["residual"] = row["LAT"] - float(value)
                oof_residual_rows.append(item)
        delay_residual = ridge_fit_predict(oof_residual_rows, test, DELAY_FEATURES, target_key="residual")
        quality_outer = ridge_fit_predict(train, test, ("pred_LQ", "pred_EXP"))
        for row, q_pred, d_pred in zip(test, quality_outer, delay_residual):
            records.append({
                "segment_id": row["segment_id"],
                "speech_group": row["speech_group"],
                "gold_residual": row["LAT"] - float(q_pred),
                "predicted_residual_from_delay": float(d_pred),
            })
    gold = [row["gold_residual"] for row in records]
    pred = [row["predicted_residual_from_delay"] for row in records]
    metrics = metric_dict(gold, pred)
    metrics["zero_baseline_mse"] = float(np.mean(np.asarray(gold) ** 2))
    metrics["delta_mse_vs_zero"] = metrics["mse"] - metrics["zero_baseline_mse"]
    return records, metrics


def descriptive_tables(rows):
    interpreters = []
    for name in sorted({row["interpreter"] for row in rows}):
        items = [row for row in rows if row["interpreter"] == name]
        record = {
            "interpreter": name,
            "n_segments": len(items),
            "n_speech_groups": len({row["speech_group"] for row in items}),
            "directions": "+".join(sorted({row["direction"] for row in items})),
        }
        for key in ("LQ", "EXP", "LAT", "delay_seconds"):
            values = np.asarray([row[key] for row in items], dtype=float)
            record.update({f"{key}_mean": values.mean(), f"{key}_std": values.std(), f"{key}_min": values.min(), f"{key}_max": values.max()})
        interpreters.append(record)
    speeches = []
    for name in sorted({row["speech_group"] for row in rows}):
        items = [row for row in rows if row["speech_group"] == name]
        speeches.append({
            "speech_group": name,
            "n_segments": len(items),
            "n_interpreters": len({row["interpreter"] for row in items}),
            "interpreters": "+".join(sorted({row["interpreter"] for row in items})),
            "direction": "+".join(sorted({row["direction"] for row in items})),
            "LAT_mean": float(np.mean([row["LAT"] for row in items])),
            "delay_mean": float(np.mean([row["delay_seconds"] for row in items])),
        })
    incidence = []
    for interpreter in sorted({row["interpreter"] for row in rows}):
        counts = Counter(row["speech_group"] for row in rows if row["interpreter"] == interpreter)
        for speech in sorted({row["speech_group"] for row in rows}):
            incidence.append({"interpreter": interpreter, "speech_group": speech, "n_segments": counts[speech]})
    return interpreters, speeches, incidence


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_shared_rows()
    evaluators = load_evaluator_rows(rows)
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))

    matrices = {
        "pooled": correlation_matrix(rows),
        "within_interpreter_centered": correlation_matrix(rows, "interpreter"),
        "within_speech_group_centered": correlation_matrix(rows, "speech_group"),
        "evaluator_R05": correlation_matrix(evaluators["R05"]),
        "evaluator_R06": correlation_matrix(evaluators["R06"]),
    }
    partial = {
        "LAT_delay_given_LQ_EXP": partial_corr(rows, "LAT", "delay_seconds", ("LQ", "EXP")),
        "LAT_LQ_given_EXP_delay": partial_corr(rows, "LAT", "LQ", ("EXP", "delay_seconds")),
        "LAT_EXP_given_LQ_delay": partial_corr(rows, "LAT", "EXP", ("LQ", "delay_seconds")),
        "LAT_quality_mean_given_delay": partial_corr(rows, "LAT", "quality_mean", ("delay_seconds",)),
    }

    human_predictions = cross_validated_human_models(rows)
    human_metrics = {name: metric_dict([row["gold"] for row in items], [row["prediction"] for row in items]) for name, items in human_predictions.items()}
    incremental = incremental_bootstrap(human_predictions["LQ_EXP"], human_predictions["LQ_EXP_delay"])
    human_residual_rows, human_residual_metrics = residual_delay_analysis(rows, ("LQ", "EXP"))

    automatic_residual = {}
    for seed in SEEDS:
        records, metrics = automatic_residual_delay_analysis(rows, manifest, seed)
        automatic_residual[seed] = metrics
        write_csv(OUT / f"automatic_residual_delay_predictions_{seed}.csv", records)

    interpreters, speeches, incidence = descriptive_tables(rows)
    write_csv(OUT / "interpreter_statistics.csv", interpreters)
    write_csv(OUT / "speech_group_statistics.csv", speeches)
    write_csv(OUT / "interpreter_speech_incidence.csv", incidence)
    write_csv(OUT / "human_LQ_EXP_predictions.csv", human_predictions["LQ_EXP"])
    write_csv(OUT / "human_LQ_EXP_delay_predictions.csv", human_predictions["LQ_EXP_delay"])
    write_csv(OUT / "human_residual_delay_predictions.csv", human_residual_rows)

    payload = {
        "protocol": {
            "n_segments": len(rows),
            "n_speech_groups": len({row["speech_group"] for row in rows}),
            "n_interpreters": len({row["interpreter"] for row in rows}),
            "lexical_length_definition": "count each CJK character and each alphanumeric word as one lexical unit",
            "bootstrap": {"unit": "source_speech_group", "samples": BOOTSTRAPS, "ci": "percentile_95", "seed": BOOTSTRAP_SEED},
            "residual_analysis": "outer speech held out; training residuals are generated by inner leave-one-speech-group-out quality regressions",
        },
        "correlation_matrices": matrices,
        "partial_correlations": partial,
        "human_incremental_validity": {"models": human_metrics, "increment": incremental},
        "residual_delay": {
            "after_human_LQ_EXP": human_residual_metrics,
            "after_cross_fitted_predicted_LQ_EXP": automatic_residual,
        },
    }
    (OUT / "p0_construct_validity_results.json").write_text(
        json.dumps(rounded(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rounded(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
