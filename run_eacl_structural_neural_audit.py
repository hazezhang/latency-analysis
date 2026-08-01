#!/usr/bin/env python3
"""Test whether cross-fitted neural quality carries signal beyond text structure.

For each outer speech group, structural-to-quality residualization and the
second-stage promptness model are fitted only on outer-training rows. Quality
features for those rows are themselves inner-OOF predictions.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/experiments/aaai_crossfitted_outer_quality_corrected"
OUT = ROOT / "experiments/eacl_structural_neural_audit_20260728_r2"
SEEDS = ("20260718", "20260719", "20260720")
BOOTSTRAPS = 10000
RNG_SEED = 20260728
STRUCTURE = (
    "source_length", "target_length", "length_ratio", "target_punctuation",
    "target_sentence_endings", "target_lexical_diversity", "very_short_output", "direction_en_zh",
)
DELAY = ("delay", "hinge_2", "hinge_4", "hinge_6", "hinge_10")


def features(row):
    source = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(row["src"]).lower())
    target = re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(row["mt"]).lower())
    delay = float(row["delay_seconds"])
    return {
        "delay": delay, "hinge_2": max(0.0, delay - 2.0), "hinge_4": max(0.0, delay - 4.0),
        "hinge_6": max(0.0, delay - 6.0), "hinge_10": max(0.0, delay - 10.0),
        "source_length": float(len(source)), "target_length": float(len(target)),
        "length_ratio": float(len(target) / max(len(source), 1)),
        "target_punctuation": float(len(re.findall(r"[.!?;,:。！？；，：]", str(row["mt"]))),),
        "target_sentence_endings": float(len(re.findall(r"[.!?。！？]", str(row["mt"]))),),
        "target_lexical_diversity": float(len(set(target)) / max(len(target), 1)),
        "very_short_output": float(len(target) < 5), "direction_en_zh": float(row["direction"] == "en-zh"),
    }


def matrix(rows, names):
    return np.asarray([[row[name] for name in names] for row in rows], dtype=float)


def fit_predict(x_train, y_train, x_test, alpha=1.0):
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1.0
    train = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
    test = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])
    penalty = np.eye(train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return test @ np.linalg.solve(train.T @ train + penalty, train.T @ y_train)


def corr(y, prediction, fn=pearsonr):
    if len(y) < 3 or np.std(y) == 0 or np.std(prediction) == 0:
        return None
    return float(fn(y, prediction).statistic)


def metric(y, prediction):
    return {
        "pearson": corr(y, prediction), "spearman": corr(y, prediction, spearmanr),
        "mse": float(np.mean((y - prediction) ** 2)), "mae": float(np.mean(np.abs(y - prediction))),
        "pred_std": float(np.std(prediction)),
    }


def clustered_ci(records, model):
    groups = sorted({row["speech_group"] for row in records})
    by_group = {group: [row for row in records if row["speech_group"] == group] for group in groups}
    rng = np.random.default_rng(RNG_SEED)
    values = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        sample = [row for group in rng.choice(groups, size=len(groups), replace=True) for row in by_group[group]]
        y = np.asarray([row["gold"] for row in sample])
        pred = np.asarray([row[model] for row in sample])
        values["pearson"].append(corr(y, pred))
        values["spearman"].append(corr(y, pred, spearmanr))
        values["mse"].append(float(np.mean((y - pred) ** 2)))
        values["mae"].append(float(np.mean(np.abs(y - pred))))
    return {key: [float(np.nanquantile(value, .025)), float(np.nanquantile(value, .975))] for key, value in values.items()}


def paired_ci(records, added, base):
    groups = sorted({row["speech_group"] for row in records})
    by_group = {group: [row for row in records if row["speech_group"] == group] for group in groups}
    rng = np.random.default_rng(RNG_SEED)
    deltas = defaultdict(list)
    for _ in range(BOOTSTRAPS):
        sample = [row for group in rng.choice(groups, size=len(groups), replace=True) for row in by_group[group]]
        y = np.asarray([row["gold"] for row in sample])
        a, b = np.asarray([row[added] for row in sample]), np.asarray([row[base] for row in sample])
        deltas["pearson"].append(corr(y, a) - corr(y, b))
        deltas["mse"].append(float(np.mean((y - a) ** 2) - np.mean((y - b) ** 2)))
    return {key: [float(np.nanquantile(value, .025)), float(np.nanquantile(value, .975))] for key, value in deltas.items()}


def partial_r2(records, added, base):
    y = np.asarray([row["gold"] for row in records])
    added_pred = np.asarray([row[added] for row in records])
    base_pred = np.asarray([row[base] for row in records])
    added_mse = float(np.mean((y - added_pred) ** 2))
    base_mse = float(np.mean((y - base_pred) ** 2))
    return float(1.0 - added_mse / base_mse)


def load_predictions(path):
    return {str(row["segment_id"]): (float(row["pred_LQ"]), float(row["pred_EXP"])) for row in json.loads(path.read_text(encoding="utf-8"))}


def attach_fold(seed, fold, rows):
    root = ROOT / f"experiments/aaai_crossfitted_corrected_seed_{seed}" / fold["name"]
    inner = {}
    for partition in fold["inner_folds"]:
        values = load_predictions(root / partition["name"] / "predictions.json")
        if set(inner) & set(values):
            raise ValueError(f"Duplicate inner OOF prediction in {fold['name']}")
        inner.update(values)
    outer = load_predictions(root / "final_outer" / "predictions.json")
    train, test = [], []
    for row in rows:
        current = dict(row)
        prediction = outer if row["speech_group"] == fold["outer_test_speech"] else inner
        current["pred_LQ"], current["pred_EXP"] = prediction[str(row["segment_id"])]
        (test if row["speech_group"] == fold["outer_test_speech"] else train).append(current)
    return train, test


def r2(y, prediction):
    return float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def run_seed(seed, rows, manifest):
    records, quality_records = [], []
    for fold in manifest["folds"]:
        train, test = attach_fold(seed, fold, rows)
        x_train, x_test = matrix(train, STRUCTURE), matrix(test, STRUCTURE)
        q_train, q_test = matrix(train, ("pred_LQ", "pred_EXP")), matrix(test, ("pred_LQ", "pred_EXP"))
        q_struct_train = np.column_stack([fit_predict(x_train, q_train[:, column], x_train) for column in range(2)])
        q_struct_test = np.column_stack([fit_predict(x_train, q_train[:, column], x_test) for column in range(2)])
        residual_train, residual_test = q_train - q_struct_train, q_test - q_struct_test
        y_train = np.asarray([float(row["perceived_latency"]) for row in train])
        y_test = np.asarray([float(row["perceived_latency"]) for row in test])
        structural_delay_train, structural_delay_test = matrix(train, (*STRUCTURE, *DELAY)), matrix(test, (*STRUCTURE, *DELAY))
        raw_train = np.column_stack([structural_delay_train, q_train])
        raw_test = np.column_stack([structural_delay_test, q_test])
        residualized_train = np.column_stack([structural_delay_train, residual_train])
        residualized_test = np.column_stack([structural_delay_test, residual_test])
        pred_struct = fit_predict(structural_delay_train, y_train, structural_delay_test)
        pred_raw = fit_predict(raw_train, y_train, raw_test)
        pred_resid = fit_predict(residualized_train, y_train, residualized_test)
        pred_residual_only = fit_predict(residual_train, y_train, residual_test)
        pred_struct_train = fit_predict(structural_delay_train, y_train, structural_delay_train)
        y_struct_residual_train = y_train - pred_struct_train
        pred_lat_residual = fit_predict(residual_train, y_struct_residual_train, residual_test)
        pred_partial = pred_struct + pred_lat_residual
        for row, gold, ps, pr, pz, pro, pp, q, qs in zip(test, y_test, pred_struct, pred_raw, pred_resid, pred_residual_only, pred_partial, q_test, q_struct_test):
            records.append({"segment_id": row["segment_id"], "speech_group": row["speech_group"], "interpreter": row["interpreter"], "direction": row["direction"], "gold": float(gold), "structural_delay": float(ps), "full_raw_quality": float(pr), "residualized_quality": float(pz), "residualized_quality_only": float(pro), "partial_residual_augmentation": float(pp), "pred_LQ": float(q[0]), "pred_EXP": float(q[1]), "struct_pred_LQ": float(qs[0]), "struct_pred_EXP": float(qs[1])})
            quality_records.append({"speech_group": row["speech_group"], "pred_LQ": float(q[0]), "pred_EXP": float(q[1]), "struct_pred_LQ": float(qs[0]), "struct_pred_EXP": float(qs[1])})
    return records, quality_records


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    rows = [dict(row, **features(row)) for row in json.loads((DATA / "all_lat_segments.json").read_text(encoding="utf-8"))]
    result, exported = {"protocol": {"outer_unit": "source_speech_group", "bootstrap_samples": BOOTSTRAPS, "residualization": "Structure-to-predicted-quality fits use outer-training rows only; the second-stage promptness models use inner-OOF training quality predictions."}, "seeds": {}}, []
    for seed in SEEDS:
        records, quality = run_seed(seed, rows, manifest)
        model_metrics = {}
        for name in ("structural_delay", "full_raw_quality", "residualized_quality", "residualized_quality_only", "partial_residual_augmentation"):
            y, prediction = np.asarray([row["gold"] for row in records]), np.asarray([row[name] for row in records])
            model_metrics[name] = {**metric(y, prediction), "ci95_speech_cluster": clustered_ci(records, name)}
        result["seeds"][seed] = {
            "models": model_metrics,
            "incremental_full_raw_minus_structure": paired_ci(records, "full_raw_quality", "structural_delay"),
            "incremental_residualized_minus_structure": paired_ci(records, "residualized_quality", "structural_delay"),
            "incremental_partial_residual_minus_structure": paired_ci(records, "partial_residual_augmentation", "structural_delay"),
            "partial_r2_over_structure_delay": partial_r2(records, "partial_residual_augmentation", "structural_delay"),
            "quality_explained_by_structure": {
                "LQ_r2": r2(np.asarray([row["pred_LQ"] for row in quality]), np.asarray([row["struct_pred_LQ"] for row in quality])),
                "EXP_r2": r2(np.asarray([row["pred_EXP"] for row in quality]), np.asarray([row["struct_pred_EXP"] for row in quality])),
            },
        }
        exported.extend({"seed": seed, **row} for row in records)
    with (OUT / "structural_neural_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (OUT / "structural_neural_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
