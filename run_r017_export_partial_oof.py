#!/usr/bin/env python3
"""Export R017 out-of-fold quality predictions from partial checkpoints.

This script is intentionally separate from evaluate.py because the remote run
keeps only LoRA/regressor weights locally, not the full 2.2GB COMET state dict.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import (
    COMETDataset,
    COMETModelWithHeads,
    LoraConfig,
    TaskType,
    compute_baseline,
    download_model,
    load_from_checkpoint,
)


FOLDS = [
    "fold_01_speech_1",
    "fold_02_speech_2",
    "fold_03_speech_3",
    "fold_04_speech_6182lawyer_3",
    "fold_05_speech_athlete_2",
]


def corr_or_nan(fn, y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(fn(y_true, y_pred)[0])


def summarize_predictions(rows):
    y_lq = np.array([r["human_LQ"] for r in rows], dtype=float)
    p_lq = np.array([r["pred_LQ"] for r in rows], dtype=float)
    y_exp = np.array([r["human_EXP"] for r in rows], dtype=float)
    p_exp = np.array([r["pred_EXP"] for r in rows], dtype=float)
    return {
        "n": int(len(rows)),
        "lq_pearson": corr_or_nan(pearsonr, y_lq, p_lq),
        "lq_spearman": corr_or_nan(spearmanr, y_lq, p_lq),
        "lq_mse": float(np.mean((p_lq - y_lq) ** 2)),
        "lq_pred_std": float(np.std(p_lq)),
        "exp_pearson": corr_or_nan(pearsonr, y_exp, p_exp),
        "exp_spearman": corr_or_nan(spearmanr, y_exp, p_exp),
        "exp_mse": float(np.mean((p_exp - y_exp) ** 2)),
        "exp_pred_std": float(np.std(p_exp)),
    }


def load_partial(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in {checkpoint_path}: {unexpected[:10]}")
    return {
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "best_score": ckpt.get("best_score") if isinstance(ckpt, dict) else None,
        "partial_keys": ckpt.get("partial_keys", len(state_dict)) if isinstance(ckpt, dict) else len(state_dict),
        "missing_keys": len(incompatible.missing_keys),
    }


def predict_fold(model, data_path, tokenizer, device, max_length, batch_size):
    dataset = COMETDataset(str(data_path), tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    pred_lq = []
    pred_exp = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predict {data_path.parent.name}"):
            out_lq, out_exp = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            pred_lq.extend(out_lq.cpu().tolist())
            pred_exp.extend(out_exp.cpu().tolist())

    rows = []
    for item, lq, exp in zip(dataset.data, pred_lq, pred_exp):
        row = dict(item)
        row["human_LQ"] = float(item["LQ"])
        row["human_EXP"] = float(item["EXP"])
        row["pred_LQ"] = round(float(lq), 6)
        row["pred_EXP"] = round(float(exp), 6)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/experiments/r017_oof_quality")
    parser.add_argument("--run-root", default="experiments/r017_oof_quality_20260712")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    data_root = Path(args.data_root)
    run_root = Path(args.run_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hf_token = None
    if Path(".hf_token").exists():
        hf_token = Path(".hf_token").read_text(encoding="utf-8").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    print("Loading COMET backbone...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory="./comet_models")
    comet_model = load_from_checkpoint(model_path)
    try:
        tokenizer = comet_model.tokenizer
    except Exception:
        from transformers import XLMRobertaTokenizer

        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/infoxlm-large", token=hf_token)

    first_train = data_root / FOLDS[0] / "train.json"
    first_dev = data_root / FOLDS[0] / "dev.json"
    baseline = compute_baseline(str(first_train), str(first_dev))
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = COMETModelWithHeads(
        comet_model,
        lora_config,
        train_lq_mean=baseline["train_lq_mean"],
        train_exp_mean=baseline["train_exp_mean"],
        pooling=args.pooling,
    ).to(device)

    all_rows = []
    summary = {"folds": [], "run_root": str(run_root), "data_root": str(data_root)}
    for fold in FOLDS:
        fold_data = data_root / fold
        fold_run = run_root / fold
        checkpoint = fold_run / "partial_best_model2.pt"
        test_data = fold_data / "test.json"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if not test_data.exists():
            raise FileNotFoundError(test_data)

        load_info = load_partial(model, checkpoint, device)
        rows = predict_fold(model, test_data, tokenizer, device, args.max_length, args.batch_size)
        for row in rows:
            row["r017_fold"] = fold
            row["quality_checkpoint"] = str(checkpoint)
        out_path = fold_run / "predictions_test.json"
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        fold_summary = {"fold": fold, **load_info, **summarize_predictions(rows)}
        summary["folds"].append(fold_summary)
        all_rows.extend(rows)
        print(json.dumps(fold_summary, ensure_ascii=False, indent=2))

    merged_json = run_root / "r017_oof_quality_predictions.json"
    merged_json.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = run_root / "r017_oof_quality_predictions.csv"
    fieldnames = sorted({key for row in all_rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary["overall"] = summarize_predictions(all_rows)
    summary_path = run_root / "r017_oof_quality_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", merged_json)
    print("Saved", csv_path)
    print("Saved", summary_path)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
