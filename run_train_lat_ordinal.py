#!/usr/bin/env python3
"""Train a direct LAT CORAL ordinal baseline with a frozen COMET-KIWI encoder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from run_train_lat_direct import LATDataset, load_rows, set_seed
from run_train_v1 import load_train_v1_symbols


# The retained shared target contains the six observed two-rater mean scores
# .5, 1.0, ..., 3.0.  Do not insert unsupported quarter-point pseudo-classes.
MIN_SCORE = 0.5
MAX_SCORE = 3.0
STEP = 0.5
LEVELS = np.arange(MIN_SCORE, MAX_SCORE + STEP / 2.0, STEP, dtype=np.float32)
THRESHOLDS = torch.tensor(LEVELS[:-1], dtype=torch.float32)
METRIC_KEYS = ("pearson", "spearman", "mse", "mae", "pred_std", "quadratic_weighted_kappa", "within_0.5_accuracy")


class OrdinalLATModel(nn.Module):
    def __init__(self, encoder_model, use_delay):
        super().__init__()
        self.encoder_model = encoder_model
        for parameter in self.encoder_model.parameters():
            parameter.requires_grad = False
        hidden_size = self.encoder_model.regressor.lq_head.in_features
        for parameter in self.encoder_model.regressor.parameters():
            parameter.requires_grad = False
        self.use_delay = use_delay
        self.ordinal_score = nn.Linear(hidden_size + int(use_delay), 1)
        nn.init.normal_(self.ordinal_score.weight, mean=0.0, std=1.5 / math.sqrt(hidden_size))
        nn.init.zeros_(self.ordinal_score.bias)
        self.raw_threshold_steps = nn.Parameter(torch.zeros(len(LEVELS) - 1))

    def forward(self, input_ids, attention_mask, delay):
        with torch.no_grad():
            embedding = self.encoder_model._encode_text(input_ids, attention_mask)
        if self.use_delay:
            embedding = torch.cat([embedding, delay.unsqueeze(-1)], dim=-1)
        threshold_steps = nn.functional.softplus(self.raw_threshold_steps)
        ordered_thresholds = torch.cumsum(threshold_steps, dim=0)
        ordered_thresholds = ordered_thresholds - ordered_thresholds.mean()
        return self.ordinal_score(embedding) - ordered_thresholds

    def trainable_ordinal_parameters(self):
        return [*self.ordinal_score.parameters(), self.raw_threshold_steps]


def expected_score(logits):
    thresholds = THRESHOLDS.to(logits.device)
    return MIN_SCORE + STEP * torch.sigmoid(logits).sum(dim=-1)


def ordinal_targets(target):
    thresholds = THRESHOLDS.to(target.device)
    return (target.unsqueeze(-1) > thresholds).float()


def correlations(gold, prediction):
    gold = np.asarray(gold, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(gold) < 3 or np.std(gold) == 0 or np.std(prediction) == 0:
        return 0.0, 0.0
    return float(pearsonr(gold, prediction).statistic), float(spearmanr(gold, prediction).statistic)


def quadratic_weighted_kappa(gold_classes, prediction_classes, n_classes=len(LEVELS)):
    gold_classes = np.asarray(gold_classes, dtype=int)
    prediction_classes = np.asarray(prediction_classes, dtype=int)
    observed = np.zeros((n_classes, n_classes), dtype=float)
    for gold, prediction in zip(gold_classes, prediction_classes):
        observed[gold, prediction] += 1.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(observed.sum(), 1.0)
    grid = np.arange(n_classes)
    weights = (grid[:, None] - grid[None, :]) ** 2 / float((n_classes - 1) ** 2)
    denominator = float(np.sum(weights * expected))
    return 1.0 - float(np.sum(weights * observed)) / denominator if denominator else 1.0


def metrics(gold, prediction):
    gold = np.asarray(gold, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    pearson, spearman = correlations(gold, prediction)
    rounded_gold = np.clip(np.rint((gold - MIN_SCORE) / STEP), 0, len(LEVELS) - 1)
    rounded_prediction = np.clip(np.rint((prediction - MIN_SCORE) / STEP), 0, len(LEVELS) - 1)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mse": float(np.mean((gold - prediction) ** 2)),
        "mae": float(np.mean(np.abs(gold - prediction))),
        "pred_std": float(np.std(prediction)),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(rounded_gold, rounded_prediction),
        "within_0.5_accuracy": float(np.mean(np.abs(gold - prediction) <= 0.5)),
    }


def evaluate(model, loader, device):
    model.eval()
    gold, prediction, indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["delay"].to(device))
            pred = expected_score(logits)
            gold.extend(batch["lat"].cpu().numpy().tolist())
            prediction.extend(pred.cpu().numpy().tolist())
            indices.extend(batch["index"].cpu().numpy().tolist())
    return {**metrics(gold, prediction), "gold": gold, "prediction": prediction, "indices": indices}


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = load_train_v1_symbols(Path(args.notebook))
    lora_config = symbols["LoraConfig"](
        r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
        bias="none", task_type=symbols["TaskType"].FEATURE_EXTRACTION,
    )
    comet_model = symbols["load_comet_base"](args.model_name, None)
    tokenizer = getattr(comet_model, "tokenizer", None)
    if tokenizer is None:
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/infoxlm-large")

    train_rows = load_rows(args.train_data, require_delay=args.use_delay)
    dev_rows = load_rows(args.dev_data, require_delay=args.use_delay)
    predict_rows = load_rows(args.predict_data, require_delay=args.use_delay)
    delay_mean = float(np.mean([row["delay_seconds"] for row in train_rows]))
    delay_std = float(np.std([row["delay_seconds"] for row in train_rows])) or 1.0

    def loader(rows, shuffle):
        dataset = LATDataset(rows, tokenizer, args.max_length, delay_mean, delay_std)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers if device.type == "cuda" else 0,
                          pin_memory=device.type == "cuda")

    encoder = symbols["COMETModelWithHeads"](comet_model, lora_config, train_lq_mean=2.0, train_exp_mean=2.0, pooling=args.pooling)
    model = OrdinalLATModel(encoder, args.use_delay).to(device)
    trainable = model.trainable_ordinal_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, dev_loader, predict_loader = loader(train_rows, True), loader(dev_rows, False), loader(predict_rows, False)

    best_score, best_epoch, history = -float("inf"), None, []
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["delay"].to(device))
                target = batch["lat"].to(device)
                ordinal_loss = nn.functional.binary_cross_entropy_with_logits(logits, ordinal_targets(target))
                pred = expected_score(logits)
                pred_c, target_c = pred - pred.mean(), target - target.mean()
                corr = torch.sum(pred_c * target_c) / torch.sqrt(torch.sum(pred_c ** 2) * torch.sum(target_c ** 2) + 1e-8)
                variance = (pred.std(unbiased=False) - target.std(unbiased=False)).pow(2)
                loss = ordinal_loss + args.corr_weight * (1.0 - corr) + args.variance_weight * variance
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        dev = evaluate(model, dev_loader, device)
        score = dev["pearson"] - args.std_penalty * max(0.0, args.std_floor - dev["pred_std"])
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_score": score,
                  "dev_metrics": {key: dev[key] for key in METRIC_KEYS}}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": epoch, "best_score": score,
                        "delay_mean": delay_mean, "delay_std": delay_std, "use_delay": args.use_delay,
                        "seed": args.seed, "levels": LEVELS.tolist()}, output_dir / "best_model.pt")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final = evaluate(model, predict_loader, device)
    predictions = []
    for index, gold, pred in zip(final["indices"], final["gold"], final["prediction"]):
        row = predict_rows[index]
        predictions.append({"segment_id": row.get("segment_id"), "speech_group": row.get("speech_group"),
                            "interpreter": row.get("interpreter"), "direction": row.get("direction"),
                            "LAT": gold, "prediction": pred, "delay_seconds": row.get("delay_seconds")})
    (output_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"variant": "ordinal_text_delay" if args.use_delay else "ordinal_text", "seed": args.seed,
               "n_train": len(train_rows), "n_dev": len(dev_rows), "n_predict": len(predict_rows),
               "best_epoch": best_epoch, "best_score": best_score, "levels": LEVELS.tolist(),
               "predict_metrics": {key: final[key] for key in METRIC_KEYS},
               "history": history}
    (output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--train-data", required=True); result.add_argument("--dev-data", required=True); result.add_argument("--predict-data", required=True); result.add_argument("--output-dir", required=True)
    result.add_argument("--use-delay", action="store_true"); result.add_argument("--seed", type=int, required=True)
    result.add_argument("--notebook", default="train_v1.ipynb"); result.add_argument("--model-name", default="Unbabel/wmt22-cometkiwi-da")
    result.add_argument("--pooling", choices=("cls", "mean"), default="cls"); result.add_argument("--num-epochs", type=int, default=10); result.add_argument("--batch-size", type=int, default=16); result.add_argument("--num-workers", type=int, default=4); result.add_argument("--max-length", type=int, default=512); result.add_argument("--learning-rate", type=float, default=5e-4); result.add_argument("--corr-weight", type=float, default=0.25); result.add_argument("--variance-weight", type=float, default=0.2); result.add_argument("--std-floor", type=float, default=0.15); result.add_argument("--std-penalty", type=float, default=2.0); result.add_argument("--max-grad-norm", type=float, default=0.5)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
