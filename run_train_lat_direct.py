#!/usr/bin/env python3
"""Train a direct perceived-latency regressor with the paper's COMET backbone.

The script is intentionally fold-local: training statistics, checkpoint
selection, and optional delay normalization are computed only from the supplied
train/dev files. It supports the two reviewer-requested baselines:

  source text + interpreted output -> LAT
  source text + interpreted output + objective delay -> LAT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from run_train_v1 import load_train_v1_symbols


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path: str, require_delay: bool = False):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = []
    for row in rows:
        lat = number(row.get("perceived_latency"))
        delay = number(row.get("delay_seconds"))
        if row.get("src") is None or row.get("mt") is None or lat is None:
            continue
        if require_delay and delay is None:
            raise ValueError(f"Missing delay_seconds in delay-enabled input: {path}")
        item = dict(row)
        item["perceived_latency"] = lat
        item["delay_seconds"] = 0.0 if delay is None else delay
        valid.append(item)
    if not valid:
        raise ValueError(f"No valid LAT rows in {path}")
    return valid


class LATDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, delay_mean, delay_std):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.delay_mean = delay_mean
        self.delay_std = delay_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["src"],
            row["mt"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "delay": torch.tensor(
                (row["delay_seconds"] - self.delay_mean) / self.delay_std,
                dtype=torch.float32,
            ),
            "lat": torch.tensor(row["perceived_latency"], dtype=torch.float32),
            "index": index,
        }


class DirectLATModel(nn.Module):
    def __init__(self, encoder_model, lat_mean, use_delay):
        super().__init__()
        self.encoder_model = encoder_model
        for parameter in self.encoder_model.parameters():
            parameter.requires_grad = False
        hidden_size = self.encoder_model.regressor.lq_head.in_features
        for parameter in self.encoder_model.regressor.parameters():
            parameter.requires_grad = False
        self.use_delay = use_delay
        self.lat_head = nn.Linear(hidden_size + int(use_delay), 1)
        nn.init.normal_(self.lat_head.weight, mean=0.0, std=1.5 / math.sqrt(hidden_size))
        nn.init.zeros_(self.lat_head.bias)
        self.register_buffer("lat_base", torch.tensor(float(lat_mean), dtype=torch.float32))

    def forward(self, input_ids, attention_mask, delay):
        with torch.no_grad():
            embedding = self.encoder_model._encode_text(input_ids, attention_mask)
        if self.use_delay:
            embedding = torch.cat([embedding, delay.unsqueeze(-1)], dim=-1)
        return self.lat_base + self.lat_head(embedding).squeeze(-1)


def correlations(gold, prediction):
    gold = np.asarray(gold, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(gold) < 3 or np.std(gold) == 0 or np.std(prediction) == 0:
        return 0.0, 0.0
    return float(pearsonr(gold, prediction).statistic), float(spearmanr(gold, prediction).statistic)


def evaluate(model, loader, device):
    model.eval()
    gold, prediction, indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["delay"].to(device),
            )
            gold.extend(batch["lat"].cpu().numpy().tolist())
            prediction.extend(pred.cpu().numpy().tolist())
            indices.extend(batch["index"].cpu().numpy().tolist())
    pearson, spearman = correlations(gold, prediction)
    gold_array = np.asarray(gold)
    pred_array = np.asarray(prediction)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mse": float(np.mean((gold_array - pred_array) ** 2)),
        "mae": float(np.mean(np.abs(gold_array - pred_array))),
        "pred_std": float(np.std(pred_array)),
        "gold": gold,
        "prediction": prediction,
        "indices": indices,
    }


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = load_train_v1_symbols(Path(args.notebook))
    lora_config = symbols["LoraConfig"](
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=symbols["TaskType"].FEATURE_EXTRACTION,
    )
    comet_model = symbols["load_comet_base"](args.model_name, None)
    try:
        tokenizer = comet_model.tokenizer
    except AttributeError:
        from transformers import XLMRobertaTokenizer

        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/infoxlm-large")

    train_rows = load_rows(args.train_data, require_delay=args.use_delay)
    dev_rows = load_rows(args.dev_data, require_delay=args.use_delay)
    predict_rows = load_rows(args.predict_data, require_delay=args.use_delay)
    lat_mean = float(np.mean([row["perceived_latency"] for row in train_rows]))
    lat_std = float(np.std([row["perceived_latency"] for row in train_rows]))
    delay_mean = float(np.mean([row["delay_seconds"] for row in train_rows]))
    delay_std = float(np.std([row["delay_seconds"] for row in train_rows])) or 1.0

    def loader(rows, shuffle):
        dataset = LATDataset(rows, tokenizer, args.max_length, delay_mean, delay_std)
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers if device.type == "cuda" else 0,
            pin_memory=device.type == "cuda",
        )

    train_loader = loader(train_rows, True)
    dev_loader = loader(dev_rows, False)
    predict_loader = loader(predict_rows, False)

    encoder = symbols["COMETModelWithHeads"](
        comet_model,
        lora_config,
        train_lq_mean=lat_mean,
        train_exp_mean=lat_mean,
        pooling=args.pooling,
    )
    model = DirectLATModel(encoder, lat_mean, args.use_delay).to(device)
    optimizer = torch.optim.AdamW(model.lat_head.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_epoch = None
    history = []
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                pred = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["delay"].to(device),
                )
                target = batch["lat"].to(device)
                mse = torch.mean((pred - target) ** 2)
                pred_centered = pred - pred.mean()
                target_centered = target - target.mean()
                denominator = torch.sqrt(
                    torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2) + 1e-8
                )
                corr = torch.sum(pred_centered * target_centered) / denominator
                variance_loss = (pred.std(unbiased=False) - target.std(unbiased=False)).pow(2)
                loss = mse + args.corr_weight * (1.0 - corr) + args.variance_weight * variance_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.lat_head.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        dev = evaluate(model, dev_loader, device)
        score = dev["pearson"] - args.std_penalty * max(0.0, args.std_floor - dev["pred_std"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            **{key: dev[key] for key in ("pearson", "spearman", "mse", "mae", "pred_std")},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "lat_mean": lat_mean,
                "lat_std": lat_std,
                "delay_mean": delay_mean,
                "delay_std": delay_std,
                "use_delay": args.use_delay,
                "seed": args.seed,
            }, output_dir / "best_model.pt")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final = evaluate(model, predict_loader, device)
    predictions = []
    for index, gold, pred in zip(final["indices"], final["gold"], final["prediction"]):
        row = predict_rows[index]
        predictions.append({
            "segment_id": row.get("segment_id"),
            "speech_group": row.get("speech_group"),
            "interpreter": row.get("interpreter"),
            "direction": row.get("direction"),
            "LAT": gold,
            "prediction": pred,
            "delay_seconds": row.get("delay_seconds"),
        })
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "variant": "direct_text_delay" if args.use_delay else "direct_text",
        "seed": args.seed,
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "n_predict": len(predict_rows),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "predict_metrics": {key: final[key] for key in ("pearson", "spearman", "mse", "mae", "pred_std")},
        "history": history,
    }
    (output_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--train-data", required=True)
    result.add_argument("--dev-data", required=True)
    result.add_argument("--predict-data", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--use-delay", action="store_true")
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--notebook", default="train_v1.ipynb")
    result.add_argument("--model-name", default="Unbabel/wmt22-cometkiwi-da")
    result.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    result.add_argument("--num-epochs", type=int, default=10)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--max-length", type=int, default=512)
    result.add_argument("--learning-rate", type=float, default=5e-4)
    result.add_argument("--corr-weight", type=float, default=0.25)
    result.add_argument("--variance-weight", type=float, default=0.2)
    result.add_argument("--std-floor", type=float, default=0.15)
    result.add_argument("--std-penalty", type=float, default=2.0)
    result.add_argument("--max-grad-norm", type=float, default=0.5)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
