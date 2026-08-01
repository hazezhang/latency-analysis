#!/usr/bin/env python3
"""Train a frozen-encoder joint LQ/EXP/LAT model for the AAAI P0 baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from run_train_lat_direct import correlations, load_rows, set_seed
from run_train_v1 import load_train_v1_symbols


class MultitaskDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, delay_mean, delay_std):
        self.rows = [row for row in rows if row.get("LQ") is not None and row.get("EXP") is not None]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.delay_mean = delay_mean
        self.delay_std = delay_std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["src"], row["mt"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "delay": torch.tensor((row["delay_seconds"] - self.delay_mean) / self.delay_std, dtype=torch.float32),
            "lq": torch.tensor(float(row["LQ"]), dtype=torch.float32),
            "exp": torch.tensor(float(row["EXP"]), dtype=torch.float32),
            "lat": torch.tensor(float(row["perceived_latency"]), dtype=torch.float32),
            "index": index,
        }


class MultitaskModel(nn.Module):
    def __init__(self, encoder_model, means, use_delay):
        super().__init__()
        self.encoder_model = encoder_model
        for parameter in self.encoder_model.parameters():
            parameter.requires_grad = False
        hidden_size = self.encoder_model.regressor.lq_head.in_features
        self.lq_head = nn.Linear(hidden_size, 1)
        self.exp_head = nn.Linear(hidden_size, 1)
        self.lat_head = nn.Linear(hidden_size + int(use_delay), 1)
        for head in (self.lq_head, self.exp_head, self.lat_head):
            nn.init.normal_(head.weight, mean=0.0, std=1.5 / math.sqrt(head.in_features))
            nn.init.zeros_(head.bias)
        self.use_delay = use_delay
        self.register_buffer("lq_base", torch.tensor(means["LQ"], dtype=torch.float32))
        self.register_buffer("exp_base", torch.tensor(means["EXP"], dtype=torch.float32))
        self.register_buffer("lat_base", torch.tensor(means["LAT"], dtype=torch.float32))

    def forward(self, input_ids, attention_mask, delay):
        with torch.no_grad():
            embedding = self.encoder_model._encode_text(input_ids, attention_mask)
        lat_embedding = torch.cat([embedding, delay.unsqueeze(-1)], dim=-1) if self.use_delay else embedding
        return (
            self.lq_base + self.lq_head(embedding).squeeze(-1),
            self.exp_base + self.exp_head(embedding).squeeze(-1),
            self.lat_base + self.lat_head(lat_embedding).squeeze(-1),
        )


def task_loss(prediction, target, corr_weight, variance_weight):
    mse = torch.mean((prediction - target) ** 2)
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2) + 1e-8)
    corr = torch.sum(pred_centered * target_centered) / denominator
    variance = (prediction.std(unbiased=False) - target.std(unbiased=False)).pow(2)
    return mse + corr_weight * (1.0 - corr) + variance_weight * variance


def evaluate(model, loader, device):
    model.eval()
    values = {task: {"gold": [], "prediction": []} for task in ("LQ", "EXP", "LAT")}
    indices = []
    with torch.no_grad():
        for batch in loader:
            predictions = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["delay"].to(device)
            )
            for task, prediction, target_key in zip(("LQ", "EXP", "LAT"), predictions, ("lq", "exp", "lat")):
                values[task]["gold"].extend(batch[target_key].numpy().tolist())
                values[task]["prediction"].extend(prediction.cpu().numpy().tolist())
            indices.extend(batch["index"].numpy().tolist())
    output = {"indices": indices}
    for task, item in values.items():
        gold = np.asarray(item["gold"], dtype=float)
        prediction = np.asarray(item["prediction"], dtype=float)
        pearson, spearman = correlations(gold, prediction)
        output[task] = {
            "pearson": pearson,
            "spearman": spearman,
            "mse": float(np.mean((gold - prediction) ** 2)),
            "mae": float(np.mean(np.abs(gold - prediction))),
            "pred_std": float(prediction.std()),
            "gold": item["gold"],
            "prediction": item["prediction"],
        }
    return output


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = load_train_v1_symbols(Path(args.notebook))
    lora_config = symbols["LoraConfig"](
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"], bias="none",
        task_type=symbols["TaskType"].FEATURE_EXTRACTION,
    )
    comet_model = symbols["load_comet_base"](args.model_name, None)
    try:
        tokenizer = comet_model.tokenizer
    except AttributeError:
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/infoxlm-large")

    train_rows, dev_rows, predict_rows = (
        load_rows(path, require_delay=args.use_delay)
        for path in (args.train_data, args.dev_data, args.predict_data)
    )
    means = {
        "LQ": float(np.mean([float(row["LQ"]) for row in train_rows])),
        "EXP": float(np.mean([float(row["EXP"]) for row in train_rows])),
        "LAT": float(np.mean([row["perceived_latency"] for row in train_rows])),
    }
    delay_mean = float(np.mean([row["delay_seconds"] for row in train_rows]))
    delay_std = float(np.std([row["delay_seconds"] for row in train_rows])) or 1.0

    def loader(rows, shuffle):
        dataset = MultitaskDataset(rows, tokenizer, args.max_length, delay_mean, delay_std)
        return DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers if device.type == "cuda" else 0,
            pin_memory=device.type == "cuda",
        )

    train_loader, dev_loader, predict_loader = loader(train_rows, True), loader(dev_rows, False), loader(predict_rows, False)
    encoder = symbols["COMETModelWithHeads"](
        comet_model, lora_config, train_lq_mean=means["LQ"], train_exp_mean=means["EXP"], pooling=args.pooling
    )
    model = MultitaskModel(encoder, means, args.use_delay).to(device)
    trainable = list(model.lq_head.parameters()) + list(model.exp_head.parameters()) + list(model.lat_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, history = -float("inf"), None, []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                lq_pred, exp_pred, lat_pred = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["delay"].to(device)
                )
                lq_loss = task_loss(lq_pred, batch["lq"].to(device), args.corr_weight, args.variance_weight)
                exp_loss = task_loss(exp_pred, batch["exp"].to(device), args.corr_weight, args.variance_weight)
                lat_loss = task_loss(lat_pred, batch["lat"].to(device), args.corr_weight, args.variance_weight)
                loss = args.lq_weight * lq_loss + args.exp_weight * exp_loss + args.lat_weight * lat_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        dev = evaluate(model, dev_loader, device)
        lat = dev["LAT"]
        score = lat["pearson"] - args.std_penalty * max(0.0, args.std_floor - lat["pred_std"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_score": score,
            "metrics": {task: {key: dev[task][key] for key in ("pearson", "spearman", "mse", "mae", "pred_std")} for task in ("LQ", "EXP", "LAT")},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({
                "model_state_dict": model.state_dict(), "best_epoch": best_epoch, "best_score": best_score,
                "means": means, "delay_mean": delay_mean, "delay_std": delay_std,
                "use_delay": args.use_delay, "seed": args.seed,
                "loss_weights": {"LQ": args.lq_weight, "EXP": args.exp_weight, "LAT": args.lat_weight},
            }, output_dir / "best_model.pt")

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final = evaluate(model, predict_loader, device)
    predictions = []
    for position, index in enumerate(final["indices"]):
        row = predict_rows[index]
        predictions.append({
            "segment_id": row.get("segment_id"), "speech_group": row.get("speech_group"),
            "interpreter": row.get("interpreter"), "direction": row.get("direction"),
            "LQ": final["LQ"]["gold"][position], "EXP": final["EXP"]["gold"][position],
            "LAT": final["LAT"]["gold"][position], "pred_LQ": final["LQ"]["prediction"][position],
            "pred_EXP": final["EXP"]["prediction"][position], "prediction": final["LAT"]["prediction"][position],
            "delay_seconds": row.get("delay_seconds"),
        })
    (output_dir / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "variant": "joint_LQ_EXP_LAT_delay" if args.use_delay else "joint_LQ_EXP_LAT",
        "seed": args.seed, "n_train": len(train_rows), "n_dev": len(dev_rows), "n_predict": len(predict_rows),
        "best_epoch": best_epoch, "best_score": best_score,
        "loss_weights": checkpoint["loss_weights"],
        "predict_metrics": {task: {key: final[task][key] for key in ("pearson", "spearman", "mse", "mae", "pred_std")} for task in ("LQ", "EXP", "LAT")},
        "history": history,
    }
    (output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
    result.add_argument("--lq-weight", type=float, default=1.0)
    result.add_argument("--exp-weight", type=float, default=1.7)
    result.add_argument("--lat-weight", type=float, default=1.0)
    result.add_argument("--corr-weight", type=float, default=0.25)
    result.add_argument("--variance-weight", type=float, default=0.2)
    result.add_argument("--std-floor", type=float, default=0.15)
    result.add_argument("--std-penalty", type=float, default=2.0)
    result.add_argument("--max-grad-norm", type=float, default=0.5)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
