"""
Train3: Single-Head Fine-Tune

核心设计：
- Single supervision target: Y = α·LQ + (1-α)·EXP
- 默认 α=0.5，即 Y = (LQ + EXP) / 2
- 若 dual-head 用了 w=1.7 强调 EXP，可设 α=1/(1+1.7)≈0.37 保持一致
- 模型结构: Encoder + LoRA → Linear → scalar output s
- Loss: L = MSE(s, Y)
"""

import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import login
try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    try:
        from unbabel_comet import download_model, load_from_checkpoint
    except ImportError:
        raise ImportError("Please install unbabel-comet: pip install unbabel-comet")
import numpy as np
from tqdm import tqdm
import os
from scipy.stats import pearsonr, spearmanr


# ========== Single-Head 回归器 ==========
class SingleHeadRegressor(nn.Module):
    """Encoder + Linear → scalar output s"""

    def __init__(self, hidden_size, init_scale=0.5):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=init_scale / hidden_size**0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(x).squeeze(-1)


# ========== 数据集：输出 Y = α·LQ + (1-α)·EXP ==========
class COMETDatasetSingleHead(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        filter_none: bool = True,
        alpha: float = 0.5,
    ):
        """
        alpha: Y = α·LQ + (1-α)·EXP
        alpha=0.5 → Y = (LQ+EXP)/2
        alpha≈0.37 → 与 dual-head w=1.7 强调 EXP 时等效
        """
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if filter_none:
            self.data = [
                item
                for item in raw_data
                if item.get("LQ") is not None
                and item.get("EXP") is not None
                and item.get("src") is not None
                and item.get("mt") is not None
            ]
            if len(self.data) < len(raw_data):
                print(f"Filtered out {len(raw_data) - len(self.data)} entries with None values")
        else:
            self.data = raw_data

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.alpha = alpha

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = item["src"]
        mt = item["mt"]

        lq = float(item.get("LQ", 0.0))
        exp = float(item.get("EXP", 0.0))

        # Y = α·LQ + (1-α)·EXP
        y = self.alpha * lq + (1.0 - self.alpha) * exp

        encoded = self.tokenizer(
            src,
            mt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "y": torch.tensor(y, dtype=torch.float32),
            "lq": torch.tensor(lq, dtype=torch.float32),
            "exp": torch.tensor(exp, dtype=torch.float32),
        }


# ========== COMET + LoRA + Single Head ==========
class COMETModelSingleHead(nn.Module):
    def __init__(self, comet_model, lora_config: LoraConfig, pooling: str = "mean"):
        super().__init__()
        self.comet_model = comet_model

        if hasattr(comet_model, "model"):
            base_model = comet_model.model
        elif hasattr(comet_model, "encoder"):
            base_model = comet_model.encoder
        else:
            raise ValueError("Cannot find base model in COMET model structure")

        if hasattr(base_model, "transformer"):
            original_transformer = base_model.transformer
        elif hasattr(base_model, "encoder"):
            original_transformer = base_model.encoder
        elif hasattr(base_model, "roberta"):
            original_transformer = base_model.roberta
        else:
            original_transformer = base_model

        for param in original_transformer.parameters():
            param.requires_grad = False

        common_targets = [
            ["query", "value"],
            ["q_proj", "v_proj"],
        ]

        lora_applied = False
        for targets in common_targets:
            try:
                temp_lora_config = LoraConfig(
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    target_modules=targets,
                    bias=lora_config.bias,
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                self.transformer = get_peft_model(original_transformer, temp_lora_config)
                print(f"✅ LoRA attached to {targets}")
                lora_applied = True
                break
            except Exception as e:
                print(f"⚠️ LoRA attach failed for {targets} -> {e}")
                continue

        if not lora_applied:
            raise RuntimeError("LoRA attach failed for all known target module sets.")

        hidden_size = self._get_hidden_size(self.transformer)
        if hidden_size is None:
            hidden_size = self._get_hidden_size(base_model)
        if hidden_size is None:
            hidden_size = 1024

        self.regressor = SingleHeadRegressor(hidden_size, init_scale=1.0)
        self.pooling = pooling

    @staticmethod
    def _get_hidden_size(module):
        config = getattr(module, "config", None)
        if config is None:
            return None
        if isinstance(config, dict):
            return config.get("hidden_size") or config.get("d_model")
        return getattr(config, "hidden_size", None) or getattr(config, "d_model", None)

    def _encode_text(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, dict):
            if "last_hidden_state" in outputs:
                hidden_states = outputs["last_hidden_state"]
            elif "hidden_states" in outputs:
                hs = outputs["hidden_states"]
                hidden_states = hs[-1] if isinstance(hs, (list, tuple)) else hs
            else:
                hidden_states = list(outputs.values())[0]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs

        if len(hidden_states.shape) == 3:
            if self.pooling == "cls":
                sentence_embedding = hidden_states[:, 0, :]
            else:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                sentence_embedding = sum_hidden / sum_mask
        else:
            sentence_embedding = hidden_states

        return sentence_embedding

    def forward(self, input_ids, attention_mask):
        emb = self._encode_text(input_ids, attention_mask)
        s = self.regressor(emb)
        return s


# ========== 训练 ==========
def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    gradient_accumulation_steps=1,
    max_grad_norm=0.5,
    use_amp=False,
    scaler=None,
):
    model.train()
    total_loss = 0
    total_n = 0
    criterion = nn.MSELoss()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y_target = batch["y"].to(device).view(-1)

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                s_pred = model(input_ids, attention_mask).view(-1)
                loss = criterion(s_pred, y_target) / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            s_pred = model(input_ids, attention_mask).view(-1)
            loss = criterion(s_pred, y_target) / gradient_accumulation_steps
            loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()

        bs = y_target.size(0)
        total_loss += loss.item() * gradient_accumulation_steps * bs
        total_n += bs
        pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

        if device.type == "cpu" and (batch_idx + 1) % 10 == 0:
            import gc
            gc.collect()

    last_batch_idx = len(dataloader) - 1
    if (last_batch_idx + 1) % gradient_accumulation_steps != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_n if total_n > 0 else 0.0


def validate(model, dataloader, device, include_predictions=False):
    model.eval()
    total_loss = 0
    total_n = 0
    criterion = nn.MSELoss()

    all_s_pred = []
    all_y_target = []
    all_lq = []
    all_exp = []
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            idx = batch["idx"].view(-1)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_target = batch["y"].to(device).view(-1)
            lq = batch["lq"].to(device).view(-1)
            exp = batch["exp"].to(device).view(-1)

            s_pred = model(input_ids, attention_mask).view(-1)

            loss = criterion(s_pred, y_target)
            bs = y_target.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            all_s_pred.extend(s_pred.cpu().tolist())
            all_y_target.extend(y_target.cpu().tolist())
            all_lq.extend(lq.cpu().tolist())
            all_exp.extend(exp.cpu().tolist())

            if include_predictions:
                for row_idx, pred, target, gold_lq, gold_exp in zip(
                    idx.cpu().tolist(),
                    s_pred.cpu().tolist(),
                    y_target.cpu().tolist(),
                    lq.cpu().tolist(),
                    exp.cpu().tolist(),
                ):
                    predictions.append(
                        {
                            "idx": int(row_idx),
                            "pred_scalar": float(pred),
                            "target_scalar": float(target),
                            "LQ": float(gold_lq),
                            "EXP": float(gold_exp),
                        }
                    )

    avg_loss = total_loss / total_n if total_n > 0 else 0.0

    s_arr = np.array(all_s_pred)
    y_arr = np.array(all_y_target)
    lq_arr = np.array(all_lq)
    exp_arr = np.array(all_exp)

    def safe_corr(x, y, fn):
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        return fn(x, y)[0]

    pearson_y = float(safe_corr(s_arr, y_arr, pearsonr))
    spearman_y = float(safe_corr(s_arr, y_arr, spearmanr))
    pearson_lq = float(safe_corr(s_arr, lq_arr, pearsonr))
    pearson_exp = float(safe_corr(s_arr, exp_arr, pearsonr))

    metrics = {
        "loss": avg_loss,
        "pearson_y": pearson_y,
        "spearman_y": spearman_y,
        "pearson_lq": pearson_lq,
        "pearson_exp": pearson_exp,
        "pred_std": float(s_arr.std()),
    }

    if include_predictions:
        metrics["predictions"] = predictions

    return metrics


def build_optimizer(model, lr_head=1e-3, lr_lora=5e-5, weight_decay=0.01):
    head_params = list(model.regressor.parameters())
    head_param_ids = {id(p) for p in head_params}

    lora_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and ("lora" in n.lower()) and id(p) not in head_param_ids
    ]

    param_groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay},
    ]

    print(f"[Optimizer] Head params: {sum(p.numel() for p in head_params):,}, lr={lr_head}")
    print(f"[Optimizer] LoRA params: {sum(p.numel() for p in lora_params):,}, lr={lr_lora}")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_parser():
    parser = argparse.ArgumentParser(description="Train scalar single-head COMET-KIWI baseline")
    parser.add_argument("--train-data", default=os.getenv("TRAIN_DATA_PATH", "train_set.json"))
    parser.add_argument("--dev-data", default=os.getenv("DEV_DATA_PATH", "dev_set.json"))
    parser.add_argument("--test-data", default=os.getenv("TEST_DATA_PATH", "test_set.json"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "./checkpoints3"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "Unbabel/wmt22-cometkiwi-da"))
    parser.add_argument("--alpha", type=float, default=float(os.getenv("ALPHA", "0.5")))
    parser.add_argument("--pooling", default=os.getenv("POOLING", "cls"), choices=["cls", "mean"])
    parser.add_argument("--num-epochs", type=int, default=int(os.getenv("NUM_EPOCHS", "10")))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("MAX_LENGTH", "512")))
    parser.add_argument("--gpu-batch-size", type=int, default=int(os.getenv("GPU_BATCH_SIZE", "16")))
    parser.add_argument("--cpu-batch-size", type=int, default=int(os.getenv("CPU_BATCH_SIZE", "2")))
    parser.add_argument("--cpu-grad-accum", type=int, default=int(os.getenv("CPU_GRAD_ACCUM", "4")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", "4")))
    parser.add_argument("--lr-head", type=float, default=float(os.getenv("LR_HEAD", "5e-4")))
    parser.add_argument("--lr-lora", type=float, default=float(os.getenv("LR_LORA", "1e-4")))
    parser.add_argument("--lora-unfreeze-epoch", type=int, default=int(os.getenv("LORA_UNFREEZE_EPOCH", "2")))
    parser.add_argument("--max-grad-norm", type=float, default=float(os.getenv("MAX_GRAD_NORM", "0.5")))
    parser.add_argument("--offline", action="store_true", default=os.getenv("TRAIN3_OFFLINE", "0").lower() in {"1", "true", "yes"})
    parser.add_argument("--check-only", action="store_true", help="Validate paths/imports without starting training")
    return parser


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = build_parser().parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    model_name = args.model_name
    train_data_path = args.train_data
    dev_data_path = args.dev_data if args.dev_data and os.path.exists(args.dev_data) else None
    test_data_path = args.test_data if args.test_data and os.path.exists(args.test_data) else None

    # Single-Head 核心参数
    alpha = args.alpha  # Y = α·LQ + (1-α)·EXP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cpu":
        batch_size = args.cpu_batch_size
        gradient_accumulation_steps = args.cpu_grad_accum
        num_workers = 0
    else:
        batch_size = args.gpu_batch_size
        gradient_accumulation_steps = 1
        num_workers = args.num_workers

    lr_head = args.lr_head
    lr_lora = args.lr_lora
    num_epochs = args.num_epochs
    max_length = args.max_length
    output_dir = args.output_dir
    max_grad_norm = args.max_grad_norm
    pooling = args.pooling
    lora_unfreeze_epoch = args.lora_unfreeze_epoch

    required = [train_data_path]
    if args.dev_data:
        required.append(args.dev_data)
    missing = [path for path in required if path and not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    os.makedirs(output_dir, exist_ok=True)
    write_json(
        os.path.join(output_dir, "train_config.json"),
        {
            "model_name": model_name,
            "train_data_path": train_data_path,
            "dev_data_path": dev_data_path,
            "test_data_path": test_data_path,
            "alpha": alpha,
            "pooling": pooling,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lr_head": lr_head,
            "lr_lora": lr_lora,
            "lora_unfreeze_epoch": lora_unfreeze_epoch,
            "max_length": max_length,
            "max_grad_norm": max_grad_norm,
            "offline": args.offline,
        },
    )

    print("=" * 60)
    print("🚀 Train3: Single-Head Fine-Tune")
    print("=" * 60)
    print(f"Y = α·LQ + (1-α)·EXP, α={alpha}")
    print(f"Loss: MSE(s, Y)")
    print(f"Output dir: {output_dir}")
    print(f"LoRA unfreeze epoch: {lora_unfreeze_epoch}")
    print("=" * 60)
    print(f"Device: {device}, batch_size={batch_size}, AMP={use_amp}")

    if args.check_only:
        print("Check passed. Training was not started.")
        return

    # HF 认证
    hf_token = None
    if os.path.exists(".hf_token"):
        try:
            with open(".hf_token", "r") as f:
                hf_token = f.read().strip()
        except Exception:
            pass
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            login(token=hf_token)
            print("✅ HF auth OK")
        except Exception as e:
            print(f"⚠️ Login failed: {e}")
    else:
        print("⚠️ HF_TOKEN not found. May need auth for model download.")

    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # 加载模型
    print("Loading COMET model...")
    model_path = download_model(model_name, saving_directory="./comet_models")
    comet_model = load_from_checkpoint(model_path)

    if hasattr(comet_model, "tokenizer"):
        tokenizer = comet_model.tokenizer
    else:
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/infoxlm-large", token=hf_token)

    # 数据集
    print("Loading datasets...")
    train_dataset = COMETDatasetSingleHead(train_data_path, tokenizer, max_length, alpha=alpha)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if dev_data_path:
        dev_dataset = COMETDatasetSingleHead(dev_data_path, tokenizer, max_length, alpha=alpha)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        dev_loader = None
        print("⚠️ No dev set. Skipping validation.")

    if test_data_path:
        test_dataset = COMETDatasetSingleHead(test_data_path, tokenizer, max_length, alpha=alpha)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        test_loader = None
        print("⚠️ No test set. Skipping final test evaluation.")

    # 模型
    print("Creating Single-Head model...")
    model = COMETModelSingleHead(comet_model, lora_config, pooling=pooling)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    best_score = -float("inf")
    optimizer = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{num_epochs}\n{'='*50}")

        if epoch < lora_unfreeze_epoch:
            for param in model.transformer.parameters():
                param.requires_grad = False
            for param in model.regressor.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=lr_head, weight_decay=0.01)
            print(f"[Freeze] Head only until epoch {lora_unfreeze_epoch}, lr={lr_head}")
        else:
            for n, p in model.transformer.named_parameters():
                p.requires_grad = "lora" in n.lower()
            for param in model.regressor.parameters():
                param.requires_grad = True
            optimizer = build_optimizer(model, lr_head=lr_head, lr_lora=lr_lora, weight_decay=0.01)
            print("[Unfreeze] LoRA + head")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            scaler=scaler,
        )
        print(f"Train Loss (MSE): {train_loss:.4f}")

        if dev_loader:
            val = validate(model, dev_loader, device)
            print(f"Val Loss: {val['loss']:.4f}")
            print(f"  Pearson(s,Y): {val['pearson_y']:.4f}, Spearman(s,Y): {val['spearman_y']:.4f}")
            print(f"  Pearson(s,LQ): {val['pearson_lq']:.4f}, Pearson(s,EXP): {val['pearson_exp']:.4f}")
            print(f"  pred_std: {val['pred_std']:.4f}")

            score = val["pearson_y"]
            if score > best_score:
                best_score = score
                path = os.path.join(output_dir, "best_model3.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_score": best_score,
                        "alpha": alpha,
                        "val_results": val,
                    },
                    path,
                )
                print(f"✅ Saved best to {path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "alpha": alpha,
            },
            os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt"),
        )

    final_path = os.path.join(output_dir, "final_model3.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "alpha": alpha},
        final_path,
    )
    print(f"\n✅ Done. Final model: {final_path}")

    best_path = os.path.join(output_dir, "best_model3.pt")
    final_metrics = {}
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics["best_epoch"] = int(checkpoint.get("epoch", -1))
        final_metrics["best_score"] = float(checkpoint.get("best_score", 0.0))
        final_metrics["alpha"] = alpha

        if dev_loader:
            dev_eval = validate(model, dev_loader, device, include_predictions=True)
            dev_predictions = dev_eval.pop("predictions")
            final_metrics["dev"] = dev_eval
            write_json(os.path.join(output_dir, "predictions_dev.json"), dev_predictions)

        if test_loader:
            test_eval = validate(model, test_loader, device, include_predictions=True)
            test_predictions = test_eval.pop("predictions")
            final_metrics["test"] = test_eval
            write_json(os.path.join(output_dir, "predictions_test.json"), test_predictions)

        write_json(os.path.join(output_dir, "metrics.json"), final_metrics)
        print(f"✅ Wrote metrics to {os.path.join(output_dir, 'metrics.json')}")
    else:
        print("⚠️ No best_model3.pt found; final metrics were not exported.")


if __name__ == "__main__":
    main()
