"""
评估脚本：加载 train3 Single-Head 模型，在 Test 集上计算 Pearson correlation

模型输出: scalar s (预测 Y = α·LQ + (1-α)·EXP)
主要指标: Pearson(s, Y)
辅助指标: Pearson(s, LQ), Pearson(s, EXP), MSE(s, Y)
"""

import argparse
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    from unbabel_comet import download_model, load_from_checkpoint

from peft import LoraConfig, TaskType

# 从 train3 导入模型和数据集
from train3 import COMETModelSingleHead, COMETDatasetSingleHead


def safe_corr(x, y, fn):
    """防止常数输入导致 pearsonr/spearmanr 报错"""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    try:
        return float(fn(x, y)[0])
    except Exception:
        return float("nan")


def evaluate(model, dataloader, device):
    """在数据集上评估，返回 Pearson 等指标"""
    model.eval()
    all_s_pred = []
    all_y = []
    all_lq = []
    all_exp = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            s_pred = model(input_ids, attention_mask).view(-1)

            all_s_pred.extend(s_pred.cpu().tolist())
            all_y.extend(batch["y"].tolist())
            all_lq.extend(batch["lq"].tolist())
            all_exp.extend(batch["exp"].tolist())

    s_arr = np.array(all_s_pred)
    y_arr = np.array(all_y)
    lq_arr = np.array(all_lq)
    exp_arr = np.array(all_exp)

    mse = float(np.mean((s_arr - y_arr) ** 2))

    pearson_y = safe_corr(s_arr, y_arr, pearsonr)
    spearman_y = safe_corr(s_arr, y_arr, spearmanr)
    pearson_lq = safe_corr(s_arr, lq_arr, pearsonr)
    pearson_exp = safe_corr(s_arr, exp_arr, pearsonr)

    return {
        "mse": mse,
        "pearson_y": pearson_y,
        "spearman_y": spearman_y,
        "pearson_lq": pearson_lq,
        "pearson_exp": pearson_exp,
        "n_samples": len(all_s_pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate train3 Single-Head model on test set")
    parser.add_argument("--checkpoint", default="best_model3.pt", help="模型文件名 (best_model3.pt 或 final_model3.pt)")
    parser.add_argument("--checkpoint_dir", default="./checkpoints3", help="checkpoint 目录")
    parser.add_argument("--test_data", default="test_set.json", help="测试集路径")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pooling", default="cls", choices=["mean", "cls"], help="与训练时一致")
    parser.add_argument("--export", default=None, help="导出预测到 JSON，如: test_predictions3.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # HF token
    hf_token = None
    if os.path.exists(".hf_token"):
        with open(".hf_token") as f:
            hf_token = f.read().strip()
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # 加载 checkpoint 获取 alpha
    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        alt = "final_model3.pt" if "best" in args.checkpoint else "best_model3.pt"
        alt_path = os.path.join(args.checkpoint_dir, alt)
        if os.path.exists(alt_path):
            ckpt_path = alt_path
            args.checkpoint = alt
            print(f"Using {alt} instead")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    alpha = ckpt.get("alpha", 0.5) if isinstance(ckpt, dict) else 0.5
    print(f"Loaded alpha={alpha} (Y = α·LQ + (1-α)·EXP)")

    # 加载 COMET
    print("Loading COMET model...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory="./comet_models")
    comet_model = load_from_checkpoint(model_path)

    if hasattr(comet_model, "tokenizer"):
        tokenizer = comet_model.tokenizer
    else:
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large", token=hf_token)

    # LoRA 配置（与 train3 一致）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # 构建 Single-Head 模型
    model = COMETModelSingleHead(comet_model, lora_config, pooling=args.pooling)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # 测试集评估
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data not found: {args.test_data}")

    print(f"\n{'='*60}")
    print(f"Evaluating on Test: {args.test_data}")
    print(f"{'='*60}")

    test_dataset = COMETDatasetSingleHead(args.test_data, tokenizer, args.max_length, alpha=alpha)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    results = evaluate(model, test_loader, device)

    print(f"\n📊 Test Set Results (n={results['n_samples']})")
    print("-" * 40)
    print(f"  MSE(s, Y):        {results['mse']:.4f}")
    print(f"  Pearson(s, Y):     {results['pearson_y']:.4f}  ← 主指标")
    print(f"  Spearman(s, Y):   {results['spearman_y']:.4f}")
    print(f"  Pearson(s, LQ):    {results['pearson_lq']:.4f}")
    print(f"  Pearson(s, EXP):   {results['pearson_exp']:.4f}")
    print("-" * 40)

    # 导出预测
    if args.export:
        model.eval()
        all_s = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Exporting"):
                s = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                ).view(-1)
                all_s.extend(s.cpu().tolist())

        with open(args.test_data, "r", encoding="utf-8") as f:
            raw = json.load(f)
        filtered = [
            i for i in raw
            if i.get("LQ") is not None and i.get("EXP") is not None
            and i.get("src") is not None and i.get("mt") is not None
        ]

        out = []
        for i, item in enumerate(filtered):
            entry = {
                "src": item["src"],
                "mt": item["mt"],
                "human_LQ": float(item["LQ"]),
                "human_EXP": float(item["EXP"]),
                "human_Y": alpha * item["LQ"] + (1 - alpha) * item["EXP"],
                "pred_s": round(all_s[i], 4) if i < len(all_s) else None,
            }
            if "segment_id" in item:
                entry["segment_id"] = item["segment_id"]
            out.append(entry)

        os.makedirs(os.path.dirname(args.export) or ".", exist_ok=True)
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nExported {len(out)} predictions to {args.export}")

    # 保存结果摘要
    summary_path = os.path.join(args.checkpoint_dir, "test_results3.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "test_data": args.test_data,
                "alpha": alpha,
                **{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in results.items()},
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
