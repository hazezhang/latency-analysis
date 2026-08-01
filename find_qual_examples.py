"""
从 Dual-Head 模型在 test set 上的预测中，找出 good examples (预测准) 和 bad examples (预测差)。
输出 LaTeX 表格，可直接插入论文 qualitative 段落后。

本脚本完全独立，不依赖 train / evaluate 等其它脚本。
仅支持 train/train2 Dual-Head 模型，Pred 为 LQ/EXP。
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    from unbabel_comet import download_model, load_from_checkpoint

from peft import LoraConfig, get_peft_model, TaskType


# ========== 工具函数 ==========

def latex_escape(s: str) -> str:
    """转义 LaTeX 特殊字符"""
    s = str(s)
    for c, r in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$")]:
        s = s.replace(c, r)
    return s


def shorten(text: str, max_len: int = 0) -> str:
    """处理文本：max_len<=0 时返回完整句子，否则截断"""
    text = text.strip().replace("\n", " ")
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _compute_baseline_means(train_path: str):
    """从 train_set.json 计算 LQ/EXP 均值，用于 Dual-Head 模型"""
    if not os.path.exists(train_path):
        return 2.0, 2.0
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lq_vals = [float(i["LQ"]) for i in data if i.get("LQ") is not None]
    exp_vals = [float(i["EXP"]) for i in data if i.get("EXP") is not None]
    lq_mean = np.mean(lq_vals) if lq_vals else 2.0
    exp_mean = np.mean(exp_vals) if exp_vals else 2.0
    return lq_mean, exp_mean


# ========== Dual-Head 模型与数据集（自包含） ==========

class DualHeadRegressor(nn.Module):
    def __init__(self, hidden_size, init_scale=0.5):
        super().__init__()
        self.lq_head = nn.Linear(hidden_size, 1)
        self.exp_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.lq_head.weight, mean=0.0, std=init_scale / hidden_size**0.5)
        nn.init.zeros_(self.lq_head.bias)
        nn.init.normal_(self.exp_head.weight, mean=0.0, std=init_scale / hidden_size**0.5)
        nn.init.zeros_(self.exp_head.bias)

    def forward(self, x):
        return self.lq_head(x).squeeze(-1), self.exp_head(x).squeeze(-1)


class COMETDatasetDual(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, filter_none: bool = True):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if filter_none:
            self.data = [i for i in raw if i.get("LQ") is not None and i.get("EXP") is not None and i.get("src") and i.get("mt")]
        else:
            self.data = raw
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src, mt = item["src"], item["mt"]
        lq = float(item.get("LQ", 0))
        exp = float(item.get("EXP", 0))
        enc = self.tokenizer(src, mt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "lq": torch.tensor(lq, dtype=torch.float32),
            "exp": torch.tensor(exp, dtype=torch.float32),
        }


class COMETModelDualHead(nn.Module):
    def __init__(self, comet_model, lora_config: LoraConfig, train_lq_mean: float = 2.0, train_exp_mean: float = 2.0, pooling: str = "cls"):
        super().__init__()
        self.comet_model = comet_model
        base = comet_model.model if hasattr(comet_model, "model") else comet_model.encoder
        orig = getattr(base, "transformer", None) or getattr(base, "encoder", None) or getattr(base, "roberta", base)
        for p in orig.parameters():
            p.requires_grad = False
        for targets in [["query", "value"], ["q_proj", "v_proj"]]:
            try:
                cfg = LoraConfig(
                    r=lora_config.r, lora_alpha=lora_config.lora_alpha, lora_dropout=lora_config.lora_dropout,
                    target_modules=targets, bias=lora_config.bias, task_type=TaskType.FEATURE_EXTRACTION,
                )
                self.transformer = get_peft_model(orig, cfg)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("LoRA attach failed")
        cfg = getattr(self.transformer, "config", None) or getattr(base, "config", None)
        h = getattr(cfg, "hidden_size", 1024) if cfg else 1024
        self.regressor = DualHeadRegressor(h, 1.5)
        self.register_buffer("lq_base", torch.tensor(train_lq_mean, dtype=torch.float32))
        self.register_buffer("exp_base", torch.tensor(train_exp_mean, dtype=torch.float32))
        self.pooling = pooling

    def _encode(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(out, dict):
            hs = out.get("last_hidden_state") or (out.get("hidden_states") or [None])[-1] or list(out.values())[0]
        elif hasattr(out, "last_hidden_state"):
            hs = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            hs = out[0]
        else:
            hs = out
        if hs.dim() == 3:
            emb = hs[:, 0, :] if self.pooling == "cls" else (hs * attention_mask.unsqueeze(-1).float()).sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
        else:
            emb = hs
        return emb

    def forward(self, input_ids, attention_mask):
        emb = self._encode(input_ids, attention_mask)
        lq_d, exp_d = self.regressor(emb)
        return self.lq_base + lq_d, self.exp_base + exp_d


# ========== 主逻辑 ==========

def load_model_and_predict(ckpt_path: str, checkpoint_dir: str, test_path: str, device, hf_token, batch_size: int):
    """加载 Dual-Head 模型并运行预测，返回 pred_lq, pred_exp, gold_lq, gold_exp, items"""
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.FEATURE_EXTRACTION)

    model_path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory="./comet_models")
    comet = load_from_checkpoint(model_path)
    tokenizer = getattr(comet, "tokenizer", None)
    if tokenizer is None:
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large", token=hf_token)

    train_path = os.path.join(os.path.dirname(checkpoint_dir) or ".", "train_set.json")
    lq_mean, exp_mean = _compute_baseline_means(train_path)
    model = COMETModelDualHead(comet, lora_cfg, train_lq_mean=lq_mean, train_exp_mean=exp_mean, pooling="cls")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    ds = COMETDatasetDual(test_path, tokenizer, max_length=512)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_lq, all_exp = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            lq, exp = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_lq.extend(lq.view(-1).cpu().tolist())
            all_exp.extend(exp.view(-1).cpu().tolist())

    with open(test_path, "r", encoding="utf-8") as f:
        items = [i for i in json.load(f) if i.get("LQ") is not None and i.get("EXP") is not None and i.get("src") and i.get("mt")]
    gold_lq = [float(i["LQ"]) for i in items]
    gold_exp = [float(i["EXP"]) for i in items]
    return all_lq, all_exp, gold_lq, gold_exp, items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model2.pt")
    parser.add_argument("--checkpoint_dir", default="./checkpoints2")
    parser.add_argument("--test_data", default="test_set.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_good", type=int, default=2)
    parser.add_argument("--n_bad", type=int, default=2)
    parser.add_argument("--max_text_len", type=int, default=0, help="0=完整句子，>0 时截断到该长度")
    parser.add_argument("--output_tex", default="qual_examples.tex")
    parser.add_argument("--output_json", default="qual_examples.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = None
    if os.path.exists(".hf_token"):
        with open(".hf_token") as f:
            hf_token = f.read().strip()
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        alt = "final_model2.pt" if "best" in args.checkpoint else "best_model2.pt"
        alt_path = os.path.join(args.checkpoint_dir, alt)
        if os.path.exists(alt_path):
            ckpt_path, args.checkpoint = alt_path, alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    pred_lq, pred_exp, gold_lq, gold_exp, items = load_model_and_predict(
        ckpt_path, args.checkpoint_dir, args.test_data, device, hf_token, args.batch_size
    )

    errors = [abs(pl - gl) + abs(pe - ge) for pl, pe, gl, ge in zip(pred_lq, pred_exp, gold_lq, gold_exp)]
    idx_sorted = np.argsort(errors)
    good_idx = idx_sorted[: args.n_good].tolist()
    bad_idx = idx_sorted[-args.n_bad :][::-1].tolist()
    max_len = args.max_text_len

    def row(ex_id, src, mt, glq, gexp, pred_str):
        g_lq = f"{glq:.1f}".rstrip("0").rstrip(".")
        g_exp = f"{gexp:.1f}".rstrip("0").rstrip(".")
        return f"{ex_id} & {latex_escape(shorten(src, max_len))} & {latex_escape(shorten(mt, max_len))} & {g_lq}/{g_exp} & {pred_str} \\\\"

    rows, examples = [], []
    for i, idx in enumerate(good_idx):
        it = items[idx]
        pred_str = f"{pred_lq[idx]:.0f}/{pred_exp[idx]:.0f}"
        rows.append(row(f"E{i+1}", it["src"], it["mt"], float(it["LQ"]), float(it["EXP"]), pred_str))
        examples.append({"id": f"E{i+1}", "type": "good", "src": it["src"], "mt": it["mt"], "gold_LQ": float(it["LQ"]), "gold_EXP": float(it["EXP"]), "pred": pred_str, "error": errors[idx]})
    for i, idx in enumerate(bad_idx):
        it = items[idx]
        pred_str = f"{pred_lq[idx]:.0f}/{pred_exp[idx]:.0f}"
        rows.append(row(f"F{i+1}", it["src"], it["mt"], float(it["LQ"]), float(it["EXP"]), pred_str))
        examples.append({"id": f"F{i+1}", "type": "bad", "src": it["src"], "mt": it["mt"], "gold_LQ": float(it["LQ"]), "gold_EXP": float(it["EXP"]), "pred": pred_str, "error": errors[idx]})

    if max_len <= 0:
        tabular_cols = r"p{0.06\linewidth}p{0.36\linewidth}p{0.36\linewidth}p{0.08\linewidth}p{0.10\linewidth}"
        h_src, h_mt = "Source", "Hypothesis"
    else:
        tabular_cols = r"p{0.10\linewidth}p{0.18\linewidth}p{0.18\linewidth}p{0.10\linewidth}p{0.12\linewidth}"
        h_src, h_mt = "Source (short)", "Hypothesis (short)"
    tex = r"""\begin{table}[t]
\caption{Qualitative examples (2 correct, 2 failure cases).}
\label{tab:qual_examples}
\centering
\small
\begin{tabular}{""" + tabular_cols + r"""}
\toprule
ID & """ + h_src + r""" & """ + h_mt + r""" & Gold (LQ/EXP) & Pred (LQ/EXP) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    os.makedirs(os.path.dirname(args.output_tex) or ".", exist_ok=True)
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Saved LaTeX table to {args.output_tex}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "examples": examples}, f, ensure_ascii=False, indent=2)
    print(f"Saved full examples to {args.output_json}")
    print("\n--- Preview ---\n" + tex + "\nDone.")


if __name__ == "__main__":
    main()
