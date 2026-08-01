"""
评估脚本：加载 final_model2.pt 或 best_model2.pt，在 Dev 和 Test 集上评估
支持导出预测结果，便于与人工打分逐条对比
"""
import json
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    from unbabel_comet import download_model, load_from_checkpoint

from peft import LoraConfig, TaskType, get_peft_model


# ========== 从train.py复制的类和函数 ==========

def quantize_half(x: float) -> float:
    """将值量化到 [0,3] 范围内的 0.5 step（{0, 0.5, 1, ..., 3}）
    纯 Python float 实现，避免 numpy scalar 混入 torch
    """
    x = float(x)
    x = max(0.0, min(3.0, x))
    return round(x * 2) / 2.0


# 双头回归器
class DualHeadRegressor(nn.Module):
    def __init__(self, hidden_size, init_scale=0.5):
        """初始化回归头，init_scale 控制 delta 的初始输出范围，缓解预测塌缩"""
        super().__init__()
        self.lq_head = nn.Linear(hidden_size, 1)
        self.exp_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.lq_head.weight, mean=0.0, std=init_scale / hidden_size**0.5)
        nn.init.zeros_(self.lq_head.bias)
        nn.init.normal_(self.exp_head.weight, mean=0.0, std=init_scale / hidden_size**0.5)
        nn.init.zeros_(self.exp_head.bias)

    def forward(self, x):
        # 输出 delta（残差），去掉 tanh 限幅，允许足够方差
        lq_delta = self.lq_head(x).squeeze(-1)
        exp_delta = self.exp_head(x).squeeze(-1)
        return lq_delta, exp_delta


# 数据集类
class COMETDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, filter_none: bool = True, use_quantize: bool = False):
        """use_quantize: 训练时用 False（原始 float 回归更稳），评估时用 True 与 baseline 一致"""
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if filter_none:
            self.data = [
                item for item in raw_data 
                if item.get('LQ') is not None and item.get('EXP') is not None
                and item.get('src') is not None and item.get('mt') is not None
            ]
            if len(self.data) < len(raw_data):
                print(f"Filtered out {len(raw_data) - len(self.data)} entries with None values")
        else:
            self.data = raw_data
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_quantize = use_quantize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = item["src"]
        mt = item["mt"]
        
        lq = item.get('LQ')
        exp = item.get('EXP')
        
        if lq is None:
            lq = 0.0
        else:
            lq = float(lq)
        
        if exp is None:
            exp = 0.0
        else:
            exp = float(exp)
        
        # 训练用原始 float 回归更稳；评估时可选 quantize 与 baseline 对齐
        if self.use_quantize:
            lq = quantize_half(lq)
            exp = quantize_half(exp)

        # Pair encoding: 一次 tokenizer 调用，合并 src 和 mt
        encoded = self.tokenizer(
            src, mt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'lq': torch.tensor(lq, dtype=torch.float32),
            'exp': torch.tensor(exp, dtype=torch.float32)
        }


# COMET 模型包装器
class COMETModelWithHeads(nn.Module):
    def __init__(self, comet_model, lora_config: LoraConfig, 
                 train_lq_mean: float = 2.0, train_exp_mean: float = 2.0,
                 pooling: str = "mean"):
        """pooling: 'cls'|'mean'，pair 输入时 mean 往往更稳，默认 mean"""
        super().__init__()
        # 使用已加载的 COMET 模型（避免重复 download_model）
        
        # 保存完整的 COMET 模型用于 encode 方法
        self.comet_model = comet_model
        
        # COMET 模型的底层 transformer 通常在 model.encoder 或 model.model 中
        # 获取底层的 transformer 模型用于 LoRA
        if hasattr(comet_model, 'model'):
            base_model = comet_model.model
        elif hasattr(comet_model, 'encoder'):
            base_model = comet_model.encoder
        else:
            raise ValueError("Cannot find base model in COMET model structure")
        
        # 找到 transformer 层（通常是 XLM-RoBERTa）
        if hasattr(base_model, 'transformer'):
            original_transformer = base_model.transformer
        elif hasattr(base_model, 'encoder'):
            original_transformer = base_model.encoder
        elif hasattr(base_model, 'roberta'):  # XLM-RoBERTa
            original_transformer = base_model.roberta
        else:
            # 如果找不到，尝试直接使用 base_model
            original_transformer = base_model
        
        # 冻结基础模型参数（只训练 LoRA）
        for param in original_transformer.parameters():
            param.requires_grad = False
        
        # 使用固定配置列表，失败时 fallback
        common_targets = [
            ["query", "value"],     # roberta/xlm-roberta
            ["q_proj", "v_proj"],   # bart/t5/llama-like
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
                    task_type=TaskType.FEATURE_EXTRACTION
                )
                self.transformer = get_peft_model(original_transformer, temp_lora_config)
                print(f"✅ LoRA attached to {targets}")
                lora_applied = True
                break
            except Exception as e:
                print(f"⚠️ LoRA attach failed for {targets} -> {e}")
                # get_peft_model 失败不会修改原始模型，所以不需要重置
                continue
        
        if not lora_applied:
            raise RuntimeError("LoRA attach failed for all known target module sets.")
        
        # 获取 hidden size；不同 COMET/transformers 版本里 config 可能是对象或 dict
        def _get_hidden_size(config):
            if isinstance(config, dict):
                return config.get("hidden_size") or config.get("d_model")
            return getattr(config, "hidden_size", None) or getattr(config, "d_model", None)

        hidden_size = None
        if hasattr(self.transformer, 'config'):
            hidden_size = _get_hidden_size(self.transformer.config)
        if hidden_size is None and hasattr(base_model, 'config'):
            hidden_size = _get_hidden_size(base_model.config)
        if hidden_size is None:
            # 默认值，wmt22-cometkiwi-da 通常使用 1024
            hidden_size = 1024
        
        self.regressor = DualHeadRegressor(hidden_size, init_scale=1.5)  # 略大缓解塌缩
        self.register_buffer("lq_base", torch.tensor(train_lq_mean, dtype=torch.float32))
        self.register_buffer("exp_base", torch.tensor(train_exp_mean, dtype=torch.float32))
        self.pooling = pooling
        
    def _encode_text(self, input_ids, attention_mask):
        """使用 COMET 模型的 encode 方法或直接调用 encoder 获取 sentence embedding"""
        # 尝试使用 COMET 模型的 encode 方法
        if hasattr(self.comet_model, 'encode'):
            # encode 方法通常接受句子列表，返回 embeddings
            # 但我们需要从 tokenized 输入获取，所以使用 forward
            pass
        
        # 使用 transformer 的 forward 方法
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取 hidden states - 处理多种输出格式
        if isinstance(outputs, dict):
            if 'last_hidden_state' in outputs:
                hidden_states = outputs['last_hidden_state']
            elif 'hidden_states' in outputs:
                hidden_states = outputs['hidden_states'][-1] if isinstance(outputs['hidden_states'], (list, tuple)) else outputs['hidden_states']
            elif 'encoder_outputs' in outputs:
                encoder_outputs = outputs['encoder_outputs']
                if isinstance(encoder_outputs, tuple):
                    hidden_states = encoder_outputs[0]
                elif hasattr(encoder_outputs, 'last_hidden_state'):
                    hidden_states = encoder_outputs.last_hidden_state
                else:
                    hidden_states = encoder_outputs
            else:
                hidden_states = list(outputs.values())[0]
                if not isinstance(hidden_states, torch.Tensor):
                    raise ValueError(f"Could not extract hidden states from outputs: {outputs.keys()}")
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        elif isinstance(outputs, torch.Tensor):
            hidden_states = outputs
        else:
            raise ValueError(f"Unexpected output type: {type(outputs)}")
        
        # pooling: cls 或 mean
        if len(hidden_states.shape) == 3:
            if self.pooling == "cls":
                sentence_embedding = hidden_states[:, 0, :]
            else:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                sentence_embedding = sum_hidden / sum_mask
        elif len(hidden_states.shape) == 2:
            sentence_embedding = hidden_states
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
        
        return sentence_embedding
    
    def forward(self, input_ids, attention_mask):
        # Pair encoding: 单次 encode，transformer 的 self-attention 能看到 src↔mt 交互
        emb = self._encode_text(input_ids, attention_mask)
        
        # 通过回归头得到 delta（残差）
        lq_delta, exp_delta = self.regressor(emb)
        
        # Residual baseline: pred = base + delta（不 clamp，避免梯度截断）
        # clamp 仅在 validate() 评估时使用
        lq_pred_raw = self.lq_base + lq_delta
        exp_pred_raw = self.exp_base + exp_delta
        
        return lq_pred_raw, exp_pred_raw


def validate(model, dataloader, device, train_exp_mean=None, train_exp_std=None, exp_weight=1.5):
    """验证函数：在验证集/测试集上评估模型"""
    model.eval()
    total_loss = 0
    lq_loss_total = 0
    exp_loss_total = 0
    total_n = 0  # 总样本数（用于加权平均）
    
    # 使用 MSE 与 baseline 口径一致
    criterion = nn.MSELoss()
    
    use_exp_zscore = False
    
    # 收集所有预测值和真实值用于计算相关系数
    all_lq_pred = []
    all_lq_target = []
    all_exp_pred = []
    all_exp_target = []
    
    # 收集原始预测值用于诊断（不 clamp）
    all_lq_pred_raw = []
    all_exp_pred_raw = []
    
    shown = 0  # 用于打印前10条样本
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lq_target = batch['lq'].to(device)
            exp_target = batch['exp'].to(device)
            
            lq_pred_raw, exp_pred_raw = model(input_ids, attention_mask)
            
            # 修复 shape bug: 强制对齐为一维向量，避免 broadcast 问题
            lq_pred_raw = lq_pred_raw.view(-1)
            exp_pred_raw = exp_pred_raw.view(-1)
            lq_target = lq_target.view(-1)
            exp_target = exp_target.view(-1)
            
            # loss 计算用 pred_raw（不 clamp，保持梯度完整性）
            loss_lq = criterion(lq_pred_raw, lq_target)
            
            # EXP loss 使用简单 MSE（不使用 z-score）
            loss_exp = criterion(exp_pred_raw, exp_target)
            
            # 收集原始预测值用于诊断
            all_lq_pred_raw.extend(lq_pred_raw.detach().cpu().tolist())
            all_exp_pred_raw.extend(exp_pred_raw.detach().cpu().tolist())
            
            # EXP loss 加权，保持与 train_epoch 口径一致
            loss = loss_lq + exp_weight * loss_exp
            
            # stats/相关/Ex 打印用 pred_eval（clamp 后的版本）
            lq_pred_eval = lq_pred_raw.clamp(0.0, 3.0)
            exp_pred_eval = exp_pred_raw.clamp(0.0, 3.0)
            
            # 按样本数加权累加（修复平均口径问题）
            bs = lq_target.size(0)
            total_loss += loss.item() * bs
            lq_loss_total += loss_lq.item() * bs
            exp_loss_total += loss_exp.item() * bs
            total_n += bs
            
            # 打印前10条样本的 (pred, gold) 对照（诊断用）
            for i in range(bs):
                if shown < 10:
                    print(f"[Ex {shown}] "
                          f"LQ_pred={lq_pred_eval[i].item():.3f} LQ_gold={lq_target[i].item():.3f} | "
                          f"EXP_pred={exp_pred_eval[i].item():.3f} EXP_gold={exp_target[i].item():.3f}")
                    shown += 1
            
            # 收集预测值和真实值用于计算相关系数
            # 使用 pred_raw（clamp 到 [-1,4]）来计算相关系数，避免 clamp(0,3) 压扁相关性
            # clamp(0,3) 仅用于打印样例
            lq_pred_for_corr = torch.clamp(lq_pred_raw, -1.0, 4.0)
            exp_pred_for_corr = torch.clamp(exp_pred_raw, -1.0, 4.0)
            all_lq_pred.extend(lq_pred_for_corr.detach().cpu().tolist())
            all_lq_target.extend(lq_target.detach().cpu().tolist())
            all_exp_pred.extend(exp_pred_for_corr.detach().cpu().tolist())
            all_exp_target.extend(exp_target.detach().cpu().tolist())
    
    # 按样本数加权平均（与 baseline 口径对齐）
    avg_loss = total_loss / total_n if total_n > 0 else 0.0
    avg_lq_loss = lq_loss_total / total_n if total_n > 0 else 0.0
    avg_exp_loss = exp_loss_total / total_n if total_n > 0 else 0.0
    
    # 计算 Pearson 和 Spearman 相关系数
    lq_pred_array = np.array(all_lq_pred)
    lq_target_array = np.array(all_lq_target)
    exp_pred_array = np.array(all_exp_pred)
    exp_target_array = np.array(all_exp_target)
    
    # 计算原始预测值的 std（用于诊断 collapse）
    lq_pred_raw_array = np.array(all_lq_pred_raw)
    exp_pred_raw_array = np.array(all_exp_pred_raw)
    lq_pred_raw_std = lq_pred_raw_array.std()
    exp_pred_raw_std = exp_pred_raw_array.std()
    
    # Sanity-check 日志：打印预测值和目标值的统计信息
    print(f"\n[Validation Stats] Total samples: {total_n}")
    print(f"[LQ] pred: mean={lq_pred_array.mean():.4f}, std={lq_pred_array.std():.4f}, min={lq_pred_array.min():.4f}, max={lq_pred_array.max():.4f}")
    print(f"[LQ] target: mean={lq_target_array.mean():.4f}, std={lq_target_array.std():.4f}, min={lq_target_array.min():.4f}, max={lq_target_array.max():.4f}")
    exp_pred_std = exp_pred_array.std()
    print(f"[EXP] pred: mean={exp_pred_array.mean():.4f}, std={exp_pred_std:.4f}, min={exp_pred_array.min():.4f}, max={exp_pred_array.max():.4f}")
    print(f"[EXP] target: mean={exp_target_array.mean():.4f}, std={exp_target_array.std():.4f}, min={exp_target_array.min():.4f}, max={exp_target_array.max():.4f}")
    
    # Collapse 诊断：打印原始预测值的 std
    print(f"\n[Collapse Diagnosis] Raw prediction std:")
    print(f"  LQ_pred_raw std: {lq_pred_raw_std:.4f}")
    print(f"  EXP_pred_raw std: {exp_pred_raw_std:.4f}")
    if lq_pred_raw_std < 0.10:
        print(f"⚠️  [Collapse Warning] LQ_pred_raw std={lq_pred_raw_std:.4f} < 0.10: LQ prediction collapse detected!")
    if exp_pred_raw_std < 0.10:
        print(f"⚠️  [Collapse Warning] EXP_pred_raw std={exp_pred_raw_std:.4f} < 0.10: EXP prediction collapse detected!")
    
    # 额外打印 EXP_pred std 是否上升（用于诊断预测塌缩问题）
    if exp_pred_std < 0.1:
        print(f"⚠️  [EXP Collapse] EXP_pred std={exp_pred_std:.4f} < 0.1: Prediction collapse detected!")
    elif exp_pred_std > 0.15:
        print(f"✓ [EXP Recovery] EXP_pred std={exp_pred_std:.4f} > 0.15: Prediction variance improved!")
    
    # Safe correlation function: 防止常数输入崩溃
    def safe_corr(x, y, fn):
        if len(x) < 2 or len(y) < 2:
            return 0.0
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        try:
            return fn(x, y)[0]
        except:
            return 0.0
    
    # 计算 Pearson 和 Spearman 相关系数
    lq_pearson = safe_corr(lq_pred_array, lq_target_array, pearsonr)
    lq_spearman = safe_corr(lq_pred_array, lq_target_array, spearmanr)
    exp_pearson = safe_corr(exp_pred_array, exp_target_array, pearsonr)
    exp_spearman = safe_corr(exp_pred_array, exp_target_array, spearmanr)
    
    # Cross-Correlation 诊断：检测 label 对调或方向错误
    cross1 = safe_corr(np.array(all_lq_pred), np.array(all_exp_target), pearsonr)
    cross2 = safe_corr(np.array(all_exp_pred), np.array(all_lq_target), pearsonr)
    print(f"[Cross Corr] Pearson(pred_LQ, gold_EXP)={cross1:.4f} | Pearson(pred_EXP, gold_LQ)={cross2:.4f}")
    
    # Pearson(pred_LQ, pred_EXP)：预测 LQ 与预测 EXP 的相关性
    pred_lq_pred_exp_pearson = safe_corr(lq_pred_array, exp_pred_array, pearsonr)
    print(f"[Pred Corr] Pearson(pred_LQ, pred_EXP)={pred_lq_pred_exp_pearson:.4f}")
    
    return {
        'loss': avg_loss,
        'lq_loss': avg_lq_loss,
        'exp_loss': avg_exp_loss,
        'lq_pearson': lq_pearson,
        'lq_spearman': lq_spearman,
        'exp_pearson': exp_pearson,
        'exp_spearman': exp_spearman,
        'pred_lq_pred_exp_pearson': pred_lq_pred_exp_pearson
    }


def compute_baseline(train_data_path: str, dev_data_path: str = None):
    """计算 baseline：永远预测训练集 LQ 均值、EXP 均值
    使用与 Dataset 完全一致的处理：quantize_half
    """
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    train_lq_values = []
    train_exp_values = []
    
    for item in train_data:
        lq = item.get('LQ')
        exp = item.get('EXP')
        if lq is not None:
            try:
                lq_val = float(lq)
                # 与 Dataset 一致：量化处理
                lq_val = quantize_half(lq_val)
                train_lq_values.append(lq_val)
            except (ValueError, TypeError):
                pass
        if exp is not None:
            try:
                exp_val = float(exp)
                # 与 Dataset 一致：量化处理
                exp_val = quantize_half(exp_val)
                train_exp_values.append(exp_val)
            except (ValueError, TypeError):
                pass
    
    train_lq_mean = np.mean(train_lq_values)
    train_exp_mean = np.mean(train_exp_values)
    train_lq_std = np.std(train_lq_values)
    train_exp_std = np.std(train_exp_values)
    
    print("\n" + "="*60)
    print("📈 Baseline: Always Predict Training Set Mean")
    print("="*60)
    print(f"Training set mean - LQ: {train_lq_mean:.3f}, EXP: {train_exp_mean:.3f}")
    print(f"Training set std  - LQ: {train_lq_std:.3f}, EXP: {train_exp_std:.3f}")
    
    if dev_data_path and os.path.exists(dev_data_path):
        with open(dev_data_path, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
        
        dev_lq_values = []
        dev_exp_values = []
        
        for item in dev_data:
            lq = item.get('LQ')
            exp = item.get('EXP')
            if lq is not None:
                try:
                    lq_val = float(lq)
                    # 与 Dataset 一致：量化处理
                    lq_val = quantize_half(lq_val)
                    dev_lq_values.append(lq_val)
                except (ValueError, TypeError):
                    pass
            if exp is not None:
                try:
                    exp_val = float(exp)
                    # 与 Dataset 一致：量化处理
                    exp_val = quantize_half(exp_val)
                    dev_exp_values.append(exp_val)
                except (ValueError, TypeError):
                    pass
        
        # 计算 baseline MSE
        dev_lq_array = np.array(dev_lq_values)
        dev_exp_array = np.array(dev_exp_values)
        
        baseline_lq_mse = np.mean((dev_lq_array - train_lq_mean) ** 2)
        baseline_exp_mse = np.mean((dev_exp_array - train_exp_mean) ** 2)
        baseline_total_mse = baseline_lq_mse + baseline_exp_mse
        
        print(f"\nDev set baseline MSE:")
        print(f"  LQ MSE: {baseline_lq_mse:.4f}")
        print(f"  EXP MSE: {baseline_exp_mse:.4f}")
        print(f"  Total MSE: {baseline_total_mse:.4f}")
        print("="*60 + "\n")
        
        return {
            'train_lq_mean': train_lq_mean,
            'train_exp_mean': train_exp_mean,
            'train_lq_std': train_lq_std,
            'train_exp_std': train_exp_std,
            'baseline_lq_mse': baseline_lq_mse,
            'baseline_exp_mse': baseline_exp_mse,
            'baseline_total_mse': baseline_total_mse
        }
    else:
        print("="*60 + "\n")
        return {
            'train_lq_mean': train_lq_mean,
            'train_exp_mean': train_exp_mean,
            'train_lq_std': train_lq_std,
            'train_exp_std': train_exp_std
        }


# ========== 评估脚本的主要功能 ==========


def predict_and_export(model, data_path, tokenizer, device, max_length, output_path, batch_size=8):
    """运行模型预测并导出，便于与人工打分逐条对比"""
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    # 过滤（与 COMETDataset 一致）
    filtered = [
        i for i in raw_data
        if i.get("LQ") is not None and i.get("EXP") is not None
        and i.get("src") is not None and i.get("mt") is not None
    ]
    if len(filtered) < len(raw_data):
        print(f"  Filtered {len(raw_data) - len(filtered)} items with None LQ/EXP")

    dataset = COMETDataset(data_path, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_pred_lq, all_pred_exp = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            out_lq, out_exp = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_pred_lq.extend(out_lq.cpu().tolist())
            all_pred_exp.extend(out_exp.cpu().tolist())

    # 构建导出列表
    results = []
    for i, item in enumerate(filtered):
        pred_lq = all_pred_lq[i] if i < len(all_pred_lq) else None
        pred_exp = all_pred_exp[i] if i < len(all_pred_exp) else None
        human_lq = float(item["LQ"]) if item.get("LQ") is not None else None
        human_exp = float(item["EXP"]) if item.get("EXP") is not None else None
        human_lq_q = quantize_half(human_lq) if human_lq is not None else None
        human_exp_q = quantize_half(human_exp) if human_exp is not None else None

        entry = {
            "src": item["src"],
            "mt": item["mt"],
            "human_LQ": human_lq,
            "human_EXP": human_exp,
            "human_LQ_quantized": human_lq_q,
            "human_EXP_quantized": human_exp_q,
            "pred_LQ": round(pred_lq, 4) if pred_lq is not None else None,
            "pred_EXP": round(pred_exp, 4) if pred_exp is not None else None,
        }
        if "segment_id" in item:
            entry["segment_id"] = item["segment_id"]
        if pred_lq is not None and human_lq_q is not None:
            entry["LQ_diff"] = round(pred_lq - human_lq_q, 4)
        if pred_exp is not None and human_exp_q is not None:
            entry["EXP_diff"] = round(pred_exp - human_exp_q, 4)
        results.append(entry)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(results)} predictions to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final_model2.pt", help="模型文件名")
    parser.add_argument("--checkpoint_dir", default="./checkpoints2")
    parser.add_argument("--dev_data", default="dev_set.json")
    parser.add_argument("--test_data", default="test_set.json")
    parser.add_argument("--train_data", default="train_set.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--export", default=None, help="导出预测到 JSON，便于与人工打分对比，如: predictions_test.json")
    parser.add_argument("--export_data", default=None, help="导出时使用的数据文件，默认 test_set.json")
    parser.add_argument("--pooling", default="cls", choices=["mean", "cls"], help="与训练时一致，GPU 默认 cls")
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

    # Load COMET
    print("Loading COMET model...")
    model_path = download_model("Unbabel/wmt22-cometkiwi-da", saving_directory="./comet_models")
    comet_model = load_from_checkpoint(model_path)

    try:
        tokenizer = comet_model.tokenizer
    except Exception:
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/infoxlm-large", token=hf_token)

    # Baseline
    print("\nComputing baseline...")
    baseline_info = compute_baseline(args.train_data, args.dev_data)

    # LoRA config (must match train.py)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    # Build model（pooling 需与训练时一致，默认 mean）
    model = COMETModelWithHeads(
        comet_model,
        lora_config,
        train_lq_mean=baseline_info["train_lq_mean"],
        train_exp_mean=baseline_info["train_exp_mean"],
        pooling=args.pooling,
    )

    # Load checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    if not os.path.exists(ckpt_path):
        # Try best_model2.pt if final not found
        alt = "final_model2.pt" if args.checkpoint == "best_model2.pt" else "best_model2.pt"
        alt_path = os.path.join(args.checkpoint_dir, alt)
        if os.path.exists(alt_path):
            ckpt_path = alt_path
            print(f"Using {alt} instead")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt)
        print(f"Loaded {ckpt_path}")
    model = model.to(device)

    exp_weight = 1.5

    # Dev
    if os.path.exists(args.dev_data):
        print(f"\n{'='*60}\nEvaluating on Dev: {args.dev_data}\n{'='*60}")
        dev_dataset = COMETDataset(args.dev_data, tokenizer, args.max_length)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        dev_results = validate(
            model, dev_loader, device,
            train_exp_mean=baseline_info.get("train_exp_mean"),
            train_exp_std=baseline_info.get("train_exp_std"),
            exp_weight=exp_weight,
        )
        if "baseline_total_mse" in baseline_info:
            imp = baseline_info["baseline_total_mse"] - dev_results["loss"]
            pct = 100 * imp / baseline_info["baseline_total_mse"]
            print(f"\nDev vs Baseline: {imp:+.4f} ({pct:+.2f}%)")

    # Test
    if os.path.exists(args.test_data):
        print(f"\n{'='*60}\nEvaluating on Test: {args.test_data}\n{'='*60}")
        test_dataset = COMETDataset(args.test_data, tokenizer, args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test_results = validate(
            model, test_loader, device,
            train_exp_mean=baseline_info.get("train_exp_mean"),
            train_exp_std=baseline_info.get("train_exp_std"),
            exp_weight=exp_weight,
        )
        print(f"\nTest Loss: {test_results['loss']:.4f}")
        print(f"Test LQ Pearson: {test_results['lq_pearson']:.4f}, EXP Pearson: {test_results['exp_pearson']:.4f}")
        print(f"Test Pearson(pred_LQ, pred_EXP): {test_results.get('pred_lq_pred_exp_pearson', 0):.4f}")

    # 导出预测结果（与人工打分逐条对比）
    if args.export:
        print(f"\n{'='*60}\nExporting predictions for human comparison\n{'='*60}")
        export_data = args.export_data or args.test_data
        if not os.path.exists(export_data):
            export_data = args.dev_data
        predict_and_export(
            model, export_data, tokenizer, device, args.max_length,
            args.export, args.batch_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
