import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
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
from typing import Dict, List, Tuple
from collections import deque
from scipy.stats import pearsonr, spearmanr

# 量化函数：clip 到 [0,3] 后 round(x*2)/2，将值量化到 0.5 step
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

        # 获取 hidden size
        if hasattr(self.transformer, 'config'):
            hidden_size = self.transformer.config.hidden_size
        elif hasattr(base_model, 'config'):
            hidden_size = base_model.config.hidden_size
        else:
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


def train_epoch(model, dataloader, optimizer, device, epoch, gradient_accumulation_steps=1,
                train_exp_mean=None, train_exp_std=None, exp_weight=1.5,
                max_grad_norm=0.5, variance_weight=0.0, variance_buffer_size=64,
                use_amp=False, scaler=None):
    model.train()
    total_loss = 0
    lq_loss_total = 0
    exp_loss_total = 0
    total_n = 0

    criterion = nn.MSELoss()
    use_exp_zscore = False

    # 滑动 buffer：跨 step 累积 (pred, target) 用于更稳的方差估计
    var_buffer_lq = deque(maxlen=variance_buffer_size)
    var_buffer_exp = deque(maxlen=variance_buffer_size)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        lq_target = batch['lq'].to(device)
        exp_target = batch['exp'].to(device)

        def _forward():
            lq_pred, exp_pred = model(input_ids, attention_mask)
            lq_pred = lq_pred.view(-1)
            exp_pred = exp_pred.view(-1)
            lq_target_ = lq_target.view(-1)
            exp_target_ = exp_target.view(-1)
            return lq_pred, exp_pred, lq_target_, exp_target_

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                lq_pred, exp_pred, lq_target, exp_target = _forward()
                loss_lq = criterion(lq_pred, lq_target)
                if use_exp_zscore:
                    exp_pred_z = (exp_pred - train_exp_mean) / (train_exp_std + 1e-6)
                    exp_tgt_z = (exp_target - train_exp_mean) / (train_exp_std + 1e-6)
                    loss_exp = criterion(exp_pred_z, exp_tgt_z)
                else:
                    loss_exp = criterion(exp_pred, exp_target)

                var_loss = 0.0
                if variance_weight > 0 and epoch >= 1:
                    for i in range(lq_pred.size(0)):
                        var_buffer_lq.append((lq_pred[i].detach().item(), lq_target[i].item()))
                        var_buffer_exp.append((exp_pred[i].detach().item(), exp_target[i].item()))
                    if len(var_buffer_lq) >= variance_buffer_size:
                        buf_lq_p = [x[0] for x in var_buffer_lq]
                        buf_lq_t = [x[1] for x in var_buffer_lq]
                        buf_exp_p = [x[0] for x in var_buffer_exp]
                        buf_exp_t = [x[1] for x in var_buffer_exp]
                        tgt_var_lq = max(np.var(buf_lq_t), 0.15)
                        tgt_var_exp = max(np.var(buf_exp_t), 0.15)
                        cur_var_lq = torch.var(lq_pred)
                        cur_var_exp = torch.var(exp_pred)
                        var_loss = variance_weight * ((cur_var_lq - tgt_var_lq) ** 2 + (cur_var_exp - tgt_var_exp) ** 2)

                loss = (loss_lq + exp_weight * loss_exp + var_loss) / gradient_accumulation_steps

            scaler.scale(loss).backward()
        else:
            lq_pred, exp_pred, lq_target, exp_target = _forward()
            if batch_idx == 0 and epoch == 1:
                print(f"\n[Shape Check] lq_pred: {lq_pred.shape}, exp_pred: {exp_pred.shape}")

            loss_lq = criterion(lq_pred, lq_target)
            loss_exp = criterion(exp_pred, exp_target)

            var_loss = 0.0
            if variance_weight > 0 and epoch >= 1:
                for i in range(lq_pred.size(0)):
                    var_buffer_lq.append((lq_pred[i].detach().item(), lq_target[i].item()))
                    var_buffer_exp.append((exp_pred[i].detach().item(), exp_target[i].item()))
                if len(var_buffer_lq) >= variance_buffer_size:
                    buf_lq_t = [x[1] for x in var_buffer_lq]
                    buf_exp_t = [x[1] for x in var_buffer_exp]
                    tgt_var_lq = max(np.var(buf_lq_t), 0.15)
                    tgt_var_exp = max(np.var(buf_exp_t), 0.15)
                    cur_var_lq = torch.var(lq_pred)
                    cur_var_exp = torch.var(exp_pred)
                    var_loss = variance_weight * ((cur_var_lq - tgt_var_lq) ** 2 + (cur_var_exp - tgt_var_exp) ** 2)

            loss = (loss_lq + exp_weight * loss_exp + var_loss) / gradient_accumulation_steps
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

        # 按样本数加权累积损失
        bs = lq_target.size(0)
        total_loss += loss.item() * gradient_accumulation_steps * bs
        lq_loss_total += loss_lq.item() * bs
        exp_loss_total += loss_exp.item() * bs
        total_n += bs

        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'lq_loss': f'{loss_lq.item():.4f}',
            'exp_loss': f'{loss_exp.item():.4f}'
        })

        # CPU 上定期清理内存
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

    # 按样本数加权平均（与 validate 口径一致）
    avg_loss = total_loss / total_n if total_n > 0 else 0.0
    avg_lq_loss = lq_loss_total / total_n if total_n > 0 else 0.0
    avg_exp_loss = exp_loss_total / total_n if total_n > 0 else 0.0

    return avg_loss, avg_lq_loss, avg_exp_loss


def validate(model, dataloader, device, train_exp_mean=None, train_exp_std=None, exp_weight=1.5):
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
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        return fn(x, y)[0]

    # 计算 Pearson 和 Spearman 相关系数
    lq_pearson = safe_corr(lq_pred_array, lq_target_array, pearsonr)
    lq_spearman = safe_corr(lq_pred_array, lq_target_array, spearmanr)
    exp_pearson = safe_corr(exp_pred_array, exp_target_array, pearsonr)
    exp_spearman = safe_corr(exp_pred_array, exp_target_array, spearmanr)

    # Cross-Correlation 诊断：检测 label 对调或方向错误
    cross1 = safe_corr(np.array(all_lq_pred), np.array(all_exp_target), pearsonr)
    cross2 = safe_corr(np.array(all_exp_pred), np.array(all_lq_target), pearsonr)
    print(f"[Cross Corr] Pearson(pred_LQ, gold_EXP)={cross1:.4f} | Pearson(pred_EXP, gold_LQ)={cross2:.4f}")

    mse_sum = avg_lq_loss + avg_exp_loss
    train_obj = avg_lq_loss + exp_weight * avg_exp_loss
    print(f"\n[Loss Metrics] mse_sum (vs baseline): {mse_sum:.4f} | train_obj (opt): {train_obj:.4f}")

    return {
        'loss': avg_loss,
        'lq_loss': avg_lq_loss,
        'exp_loss': avg_exp_loss,
        'mse_sum': mse_sum,
        'train_obj': train_obj,
        'lq_pearson': lq_pearson,
        'lq_spearman': lq_spearman,
        'exp_pearson': exp_pearson,
        'exp_spearman': exp_spearman,
        'lq_pred_std': float(lq_pred_raw_std),
        'exp_pred_std': float(exp_pred_raw_std),
    }


def build_optimizer(model, lr_head=1e-3, lr_lora=5e-5, weight_decay=0.01):
    """构建参数组优化器：regressor 和 LoRA 使用不同学习率

    Args:
        model: COMETModelWithHeads 模型
        lr_head: regressor 参数的学习率（默认 1e-3）
        lr_lora: LoRA 参数的学习率（默认 5e-5）
        weight_decay: 权重衰减（默认 0.01）

    Returns:
        AdamW optimizer with parameter groups
    """
    # regressor 参数
    head_params = list(model.regressor.parameters())
    head_param_ids = {id(p) for p in head_params}

    # LoRA 参数：所有 requires_grad=True 且包含 "lora" 的参数，排除 head_params（使用 id 去重）
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and ("lora" in n.lower())
        and id(p) not in head_param_ids
    ]

    # 创建参数组
    param_groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay}
    ]

    print(f"[Optimizer] Head params: {sum(p.numel() for p in head_params):,} params, lr={lr_head}")
    print(f"[Optimizer] LoRA params: {sum(p.numel() for p in lora_params):,} params, lr={lr_lora}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # 调试输出：打印 transformer 中可训练的参数
    print("[Trainable in transformer]")
    for n, p in model.transformer.named_parameters():
        if p.requires_grad:
            print("  ", n)

    return optimizer


def check_label_distribution(data_path: str):
    """检查标签分布：直方图、均值、方差"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lq_values = []
    exp_values = []

    for item in data:
        lq = item.get('LQ')
        exp = item.get('EXP')
        if lq is not None:
            try:
                lq_values.append(float(lq))
            except (ValueError, TypeError):
                pass
        if exp is not None:
            try:
                exp_values.append(float(exp))
            except (ValueError, TypeError):
                pass

    lq_values = np.array(lq_values)
    exp_values = np.array(exp_values)

    print("\n" + "="*60)
    print(f"📊 Label Distribution Analysis: {os.path.basename(data_path)}")
    print("="*60)

    if len(lq_values) > 0:
        print(f"\nLQ (Language Quality):")
        print(f"  Count: {len(lq_values)}")
        print(f"  Mean: {np.mean(lq_values):.3f}")
        print(f"  Std: {np.std(lq_values):.3f}")
        print(f"  Min: {np.min(lq_values):.3f}")
        print(f"  Max: {np.max(lq_values):.3f}")
        print(f"  Median: {np.median(lq_values):.3f}")

        # 统计每个值的分布
        unique_lq, counts_lq = np.unique(lq_values, return_counts=True)
        print(f"\n  Value distribution:")
        for val, cnt in zip(unique_lq, counts_lq):
            print(f"    {val:.1f}: {cnt} ({100*cnt/len(lq_values):.1f}%)")
    else:
        print(f"\nLQ (Language Quality): No valid values found")

    if len(exp_values) > 0:
        print(f"\nEXP (Expressiveness):")
        print(f"  Count: {len(exp_values)}")
        print(f"  Mean: {np.mean(exp_values):.3f}")
        print(f"  Std: {np.std(exp_values):.3f}")
        print(f"  Min: {np.min(exp_values):.3f}")
        print(f"  Max: {np.max(exp_values):.3f}")
        print(f"  Median: {np.median(exp_values):.3f}")

        # 统计每个值的分布
        unique_exp, counts_exp = np.unique(exp_values, return_counts=True)
        print(f"\n  Value distribution:")
        for val, cnt in zip(unique_exp, counts_exp):
            print(f"    {val:.1f}: {cnt} ({100*cnt/len(exp_values):.1f}%)")
    else:
        print(f"\nEXP (Expressiveness): No valid values found")

    print("="*60 + "\n")

    return lq_values, exp_values


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


def main():
    # 配置
    model_name = "Unbabel/wmt22-cometkiwi-da"
    train_data_path = "train_set.json"
    dev_data_path = "dev_set.json" if os.path.exists("dev_set.json") else None

    # GPU 配置：大 batch + AMP；CPU 小 batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cpu":
        batch_size = 2
        gradient_accumulation_steps = 4
        num_workers = 0
    else:
        batch_size = 16  # GPU 大 batch 利于方差估计稳定
        gradient_accumulation_steps = 1
        num_workers = 4

    # 统一学习率来源（lr_head 5e-4 帮助 regressor，lr_lora 降低避免 LoRA 解冻后压制 head 导致塌缩）
    lr_head = 5e-4
    lr_lora = 1.5e-4  # 2e-4->1e-4，epoch2 解冻 LoRA 后塌缩加重，降低 LoRA 学习率

    # EXP loss 权重（增加 EXP 损失权重，帮助恢复方差）
    exp_weight = 1.7
    num_epochs = 10
    max_length = 512
    output_dir = "./checkpoints2"
    max_grad_norm = 0.5
    pooling = "cls"  # GPU 上 CLS 对 pair 编码更稳，塌缩时可选 mean
    # 滑动方差正则：缓解塌缩，GPU 大 batch 时更稳定
    variance_weight = 0.05 if device.type == "cuda" else 0.0
    variance_buffer_size = 64  # 跨 step 累积样本再算方差

    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}, AMP: {use_amp}")

    # Hugging Face 认证
    # 方法1: 从配置文件读取
    token_file = ".hf_token"
    hf_token = None

    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                hf_token = f.read().strip()
            print("Using HF_TOKEN from .hf_token file...")
        except Exception as e:
            print(f"Warning: Could not read .hf_token file: {e}")

    # 方法2: 从环境变量读取
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print("Using HF_TOKEN from environment variable...")

    # 方法3: 尝试登录
    if hf_token:
        try:
            login(token=hf_token)
            print("✅ Authentication successful!")
        except Exception as e:
            print(f"⚠️  Warning: Login failed: {e}")
            print("Will try to use token directly when loading model...")
    else:
        print("=" * 60)
        print("⚠️  Warning: HF_TOKEN not found.")
        print("This model requires Hugging Face authentication.")
        print("\nPlease do one of the following:")
        print("1. Create .hf_token file with your token")
        print("2. Set environment variable: export HF_TOKEN=your_token_here")
        print("3. Get your token from: https://huggingface.co/settings/tokens")
        print("4. Request access at: https://huggingface.co/Unbabel/wmt22-cometkiwi-da")
        print("=" * 60)
        try:
            # 尝试使用已保存的 token
            login()
            hf_token = os.getenv("HF_TOKEN")
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
            print("\nPlease:")
            print("1. Get your Hugging Face token from: https://huggingface.co/settings/tokens")
            print("2. Request access to the model: https://huggingface.co/Unbabel/wmt22-cometkiwi-da")
            print("3. Create .hf_token file or set environment variable: export HF_TOKEN=your_token_here")
            return

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "v_proj"
        ],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # 加载 tokenizer
    # COMET 模型通常使用 XLM-RoBERTa tokenizer
    print("Loading tokenizer...")
    # 设置 token 到环境变量（COMET 库会读取）
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token

    # 先下载模型以获取正确的 tokenizer（避免重复 download_model）
    print("Downloading/loading COMET model...")
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
    model_path = download_model(model_name, saving_directory="./comet_models")
    comet_model = load_from_checkpoint(model_path)

    try:
        if hasattr(comet_model, 'tokenizer'):
            tokenizer = comet_model.tokenizer
        else:
            # 如果没有，使用 XLM-RoBERTa tokenizer
            from transformers import XLMRobertaTokenizer
            tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large", token=hf_token)
    except Exception as e:
        print(f"Warning: Could not load COMET tokenizer, using XLM-RoBERTa: {e}")
        from transformers import XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large", token=hf_token)

    # A. 检查标签分布
    print("\n" + "="*60)
    print("A. Checking Label Distribution")
    print("="*60)
    check_label_distribution(train_data_path)
    if dev_data_path:
        check_label_distribution(dev_data_path)

    # B. 计算 baseline
    print("\n" + "="*60)
    print("B. Computing Baseline")
    print("="*60)
    baseline_info = compute_baseline(train_data_path, dev_data_path)

    # 打印 baseline 均值（用于 residual baseline）
    print(f"[Baseline Means] LQ={baseline_info['train_lq_mean']:.3f} EXP={baseline_info['train_exp_mean']:.3f}")

    # 创建数据集（训练用原始 float，评估用 quantize 与 baseline 对齐）
    print("Loading datasets...")
    train_dataset = COMETDataset(train_data_path, tokenizer, max_length, use_quantize=False)

    # Sanity check: 验证 EXP 标签范围正确（允许 0.0-3.0，步长 0.5）
    exp_values_check = []
    check_samples = min(500, len(train_dataset))
    for i in range(check_samples):
        sample = train_dataset[i]
        exp_val = sample['exp'].item()
        exp_values_check.append(exp_val)

    # 归一化到 0.5 step 再统计
    exp_values_norm = [round(v * 2) / 2.0 for v in exp_values_check]
    exp_unique = sorted(set(exp_values_norm))
    exp_min = min(exp_values_norm)
    exp_max = max(exp_values_norm)
    expected_exp_values = {i / 2.0 for i in range(0, 7)}  # {0.0, 0.5, ..., 3.0}

    print(f"[Sanity Check] EXP values (first {check_samples} samples):")
    print(f"  Range: [{exp_min:.1f}, {exp_max:.1f}]")
    print(f"  Unique values: {exp_unique}")

    # 检查范围是否超出 [0, 3]
    if exp_min < -1e-6 or exp_max > 3.0 + 1e-6:
        raise ValueError(
            f"❌ EXP out of [0,3] range! EXP range is [{exp_min:.1f}, {exp_max:.1f}]. "
            f"Expected range [0.0, 3.0] with step=0.5."
        )

    # 训练集 use_quantize=False 时不做 0.5 step 检查
    if train_dataset.use_quantize:
        unexpected_values = set(exp_unique) - expected_exp_values
        if unexpected_values:
            raise ValueError(
                f"❌ Unexpected EXP values (not 0.5 step): {unexpected_values}. "
                f"Expected range [0.0, 3.0] with step=0.5."
            )
    print(f"✓ EXP values check done (use_quantize={train_dataset.use_quantize})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")  # 只在 GPU 上使用 pin_memory
    )

    if dev_data_path:
        dev_dataset = COMETDataset(dev_data_path, tokenizer, max_length, use_quantize=True)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda")
        )
    else:
        dev_loader = None
        print("⚠️  Warning: No dev set found. Cannot evaluate during training!")

    # 创建模型
    print("Loading model...")
    model = COMETModelWithHeads(
        comet_model,
        lora_config,
        train_lq_mean=baseline_info["train_lq_mean"],
        train_exp_mean=baseline_info["train_exp_mean"],
        pooling=pooling,
    )
    model = model.to(device)

    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 训练循环
    print("Starting training...")
    os.makedirs(output_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 1
    best_score = -float('inf')  # 用 pearson 和选 best，不用 loss
    optimizer = None

    # 查找最新的 checkpoint
    checkpoint_files = [f for f in os.listdir(output_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if checkpoint_files:
        # 提取 epoch 编号并找到最新的
        epochs = []
        for f in checkpoint_files:
            try:
                epoch_num = int(f.replace("checkpoint_epoch_", "").replace(".pt", ""))
                epochs.append((epoch_num, f))
            except ValueError:
                continue

        if epochs:
            latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
            checkpoint_path = os.path.join(output_dir, latest_file)
            try:
                print(f"\n📂 Found checkpoint: {latest_file}")
                print(f"   Attempting to resume from epoch {latest_epoch}...")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = latest_epoch + 1
                print(f"✅ Successfully loaded checkpoint from epoch {latest_epoch}")
                print(f"   Resuming training from epoch {start_epoch}")

                if 'best_score' in checkpoint:
                    best_score = checkpoint['best_score']
                    print(f"   Restored best_score: {best_score:.4f}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to load checkpoint: {e}")
                print(f"   Starting training from epoch 1...")
                start_epoch = 1

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")

        # 训练日程：epoch1 head only；epoch2+ LoRA+head 全解冻
        if epoch == 1:
            print("[Freeze] Epoch 1: Head only")
            for param in model.transformer.parameters():
                param.requires_grad = False
            for param in model.regressor.parameters():
                param.requires_grad = True
            head_params = list(model.regressor.parameters())
            optimizer = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=0.01)
            print(f"[Optimizer] Head only, lr={lr_head}")
        else:
            print(f"[Unfreeze] Epoch {epoch}+: LoRA + head")
            for n, p in model.transformer.named_parameters():
                p.requires_grad = ("lora" in n.lower())
            for param in model.regressor.parameters():
                param.requires_grad = True
            if epoch == 2:
                num_trainable = sum(1 for n, p in model.transformer.named_parameters() if p.requires_grad)
                if num_trainable == 0:
                    raise RuntimeError("No LoRA params trainable!")
            optimizer = build_optimizer(model, lr_head=lr_head, lr_lora=lr_lora, weight_decay=0.01)

        train_loss, train_lq_loss, train_exp_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, gradient_accumulation_steps,
            train_exp_mean=baseline_info.get('train_exp_mean'),
            train_exp_std=baseline_info.get('train_exp_std'),
            exp_weight=exp_weight,
            max_grad_norm=max_grad_norm,
            variance_weight=variance_weight,
            variance_buffer_size=variance_buffer_size,
            use_amp=use_amp,
            scaler=scaler,
        )
        print(f"Train Loss: {train_loss:.4f} (LQ: {train_lq_loss:.4f}, EXP: {train_exp_loss:.4f})")

        # 验证
        if dev_loader:
            val_results = validate(
                model, dev_loader, device,
                train_exp_mean=baseline_info.get('train_exp_mean'),
                train_exp_std=baseline_info.get('train_exp_std'),
                exp_weight=exp_weight
            )
            val_loss = val_results['loss']
            val_lq_loss = val_results['lq_loss']
            val_exp_loss = val_results['exp_loss']

            print(f"\nVal Loss: {val_loss:.4f} (LQ: {val_lq_loss:.4f}, EXP: {val_exp_loss:.4f})")
            print(f"Val Correlations:")
            print(f"  LQ - Pearson: {val_results['lq_pearson']:.4f}, Spearman: {val_results['lq_spearman']:.4f}")
            print(f"  EXP - Pearson: {val_results['exp_pearson']:.4f}, Spearman: {val_results['exp_spearman']:.4f}")

            # 与 baseline 对比（用 mse_sum 与 baseline_total_mse 口径一致）
            mse_sum = val_results.get('mse_sum', val_lq_loss + val_exp_loss)
            if 'baseline_total_mse' in baseline_info:
                print(f"\nBaseline Comparison (mse_sum vs baseline_total_mse):")
                print(f"  Baseline: {baseline_info['baseline_total_mse']:.4f}")
                print(f"  Model mse_sum: {mse_sum:.4f}")
                improvement = baseline_info['baseline_total_mse'] - mse_sum
                improvement_pct = 100 * improvement / baseline_info['baseline_total_mse'] if baseline_info['baseline_total_mse'] > 0 else 0.0
                print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

            # 用 pearson 和选 best，避免 loss 下降但相关变差
            score = val_results['lq_pearson'] + val_results['exp_pearson']
            print(f"  Best score (LQ+EXP Pearson): {best_score:.4f} | Current: {score:.4f} | pred_std LQ={val_results.get('lq_pred_std', 0):.3f} EXP={val_results.get('exp_pred_std', 0):.3f}")
            if score > best_score:
                best_score = score
                checkpoint_path = os.path.join(output_dir, "best_model2.pt")
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_score': best_score,
                        'val_results': val_results,
                    }, checkpoint_path)
                    print(f"\n✅ Saved best model to {checkpoint_path} (score={score:.4f})")
                except Exception as e:
                    print(f"\n⚠️  Warning: Failed to save best model: {e}")
                    print(f"   Continuing training without saving checkpoint...")

        # 保存 checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to save checkpoint for epoch {epoch}: {e}")
            print(f"   Continuing training without saving checkpoint...")

    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model2.pt")
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"\nTraining completed! Final model saved to {final_model_path}")
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to save final model: {e}")
        print(f"   Training completed but final model was not saved.")


if __name__ == "__main__":
    main()

