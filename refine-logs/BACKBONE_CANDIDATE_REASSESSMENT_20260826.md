# Backbone Candidate Reassessment

## 选型依据

候选不是按“模型越新越好”选择，而是按五个条件排序：

1. 是否保持 encoder paradigm，能公平替换当前 InfoXLM encoder；
2. 是否支持 source + MT pair 输入，并能输出稳定的 sequence representation；
3. 是否能在当前 LoRA + LQ/EXP 双头回归代码中复现；
4. 是否有足够的 multilingual 覆盖和公开 checkpoint，便于重复实验；
5. 是否能在现有 GPU 和小数据规模下控制显存、过拟合与训练时间。

## 重新排序

### 1. mmBERT-base：主更新候选

你的判断基本正确。它最适合回答“更新的 multilingual encoder 是否能提高绝对性能，同时保持学生监督规律”。

- 优点：现代 multilingual encoder、保持 encoder 范式；比 decoder-style embedding 模型更容易和当前 pair regression 任务对齐。
- 研究价值：如果 mmBERT-base 的 S0/S1/S3 规律与当前模型一致，说明监督结论不是某个旧 COMET backbone 的特例；如果绝对 Pearson/MAE 也提高，则能支持 backbone upgrade claim。
- 需要确认：具体 Hugging Face repo id、参数量、hidden size、tokenizer 语言覆盖、许可证和 Transformers/PEFT 支持。`mmBERT-base` 不是一个足够精确的 checkpoint 名称，正式实验前必须冻结完整 repo id 和 revision。
- 适配工作：新增通用 `AutoModel` loader，明确 src/mt pair 模板和 pooling；重新核对 LoRA target modules，不应默认沿用 `q_proj/v_proj`。
- 结论：**P0 主候选**。

### 2. XLM-R-large：可信 strong baseline

- 优点：成熟、公开、稳定，仍是 standard multilingual encoder；约 550M 级别，硬件成本接近当前模型；tokenizer 和 LoRA 适配风险低。
- 研究价值：它不是 novelty，而是控制变量。若 XLM-R-large 明显超过当前模型，说明收益可能来自更强的通用 encoder；若不超过，则 WMT23 COMETKiwi XL 的收益才更能归因于 QE-specific pretraining。
- 适配工作：需要将当前 COMET wrapper 改为 raw `AutoModel` 路径，但模型本身的 hidden size、attention target 和 pair encoding 都容易验证。
- 结论：**P0 强基线，建议与 mmBERT-base 同时做 smoke 和 100% professional screening**。

### 3. Unbabel/wmt23-cometkiwi-da-xl：QE-specific upper-bound candidate

- 优点：仍然是 COMETKiwi 质量估计范式，最接近当前任务；比 raw encoder 更能测试“更强 QE 预训练是否提升上限”。
- 风险：约 3.5B 级别，显存和训练时间显著增加；它同时改变了 backbone 规模和 QE pretraining，解释上不如 XLM-R/mmBERT 干净。
- 结论：**P0/P1 之间**。用于性能上限，不作为唯一的跨 backbone 泛化证据。

### 4. LFM2.5 Encoder：探索性候选

- 你提出它是合理的“前沿架构”方向，但它不能默认视为普通 Transformer encoder 的 drop-in replacement。
- 需要确认：确切模型名称是否为 encoder checkpoint，而不是 LFM2.5 decoder/instruct/embedding 变体；官方 Transformers 支持、hidden-state 接口、attention/pooling 方式、LoRA/PEFT target 和 multilingual coverage。
- 如果它是 hybrid/recurrent 或特殊 state-space 结构，`src + mt` 的 pair interaction、sequence position 和 pooling 都可能与当前 encoder 不同；结果将同时混入架构、输入模板和训练接口变量。
- 结论：**P1/P2 探索候选**。先做单 fold接口和非塌缩 smoke，不通过就不进入主实验矩阵。

## 推荐的最小比较矩阵

第一轮不要把所有候选都跑完整 learning curve，而是：

| 阶段 | Model | Setting | 目的 |
|---|---|---|---|
| M0 | Current COMETKiwi | S0, outer_01 | 复现控制 |
| M0 | XLM-R-large | S0, outer_01 | 标准 encoder 兼容性 |
| M0 | mmBERT-base | S0, outer_01 | 新 multilingual encoder 兼容性 |
| M0 | WMT23 COMETKiwi XL | S0, outer_01 | QE-specific 强模型兼容性 |
| M0 | LFM2.5 Encoder | S0, outer_01 | 仅接口可行性，不承诺完整实验 |
| M1 | Current / XLM-R / mmBERT / XL | S0, 100%, seed 20260825 | 绝对性能 screening |
| M2 | winner | S0/S1/S3, 3 seeds | 监督规律泛化 |
| M3 | winner | 10/25/100%, S0/S3, 3 seeds | label-efficiency |
| M4 | winner | EXP weight 1.00/0.50, 3 seeds | EXP reliability |

## 决策规则

- 如果 mmBERT-base 超过 XLM-R-large 且保持非塌缩：它是主 backbone。
- 如果 XLM-R-large 超过 mmBERT-base：优先采用 XLM-R-large，mmBERT 作为负结果或附录对照。
- 如果 WMT23 COMETKiwi XL 只提升 Pearson 但 MAE、稳定性或成本明显恶化：报告为性能上限候选，不作为最终部署模型。
- 如果 LFM2.5 需要大量架构特化适配：不要把它与标准 encoder 放在同一个主表中，转为 future work 或单独 exploratory result。

## 当前建议

正式优先级改为：

1. `mmBERT-base`：主新模型；
2. `XLM-R-large`：强而可信的控制基线；
3. `Unbabel/wmt23-cometkiwi-da-xl`：QE-specific 上限；
4. `LFM2.5 Encoder`：接口 smoke 后再决定是否继续。
