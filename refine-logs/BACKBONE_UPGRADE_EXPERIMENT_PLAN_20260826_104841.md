# Multilingual Backbone Upgrade Experiment Plan

**问题**：更强的 multilingual backbone 能否提升 LQ/EXP 的绝对预测性能，并验证学生监督规律是否跨 backbone 成立？
**方法主张**：将 backbone 效应与 supervision 效应分开评估；先筛选 backbone，再只对胜出模型复验核心监督结论。
**日期**：2026-08-26

## Claim Map

| Claim | 为什么重要 | 最低可信证据 | 实验块 |
|---|---|---|---|
| C1：监督规律跨 backbone 稳定 | 排除当前结论只是 InfoXLM/COMETKiwi 特例 | 新 backbone 上 S1 仍弱于 S0；S3 在 10%/25% 专业数据下的收益大于 100%；EXP 谨慎降权方向在多数 seed 一致 | B2-B4 |
| C2：新 backbone 提高系统性能上限 | 最终目标是更强、更可靠的评价模型 | 相同 outer folds 上，专业测试 LQ/EXP Pearson 有实质提升，同时 MAE 不明显恶化、prediction SD 不塌缩 | B1-B4 |
| Anti-claim：收益只是参数量或训练协议变化 | 保证比较可解释 | 相同数据、fold、head、outer test 和评价脚本；所有选择只用 inner/dev；同时报告参数量、显存、速度 | B1-B5 |

## Backbone 候选

### P0：Unbabel/wmt23-cometkiwi-da-xl

- 类型：QE-specific COMET，底层 `facebook/xlm-roberta-xl`，约 3.5B 参数。
- 优点：与当前 `Unbabel/wmt22-cometkiwi-da` 最接近；仍是 source+MT quality estimation，最适合做“只换 backbone”的主比较。
- 风险：训练成本和显存明显增加；当前 LoRA target、hidden size 和 encoder 路径必须做兼容检查。
- 建议：**首选核心候选**。先在 RTX 5090 32GB 做单 fold smoke；通过后做单 seed screening。

### P1：BAAI/bge-m3

- 类型：通用 multilingual encoder，约 568M 参数，支持长上下文。
- 优点：规模接近当前 InfoXLM-large，但训练更新、跨语言表示能力强；可以检验规律是否跨越 COMET/QE 模型族。
- 风险：不是专门为 translation quality estimation 训练；需要新增通用 `AutoModel` loader、模型特定 pooling 和明确的 src/mt 输入模板。
- 建议：作为**跨模型族泛化候选**，在 P0 之后做单 seed 筛选。

### P1：intfloat/multilingual-e5-large-instruct

- 类型：multilingual instruction embedding encoder，约 560M 参数。
- 优点：硬件成本接近当前模型；适合判断强 multilingual sentence representation 是否足以替代 QE-specific pretraining。
- 风险：retrieval/embedding objective 与 LQ/EXP prediction 不完全对齐；pair encoding 和 instruction template 会引入额外设计变量。
- 建议：与 BGE-M3 二选一，优先选择在中文/英文 direction smoke 中更稳定者，不必两个都跑完整实验。

### P2：Qwen/Qwen3-Embedding-0.6B

- 类型：较新的 decoder-style multilingual embedding model，约 0.6B 参数。
- 优点：模型较新，语言覆盖和语义表示潜力高，成本仍可控。
- 风险：不是现有 encoder 的 drop-in replacement；需要 EOS/last-token pooling、输入模板和 LoRA target 重写，比较不再是纯 backbone substitution。
- 建议：仅作为第二轮探索，不进入第一轮核心矩阵。

### 暂不作为核心：wmt23-cometkiwi-da-xxl / XCOMET-XL/XXL

- `wmt23-cometkiwi-da-xxl` 底层 XLM-R XXL，约 10.7B 参数；在单张 24-32GB GPU 上按当前训练流程风险过高，需要量化、gradient checkpointing 或多卡。
- XCOMET 强，但其训练目标包含 reference/error-span 结构。当前主输入只有 `src + mt`，直接替换会改变任务定义，不适合作为第一轮公平 backbone 对照。
- 可在 XL 明显成功后作为上限实验，不应阻塞主线。

## 实验块

### B0：兼容性与资源 smoke

- Claim：候选能在当前数据和代码路径中产生非塌缩预测。
- 数据：固定 `outer_01`，professional-only；再取一个 student-pretrain fold。
- 系统：current WMT22 COMETKiwi、WMT23 COMETKiwi XL；通用 encoder 候选最多一个。
- 检查：tokenizer pair encoding、hidden size、LoRA target、forward shape、checkpoint save/load、prediction export。
- 指标：显存峰值、每 epoch 时间、prediction SD、dev Pearson/MAE、fatal scan。
- Gate：完整跑通；LQ/EXP prediction SD 均 > 0.15；无 NaN/OOM；单 fold 性能不比当前模型低超过 Pearson 0.05。
- 优先级：MUST-RUN。

### B1：Backbone screening（只比较绝对性能）

- Claim：新 backbone 能提高 evaluation model 的性能上限。
- 数据：完全相同的 professional-only outer folds，先用 seed 20260825、100% professional data。
- 系统：current backbone、WMT23 COMETKiwi XL、通过 B0 的一个通用 encoder。
- 固定项：同一 fold、训练/测试数据、双头定义、评价代码、outer-test 隔离。
- 允许变化：每个 backbone 仅可在 inner/dev 上选择 batch size、LoRA target 和一个小范围 learning rate；不得用 outer-test 选模型。
- Primary：LQ/EXP Pearson。
- Secondary：MAE、MSE、Spearman、prediction SD、方向分组结果、GPU-hours。
- Advance gate：两维平均 Pearson 相对 current backbone 至少 +0.015；任一维 MAE 恶化不超过 0.01；无 prediction collapse；两个语言方向无明显反转。
- 优先级：MUST-RUN。

### B2：Full-data supervision transfer

- Claim：监督规律不是当前 backbone 特例。
- 只对 B1 胜出模型运行：
  - S0：Professional-only。
  - S1：raw Student-only，直接在 professional outer test 上评估，不做专业校准。
  - S3：Student pretrain -> Professional finetune。
- Seeds：20260825、20260826、20260827。
- 关键比较：S1 vs S0；S3 vs S0 的 paired seed delta。
- Gate：S1 在多数 seed 明显弱于 S0；S3 至少保持 current-backbone 的 LQ/EXP 水平，且绝对性能达到 B1 gate。
- 优先级：MUST-RUN。

### B3：Low-resource professional-data test

- Claim：student supervision 的主要价值出现在 expert-label scarcity。
- 第一阶段比例：10%、25%、100%；只比较 S0 与 S3，使用与现有 learning curve 相同的 nested subsets。
- Seeds：3 seeds，paired by seed/fold/subset。
- 只有在 10%/25% 显示正向趋势后，才补 50%、75%。
- Primary：每个比例的 paired `Delta = S3 - S0`，分别报告 LQ/EXP Pearson。
- Secondary：MAE/MSE 和 prediction SD。
- Gate：10% 或 25% 下至少 2/3 seeds 的 Pearson delta 为正；低资源平均增益大于 100% 增益；MAE 恶化不超过 0.02。
- 优先级：MUST-RUN，但采用 10/25/100 的节省版起步。

### B4：Dimension-specific EXP weighting

- Claim：student EXP 的 reliability 低于 student LQ，这一现象跨 backbone 存在。
- 系统：winner backbone 上 matched `Student EXP weight = 1.00 vs 0.50`。
- Seeds：3 seeds；相同 folds、初始化 seed、student/professional 数据和训练流程。
- Primary：EXP Pearson、EXP MAE 的 paired delta。
- 约束：LQ Pearson 不应明显降低（容忍 -0.01）。
- Gate：至少 2/3 seeds 的 EXP Pearson 提升或 EXP MAE 改善，且平均结果不反向。
- 优先级：MUST-RUN；不继续搜索 0.2/0.3/0.4 等细网格。

### B5：效率和可靠性报告

- 报告：参数量、trainable 参数、峰值显存、训练时间、推理吞吐、预测 SD、bootstrap CI、语言方向分组。
- 目的：避免只报告 Pearson，并量化更强 backbone 的实际代价。
- 优先级：MUST-RUN。

## 执行顺序

| Milestone | 运行内容 | Stop/Go gate | 相对成本 |
|---|---|---|---|
| M0 | 当前模型 + WMT23 XL 单 fold smoke | 兼容、无 OOM、非塌缩 | 低 |
| M1 | 100% professional-only 单 seed screening | 平均 Pearson +0.015，MAE 受控 | 中 |
| M2 | Winner 的 S0/S1/S3 三 seed | 监督主规律可复现 | 中高 |
| M3 | Winner 的 10/25/100 learning curve 三 seed | 低资源收益方向稳定 | 高 |
| M4 | Winner 的 EXP weight 1.00/0.50 三 seed | EXP 谨慎降权方向稳定 | 中 |
| M5 | 如 M3 通过，再补 50/75% | 完整 learning-curve figure | 中高 |

## 计算预算建议

- 当前模型 InfoXLM-large 约 550M，作为 1x 参考。
- WMT23 COMETKiwi XL 约 3.5B，预计参数量约 6x；LoRA 训练的实际时间/显存需由 M0 实测，不能直接按参数量线性估计。
- WMT23 XXL 约 10.7B，不建议在现有单卡 24-32GB 环境直接进入核心实验。
- BGE-M3 / multilingual-e5-large-instruct / Qwen3-Embedding-0.6B 约为当前模型同一数量级，但需要额外适配和公平输入模板。
- 不应一开始复制当前完整 `160 folds x seed`。先通过 M0/M1，再扩大到三 seed 和 learning curve。

## 主要风险与缓解

- **OOM / 训练过慢**：先降低 batch size，启用 gradient accumulation 和 gradient checkpointing；仍不稳定则停止 XL，而不是改变 outer protocol。
- **模型接口不兼容**：将 COMET loader 与通用 `AutoModel` loader 分离；不要在 notebook 内继续堆条件分支。
- **比较不公平**：固定 outer folds 和 head；所有 backbone-specific 超参数仅在 inner/dev 决定。
- **大模型过拟合小数据**：坚持 LoRA/head-first，监控 prediction SD、seed variance 和 group bootstrap。
- **只提高 correlation、不改善 MAE**：分别报告 representation/ranking 与 absolute calibration，不把 mixed MAE 包装成全面提升。
- **许可证**：Unbabel WMT23 模型标注为 CC-BY-NC-SA-4.0；适合研究比较，但部署前必须单独核对许可要求。

## 最终建议

第一轮只启动两个候选：

1. **Unbabel/wmt23-cometkiwi-da-xl**：核心、低解释风险，回答“更强 QE backbone 是否提高上限”。
2. **BAAI/bge-m3**：跨模型族候选，回答“监督规律是否依赖 COMET-specific pretraining”。

暂不并行跑 XXL、XCOMET 和 Qwen3-Embedding。只有 WMT23 XL 通过 smoke 和 professional-only screening 后，才进入 S0/S1/S3、low-resource 和 EXP weighting 的完整复验。
