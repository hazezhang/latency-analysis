# 学生弱监督与专业校准路线图

**日期**：2026-08-04
**问题**：在专业评分昂贵的条件下，学生评分能否作为弱监督，提高并稳定专业同传自动评价；并以可验证的诊断证据为基础，逐步扩展到教学反馈。
**方法主张**：先以学生评分学习可迁移表示和相对排序，再只用专业训练折完成最终校准；将非线性质量-时间关系、评分者差异和局部时间序列特征纳入同一套泄漏受控评测。

## 已确认的起点与边界

- 共享专业主队列已有 622 个片段、16 个 source-speech groups；主结果继续采用 source-speech-group-held-out 外层交叉验证，所有上游质量预测必须是该外层折的 OOF 预测。
- 已有学生预训练集 622 行（R01=207、R02=73、R04=342）；构建时已排除与专业 dev/test 文本重叠的 207 行。该排除规则必须保留并重新审计。
- R05/R06 的单独标签有明显 evaluator-by-interpreter 标度差异。现有可严格对齐的双评分子集只有 125 行、5 个 speech groups 且仅 zh-en，故 rater-aware 结果只能是探索性分析，不能选择或替换主模型。
- `file_id` 与 `original_segment_id` 足以恢复片段顺序。序列特征只能使用当前片段之前的观测量；跨 fold 统计量必须只由训练折拟合。
- 已有非线性延迟、共享编码器多任务、rater-aware 数据集构建等代码。新工作先复用并统一协议，不把新的大模型作为首要变量。

## Claim Map

| 主张 | 最小可信证据 | 反证/失败含义 |
|---|---|---|
| C1：学生评分在专业标签有限时提供额外监督。 | 在完全相同的专业外层测试片段上，`student pretrain -> professional calibration` 优于 `professional only`，且三 seed、group-cluster CI 与方向切片不相互矛盾。 | 若只提升学生标签预测或只有 raw pooling 有增益，说明没有可迁移的专业监督价值。 |
| C2：收益来自正确建模评分标度和非线性/局部时间信息，而非额外行数或泄漏。 | 不混合 raw scores 的多源模型优于 raw pooling；非线性/序列特征的增益在固定 OOF 协议下成立。 | 若仅 raw pooling 有效或跨组不稳，学生尺度偏差或数据量解释更合理。 |
| C3：自动评分可安全地向诊断反馈扩展。 | 错误类型、证据跨度和反馈建议均有可核验标注；高不确定样本转人工复核。 | 若只有分数相关而缺少诊断真值，系统应保留为评分器，不宣称教学反馈有效。 |

## 阶段一：现在可执行的核心实验

### B1. 冻结数据与协议

- 输入：`data/evaluation/profess_eval_delay_enriched.json`、`data/evaluation/student_eval.json`、现有共享专业 fold。
- 产物：新的 manifest，逐折记录 `segment_id/file_id/original_segment_id/speech_group/interpreter/evaluator_id/supervision_domain`，以及每种监督在 train/dev/test 的样本数。
- 不变量：学生预训练不得包含专业外层 dev/test 的任一 source-interpretation 文本对；模型选择和所有标准化/特征统计仅使用内层训练/dev；测试标签不参与校准。
- Gate：逐折 overlap=0、顺序唯一、每个专业测试组有上游 OOF 预测后，才进入 B2-B5。

### B2. 专业校准与学生弱监督主矩阵

所有系统使用相同 backbone、相同专业外层 folds、三 seed；最终仅在专业测试标签上评价。

| ID | 系统 | 目的 | 地位 |
|---|---|---|---|
| S0 | professional-only, aggregate LQ/EXP | 当前可比基线 | 必跑 |
| S1 | student-only -> professional test | 量化直接域偏移 | 必跑 |
| S2 | raw student + professional pooling | 明确的反例基线 | 必跑，但不作为部署候选 |
| S3 | student pretrain -> professional calibration | 主要候选 | 必跑 |
| S4 | shared encoder + student/professional heads + rater-type embedding | 分离监督来源的多源候选 | 必跑 |
| S5 | within-rater normalized / pairwise student pretrain -> professional calibration | 判断学生绝对分数噪声是否掩盖有效排序 | 次优先 |

- 质量任务：LQ、EXP 分别报告 Pearson、Spearman、MAE、QWK、within-0.5、预测方差、校准曲线；不把 LQ/EXP 压成单一分数作为主结论。
- 主比较：S3/S4 对 S0 的逐 group paired delta，cluster bootstrap CI，三 seed 的均值和标准差；S2 只检验“混合尺度”这一错误假设。
- 成功标准：S3 或 S4 在 LQ 和/或 EXP 上相对 S0 有稳定、可重复的专业测试增益，且不是少数 speech group 或单一方向驱动。

### B3. 非线性与局部序列 baselines

在 B2 选出的最稳定质量预测上，比较 promptness/LAT 或相应专业目标的下游模型。先使用可解释、低容量方法：Ridge/elastic net、GAM、CatBoost 或 LightGBM（三者择二，优先以当前环境可用者为准）；小型 MLP 仅作附录上界。

- 特征族 A：当前片段的 OOF LQ/EXP、delay、长度/压缩率、方向与已验证的结构特征。
- 特征族 B：只含过去信息的序列特征：`previous_delay`、过去 2/3 片段 delay rolling mean、delay slope、先前输出压缩率、speech 内位置、catch-up 指示器。首片段使用明确缺失标志，不能用未来片段填充。
- 比较：A linear、A nonlinear、A+B linear、A+B nonlinear；每个模型在相同 OOF 特征和外层测试片段上评估。
- 成功标准：B 的增益必须在 group-held-out 及解释器留出敏感性中保留；否则只作为描述性分析，不将“trajectory”写为主要贡献。

### B4. Rater-aware 与不确定性分析

- 主队列仍以共享专业均值作为 target，避免把评分者身份当作不可泛化捷径。
- 探索子集：rater-specific heads、rater embedding、mixed-effects/ordinal baseline、latent item score + rater severity、disagreement prediction。
- 输出：aggregate 预测、评分者条件预测、预测分歧/置信度，以及“需要教师复核”的 abstention 曲线。
- 限制：125 行子集的任何改善均不能与 622 片段主结果直接比较；不用于选择 B2/B3 超参数。

### B5. 决策与论文表格

主文仅保留：学生弱监督主矩阵、最强线性/非线性/序列比较、严格协议与方向/组别稳健性。rater-aware、pairwise、MLP、详细校准与 hard-case 分类进附录。

## 阶段二：从评分到诊断反馈的数据建设

先建立 150--250 条分层 hard-case 标注集，按模型大误差、专业评分分歧、学生-专业分歧、短延迟低分、长延迟高分、方向和 interpreter 抽样。每个样本记录：

- 错误类型及严重度：omission、distortion、number/entity/negation、过度压缩、不完整句、repair/disfluency、lag accumulation 等；
- 可复核证据：source/interpretation 的 span、时间戳或对齐证据；
- 最主要问题、教师反馈、可执行练习建议、自动展示是否安全、教师复核需求；
- 标注者 ID、信心和分歧。

系统先输出结构化诊断 JSON，LLM 只负责受约束表述，不能凭空给出错误或建议。评价包括 detection F1、evidence-span agreement、教师正确性/具体性/可操作性/潜在伤害评分，以及高不确定样本的人工转交覆盖率。

## 阶段三：平台闭环验证

- 记录同一学生的练习轮次、自动反馈、采纳情况、重练结果、教师评分和延迟轨迹。
- 主要终点：后测专业评分提升、错误类型改善、教师审核时间；次要终点：学生感知有用性与保留效果。
- 采用教师反馈或无反馈对照，预注册分配/分析，避免把模型预测相关性误写成教学效果。

## 执行顺序与决策门

| 里程碑 | 先决产物 | 运行 | Gate | 风险与处理 |
|---|---|---|---|---|
| M0 | 新的 leakage/order manifest | B1 | overlap=0、fold 可重建 | 身份不一致时先修数据，不运行模型 |
| M1 | 专业与学生训练入口 | S0--S4 单 seed sanity | 非塌缩、全折产物齐全 | 先冻结最小可运行设置 |
| M2 | 完整 quality OOF 预测 | S0--S4 三 seed | S3/S4 是否超过 S0 | 无稳定增益则停止扩展模型，转向学生排序/标注质量分析 |
| M3 | 固定的 B2 最佳候选 | B3 | 非线性/序列是否有跨组增益 | 只保留可解释且稳健的模型 |
| M4 | rater subset | B4 | 是否值得作为不确定性功能 | 仅附录，不影响主模型选择 |
| M5 | 主表、appendix 表 | B5 | 证据闭环 | 结论严格限定为 professional-held-out 结果 |
| M6 | hard-case sampling frame | 阶段二标注试点 | 标注一致性与反馈安全性达到预设门槛 | 未达标时不向学生自动展示自由生成反馈 |

## 必跑与暂缓

**必跑**：B1、S0--S4、B3 的 linear/nonlinear 与无/有序列四格比较、三 seed/group-cluster 报告。
**暂缓**：decoder-only 大模型蒸馏、全量 LoRA/全参微调、音频特征大规模重建、学生反馈偏好优化。这些都应在弱监督主张成立且诊断标注可用后再启动。

## 可复现与 GitHub 规则

- 每个 run 固定记录数据 manifest hash、fold、seed、配置、代码 commit、输出预测路径和失败日志扫描。
- 先写时间戳目录，再生成仅指向已完成产物的汇总表；不以单一最佳 seed 选模型。
- 每个里程碑完成后，将计划、manifest、运行入口、汇总指标和不含受限原始数据的说明分批提交并推送；原始评分、音频、可识别文本不上传公开仓库。
