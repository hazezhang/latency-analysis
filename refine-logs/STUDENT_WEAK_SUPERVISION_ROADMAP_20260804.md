# 学生弱监督与专业校准路线图

**日期**：2026-08-04
**问题**：在专业评分昂贵的条件下，学生评分能否作为弱监督，提高并稳定专业同传自动评价；并以可验证的诊断证据为基础，逐步扩展到教学反馈。
**方法主张**：先以学生评分学习可迁移表示和相对排序，再只用专业训练折完成最终校准；将非线性质量-时间关系、评分者差异和局部时间序列特征纳入同一套泄漏受控评测。

## 已确认的起点与边界

- 共享专业主队列由专业评分者对译员表现的评分构成，共 622 个片段、16 个 source-speech groups；主结果继续采用 source-speech-group-held-out 外层交叉验证，所有上游质量预测必须是该外层折的 OOF 预测。
- 学生原始弱监督集有 829 条完整 LQ/EXP 评分行（R01=243、R02=102、R04=484），不是专业主队列的定义。旧固定 dev/test 切分在排除 207 条重叠行后剩 622 条；新的 16-fold 协议改为按每个专业 outer test speech 单独排除相同 source-target 对，每折保留 763--829 条学生训练行，且 16/16 folds overlap=0。
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
| S1 | raw-student-only -> professional test | 每个 outer fold 以该 fold 隔离后的 763--829 条学生原始 LQ/EXP 分数训练同一架构；不做专业校准，直接测试对应专业 outer test | 必跑 |
| S2 | raw student + professional pooling | 明确的反例基线 | 必跑，但不作为部署候选 |
| S3 | student pretrain -> professional calibration | 主要候选 | 必跑 |
| S4a | shared encoder + separate student/professional heads | 最小多源模型：学生监督共享表示，专业 head 负责尺度校准 | 必跑 |
| S4b | shared encoder + rater-type embedding + shared output head | 评分者类型条件化的附录变体 | 次优先，仅在 S4a 有信号时运行 |
| S5 | within-student-rater normalized or pairwise student pretrain -> professional calibration | 将学生标度/排序定义与域迁移区分开 | 次优先 |

- 质量任务：LQ、EXP 分别报告 Pearson、Spearman、MAE、QWK、within-0.5、预测方差、校准曲线；不把 LQ/EXP 压成单一分数作为主结论。
- S1 的标签定义固定为学生原始分数；不进行学生评分者内标准化、不聚合学生评分、不假设未见 student evaluator 可泛化。学生内标准化/排序只属于 S5，学生 aggregate 仅作为 S5 的附录敏感性分析。
- 主比较：S3/S4a 对 S0 的逐 group paired delta，group-cluster bootstrap CI，三 seed 的均值和标准差；S2 只检验“混合尺度”这一错误假设。S4b 不参与首轮主比较。
- 预先注册的质量 gate（逐 LQ/EXP 判断）：候选相对 S0 的平均 Pearson delta >= +0.02，三个 seed 至少两个 delta > 0，group-cluster 95% CI 的下界 >= -0.02 且上界 > 0；MAE 增加不得超过 0.02；预测 SD 不得低于 0.10 或 S0 的 80%（取更严格者）；两个方向的平均 Pearson delta 都不得 < -0.02。满足至少一个维度，才可称为学生监督的候选增益。

### B3. 非线性与局部序列 baselines

只有 B2 中至少一种学生监督方法通过上述 LQ 或 EXP 的质量 gate，才以该方法的 OOF 质量预测进入 promptness/LAT 下游实验；否则 B3 只使用 S0 的 OOF 质量预测，学生监督主张停止推进，不能以后续偶然的下游相关性补救解释。

先以可解释、低容量模型完成两阶段比较，避免把模型族、特征族和质量模型同时搜索。小型 MLP 仅作附录上界。

- 特征族 A：当前片段的 OOF LQ/EXP、delay、长度/压缩率、方向与已验证的结构特征。
- 特征族 B（online-compatible）：真实已观测的 `previous_delay`、过去 2/3 片段 rolling past delay、past-delay slope、已完成前一片段的 compression ratio、按已完成片段数计算的 speech position、catch-up 指示器。首片段使用明确缺失标志；speech position 按真实 `file_id + original_segment_id` 顺序计算。
- 只允许 B 中部署时可计算的真实 timing/text 特征。先前 LQ/EXP 若作为特征，必须来自同一外层 fold 的 OOF 模型；全 speech 均值、未来片段、future-aware trajectory 和完整 speech 归一化只能标为 post-hoc，不进入主模型。
- B3a（非线性）：固定特征族 A，比较 Ridge、GAM、CatBoost；选择以三 seed/group 稳健性而非单点最优为准。
- B3b（序列增益）：固定 B3a 中一个最稳健线性模型和一个最稳健非线性模型，仅比较 A linear、A+B linear、A nonlinear、A+B nonlinear。
- 成功标准：B 的平均 Pearson 增益 >= +0.02、group-cluster CI 下界 >= -0.02、MAE 增加 <= 0.02，并在解释器留出敏感性中不反向；否则只作为描述性分析，不将“trajectory”写为主要贡献。

### B4. Rater-aware 与不确定性分析

- 主队列仍以共享专业均值作为 target，避免把评分者身份当作不可泛化捷径。
- 探索子集仅比较：rater-specific heads、简单 mixed-effects baseline、disagreement classifier/regressor、aggregate prediction uncertainty 与 abstention curve。
- 输出：aggregate 预测、评分者条件预测、预测分歧/置信度，以及“需要教师复核”的 abstention 曲线。
- 限制：125 行子集的任何改善均不能与 622 片段主结果直接比较；不用于选择 B2/B3 超参数。复杂 Bayesian hierarchical、many-facet Rasch 和 latent neural crowd 模型暂缓至新增专业评分者后。

### B5. 决策与论文表格

主文仅保留：学生弱监督主矩阵、最强线性/非线性/序列比较、严格协议与方向/组别稳健性。rater-aware、pairwise、MLP、详细校准与 hard-case 分类进附录。

## 阶段二：从评分到诊断反馈的数据建设

先建立 150--250 条分层 hard-case 标注集，按模型大误差、专业评分分歧、学生-专业分歧、短延迟低分、长延迟高分、方向和 interpreter 抽样。标注必须分两轮，避免把错误识别分歧和教学建议分歧混为一谈。

- 第一轮（错误与证据）：error type、severity、source span、interpretation span、timestamp、confidence、teacher-review required，以及标注者 ID。先测 taxonomy 的类别一致性与证据跨度一致性。
- Gate：核心 taxonomy 达到预先设定的一致性门槛后才大规模进入第二轮。建议试点门槛为核心类别的 Krippendorff alpha 或加权 kappa >= 0.60，且 evidence-span overlap F1 >= 0.60；若不足，先修订定义和标注指南。
- 第二轮（反馈与练习）：仅对第一轮稳定的条目标注 main problem、feedback text、recommended exercise、safe to display 与 potential harm。

系统先输出结构化诊断 JSON，LLM 只负责受约束表述，不能凭空给出错误或建议。评价包括 detection F1、evidence-span agreement、教师正确性/具体性/可操作性/潜在伤害评分，以及高不确定样本的人工转交覆盖率。

## 阶段三：平台闭环验证

- 采用 matched material sets 的 pretest-posttest crossover：每名学生在不同但难度匹配的材料上经历自动反馈与对照条件，记录练习轮次、反馈采纳、重练结果、教师评分和延迟轨迹。
- 主要终点：blinded 专业后测评分提升、错误类型改善、教师审核时间；次要终点：学生感知有用性、延迟 retention test 与条件顺序效应。
- 预注册材料配对、随机化/顺序平衡、盲评和分析；不将模型预测相关性写成教学效果。

## 执行顺序与决策门

| 里程碑 | 先决产物 | 运行 | Gate | 风险与处理 |
|---|---|---|---|---|
| M0 | 新的 leakage/order manifest | B1 | overlap=0、fold 可重建 | 身份不一致时先修数据，不运行模型 |
| M1 | 专业与学生训练入口 | S0--S4a 单 seed sanity | 非塌缩、全折产物齐全 | 先冻结最小可运行设置；S4b 暂缓 |
| M2 | 完整 quality OOF 预测 | S0--S4a 三 seed | 仅在任一学生方法满足预先注册 quality gate 时进入 B3；否则 B3 仅用 S0，停止学生监督主张 | 无稳定增益则优先 S5 的标签定义敏感性，不扩张模型 |
| M3 | 固定的 B2 候选或 S0 | B3a 后 B3b | 非线性/序列是否满足预设跨组门槛 | 只保留可解释且稳健的模型 |
| M4 | rater subset | B4 | 是否值得作为不确定性功能 | 仅附录，不影响主模型选择 |
| M5 | 主表、appendix 表 | B5 | 证据闭环 | 结论严格限定为 professional-held-out 结果 |
| M6 | hard-case sampling frame | 第一轮 taxonomy/evidence 标注试点 | 先达到 taxonomy/evidence 一致性门槛，才进入第二轮反馈标注 | 未达标时不向学生自动展示生成式反馈 |

## 必跑与暂缓

**必跑**：B1、S0--S4a、B3a 的 Ridge/GAM/CatBoost、B3b 的四格比较、三 seed/group-cluster 报告。S1 固定为原始学生分数直接迁移；S4b 和 S5 由 M2 决策门触发。
**暂缓**：decoder-only 大模型蒸馏、全量 LoRA/全参微调、音频特征大规模重建、学生反馈偏好优化。这些都应在弱监督主张成立且诊断标注可用后再启动。

## 可复现与 GitHub 规则

- 每个 run 固定记录数据 manifest hash、fold、seed、配置、代码 commit、输出预测路径和失败日志扫描。
- 先写时间戳目录，再生成仅指向已完成产物的汇总表；不以单一最佳 seed 选模型。
- 每个里程碑完成后，将计划、manifest、运行入口、汇总指标和不含受限原始数据的说明分批提交并推送；原始评分、音频、可识别文本不上传公开仓库。
