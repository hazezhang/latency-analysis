# 学生弱监督与专业校准执行追踪表

| Run ID | 里程碑 | 目的 | 系统 / 变体 | 固定评测 | 指标 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| SW0 | M0 | 数据与泄漏审计 | `run_student_weak_supervision_m0.py` 构建 per-outer-fold 学生训练集与 manifest | 16-group 专业外层 folds | overlap、样本数、顺序唯一性 | MUST | DONE | 专业 622 segments/16 groups；学生 raw 829 rows；每折 763--829 行，16/16 overlap=0。 |
| SW1 | M1 | 专业基线 sanity | S0 professional-only aggregate 双头 | professional dev（sanity only） | LQ/EXP Pearson、raw std | MUST | DONE | seed 20260804；best dev score=.7084；最终 dev LQ=.3154、EXP=.3286；未塌缩。 |
| SW1-formal | M1 | 专业基线正式外折 | S0：professional outer-train + professional inner-dev；professional outer speech test | 16-group professional outer folds | LQ/EXP Pearson、MAE、MSE、raw std、fold summary | MUST | DONE | 16/16 folds、622 segments、fatal=0；seed 20260805。fold-mean LQ r=.6088、EXP r=.5356；LQ MAE=.4012、EXP MAE=.3449；raw pred std=.4628/.3672。 |
| SW2 | M1 | 学生直接迁移 sanity | S1：仅学生原始 LQ/EXP 分数训练，无专业校准 | professional dev（sanity only） | LQ/EXP Pearson、raw std | MUST | DONE | outer_01 student train 807 行；best dev score=.5808；最终 dev LQ=.3622、EXP=.1775；未塌缩。正式 outer-test 不使用专业 dev 选模。 |
| SW2-formal | M1 | 学生直接迁移正式外折 | S1：raw student fit + deterministic student-only dev；professional outer speech test | 16-group professional outer folds | LQ/EXP Pearson、MAE、MSE、raw std、fold summary | MUST | DONE | 16/16 folds、622 segments；raw student labels、无专业 calibration；fold-mean LQ r=.5334、EXP r=.3192；LQ MAE=.4897、EXP MAE=.6976。 |
| SW3 | M1 | 错误池化反例 | S2 raw student+professional pooling | 同 SW1 | 同 SW1 + 标度偏差 | MUST | TODO | 不作为部署候选。 |
| SW4 | M1/M2 | 主弱监督候选 | S3 student pretrain -> professional calibration | 同 SW1 | 同 SW1 + 对 SW1 paired delta | MUST | PRELIMINARY | 旧版 16/16 结果 LQ r=.6049、EXP r=.5196，但审计发现 student fit/dev 与 professional inner-dev/test 有 1,035 条 text-pair overlap，不能作为正式结论；严格无 overlap 版本已启动。 |
| SW5 | M1/M2 | 最小多源候选 | S4a shared encoder + separate student/professional heads | 同 SW1 | 同 SW1 + 对 SW1 paired delta | MUST | PRELIMINARY | 旧版 16/16 结果 LQ/EXP r=.6039/.5220，但同样存在 student fit/dev 与 professional inner-dev/test overlap；严格无 overlap 版本待 S3 完成后启动。 |
| SW5b | M2 | 评分者类型敏感性 | S4b rater-type embedding + shared head | 同 SW1 | 同 SW1 | SHOULD | TODO | 仅在 S4a 通过 quality gate 后运行，列入附录。 |
| SW6 | M2 | 学生排序敏感性 | S5 within-rater normalized/pairwise pretrain | 同 SW1 | 同 SW1 | SHOULD | TODO | 标签定义敏感性，不与 S1 混用。 |
| SW7 | M3 | 非线性基线 | B3a：固定 A，Ridge vs GAM vs CatBoost | 固定 B2 OOF + 专业外层 | r、MAE、MSE、group CI | MUST | TODO | 只在 M2 gate 通过后使用学生 OOF，否则用 S0。 |
| SW8 | M3 | 序列特征 | B3b：A/A+B x 最稳健 linear/nonlinear | 同 SW7 | 增量 r/MSE、组别敏感性 | MUST | TODO | 仅 online-compatible 过去信息；首片段显式缺失。 |
| SW9 | M4 | rater-aware 探索 | rater heads/simple mixed effects/disagreement | 125-row 配对子集 | NLL、QWK、校准、abstention | SHOULD | TODO | 不运行 latent/Bayesian/Rasch；限定 zh-en、5 groups。 |
| SW10 | M5 | 论文证据审计 | 主表、附录、claim-to-evidence 清单 | 所有完成 runs | 完整性、可重现性 | MUST | TODO | 仅报告已经完成且有产物的结果。 |
| SW11 | M6 | 第一轮诊断标注 | 150--250 hard cases：taxonomy 与证据 | 独立标注集 | kappa/alpha、span F1 | NEXT | TODO | 达到 >=.60 的 taxonomy 与 span F1 gate 后，才进入反馈标注。 |
| SW12 | M6 | 第二轮反馈标注 | main problem、feedback、exercise、安全性 | taxonomy 通过的条目 | 反馈一致性、潜在伤害 | NEXT | TODO | 不与第一轮同时扩大。 |
