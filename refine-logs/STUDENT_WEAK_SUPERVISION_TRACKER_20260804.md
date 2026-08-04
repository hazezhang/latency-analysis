# 学生弱监督与专业校准执行追踪表

| Run ID | 里程碑 | 目的 | 系统 / 变体 | 固定评测 | 指标 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| SW0 | M0 | 数据与泄漏审计 | 重建 student-professional manifest 与 fold 映射 | 16-group 专业外层 folds | overlap、样本数、顺序唯一性 | MUST | TODO | 复用 `build_rater_aware_experiment_sets.py` 的文本排除规则。 |
| SW1 | M1 | 专业基线 | S0 professional-only aggregate 双头 | 专业外层 test | LQ/EXP 全套质量指标 | MUST | TODO | 三 seed 前先做单 seed sanity。 |
| SW2 | M1 | 学生直接迁移 | S1 student-only | 同 SW1 | 同 SW1 | MUST | TODO | 只用于测域偏移。 |
| SW3 | M1 | 错误池化反例 | S2 raw student+professional pooling | 同 SW1 | 同 SW1 + 标度偏差 | MUST | TODO | 不作为部署候选。 |
| SW4 | M1/M2 | 主弱监督候选 | S3 student pretrain -> professional calibration | 同 SW1 | 同 SW1 + 对 SW1 paired delta | MUST | TODO | 所有学生预训练样本须排除专业 dev/test 文本。 |
| SW5 | M1/M2 | 多源监督候选 | S4 shared encoder + domain heads + rater type | 同 SW1 | 同 SW1 + 对 SW1 paired delta | MUST | TODO | 不将 rater ID 用作跨新评分者的主输入。 |
| SW6 | M2 | 学生排序敏感性 | S5 normalized/pairwise student pretrain | 同 SW1 | 同 SW1 | SHOULD | TODO | 仅当 SW4 未稳定提升时优先。 |
| SW7 | M3 | 非线性基线 | Ridge/elastic net vs GAM/CatBoost 或 LightGBM | 固定 B2 OOF + 专业外层 | r、MAE、MSE、group CI | MUST | TODO | 确保全部模型接收相同 OOF 特征。 |
| SW8 | M3 | 序列特征 | 无/有历史序列特征 x linear/nonlinear | 同 SW7 | 增量 r/MSE、组别敏感性 | MUST | TODO | 不用未来片段；首片段显式缺失。 |
| SW9 | M4 | rater-aware 探索 | rater heads/ordinal/mixed effects/disagreement | 125-row 配对子集 | NLL、QWK、校准、abstention | SHOULD | TODO | 限定为 zh-en、5 groups 的探索性结果。 |
| SW10 | M5 | 论文证据审计 | 主表、附录、claim-to-evidence 清单 | 所有完成 runs | 完整性、可重现性 | MUST | TODO | 仅报告已经完成且有产物的结果。 |
| SW11 | M6 | 诊断标注试点 | 150--250 hard cases 分层抽样 | 独立标注集 | 一致性、span evidence、feedback 安全 | NEXT | TODO | 先于生成式反馈训练。 |
