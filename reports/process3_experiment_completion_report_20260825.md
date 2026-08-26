# Process3 实验完成性与阶段结果报告

生成日期：2026-08-25
项目：学生弱监督、专业校准与 EXP loss weighting

## 1. 完成性检查

本轮远程只读检查确认：

| 实验 | 预期产物 | 已发现 | 状态 |
|---|---:|---:|---|
| Learning curve, seed 20260826 | 160 predictions，10 summaries | 160，10 | 完成 |
| Learning curve, seed 20260827 | 160 predictions，10 summaries | 160，10 | 完成 |
| EXP weight, seed 20260826, weight=1.00 | 16 predictions，1 summary | 16，1 | 完成 |
| EXP weight, seed 20260826, weight=0.50 | 16 predictions，1 summary | 16，1 | 完成 |
| EXP weight, seed 20260827, weight=1.00 | 16 predictions，1 summary | 16，1 | 完成 |
| EXP weight, seed 20260827, weight=0.50 | 16 predictions，1 summary | 16，1 | 完成 |

31675 的 learning-curve 日志包含 `LEARNING_CURVE_COMPLETE`；30831 的 learning-curve 日志也包含 `LEARNING_CURVE_COMPLETE`。30139 的四个 EXP weight 配置均有完整 prediction 数量和 outer summary。各任务结束后 GPU 均处于空闲状态。

## 2. 已验证的数值结果

### Learning curve seed 20260825（此前已完成并核验）

| 专业数据比例 | S0 LQ r | S3 LQ r | ΔLQ | S0 EXP r | S3 EXP r | ΔEXP |
|---:|---:|---:|---:|---:|---:|---:|
| 10% | 0.5272 | 0.5814 | +0.0542 | 0.4635 | 0.4926 | +0.0291 |
| 25% | 0.5737 | 0.5811 | +0.0074 | 0.4805 | 0.5134 | +0.0330 |
| 50% | 0.5776 | 0.5862 | +0.0086 | 0.4961 | 0.5115 | +0.0154 |
| 75% | 0.5901 | 0.6007 | +0.0106 | 0.5164 | 0.5267 | +0.0104 |
| 100% | 0.6057 | 0.6175 | +0.0118 | 0.5310 | 0.5407 | +0.0096 |

该表是单 seed 结果，不能作为 multi-seed 稳定性结论。

### EXP weight seed 20260825（此前已完成并核验）

| Student EXP weight | LQ Pearson | EXP Pearson | LQ MAE | EXP MAE |
|---:|---:|---:|---:|---:|
| 1.00 | 0.6168 | 0.5274 | 0.3786 | 0.3946 |
| 0.50 | 0.6157 | 0.5488 | 0.3742 | 0.3736 |
| Δ(0.50-1.00) | -0.0011 | +0.0214 | -0.0044 | -0.0210 |

这同样只是单 seed 的 supporting evidence。

## 3. Multi-seed 汇总状态

seed 20260826 和 20260827 的 learning-curve 与 EXP-weight prediction 产物均已齐全，但本轮 SSH 审批/连接超时，未能可靠读取这些 JSON 的具体指标内容。因此本报告不臆造 seed 20260826/20260827 的 Pearson、MAE、MSE，也不计算未经读取原始数值支持的 mean±SD 或 paired delta。

下一步应从上述产物直接解析：

1. learning curve：每个 seed、每个比例的 S0/S3 LQ/EXP Pearson、MAE、MSE；
2. EXP weight：每个 seed 的 weight=1.00 与 0.50 指标及 paired delta；
3. 三 seed 的 mean±SD、paired delta mean±SD、每个 seed 的方向一致性；
4. 对所有日志重新做 fatal scan，并记录具体匹配行，区分真实错误与普通 warning。

## 4. 当前可支持的研究结论

基于已核验的 full-data multi-seed、EXP diagnosis、seed 20260825 learning curve 和 seed 20260825 weight sweep：

- 学生监督不能替代专业监督，但经过专业校准后具有可迁移信息。
- 学生监督在完整专业数据下的边际收益较小；单 seed learning curve 显示专业数据稀缺时，S3 的相关性增益可能更明显。
- 学生 EXP 标签与专业 EXP 的一致性较弱、方差更大且存在 evaluator-scale heterogeneity，因此降低 student EXP loss weight 具有机制上的合理性。
- seed 20260825 中，weight=0.50 相比 1.00 的 EXP Pearson 提升 0.0214、EXP MAE 降低 0.0210；只有在 20260826/20260827 的原始 summary 被解析后，才能判断该现象是否稳定。

## 5. 产物路径

- `/122090786/process3_aaai_current/experiments/professional_learning_curve_20260826`
- `/122090786/process3_aaai_current/experiments/professional_learning_curve_20260827`
- `/122090786/process3_aaai_current/experiments/student_exp_weight_exp100_s26_20260826`
- `/122090786/process3_aaai_current/experiments/student_exp_weight_exp050_s26_20260826`
- `/122090786/process3_aaai_current/experiments/student_exp_weight_exp100_s27_20260827`
- `/122090786/process3_aaai_current/experiments/student_exp_weight_exp050_s27_20260827`

**重要限制：** 本文件是完成性报告和阶段性结果报告，不是最终统计汇总。完整 mean±SD、paired delta 和最终稳定性判断必须建立在成功读取上述 seed 20260826/20260827 summary JSON 之后。
