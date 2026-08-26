# Multilingual Backbone Upgrade Experiment Tracker

| Run ID | Milestone | 目的 | 模型 / Setting | 数据 | Seeds | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| BB001 | M0 | 当前基线 smoke | WMT22 COMETKiwi, S0 | outer_01 | 20260825 | MUST | TODO | 验证新 runner 与旧结果一致 |
| BB002 | M0 | XL 兼容性和显存 | WMT23 COMETKiwi XL, S0 | outer_01 | 20260825 | MUST | TODO | 记录显存、速度、prediction SD |
| BB003 | M0 | XL student pretrain smoke | WMT23 COMETKiwi XL, S3 | outer_01 | 20260825 | MUST | TODO | 验证 student -> professional 流程 |
| BB004 | M0 | 跨族兼容性 | BGE-M3, S0 | outer_01 | 20260825 | SHOULD | TODO | 需要 AutoModel loader 和输入模板 |
| BB005 | M1 | 绝对性能筛选 | WMT22 COMETKiwi, S0 100% | 16 outer folds | 20260825 | MUST | TODO | matched baseline |
| BB006 | M1 | 绝对性能筛选 | WMT23 COMETKiwi XL, S0 100% | 16 outer folds | 20260825 | MUST | TODO | advance gate +0.015 mean Pearson |
| BB007 | M1 | 跨族筛选 | BGE-M3, S0 100% | 16 outer folds | 20260825 | SHOULD | TODO | 仅在 BB004 通过后 |
| BB008 | M2 | Student-only 可替代性 | winner, S1 raw student-only | 16 outer folds | 3 | MUST | BLOCKED | 等 M1 winner |
| BB009 | M2 | Full-data transfer | winner, S0 vs S3 | 16 outer folds | 3 | MUST | BLOCKED | paired seeds |
| BB010 | M3 | Low-resource core | winner, S0 vs S3, 10/25/100% | nested outer folds | 3 | MUST | BLOCKED | 先不跑 50/75% |
| BB011 | M4 | EXP reliability | winner, EXP weight 1.00 vs 0.50 | 16 outer folds | 3 | MUST | BLOCKED | 不做细粒度 sweep |
| BB012 | M5 | 完整 learning curve | winner, S0 vs S3, 50/75% | nested outer folds | 3 | SHOULD | BLOCKED | 仅在 BB010 通过 gate 后 |
