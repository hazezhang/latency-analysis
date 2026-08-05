# M1 单 Seed Sanity 结果

**日期**：2026-08-05
**远端**：`root@10.27.130.23:31675`，持久目录 `/122090786/process3_aaai_current`
**环境**：`.venv_aaai`，torch `2.4.1+cu121`，PEFT `0.13.2`，Transformers `4.46.3`，两张 NVIDIA A100-SXM4-40GB
**代码/配置**：`run_train_v1.py`，COMET-KIWI，双头 LQ/EXP，seed `20260804`，mean pooling，10 epochs，GPU batch 16，LoRA 从 epoch 2 解冻。

## 运行定义

| Run | 训练数据 | 验证数据 | GPU | 监督定义 |
|---|---|---|---:|---|
| S0 | `professional_shared_train.json` | `professional_shared_dev.json` | 0 | 专业评分者 aggregate LQ/EXP |
| S1 | `outer_01_speech_1/student_raw_train.json`（807 行） | `professional_shared_dev.json`（192 行） | 1 | 学生 raw LQ/EXP，未做专业校准；此处是 sanity dev，不是 outer test |

## 结果

| Run | Best epoch | Best dev score | 最终 dev LQ Pearson | 最终 dev EXP Pearson | 最终 raw LQ std | 最终 raw EXP std | Collapse |
|---|---:|---:|---:|---:|---:|---:|---|
| S0 | 6 | 0.7084 | 0.3154 | 0.3286 | 0.2263 | 0.2340 | No |
| S1 | 5 | 0.5808 | 0.3622 | 0.1775 | 0.5546 | 0.4536 | No |

## 解释边界

- S1 的 dev 结果低于 S0 的 best score，但这是不同训练监督域的单次 sanity，不能解释为最终 student-to-professional transfer 结论。
- S1 当前使用专业 dev 作为验证集，未使用 professional outer test；正式 S1 必须逐 outer fold 训练并在对应 professional outer test 上评估。
- 两个模型均通过当前非塌缩检查；S0 最终 epoch 的 score 低于 epoch 6，保留 `best_model2.pt` 而非 `final_model2.pt`。

## Provenance

- S0 log: `experiments/student_weak_supervision_m1_sanity_20260804/S0_professional_only.log`
- S1 log: `experiments/student_weak_supervision_m1_sanity_20260804/S1_raw_student_outer01.log`
- M0 manifest: `data/experiments/student_weak_supervision_m0_20260804/manifest.json`
- Remote InfoXLM cache: `/122090786/process3_aaai_current/hf_cache`
