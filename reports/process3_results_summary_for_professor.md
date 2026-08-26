# Based on the current experiments, I feel that the main patterns related to supervision are becoming relatively clear. For the next stage, I would like to try replacing the current base model with a newer and stronger multilingual model.
I see two goals for this experiment. First, I would like to check whether the patterns we observed are consistent across different backbones—for example, whether student supervision is still more useful in low-resource professional-data settings, whether student-only supervision remains insufficient, and whether EXP still benefits from more cautious weighting.
Second, and more importantly, I would like to see whether a stronger base model can improve the absolute performance of the evaluation model itself. Since our final goal is to build a stronger and more reliable quality prediction model, we should evaluate not only whether the trends are consistent with the current model, but also whether the new backbone can provide meaningful gains in LQ/EXP Pearson and reduce errors such as MAE.
Ideally, this experiment would help us answer both questions: whether our findings about student supervision generalize beyond the current backbone, and whether we can raise the overall performance ceiling of the evaluation system.

## 7. 产物位置

- `/122090786/process3_aaai_current/experiments/professional_learning_curve_20260826`
- `/122090786/process3_aaai_current/experiments/professional_learning_curve_20260827`
- `/122090786/process3_aaai_current/experiments/student_exp_weight_exp100_s26_20260826`
- `/122090786/process3_aaai_current/experiments/student_exp_weight_exp050_s26_20260826`
- `/122090786/process3_aaai_current/experiments/student_exp_weight_exp100_s27_20260827`
- `/122090786/process3_aaai_current/experiments/student_exp_weight_exp050_s27_20260827`

**限制：** 本 summary 严格区分“已读取的数值结果”和“已确认存在但尚未成功读取的完成产物”。因此没有对 seed 20260826/20260827 的具体指标作未经验证的统计推断。
