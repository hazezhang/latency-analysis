# AI Scientist-lite for this project

This repository is not the Sakana AI Scientist codebase. The useful move here is to apply the same loop to the current SI evaluation project:

1. Generate research hypotheses.
2. Convert each hypothesis into a concrete `run_train_v1.py` ablation.
3. Keep each run in an isolated output directory.
4. Evaluate with the same split and pooling setting.
5. Summarize metrics into paper-ready evidence.

The local entry point is:

```bash
python ai_scientist_lite.py
```

By default this only writes:

- `experiments/ai_scientist_lite/experiment_plan.json`
- `experiments/ai_scientist_lite/report.md`

It does not start training unless you explicitly pass `--execute`.

## Commands

Refresh the plan and report:

```bash
python ai_scientist_lite.py --plan --report
```

Validate paths and imports for one planned experiment:

```bash
python ai_scientist_lite.py --check-only --ids root_current_dual_head
```

Run selected experiments on a GPU machine:

```bash
python ai_scientist_lite.py --execute --ids root_current_dual_head pooling_mean_vs_cls
```

Rebuild the Markdown report after runs finish:

```bash
python ai_scientist_lite.py --report
```

## Current experiment tree

- `root_current_dual_head`: reproduce the current proposed system.
- `pooling_mean_vs_cls`: test whether mean pooling gives steadier pair representations.
- `stronger_disentanglement`: increase variance and correlation regularization to reduce rubric collapse.
- `no_disentanglement_penalty`: negative control for the disentanglement losses.
- `balanced_exp_weight`: test whether equal LQ/EXP weighting is enough.
- `professional_augmented_training`: test original plus professional supervision.

## Paper alignment

Use the Interspeech paper as the LQ/EXP evaluation target: the key claim is that rubric dimensions need separate signals and should not collapse into one scalar. Keep the LAT paper separate: LAT can be discussed as perceived cognitive delay, but these LQ/EXP ablations should not claim to solve LAT unless a dedicated LAT target is added.

## Metric discipline

For each experiment, compare:

- Test LQ Pearson.
- Test EXP Pearson.
- Pearson between predicted LQ and predicted EXP.
- Whether the result preserves the talk-level split.
- Whether it improves correlation without simply coupling the two predicted dimensions.
