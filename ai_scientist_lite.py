#!/usr/bin/env python3
"""AI Scientist-lite workflow for the SI rubric evaluation project.

This script keeps the AI-Scientist idea practical for this repository:
generate a hypothesis tree, turn each leaf into a reproducible train/eval
command, collect metrics from run logs, and draft a paper-facing report.
Training is opt-in: without --execute it only writes the plan and commands.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "experiments" / "ai_scientist_lite"
DEFAULT_PLAN = RUN_ROOT / "experiment_plan.json"
DEFAULT_REPORT = RUN_ROOT / "report.md"


def project_python() -> str:
    """Prefer the repository virtualenv so generated commands match this project."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@dataclass(frozen=True)
class ExperimentIdea:
    id: str
    parent: str | None
    research_question: str
    hypothesis: str
    rationale: str
    train_data: str
    dev_data: str
    test_data: str
    params: dict[str, str | int | float]
    expected_signal: str
    paper_use: str
    risk: str


def project_context() -> dict[str, object]:
    """Collect lightweight, stable facts used to condition the plan."""
    files = {
        "train": ROOT / "train_set.json",
        "dev": ROOT / "dev_set.json",
        "test": ROOT / "test_set.json",
    }
    counts = {}
    for name, path in files.items():
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                counts[name] = len(json.load(f))

    existing_results = {}
    for path in [
        ROOT / "table1_results.json",
        ROOT / "baseline_correlation_results.json",
        ROOT / "latency_stability_results.json",
    ]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                existing_results[path.name] = json.load(f)

    return {
        "dataset_counts": counts,
        "existing_results": existing_results,
        "paper_claims_to_protect": [
            "Rubric dimensions should not collapse into one scalar.",
            "LQ and EXP need separate signals despite noisy human supervision.",
            "Talk-level split is required to avoid leakage.",
            "LAT is analyzed separately as perceived cognitive delay.",
        ],
    }


def generate_ideas() -> list[ExperimentIdea]:
    """Create a small tree of repository-native ablations."""
    base = {
        "train_data": "train_set.json",
        "dev_data": "dev_set.json",
        "test_data": "test_set.json",
        "params": {
            "pooling": "cls",
            "gpu_batch_size": 8,
            "num_epochs": 10,
            "lr_head": 5e-4,
            "lr_lora": 1.5e-4,
            "exp_weight": 1.7,
            "variance_weight": 0.2,
            "corr_weight": 0.25,
        },
    }
    ideas = [
        ExperimentIdea(
            id="root_current_dual_head",
            parent=None,
            research_question="Can the current dual-head COMET-KIWI setup recover separate LQ and EXP ranking signals?",
            hypothesis="The existing dual-head LoRA model is the anchor system and should reproduce the reported LQ/EXP correlations.",
            rationale="This protects the current paper claim before exploring variants.",
            expected_signal="LQ and EXP Pearson should be close to the reported dual-head values in table1_results.json.",
            paper_use="Main system sanity check; Table 1 proposed row.",
            risk="If this drifts, later ablations are not interpretable.",
            **base,
        ),
        ExperimentIdea(
            id="pooling_mean_vs_cls",
            parent="root_current_dual_head",
            research_question="Does mean pooling stabilize pair encoding better than CLS pooling for SI segment pairs?",
            hypothesis="Mean pooling may improve EXP because delivery quality is distributed over the full source-MT pair.",
            rationale="The code comments already note mean pooling can be steadier for pair inputs; this makes it a direct ablation.",
            expected_signal="Higher EXP Pearson or lower pred_LQ/pred_EXP coupling without hurting LQ.",
            paper_use="Architecture ablation: representation pooling.",
            risk="Must evaluate with the same pooling value used at train time.",
            **{
                **base,
                "params": {
                    **base["params"],
                    "pooling": "mean",
                },
            },
        ),
        ExperimentIdea(
            id="stronger_disentanglement",
            parent="root_current_dual_head",
            research_question="Does stronger variance/correlation regularization prevent rubric collapse?",
            hypothesis="Increasing correlation and variance regularization should reduce pred_LQ/pred_EXP coupling.",
            rationale="The paper's central failure mode is dimension collapse, so this tests parameter-level disentanglement.",
            expected_signal="Lower Pearson(pred_LQ, pred_EXP) with stable or improved LQ/EXP Pearson.",
            paper_use="Disentanglement ablation and discussion of collapse control.",
            risk="Over-regularization can reduce absolute predictive accuracy.",
            **{
                **base,
                "params": {
                    **base["params"],
                    "variance_weight": 0.35,
                    "corr_weight": 0.45,
                },
            },
        ),
        ExperimentIdea(
            id="no_disentanglement_penalty",
            parent="stronger_disentanglement",
            research_question="Are the explicit disentanglement losses necessary?",
            hypothesis="Removing variance/correlation penalties should increase dimension coupling or reduce one target's stability.",
            rationale="A clean negative control strengthens the claim that structure matters beyond dual heads alone.",
            expected_signal="Higher pred_LQ/pred_EXP coupling or weaker EXP Pearson.",
            paper_use="Negative-control ablation.",
            risk="If performance improves, the current regularizers need retuning rather than removal from the claim.",
            **{
                **base,
                "params": {
                    **base["params"],
                    "variance_weight": 0.0,
                    "corr_weight": 0.0,
                },
            },
        ),
        ExperimentIdea(
            id="balanced_exp_weight",
            parent="root_current_dual_head",
            research_question="Is EXP overweighting needed, or does equal LQ/EXP loss preserve both dimensions?",
            hypothesis="Reducing EXP weight to 1.0 may improve LQ without destroying EXP if target scales are already balanced.",
            rationale="The current training recipe emphasizes EXP; a balanced-loss leaf tests whether this is essential.",
            expected_signal="Trade-off curve between LQ Pearson and EXP Pearson.",
            paper_use="Loss weighting ablation.",
            risk="A lower EXP weight may reintroduce EXP underfitting.",
            **{
                **base,
                "params": {
                    **base["params"],
                    "exp_weight": 1.0,
                },
            },
        ),
        ExperimentIdea(
            id="professional_augmented_training",
            parent="root_current_dual_head",
            research_question="Does adding professional evaluation data improve rubric-aligned generalization?",
            hypothesis="Original plus professional training data should improve robustness if label scales are compatible.",
            rationale="The repository already contains merged professional experiment splits, making this an immediate data ablation.",
            train_data="data/experiments/lqexp/train_original_plus_professional.json",
            dev_data="data/experiments/lqexp/dev_original_plus_professional.json",
            test_data="test_set.json",
            params=base["params"],
            expected_signal="Improved held-out LQ/EXP Pearson or better stability across dimensions.",
            paper_use="Data-scaling / professional-supervision ablation.",
            risk="Scale mismatch can inflate loss or hurt held-out original-test performance.",
        ),
    ]
    return ideas


def train_command(idea: ExperimentIdea, output_dir: Path, check_only: bool = False) -> list[str]:
    cmd = [
        project_python(),
        "run_train_v1.py",
        "--train-data",
        idea.train_data,
        "--dev-data",
        idea.dev_data,
        "--output-dir",
        str(output_dir),
        "--pooling",
        str(idea.params["pooling"]),
        "--gpu-batch-size",
        str(idea.params["gpu_batch_size"]),
        "--num-epochs",
        str(idea.params["num_epochs"]),
        "--lr-head",
        str(idea.params["lr_head"]),
        "--lr-lora",
        str(idea.params["lr_lora"]),
        "--exp-weight",
        str(idea.params["exp_weight"]),
        "--variance-weight",
        str(idea.params["variance_weight"]),
        "--corr-weight",
        str(idea.params["corr_weight"]),
    ]
    if check_only:
        cmd.append("--check-only")
    return cmd


def eval_command(idea: ExperimentIdea, output_dir: Path) -> list[str]:
    return [
        project_python(),
        "evaluate.py",
        "--checkpoint",
        "best_model2.pt",
        "--checkpoint_dir",
        str(output_dir),
        "--train_data",
        idea.train_data,
        "--dev_data",
        idea.dev_data,
        "--test_data",
        idea.test_data,
        "--pooling",
        str(idea.params["pooling"]),
        "--export",
        str(output_dir / "predictions_test.json"),
        "--export_data",
        idea.test_data,
    ]


def command_string(cmd: Iterable[str]) -> str:
    return " ".join(quote_arg(part) for part in cmd)


def quote_arg(part: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", part):
        return part
    return "'" + part.replace("'", "'\"'\"'") + "'"


def write_plan(path: Path, ideas: list[ExperimentIdea]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": "rubric-aligned SI evaluation with dual-head COMET-KIWI",
        "context": project_context(),
        "experiments": [],
    }
    for idea in ideas:
        output_dir = RUN_ROOT / idea.id
        item = asdict(idea)
        item["output_dir"] = str(output_dir)
        item["train_command"] = command_string(train_command(idea, output_dir))
        item["check_command"] = command_string(train_command(idea, output_dir, check_only=True))
        item["eval_command"] = command_string(eval_command(idea, output_dir))
        payload["experiments"].append(item)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + command_string(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"\n[exit_code] {proc.returncode}\n")
        return proc.returncode


def parse_eval_log(log_path: Path) -> dict[str, float | str]:
    if not log_path.exists():
        return {"status": "missing_eval_log"}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, float | str] = {"status": "parsed"}
    patterns = {
        "test_loss": r"Test Loss:\s*([-+0-9.eE]+)",
        "test_lq_pearson": r"Test LQ Pearson:\s*([-+0-9.eE]+)",
        "test_exp_pearson": r"EXP Pearson:\s*([-+0-9.eE]+)",
        "pred_lq_pred_exp_pearson": r"Test Pearson\(pred_LQ,\s*pred_EXP\):\s*([-+0-9.eE]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def run_experiments(ideas: list[ExperimentIdea], ids: set[str], check_only: bool) -> None:
    selected = [idea for idea in ideas if not ids or idea.id in ids]
    for idea in selected:
        output_dir = RUN_ROOT / idea.id
        if check_only:
            code = run_command(train_command(idea, output_dir, check_only=True), output_dir / "check.log")
            if code != 0:
                print(f"[check failed] {idea.id}: see {output_dir / 'check.log'}")
            else:
                print(f"[check ok] {idea.id}")
            continue

        train_code = run_command(train_command(idea, output_dir), output_dir / "train.log")
        if train_code != 0:
            print(f"[train failed] {idea.id}: see {output_dir / 'train.log'}")
            continue
        eval_code = run_command(eval_command(idea, output_dir), output_dir / "eval.log")
        if eval_code != 0:
            print(f"[eval failed] {idea.id}: see {output_dir / 'eval.log'}")
        else:
            print(f"[done] {idea.id}")


def write_report(path: Path, ideas: list[ExperimentIdea]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AI Scientist-lite Experiment Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Goal",
        "",
        "Apply an AI-Scientist-style loop to the SI rubric evaluation project: propose hypotheses, convert them to reproducible ablations, collect evidence, and draft paper-ready findings.",
        "",
        "## Paper Alignment",
        "",
        "- Interspeech-style SI paper claim to protect: LQ and EXP require separate rubric-aligned signals rather than a collapsed scalar metric.",
        "- Current Overleaf LAT paper claim to keep separate: perceived LAT is a cognitive-quality phenomenon and should not be conflated with LQ/EXP model training.",
        "- Evaluation emphasis: held-out talk-level Pearson/Spearman, cross-dimension coupling, and human-consistency framing.",
        "",
        "## Experiment Tree",
        "",
    ]
    for idea in ideas:
        output_dir = RUN_ROOT / idea.id
        metrics = parse_eval_log(output_dir / "eval.log")
        lines.extend(
            [
                f"### {idea.id}",
                "",
                f"- Parent: {idea.parent or 'root'}",
                f"- Research question: {idea.research_question}",
                f"- Hypothesis: {idea.hypothesis}",
                f"- Expected signal: {idea.expected_signal}",
                f"- Paper use: {idea.paper_use}",
                f"- Risk: {idea.risk}",
                f"- Train command: `{command_string(train_command(idea, output_dir))}`",
                f"- Eval command: `{command_string(eval_command(idea, output_dir))}`",
                f"- Current metrics: `{json.dumps(metrics, ensure_ascii=False)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Draft Result Language",
            "",
            "Use this only after the corresponding metrics are populated:",
            "",
            "The ablation tree evaluates whether rubric-aligned supervision benefits from explicit structural separation. Starting from the current dual-head COMET-KIWI model, we vary representation pooling, disentanglement regularization, loss weighting, and professional-data augmentation. We report each variant with the same talk-level split and compare not only LQ/EXP Pearson but also the coupling between predicted dimensions. This separates improvements in absolute correlation from undesirable rubric collapse.",
            "",
            "## Next Actions",
            "",
            "1. Run `python ai_scientist_lite.py --plan` to refresh the plan.",
            "2. Run `python ai_scientist_lite.py --check-only --ids root_current_dual_head` to validate imports and paths.",
            "3. On a GPU machine, run selected experiments with `python ai_scientist_lite.py --execute --ids root_current_dual_head pooling_mean_vs_cls`.",
            "4. Re-run `python ai_scientist_lite.py --report` after experiments finish.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Scientist-lite planner/runner for process3")
    parser.add_argument("--plan", action="store_true", help="Write the experiment plan JSON")
    parser.add_argument("--report", action="store_true", help="Write a Markdown report from the plan and logs")
    parser.add_argument("--execute", action="store_true", help="Run selected train/eval experiments")
    parser.add_argument("--check-only", action="store_true", help="Run selected experiments in run_train_v1 --check-only mode")
    parser.add_argument("--ids", nargs="*", default=[], help="Optional experiment ids to execute/check")
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ideas = generate_ideas()
    plan_path = Path(args.plan_path)
    report_path = Path(args.report_path)

    if not any([args.plan, args.report, args.execute, args.check_only]):
        args.plan = True
        args.report = True

    if args.plan:
        write_plan(plan_path, ideas)
        print(f"Wrote plan: {plan_path}")

    if args.check_only:
        run_experiments(ideas, set(args.ids), check_only=True)

    if args.execute:
        run_experiments(ideas, set(args.ids), check_only=False)

    if args.report:
        write_report(report_path, ideas)
        print(f"Wrote report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
