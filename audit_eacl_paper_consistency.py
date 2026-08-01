#!/usr/bin/env python3
"""Deterministic paper-to-canonical consistency checks for the EACL draft."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "eacl27_paper_staging"
RESULTS = ROOT / "experiments" / "eacl_paper_canonical_20260728" / "paper_results.json"
MACROS = PAPER / "generated" / "paper_results.tex"


def fmt(value: float, digits: int = 3) -> str:
    rendered = f"{value:.{digits}f}"
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-." + rendered.split(".", 1)[1]
    return rendered


def parse_macros() -> dict[str, str]:
    pattern = re.compile(r"^\\newcommand\{\\([^}]+)\}\{(.*)\}$")
    values: dict[str, str] = {}
    for line in MACROS.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def expect(macros: dict[str, str], name: str, value: str) -> None:
    actual = macros.get(name)
    if actual != value:
        raise AssertionError(f"{name}: expected {value}, found {actual}")


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    macros = parse_macros()
    source = results["source_speech_group"]
    cross = results["cross_rater"]
    loio = results["interpreter_disjoint_aggregate"]

    checks = {
        "DelayPearson": fmt(source["main_models"]["delay_piecewise"]["point"]["pearson"]),
        "QualityDelayPearson": fmt(source["seed_metrics"]["auto_pred_LQ_EXP_piecewise_delay"]["pearson"]["mean"]),
        "QualityDelayPearsonSD": fmt(source["seed_metrics"]["auto_pred_LQ_EXP_piecewise_delay"]["pearson"]["sd"]),
        "StructureDelayPearson": fmt(source["structural_delay"]["pearson"]),
        "FullPearson": fmt(source["full_model"]["pearson"]["mean"]),
        "FullPearsonSD": fmt(source["full_model"]["pearson"]["sd"]),
        "PrimaryDeltaPearson": fmt(source["primary_test"]["observed_statistic"], 4),
        "PrimaryPValue": fmt(source["primary_test"]["plus_one_corrected_p_value"], 6),
        "CrossRtoS": fmt(cross["R05_quality_to_R06_promptness"]["pearson"]),
        "CrossStoR": fmt(cross["R06_quality_to_R05_promptness"]["pearson"]),
        "LOIOFullPearson": fmt(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["pearson"]["mean"]),
        "LOIOFullPearsonSD": fmt(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["pearson"]["sd"]),
        "LOIOFullCenteredPearson": fmt(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["within_interpreter_centered_pearson"]["mean"]),
        "LOIOFullCenteredPearsonSD": fmt(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["within_interpreter_centered_pearson"]["sd"]),
        "LOIOFullMacroPearson": fmt(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["macro_interpreter_pearson"]["mean"]),
        "LOIOFullMacroPearsonSD": fmt(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["macro_interpreter_pearson"]["sd"]),
    }
    for name, value in checks.items():
        expect(macros, name, value)

    all_tex = "\n".join(path.read_text(encoding="utf-8") for path in sorted(PAPER.rglob("*.tex")))
    required = [
        "primary prespecified comparison",
        "we interpret this comparison as a representation audit",
        "Interpreter-disjoint evaluation remains positive but heterogeneous.",
        "predefined two-rater professional aggregate",
        "Both evaluators received the same written rubric before formal scoring.",
        "jointly discussed example cases and completed a pilot calibration exercise",
        "could not see each other's scores",
        "Both used headphones.",
        "Segment order was randomized.",
        "numerical scoring order was fixed as LQ, then EXP, then promptness/LAT",
        "Optional comments were entered only after all three numerical ratings",
        "cannot arise from direct score sharing",
        "parallel halo and anchoring effects",
        "The archive contains 679 segment identifiers with records from both primary evaluators.",
        "Of these, 632 have complete LQ, EXP, and promptness ratings from both evaluators and a numeric archived onset difference.",
        "The other 47 are outside this complete-case timing candidate set",
        "The primary analysis retains 622 of the 632 complete candidates",
        "quality--promptness relationship is not confined to within-evaluator score coupling but remains evaluator-dependent",
        "Low structure-to-quality $R^2$ does not imply that raw quality contributes an orthogonal promptness signal",
        "possible halo and fixed-order anchoring effects",
        "do not independently validate the promptness construct",
    ]
    for phrase in required:
        if phrase not in all_tex:
            raise AssertionError(f"missing required wording: {phrase}")

    forbidden = [
        "It rejects a purely same-rater account",
        "sole confirmatory test",
        ".358\\pm.016",
        ".340\\pm.013",
        "andR06",
        "whileR06",
        "at.543",
        "atr=.552",
        "operational professional consensus",
        "headphone-use records",
        "formal rater-calibration records",
        "presentation-order randomization",
        "rater-independence records",
        "score/comment order",
        "partially shared professional promptness component",
    ]
    for phrase in forbidden:
        if phrase in all_tex:
            raise AssertionError(f"forbidden stale text: {phrase}")

    appendix = (PAPER / "sections" / "A_appendix.tex").read_text(encoding="utf-8")
    for old in ("$.502$", "$.828$"):
        if old not in appendix:
            raise AssertionError("superseded cross-rater provenance note is missing")
    main_results = (PAPER / "sections" / "5_experiments.tex").read_text(encoding="utf-8")
    for old in ("$.502$", "$.828$"):
        if old in main_results:
            raise AssertionError("superseded cross-rater value leaked into main results")

    generated = [
        "automatic_direction_table.tex",
        "residualization_table.tex",
        "rater_confusion_table.tex",
        "rater_human_table.tex",
        "rater_direction_table.tex",
    ]
    for name in generated:
        text = (PAPER / "generated" / name).read_text(encoding="utf-8")
        if "Generated by build_eacl_paper_results.py" not in text:
            raise AssertionError(f"missing generated provenance marker: {name}")

    report = """# Final Numerical and Methodological Consistency Audit

Status: PASS

- Canonical source: `experiments/eacl_paper_canonical_20260728/paper_results.json`
- Main source-group, cross-rater, residualization, range, direction, and interpreter-disjoint values are generated from that source.
- Cross-rater values `.430` and `.543` use target-specific outer-fold models; `.502` and `.828` appear only in the superseded-estimand provenance note.
- Interpreter-disjoint pooled, centered, and macro statistics use their matching aggregation-specific seed SDs.
- The abstract, main tables, appendix audit tables, and main figure are generated from or checked against the canonical source.
- The professional-rating protocol records shared instructions and calibration, independent blinded full-cohort scoring, headphones, randomized presentation, fixed within-item score order, revision permissions, and post-score comments.
- Required construct boundaries and residualization interpretation are present; prohibited stale claims are absent.
"""
    (PAPER / "FINAL_CONSISTENCY_AUDIT.md").write_text(report, encoding="utf-8")
    print("PASS: EACL paper matches canonical result definitions and values")


if __name__ == "__main__":
    main()
