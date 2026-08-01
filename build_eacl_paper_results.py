#!/usr/bin/env python3
"""Build the canonical numerical source used by the EACL manuscript."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "eacl_paper_canonical_20260728"
GENERATED = ROOT / "eacl27_paper_staging" / "generated"

SPEECH_CI = ROOT / "experiments" / "aaai_speech_group_clustered_ci_20260726" / "speech_group_clustered_ci.json"
SEED_SUMMARY = ROOT / "experiments" / "aaai_additional_analysis_20260722" / "additional_analysis_summary.json"
STRUCTURAL = ROOT / "experiments" / "eacl_structural_neural_audit_20260728_r2" / "structural_neural_results.json"
STRUCTURAL_PRED = ROOT / "experiments" / "eacl_structural_neural_audit_20260728_r2" / "structural_neural_predictions.csv"
RATER = ROOT / "experiments" / "eacl_rater_calibration_audit_20260728" / "rater_calibration_results.json"
LOIO = ROOT / "experiments" / "aaai_loio_structural_audit_20260722" / "loio_structural_summary.csv"
LOIO_AUDIT = ROOT / "experiments" / "aaai_loio_structural_audit_20260722" / "loio_structural_audit.json"
AUTO_DIRECTION = ROOT / "experiments" / "aaai_reviewer_priority_audits_20260722" / "automatic_direction_results.csv"
SECONDARY = ROOT / "experiments" / "aaai_key_incremental_comparison_20260726" / "key_incremental_comparison.json"
BOUNDED = ROOT / "experiments" / "aaai_bounded_lat_sensitivity_20260726_r2" / "bounded_lat_sensitivity.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def point_metrics(gold: pd.Series, pred: pd.Series) -> dict[str, float]:
    y = gold.to_numpy(float)
    p = pred.to_numpy(float)
    return {
        "n": int(len(y)),
        "pearson": float(pearsonr(y, p).statistic),
        "spearman": float(spearmanr(y, p).statistic),
        "mse": float(np.mean((y - p) ** 2)),
    }


def aggregate_seed_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {"n": rows[0]["n"]}
    for metric in ("pearson", "spearman", "mse"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "per_seed": [float(value) for value in values],
        }
    return result


def automatic_direction_results() -> dict:
    legacy = pd.read_csv(AUTO_DIRECTION)
    structural = pd.read_csv(STRUCTURAL_PRED)
    result: dict[str, dict] = {}
    mapping = {
        "delay": (legacy, "delay_piecewise"),
        "predicted_quality_delay": (legacy, "auto_pred_LQ_EXP_piecewise_delay"),
    }
    for output_name, (frame, model_name) in mapping.items():
        model_rows = frame[frame["model"] == model_name]
        result[output_name] = {}
        for direction, subset in model_rows.groupby("direction"):
            per_seed = [
                {
                    "n": int(row.n),
                    "pearson": float(row.pearson),
                    "spearman": float(row.spearman),
                    "mse": float(row.mse),
                }
                for row in subset.itertuples(index=False)
            ]
            result[output_name][direction] = aggregate_seed_rows(per_seed)

    for output_name, column in {
        "structure_delay": "structural_delay",
        "full_model": "full_raw_quality",
    }.items():
        result[output_name] = {}
        for direction, direction_rows in structural.groupby("direction"):
            per_seed = []
            for _, seed_rows in direction_rows.groupby("seed"):
                per_seed.append(point_metrics(seed_rows["gold"], seed_rows[column]))
            result[output_name][direction] = aggregate_seed_rows(per_seed)
    return result


def fmt(value: float, digits: int = 3) -> str:
    rendered = f"{value:.{digits}f}"
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-." + rendered.split(".", 1)[1]
    return rendered


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    speech_ci = read_json(SPEECH_CI)
    seed_summary = read_json(SEED_SUMMARY)
    structural = read_json(STRUCTURAL)
    rater = read_json(RATER)
    secondary = read_json(SECONDARY)
    bounded = read_json(BOUNDED)
    loio_audit = read_json(LOIO_AUDIT)
    loio_rows = {row["model"]: row for row in csv.DictReader(LOIO.open(encoding="utf-8"))}
    loio_aggregate = loio_audit["aggregate"]
    direction = automatic_direction_results()
    cohort_rows = pd.read_csv(STRUCTURAL_PRED).drop_duplicates(subset=["segment_id"])
    cohort = {
        "n_segments": int(len(cohort_rows)),
        "n_speech_groups": int(cohort_rows["speech_group"].nunique()),
        "n_interpreters": int(cohort_rows["interpreter"].nunique()),
        "direction_counts": {
            str(key): int(value)
            for key, value in cohort_rows["direction"].value_counts().to_dict().items()
        },
    }
    bounded_rows = {
        (row["protocol"], row["method"]): row
        for row in bounded["three_seed_summary"]
    }

    full_seed = [seed["models"]["full_raw_quality"] for seed in structural["seeds"].values()]
    full = {
        metric: {
            "mean": float(np.mean([row[metric] for row in full_seed])),
            "sd": float(np.std([row[metric] for row in full_seed], ddof=1)),
        }
        for metric in ("pearson", "spearman", "mse", "mae")
    }
    residual = {
        seed: {
            "quality_explained_by_structure": values["quality_explained_by_structure"],
            "residualized_quality_only": values["models"]["residualized_quality_only"],
            "partial_residual_augmentation": values["models"]["partial_residual_augmentation"],
            "partial_r2_over_structure_delay": values["partial_r2_over_structure_delay"],
            "incremental_partial_residual_minus_structure": values["incremental_partial_residual_minus_structure"],
            "full_raw_quality": values["models"]["full_raw_quality"],
            "full_residualized_quality": values["models"]["residualized_quality"],
        }
        for seed, values in structural["seeds"].items()
    }

    canonical = {
        "generated_from": {
            "speech_group_clustered_metrics": str(SPEECH_CI.relative_to(ROOT)),
            "seed_summaries": str(SEED_SUMMARY.relative_to(ROOT)),
            "structural_and_residualization": str(STRUCTURAL.relative_to(ROOT)),
            "structural_predictions": str(STRUCTURAL_PRED.relative_to(ROOT)),
            "rater_calibration": str(RATER.relative_to(ROOT)),
            "loio": str(LOIO.relative_to(ROOT)),
            "loio_aggregate": str(LOIO_AUDIT.relative_to(ROOT)),
            "automatic_direction": str(AUTO_DIRECTION.relative_to(ROOT)),
            "secondary_increment": str(SECONDARY.relative_to(ROOT)),
            "range_sensitivity": str(BOUNDED.relative_to(ROOT)),
        },
        "source_speech_group": {
            "main_models": speech_ci["models"],
            "seed_metrics": {
                name: seed_summary["speech_models"][name]
                for name in ("auto_pred_LQ_EXP", "auto_pred_LQ_EXP_piecewise_delay", "human_LQ_EXP")
            },
            "structural_delay": next(iter(structural["seeds"].values()))["models"]["structural_delay"],
            "full_model": full,
            "primary_test": speech_ci["primary_permutation_test"],
            "paired_differences": speech_ci["paired_differences"],
            "secondary_increment": secondary,
        },
        "cohort": cohort,
        "cross_rater": rater["cross_rater_human_label_models"],
        "within_rater_human_label_models": rater["within_rater_human_label_models"],
        "direction_specific_rater": rater["direction_results"],
        "interrater_promptness": rater["interrater_promptness"],
        "interpreter_disjoint": loio_rows,
        "interpreter_disjoint_aggregate": loio_aggregate,
        "automatic_direction": direction,
        "residualization": residual,
        "range_sensitivity": bounded,
    }
    (OUT / "paper_results.json").write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")

    speech_seed = canonical["source_speech_group"]["seed_metrics"]
    struct = canonical["source_speech_group"]["structural_delay"]
    cross = canonical["cross_rater"]
    loio = canonical["interpreter_disjoint"]
    loio_stats = canonical["interpreter_disjoint_aggregate"]
    quality = speech_seed["auto_pred_LQ_EXP"]
    quality_delay = speech_seed["auto_pred_LQ_EXP_piecewise_delay"]
    human = speech_seed["human_LQ_EXP"]
    secondary_aggregate = secondary["aggregate"]
    paired_delay = speech_ci["paired_differences"]["predicted_quality_plus_delay_minus_delay_only"]
    paired_quality = speech_ci["paired_differences"]["predicted_quality_plus_delay_minus_predicted_quality"]
    interrater = canonical["interrater_promptness"]
    source_raw = bounded_rows[("source_speech_group_held_out", "ridge_raw")]
    source_clipped = bounded_rows[("source_speech_group_held_out", "ridge_clipped")]
    source_bounded = bounded_rows[("source_speech_group_held_out", "bounded_sigmoid")]
    loio_raw = bounded_rows[("interpreter_disjoint", "ridge_raw")]
    loio_clipped = bounded_rows[("interpreter_disjoint", "ridge_clipped")]

    residual_values = list(residual.values())
    residual_lq = [row["quality_explained_by_structure"]["LQ_r2"] for row in residual_values]
    residual_exp = [row["quality_explained_by_structure"]["EXP_r2"] for row in residual_values]
    residual_only_r = [row["residualized_quality_only"]["pearson"] for row in residual_values]
    residual_partial_r2 = [row["partial_r2_over_structure_delay"] for row in residual_values]

    def ci_macros(prefix: str, values: list[float]) -> list[str]:
        return [macro(f"{prefix}Low", fmt(values[0])), macro(f"{prefix}High", fmt(values[1]))]

    macros = [
        "% Generated by build_eacl_paper_results.py. Do not edit by hand.",
        macro("DelayPearson", fmt(speech_ci["models"]["delay_piecewise"]["point"]["pearson"])),
        macro("DelaySpearman", fmt(speech_ci["models"]["delay_piecewise"]["point"]["spearman"])),
        macro("DelayMSE", fmt(speech_ci["models"]["delay_piecewise"]["point"]["mse"])),
        macro("QualityPearson", fmt(quality["pearson"]["mean"])),
        macro("QualityPearsonSD", fmt(quality["pearson"]["sd"])),
        macro("QualitySpearman", fmt(quality["spearman"]["mean"])),
        macro("QualitySpearmanSD", fmt(quality["spearman"]["sd"])),
        macro("QualityMSE", fmt(quality["mse"]["mean"])),
        macro("QualityMSESD", fmt(quality["mse"]["sd"])),
        macro("QualityDelayPearson", fmt(quality_delay["pearson"]["mean"])),
        macro("QualityDelayPearsonSD", fmt(quality_delay["pearson"]["sd"])),
        macro("QualityDelaySpearman", fmt(quality_delay["spearman"]["mean"])),
        macro("QualityDelaySpearmanSD", fmt(quality_delay["spearman"]["sd"])),
        macro("QualityDelayMSE", fmt(quality_delay["mse"]["mean"])),
        macro("QualityDelayMSESD", fmt(quality_delay["mse"]["sd"])),
        macro("StructureDelayPearson", fmt(struct["pearson"])),
        macro("StructureDelaySpearman", fmt(struct["spearman"])),
        macro("StructureDelayMSE", fmt(struct["mse"])),
        macro("FullPearson", fmt(full["pearson"]["mean"])),
        macro("FullPearsonSD", fmt(full["pearson"]["sd"])),
        macro("FullSpearman", fmt(full["spearman"]["mean"])),
        macro("FullSpearmanSD", fmt(full["spearman"]["sd"])),
        macro("FullMSE", fmt(full["mse"]["mean"])),
        macro("FullMSESD", fmt(full["mse"]["sd"])),
        macro("HumanPearson", fmt(human["pearson"]["mean"])),
        macro("HumanSpearman", fmt(human["spearman"]["mean"])),
        macro("HumanMSE", fmt(human["mse"]["mean"])),
        macro("PrimaryDeltaPearson", fmt(speech_ci["primary_permutation_test"]["observed_statistic"], 4)),
        macro("PrimaryPValue", fmt(speech_ci["primary_permutation_test"]["plus_one_corrected_p_value"], 6)),
        macro("SecondaryDeltaPearson", fmt(secondary_aggregate["delta_pearson"])),
        macro("SecondaryDeltaMSE", fmt(secondary_aggregate["delta_mse"])),
        macro("CrossRtoR", fmt(cross["R05_quality_to_R05_promptness"]["pearson"])),
        macro("CrossRtoS", fmt(cross["R05_quality_to_R06_promptness"]["pearson"])),
        macro("CrossStoR", fmt(cross["R06_quality_to_R05_promptness"]["pearson"])),
        macro("CrossStoS", fmt(cross["R06_quality_to_R06_promptness"]["pearson"])),
        macro("LOIOQualityDelayPearson", fmt(float(loio["auto_pred_LQ_EXP_piecewise_delay"]["pearson_mean"]))),
        macro("LOIOQualityDelayPearsonSD", fmt(float(loio["auto_pred_LQ_EXP_piecewise_delay"]["pearson_sd"]))),
        macro("LOIOStructureDelayPearson", fmt(float(loio["lexical_structural_piecewise_delay"]["pearson_mean"]))),
        macro("LOIOStructureDelayMacroPearson", fmt(float(loio["lexical_structural_piecewise_delay"]["macro_interpreter_pearson_mean"]))),
        macro("LOIOFullPearson", fmt(float(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["pearson_mean"]))),
        macro("LOIOFullPearsonSD", fmt(float(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["pearson_sd"]))),
        macro("LOIODelayPearson", fmt(float(loio["delay_piecewise"]["pearson_mean"]))),
        macro("LOIODelayCenteredPearson", fmt(float(loio["delay_piecewise"]["within_interpreter_centered_pearson_mean"]))),
        macro("LOIODelayMacroPearson", fmt(float(loio["delay_piecewise"]["macro_interpreter_pearson_mean"]))),
        macro("LOIOQualityMacroPearson", fmt(float(loio["auto_pred_LQ_EXP"]["macro_interpreter_pearson_mean"]))),
        macro("LOIOQualityMacroPearsonSD", fmt(loio_stats["auto_pred_LQ_EXP"]["macro_interpreter_pearson"]["sd"])),
        macro("LOIOQualityDelayMacroPearson", fmt(float(loio["auto_pred_LQ_EXP_piecewise_delay"]["macro_interpreter_pearson_mean"]))),
        macro("LOIOQualityDelayMacroPearsonSD", fmt(loio_stats["auto_pred_LQ_EXP_piecewise_delay"]["macro_interpreter_pearson"]["sd"])),
        macro("LOIOFullCenteredPearson", fmt(float(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["within_interpreter_centered_pearson_mean"]))),
        macro("LOIOFullCenteredPearsonSD", fmt(loio_stats["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["within_interpreter_centered_pearson"]["sd"])),
        macro("LOIOFullMacroPearson", fmt(float(loio["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["macro_interpreter_pearson_mean"]))),
        macro("LOIOFullMacroPearsonSD", fmt(loio_stats["auto_pred_LQ_EXP_piecewise_delay_lexical_structural"]["macro_interpreter_pearson"]["sd"])),
        macro("InterraterExactPercent", f"{100 * interrater['exact_agreement']:.1f}"),
        macro("InterraterWithinOnePercent", f"{100 * interrater['within_one_point_agreement']:.1f}"),
        macro("InterraterPearson", fmt(interrater["pearson"])),
        macro("InterraterSpearman", fmt(interrater["spearman"])),
        macro("InterraterICC", fmt(interrater["icc_2_1"])),
        macro("InterraterKappa", fmt(interrater["quadratic_weighted_kappa"])),
        macro("CohortN", str(cohort["n_segments"])),
        macro("SpeechGroupN", str(speech_ci["primary_permutation_test"]["n_source_speech_groups"])),
        macro("InterpreterN", str(cohort["n_interpreters"])),
        macro("ZhEnN", str(direction["delay"]["zh-en"]["n"])),
        macro("EnZhN", str(direction["delay"]["en-zh"]["n"])),
        macro("BootstrapDraws", "10{,}000"),
        macro("PermutationCount", f"{speech_ci['primary_permutation_test']['permutations']:,}".replace(",", "{,}")),
        macro("RZeroCount", str(interrater["r05_distribution"].get("0.0", 0))),
        macro("ROneCount", str(interrater["r05_distribution"].get("1.0", 0))),
        macro("RTwoCount", str(interrater["r05_distribution"].get("2.0", 0))),
        macro("RThreeCount", str(interrater["r05_distribution"].get("3.0", 0))),
        macro("SZeroCount", str(interrater["r06_distribution"].get("0.0", 0))),
        macro("SOneCount", str(interrater["r06_distribution"].get("1.0", 0))),
        macro("STwoCount", str(interrater["r06_distribution"].get("2.0", 0))),
        macro("SThreeCount", str(interrater["r06_distribution"].get("3.0", 0))),
        macro("SourceUpperViolationMean", f"{source_raw['above_3_0_count_mean']:.1f}"),
        macro("SourceUpperViolationPercent", f"{100 * source_raw['above_3_0_rate_mean']:.1f}"),
        macro("SourceClippedPearson", fmt(source_clipped["pearson_mean"])),
        macro("SourceClippedPearsonSD", fmt(source_clipped["pearson_sd"])),
        macro("SourceClippedMSE", fmt(source_clipped["mse_mean"])),
        macro("SourceClippedMSESD", fmt(source_clipped["mse_sd"])),
        macro("SourceBoundedPearson", fmt(source_bounded["pearson_mean"])),
        macro("SourceBoundedPearsonSD", fmt(source_bounded["pearson_sd"])),
        macro("LOIOUpperViolationMean", f"{loio_raw['above_3_0_count_mean']:.1f}"),
        macro("LOIOUpperViolationPercent", f"{100 * loio_raw['above_3_0_rate_mean']:.1f}"),
        macro("LOIOClippedPearson", fmt(loio_clipped["pearson_mean"])),
        macro("LOIOClippedPearsonSD", fmt(loio_clipped["pearson_sd"])),
        macro("ResidualLQLow", fmt(min(residual_lq))),
        macro("ResidualLQHigh", fmt(max(residual_lq))),
        macro("ResidualEXPLow", fmt(min(residual_exp))),
        macro("ResidualEXPHigh", fmt(max(residual_exp))),
        macro("ResidualOnlyPearsonLow", fmt(min(residual_only_r))),
        macro("ResidualOnlyPearsonHigh", fmt(max(residual_only_r))),
        macro("ResidualPartialRLow", fmt(min(residual_partial_r2))),
        macro("ResidualPartialRHigh", fmt(max(residual_partial_r2))),
        *ci_macros("SecondaryDeltaPearsonCI", secondary_aggregate["delta_pearson_ci95"]),
        *ci_macros("SecondaryDeltaMSECI", secondary_aggregate["delta_mse_ci95"]),
        *ci_macros("QualityDelayPearsonCI", speech_ci["models"]["auto_pred_LQ_EXP_piecewise_delay"]["ci95_speech_group_cluster"]["pearson"]),
        *ci_macros("QualityDelaySpearmanCI", speech_ci["models"]["auto_pred_LQ_EXP_piecewise_delay"]["ci95_speech_group_cluster"]["spearman"]),
        *ci_macros("QualityDelayMSECI", speech_ci["models"]["auto_pred_LQ_EXP_piecewise_delay"]["ci95_speech_group_cluster"]["mse"]),
        *ci_macros("QualityDelayMAECI", speech_ci["models"]["auto_pred_LQ_EXP_piecewise_delay"]["ci95_speech_group_cluster"]["mae"]),
        macro("QualityDelayMAE", fmt(quality_delay["mae"]["mean"])),
        macro("PrimaryBootstrapDeltaPearson", fmt(paired_delay["point"]["pearson"])),
        macro("PrimaryBootstrapDeltaSpearman", fmt(paired_delay["point"]["spearman"])),
        macro("PrimaryBootstrapDeltaMSE", fmt(paired_delay["point"]["mse"])),
        macro("PrimaryBootstrapDeltaMAE", fmt(paired_delay["point"]["mae"])),
        *ci_macros("PrimaryBootstrapDeltaPearsonCI", paired_delay["ci95_speech_group_cluster"]["pearson"]),
        *ci_macros("PrimaryBootstrapDeltaSpearmanCI", paired_delay["ci95_speech_group_cluster"]["spearman"]),
        *ci_macros("PrimaryBootstrapDeltaMSECI", paired_delay["ci95_speech_group_cluster"]["mse"]),
        *ci_macros("PrimaryBootstrapDeltaMAECI", paired_delay["ci95_speech_group_cluster"]["mae"]),
        macro("QualityAddedDelayDeltaPearson", fmt(paired_quality["point"]["pearson"])),
        *ci_macros("QualityAddedDelayDeltaPearsonCI", paired_quality["ci95_speech_group_cluster"]["pearson"]),
    ]
    for index, (suffix, seed) in enumerate(zip(("One", "Two", "Three"), ("20260718", "20260719", "20260720")), start=1):
        seed_result = secondary["per_seed"][seed]
        macros.extend([
            macro(f"SecondarySeed{suffix}DeltaPearson", fmt(seed_result["pooled_delta_pearson"])),
            macro(f"SecondarySeed{suffix}PositiveGroups", str(seed_result["speech_group_delta_pearson_distribution"]["positive_count"])),
            macro(f"SecondarySeed{suffix}ValidGroups", str(seed_result["speech_group_delta_pearson_distribution"]["n"])),
        ])
        macros.extend(ci_macros(
            f"ResidualSeed{suffix}DeltaPearsonCI",
            residual[seed]["incremental_partial_residual_minus_structure"]["pearson"],
        ))
    for prefix, key in (
        ("CrossRtoR", "R05_quality_to_R05_promptness"),
        ("CrossRtoS", "R05_quality_to_R06_promptness"),
        ("CrossStoR", "R06_quality_to_R05_promptness"),
        ("CrossStoS", "R06_quality_to_R06_promptness"),
    ):
        macros.extend(ci_macros(f"{prefix}CI", cross[key]["ci95_speech_cluster"]["pearson"]))
    (GENERATED / "paper_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    labels = {
        "delay": "Segment-onset delay",
        "predicted_quality_delay": "Predicted LQ+EXP + delay",
        "structure_delay": "Lexical/structural + delay",
        "full_model": "Full model",
    }
    rows = []
    for system in ("delay", "predicted_quality_delay", "structure_delay", "full_model"):
        for direction_key, direction_label in (("zh-en", "Zh$\\rightarrow$En"), ("en-zh", "En$\\rightarrow$Zh")):
            item = direction[system][direction_key]
            seeded = system in {"predicted_quality_delay", "full_model"}
            def cell(metric: str) -> str:
                mean = fmt(item[metric]["mean"])
                return f"${mean}\\pm{fmt(item[metric]['sd'])}$" if seeded else mean
            rows.append(
                f"{labels[system]} & {direction_label} ({item['n']}) & {cell('pearson')} & {cell('spearman')} & {cell('mse')} \\\\"
            )
    table = "\n".join([
        "% Generated by build_eacl_paper_results.py. Do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Direction-specific results for the actual automatic systems. Seeded rows report mean $\\pm$ sample SD across three fixed training seeds; deterministic rows are shown once. These descriptive subgroups are imbalanced and are not direction-level significance tests.}",
        "\\label{tab:automatic-direction}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "System & Direction ($n$) & Pearson $r$ & Spearman $r_S$ & MSE \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    (GENERATED / "automatic_direction_table.tex").write_text(table + "\n", encoding="utf-8")

    residual_rows = []
    for seed, values in residual.items():
        explained = values["quality_explained_by_structure"]
        residual_only = values["residualized_quality_only"]
        partial = values["partial_residual_augmentation"]
        residual_rows.append(
            f"{seed} & {fmt(explained['LQ_r2'])}/{fmt(explained['EXP_r2'])} & "
            f"{fmt(residual_only['pearson'])}/{fmt(residual_only['mse'])} & "
            f"{fmt(partial['pearson'])}/{fmt(partial['mse'])} & "
            f"{fmt(values['partial_r2_over_structure_delay'])} \\\\"
        )
    residual_table = "\n".join([
        "% Generated by build_eacl_paper_results.py. Do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Fold-safe residualization representation audit. $R^2_Q$ gives structure-to-predicted-LQ/EXP held-out $R^2$; model cells report Pearson $r$/MSE. Partial $R^2$ is the held-out error reduction when quality residuals predict the structure+delay professional-promptness residual.}",
        "\\label{tab:residualization}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Seed & $R^2_Q$ LQ/EXP & Residual Q only & Residual augmentation & Partial $R^2$ \\\\",
        "\\midrule",
        *residual_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    (GENERATED / "residualization_table.tex").write_text(residual_table + "\n", encoding="utf-8")

    confusion = interrater["confusion_rows_r05_columns_r06"]
    confusion_rows = [
        f"{score} & " + " & ".join(str(value) for value in row) + " \\\\"
        for score, row in zip(interrater["score_order"], confusion)
    ]
    confusion_table = "\n".join([
        "% Generated by build_eacl_paper_results.py. Do not edit by hand.",
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\caption{Promptness-score confusion matrix: R05 rows, R06 columns.}",
        "\\label{tab:rater-confusion}",
        "\\begin{tabular}{c|rrrr}",
        "\\toprule",
        "R05 $\\backslash$ R06 & 0 & 1 & 2 & 3 \\\\",
        "\\midrule",
        *confusion_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    (GENERATED / "rater_confusion_table.tex").write_text(confusion_table + "\n", encoding="utf-8")

    within = canonical["within_rater_human_label_models"]
    within_systems = (
        ("delay_only", "Delay"),
        ("same_rater_human_quality_delay", "Same-rater LQ+EXP+delay"),
        ("structural_delay", "Structure+delay"),
        ("full_human_quality_structural_delay", "Full human reference"),
    )
    within_rows = []
    for rater_name in ("R05", "R06"):
        for system_key, system_label in within_systems:
            item = within[rater_name][system_key]
            ci = item["ci95_speech_cluster"]["pearson"]
            within_rows.append(
                f"{rater_name} & {system_label} & {fmt(item['pearson'])} "
                f"[{fmt(ci[0])},{fmt(ci[1])}] & {fmt(item['spearman'])} & {fmt(item['mse'])} \\\\"
            )
    within_table = "\n".join([
        "% Generated by build_eacl_paper_results.py. Do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Individual-rater human-label audit. Models are fitted within the same outer speech-group folds; intervals are speech-group clustered 95\\% CIs.}",
        "\\label{tab:rater-human}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Target & System & Pearson $r$ & Spearman $r_S$ & MSE \\\\",
        "\\midrule",
        *within_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    (GENERATED / "rater_human_table.tex").write_text(within_table + "\n", encoding="utf-8")

    rater_direction = canonical["direction_specific_rater"]
    direction_rows = []
    for rater_name in ("R05", "R06"):
        for system_key, system_label in (
            ("same_rater_human_quality_delay", "LQ+EXP+delay"),
            ("structural_delay", "Structure+delay"),
        ):
            for direction_key, direction_label in (("zh-en", "Zh$\\rightarrow$En"), ("en-zh", "En$\\rightarrow$Zh")):
                item = rater_direction[rater_name][system_key][direction_key]
                ci = item["ci95_speech_cluster"]["pearson"]
                direction_rows.append(
                    f"{rater_name} & {system_label} & {direction_label} ({item['n']}) & "
                    f"{fmt(item['pearson'])} [{fmt(ci[0])},{fmt(ci[1])}] & {fmt(item['mse'])} \\\\"
                )
    direction_table = "\n".join([
        "% Generated by build_eacl_paper_results.py. Do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Direction-specific human-label sensitivity. Brackets are speech-group clustered 95\\% CIs.}",
        "\\label{tab:rater-direction}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Target & System & Direction ($n$) & Pearson $r$ & MSE \\\\",
        "\\midrule",
        *direction_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    (GENERATED / "rater_direction_table.tex").write_text(direction_table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
