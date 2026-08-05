#!/usr/bin/env python3
"""Prepare and summarize formal S1 raw-student-only outer-fold evaluation.

S1 trains on raw student LQ/EXP labels, uses a deterministic student-only
development split for checkpoint selection, and evaluates once on the
professional outer speech held out for that fold. No professional labels are
used during training or calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


BASE = Path(__file__).resolve().parent
DEFAULT_PROFESSIONAL = BASE / "data/experiments/aaai_crossfitted_outer_quality_corrected/all_lat_segments.json"
DEFAULT_M0 = BASE / "data/experiments/student_weak_supervision_m0_20260804"
DEFAULT_OUTPUT = BASE / "data/experiments/student_weak_supervision_s1_20260805"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pair(row: dict) -> tuple[str, str]:
    src = row.get("src") or row.get("source_chinese") or row.get("source_english")
    mt = row.get("mt") or row.get("target_english") or row.get("target_chinese")
    if not src or not mt:
        raise ValueError(f"Missing text pair in row: {row}")
    return (" ".join(str(src).split()), " ".join(str(mt).split()))


def prepare(args: argparse.Namespace) -> int:
    professional = load(args.professional)
    m0_manifest = load(args.m0 / "manifest.json")
    if m0_manifest["professional_main_cohort"]["n_segments"] != 622:
        raise ValueError("S1 expects the 622-segment professional main cohort")

    folds = []
    for fold in m0_manifest["folds"]:
        fold_name = fold["name"]
        source_dir = args.m0 / fold_name
        student_rows = load(source_dir / "student_raw_train.json")
        file_ids = sorted({str(row["file_id"]) for row in student_rows})
        # Fixed, data-only split: every fifth file is dev. This is independent
        # of professional scores and gives a stable checkpoint-selection set.
        dev_files = {file_id for index, file_id in enumerate(file_ids) if index % 5 == 0}
        student_dev = [row for row in student_rows if str(row["file_id"]) in dev_files]
        student_fit = [row for row in student_rows if str(row["file_id"]) not in dev_files]
        if not student_fit or not student_dev:
            raise ValueError(f"Empty student fit/dev split for {fold_name}")

        outer_speech = str(fold["outer_test_speech"])
        professional_test = [row for row in professional if str(row["speech_group"]) == outer_speech]
        protected = {pair(row) for row in professional_test}
        fit_overlap = {pair(row) for row in student_fit} & protected
        dev_overlap = {pair(row) for row in student_dev} & protected
        if fit_overlap or dev_overlap:
            raise ValueError(f"S1 leakage in {fold_name}: fit={len(fit_overlap)} dev={len(dev_overlap)}")

        out_dir = args.output / fold_name
        dump(out_dir / "student_raw_fit.json", student_fit)
        dump(out_dir / "student_raw_dev.json", student_dev)
        dump(out_dir / "professional_outer_test.json", professional_test)
        folds.append({
            "name": fold_name,
            "outer_test_speech": outer_speech,
            "n_student_raw": len(student_rows),
            "n_student_fit": len(student_fit),
            "n_student_dev": len(student_dev),
            "student_dev_file_ids": sorted(dev_files),
            "n_professional_outer_test": len(professional_test),
            "fit_overlap": len(fit_overlap),
            "dev_overlap": len(dev_overlap),
        })

    dump(args.output / "manifest.json", {
        "protocol": "S1 raw student-only training; student-only dev selection; professional outer speech test",
        "professional_input": str(args.professional.relative_to(BASE)),
        "m0_input": str(args.m0.relative_to(BASE)),
        "n_folds": len(folds),
        "folds": folds,
    })
    print(json.dumps({"output": str(args.output), "n_folds": len(folds), "folds": folds}, ensure_ascii=False, indent=2))
    return 0


def summarize(args: argparse.Namespace) -> int:
    rows = []
    fold_summaries = []
    for pred_path in sorted(args.predictions.glob("outer_*/predictions_outer_test.json")):
        pred = load(pred_path)
        fold_name = pred_path.parent.name
        lq_y = np.array([float(row["human_LQ_quantized"]) for row in pred])
        exp_y = np.array([float(row["human_EXP_quantized"]) for row in pred])
        lq_p = np.array([float(row["pred_LQ"]) for row in pred])
        exp_p = np.array([float(row["pred_EXP"]) for row in pred])

        def corr(y, p):
            return float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else 0.0

        result = {
            "fold": fold_name,
            "n": len(pred),
            "LQ_pearson": corr(lq_y, lq_p),
            "EXP_pearson": corr(exp_y, exp_p),
            "LQ_MAE": float(np.mean(np.abs(lq_p - lq_y))),
            "EXP_MAE": float(np.mean(np.abs(exp_p - exp_y))),
            "LQ_MSE": float(np.mean((lq_p - lq_y) ** 2)),
            "EXP_MSE": float(np.mean((exp_p - exp_y) ** 2)),
            "LQ_pred_std": float(np.std(lq_p)),
            "EXP_pred_std": float(np.std(exp_p)),
        }
        fold_summaries.append(result)
        rows.extend({"fold": fold_name, **row} for row in pred)

    if not fold_summaries:
        raise FileNotFoundError(f"No outer-fold predictions under {args.predictions}")
    summary = {
        "protocol": "S1 raw student-only; no professional calibration",
        "n_folds": len(fold_summaries),
        "n_segments": sum(row["n"] for row in fold_summaries),
        "folds": fold_summaries,
        "mean": {key: float(mean(row[key] for row in fold_summaries)) for key in fold_summaries[0] if key not in {"fold", "n"}},
        "sd_across_folds": {key: float(pstdev(row[key] for row in fold_summaries)) for key in fold_summaries[0] if key not in {"fold", "n"}},
    }
    dump(args.predictions / "s1_outer_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--professional", type=Path, default=DEFAULT_PROFESSIONAL)
    prep.add_argument("--m0", type=Path, default=DEFAULT_M0)
    prep.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prep.set_defaults(func=prepare)
    summ = sub.add_parser("summarize")
    summ.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT)
    summ.set_defaults(func=summarize)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
