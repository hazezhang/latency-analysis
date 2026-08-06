#!/usr/bin/env python3
"""Build per-fold student data excluding professional inner-dev and outer-test text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pair(row: dict) -> tuple[str, str]:
    src = row.get("src") or row.get("source_chinese") or row.get("source_english")
    mt = row.get("mt") or row.get("target_english") or row.get("target_chinese")
    if not src or not mt:
        raise ValueError(f"Missing text pair: {row}")
    return (" ".join(str(src).split()), " ".join(str(mt).split()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-root", type=Path, default=Path("data/experiments/student_weak_supervision_s1_20260805"))
    parser.add_argument("--professional-root", type=Path, default=Path("data/experiments/aaai_crossfitted_outer_quality_corrected"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/student_weak_supervision_strict_20260806"))
    args = parser.parse_args()

    report = {"protocol": "student pretraining excludes professional inner-dev and outer-test normalized text pairs", "folds": []}
    for fold_dir in sorted(args.student_root.glob("outer_*")):
        fold = fold_dir.name
        professional_dir = args.professional_root / fold / "final_outer"
        protected = {pair(row) for name in ("dev.json", "predict.json") for row in load(professional_dir / name)}
        fit = load(fold_dir / "student_raw_fit.json")
        dev = load(fold_dir / "student_raw_dev.json")
        fit_keep = [row for row in fit if pair(row) not in protected]
        dev_keep = [row for row in dev if pair(row) not in protected]
        if not fit_keep or not dev_keep:
            raise ValueError(f"Strict split empty for {fold}: fit={len(fit_keep)} dev={len(dev_keep)}")
        fit_overlap = len({pair(row) for row in fit_keep} & protected)
        dev_overlap = len({pair(row) for row in dev_keep} & protected)
        if fit_overlap or dev_overlap:
            raise AssertionError(f"Residual overlap for {fold}: fit={fit_overlap} dev={dev_overlap}")
        out = args.output / fold
        dump(out / "student_fit.json", fit_keep)
        dump(out / "student_dev.json", dev_keep)
        report["folds"].append({
            "fold": fold,
            "n_fit_before": len(fit), "n_fit_after": len(fit_keep),
            "n_dev_before": len(dev), "n_dev_after": len(dev_keep),
            "removed_fit": len(fit) - len(fit_keep),
            "removed_dev": len(dev) - len(dev_keep),
            "protected_pairs": len(protected),
            "fit_overlap": 0, "dev_overlap": 0,
        })
    report["n_folds"] = len(report["folds"])
    report["n_fit_before"] = sum(x["n_fit_before"] for x in report["folds"])
    report["n_fit_after"] = sum(x["n_fit_after"] for x in report["folds"])
    report["n_dev_before"] = sum(x["n_dev_before"] for x in report["folds"])
    report["n_dev_after"] = sum(x["n_dev_after"] for x in report["folds"])
    dump(args.output / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
