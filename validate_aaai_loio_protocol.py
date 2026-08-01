#!/usr/bin/env python3
"""Validate interpreter-disjoint LOIO folds and optional model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE / "data/experiments/aaai_loio_outer_quality_corrected"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE))
    except ValueError:
        return str(path)


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return rows


def ids(rows: list[dict]) -> set[str]:
    values = {str(row["segment_id"]) for row in rows}
    if len(values) != len(rows):
        raise ValueError("Duplicate segment IDs within a split")
    return values


def assert_outer_absent(rows: list[dict], outer: str, label: str) -> None:
    offenders = [str(row["segment_id"]) for row in rows if str(row["interpreter"]) == outer]
    if offenders:
        raise ValueError(f"Outer interpreter leaked into {label}: {offenders[:3]}")


def prediction_ids(path: Path) -> set[str]:
    return ids(load_rows(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT.relative_to(BASE)))
    parser.add_argument("--prediction-root", help="Optional completed seed root to validate")
    args = parser.parse_args()

    data_root = BASE / args.data_root
    prediction_root = BASE / args.prediction_root if args.prediction_root else None
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    all_rows = load_rows(data_root / "all_lat_segments.json")
    all_ids = ids(all_rows)
    all_interpreters = {str(row["interpreter"]) for row in all_rows}
    if len(all_rows) != 622 or len(all_interpreters) != 7 or len(manifest["folds"]) != 7:
        raise ValueError("Unexpected LOIO cohort dimensions")

    report = []
    for fold in manifest["folds"]:
        outer_name = fold["name"]
        outer = str(fold["outer_test_interpreter"])
        expected_train_ids = {str(row["segment_id"]) for row in all_rows if str(row["interpreter"]) != outer}
        expected_test_ids = all_ids - expected_train_ids
        covered_train_ids: set[str] = set()

        for inner in fold["inner_folds"]:
            split_root = data_root / outer_name / inner["name"]
            train = load_rows(split_root / "train.json")
            dev = load_rows(split_root / "dev.json")
            predict = load_rows(split_root / "predict.json")
            assert_outer_absent(train, outer, f"{outer_name}/{inner['name']}/train")
            assert_outer_absent(dev, outer, f"{outer_name}/{inner['name']}/dev")
            assert_outer_absent(predict, outer, f"{outer_name}/{inner['name']}/predict")
            if ids(train) & ids(dev) or ids(train) & ids(predict) or ids(dev) & ids(predict):
                raise ValueError(f"Split overlap in {outer_name}/{inner['name']}")
            if covered_train_ids & ids(predict):
                raise ValueError(f"Repeated inner-OOF coverage in {outer_name}")
            covered_train_ids.update(ids(predict))
            if prediction_root:
                actual = prediction_ids(prediction_root / outer_name / inner["name"] / "predictions.json")
                if actual != ids(predict):
                    raise ValueError(f"Prediction/data mismatch in {outer_name}/{inner['name']}")

        if covered_train_ids != expected_train_ids:
            raise ValueError(f"Incomplete inner-OOF coverage in {outer_name}")

        final_root = data_root / outer_name / "final_outer"
        final_train = load_rows(final_root / "train.json")
        final_dev = load_rows(final_root / "dev.json")
        final_predict = load_rows(final_root / "predict.json")
        assert_outer_absent(final_train, outer, f"{outer_name}/final_outer/train")
        assert_outer_absent(final_dev, outer, f"{outer_name}/final_outer/dev")
        if ids(final_predict) != expected_test_ids:
            raise ValueError(f"Outer test coverage mismatch in {outer_name}")
        if {str(row["interpreter"]) for row in final_predict} != {outer}:
            raise ValueError(f"Wrong interpreter in {outer_name}/final_outer/predict")
        if prediction_root:
            actual = prediction_ids(prediction_root / outer_name / "final_outer" / "predictions.json")
            if actual != expected_test_ids:
                raise ValueError(f"Final prediction/data mismatch in {outer_name}")
        report.append({
            "outer_fold": outer_name,
            "outer_interpreter": outer,
            "n_lat_train": len(expected_train_ids),
            "n_lat_test": len(expected_test_ids),
            "n_inner_oof": len(covered_train_ids),
        })

    print(json.dumps({
        "status": "PASS",
        "scope": "unseen interpreter; source speeches may appear via other interpreters",
        "prediction_root_checked": display_path(prediction_root) if prediction_root else None,
        "folds": report,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
