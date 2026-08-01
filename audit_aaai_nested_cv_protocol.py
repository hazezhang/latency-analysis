#!/usr/bin/env python3
"""Create an anonymous, evidence-backed record of the nested CV protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE / "data/experiments/aaai_crossfitted_outer_quality_corrected"
SEEDS = (20260718, 20260719, 20260720)


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def text_pair(row: dict) -> tuple[str, str]:
    return (" ".join(str(row["src"]).split()), " ".join(str(row["mt"]).split()))


def summary(rows: list[dict], speech_alias: dict[str, str], interpreter_alias: dict[str, str]) -> dict:
    by_direction: dict[str, int] = {}
    for row in rows:
        direction = str(row["direction"])
        by_direction[direction] = by_direction.get(direction, 0) + 1
    return {
        "n_segments": len(rows),
        "n_speech_groups": len({str(row["speech_group"]) for row in rows}),
        "n_interpreters": len({str(row["interpreter"]) for row in rows}),
        "directions": by_direction,
        "speech_groups": sorted({speech_alias[str(row["speech_group"])] for row in rows}),
        "interpreters": sorted({interpreter_alias[str(row["interpreter"])] for row in rows}),
    }


def ids(rows: list[dict]) -> set[str]:
    return {str(row["segment_id"]) for row in rows}


def main() -> int:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    cohort = load(DATA_ROOT / "all_lat_segments.json")
    speech_alias = {speech: f"speech_{i:02d}" for i, speech in enumerate(sorted({str(row["speech_group"]) for row in cohort}), 1)}
    interpreter_alias = {name: f"interpreter_{i:02d}" for i, name in enumerate(sorted({str(row["interpreter"]) for row in cohort}), 1)}
    output_folds, csv_rows = [], []

    for fold in manifest["folds"]:
        fold_dir = DATA_ROOT / fold["name"]
        final = {part: load(fold_dir / "final_outer" / f"{part}.json") for part in ("train", "dev", "predict")}
        final_ids = {part: ids(rows) for part, rows in final.items()}
        if final_ids["train"] & final_ids["dev"] or final_ids["train"] & final_ids["predict"] or final_ids["dev"] & final_ids["predict"]:
            raise ValueError(f"Final split ID overlap in {fold['name']}")
        test_pairs = {text_pair(row) for row in final["predict"]}
        if any(text_pair(row) in test_pairs for row in final["train"] + final["dev"]):
            raise ValueError(f"Final exact text-pair leak in {fold['name']}")

        inner_rows = []
        oof_ids: set[str] = set()
        for inner in fold["inner_folds"]:
            split = {part: load(fold_dir / inner["name"] / f"{part}.json") for part in ("train", "dev", "predict")}
            split_ids = {part: ids(rows) for part, rows in split.items()}
            if split_ids["train"] & split_ids["dev"] or split_ids["train"] & split_ids["predict"] or split_ids["dev"] & split_ids["predict"]:
                raise ValueError(f"Inner split ID overlap in {fold['name']}/{inner['name']}")
            if (split_ids["train"] | split_ids["dev"] | split_ids["predict"]) & final_ids["predict"]:
                raise ValueError(f"Outer test leakage in {fold['name']}/{inner['name']}")
            if any(text_pair(row) in test_pairs for row in split["train"] + split["dev"]):
                raise ValueError(f"Outer exact text-pair leak in {fold['name']}/{inner['name']}")
            if oof_ids & split_ids["predict"]:
                raise ValueError(f"Duplicate inner OOF IDs in {fold['name']}")
            oof_ids |= split_ids["predict"]
            held_out = [speech_alias[str(value)] for value in inner["held_out_speeches"]]
            entry = {
                "inner_fold": inner["name"],
                "held_out_speech_groups": held_out,
                "dev_speech_group": speech_alias[str(inner["dev_speech"])],
                "train": summary(split["train"], speech_alias, interpreter_alias),
                "dev": summary(split["dev"], speech_alias, interpreter_alias),
                "predict": summary(split["predict"], speech_alias, interpreter_alias),
            }
            inner_rows.append(entry)
            for part in ("train", "dev", "predict"):
                csv_rows.append({"outer_fold": fold["name"].split("_speech_")[0], "fold_kind": "inner", "fold": inner["name"], "partition": part, **entry[part]})
        expected_oof = final_ids["train"] | final_ids["dev"]
        if oof_ids != expected_oof:
            raise ValueError(f"Inner OOF coverage mismatch in {fold['name']}: {len(oof_ids)} vs {len(expected_oof)}")

        final_entry = {part: summary(rows, speech_alias, interpreter_alias) for part, rows in final.items()}
        output_folds.append({
            "outer_fold": fold["name"].split("_speech_")[0],
            "outer_test_speech_group": speech_alias[str(fold["outer_test_speech"])],
            "inner_folds": inner_rows,
            "final_outer": {
                "dev_speech_group": speech_alias[str(fold["final_outer"]["dev_speech"])],
                **final_entry,
            },
        })
        for part in ("train", "dev", "predict"):
            csv_rows.append({"outer_fold": fold["name"].split("_speech_")[0], "fold_kind": "final_outer", "fold": "final_outer", "partition": part, **final_entry[part]})

    metadata = {
        "cohort": {"n_segments": len(cohort), "n_speech_groups": len(speech_alias), "n_interpreters": len(interpreter_alias)},
        "outer_protocol": "16 deterministic leave-one-source-speech-group-out folds",
        "inner_protocol": "four deterministic group-level partitions within each outer-training set; greedy balance by segment count with stable speech-name tie-breaks",
        "stratification": "No explicit stratification by direction or interpreter; group-level segment-count balancing only.",
        "seeds": {"values": SEEDS, "split_manifest_identical_across_seeds": True},
        "quality_training": {
            "epochs": 10,
            "early_stopping": "Not used. All runs train for 10 epochs.",
            "checkpoint_selection": "best_model2.pt is selected by the maximum dev LQ Pearson + dev EXP Pearson within the fold; the outer test partition is never used.",
            "hyperparameters": "Fixed runner configuration for all folds/seeds: head-only, lr_head=5e-4, lr_lora=0, LoRA unfreeze epoch=999, batch=16, exp_weight=1.7, variance_weight=.2, corr_weight=.25, selection_metric=sum.",
        },
        "preprocessing": {
            "exact_duplicate_removal": "For each outer fold, exact normalized source--interpreted-output pairs from the outer test speech are removed from every upstream quality train/dev split. The audit asserted zero remaining outer test text-pair overlap.",
            "ridge_scaling": "For every outer fold, the second-stage feature mean and SD are fitted on its LAT-training rows only and applied to its outer test rows.",
            "pca": "PCA/one-factor analyses are diagnostic sensitivities, not inputs to the primary two-stage pipeline; no PCA transform is fitted or applied in the primary system.",
            "tokenization": "The frozen pretrained tokenizer is applied row-wise; it does not fit corpus statistics.",
        },
        "audit_assertions": [
            "final train/dev/test segment IDs are disjoint in all 16 outer folds",
            "inner train/dev/predict segment IDs are disjoint in all 64 inner folds",
            "no outer-test IDs occur in any inner split",
            "inner OOF prediction IDs exactly cover final outer train+dev IDs once",
            "no exact normalized outer test source--interpreted-output pair remains in upstream train/dev",
        ],
    }
    payload = {"metadata": metadata, "outer_folds": output_folds}
    out_dir = BASE / "experiments/aaai_nested_cv_protocol_audit_20260726"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "nested_cv_protocol_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = ["outer_fold", "fold_kind", "fold", "partition", "n_segments", "n_speech_groups", "n_interpreters", "directions", "speech_groups", "interpreters"]
    with (out_dir / "nested_cv_fold_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in csv_rows:
            row = dict(row)
            row["directions"] = json.dumps(row["directions"], sort_keys=True)
            row["speech_groups"] = "+".join(row["speech_groups"])
            row["interpreters"] = "+".join(row["interpreters"])
            writer.writerow(row)
    digest = hashlib.sha256((DATA_ROOT / "manifest.json").read_bytes()).hexdigest()
    (out_dir / "manifest_sha256.txt").write_text(f"sha256:{digest}\n", encoding="ascii")
    print(json.dumps({"output_dir": str(out_dir.relative_to(BASE)), "manifest_sha256": digest, "n_outer": len(output_folds)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
