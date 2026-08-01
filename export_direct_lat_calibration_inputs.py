#!/usr/bin/env python3
"""Export fold-local train/dev predictions from completed direct-LAT checkpoints.

The exported development predictions are used to fit post-hoc calibration.
Outer-test labels are never read by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_train_lat_direct import DirectLATModel, LATDataset, evaluate, load_rows
from run_train_v1 import load_train_v1_symbols


def prediction_rows(rows, evaluated):
    output = []
    for index, gold, prediction in zip(evaluated["indices"], evaluated["gold"], evaluated["prediction"]):
        row = rows[index]
        output.append({
            "segment_id": row.get("segment_id"),
            "speech_group": row.get("speech_group"),
            "interpreter": row.get("interpreter"),
            "direction": row.get("direction"),
            "LAT": float(gold),
            "prediction": float(prediction),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--data-root", default="data/experiments/aaai_crossfitted_outer_quality_corrected")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", choices=("text", "text_delay"), required=True)
    parser.add_argument("--notebook", default="train_v1.ipynb")
    parser.add_argument("--model-name", default="Unbabel/wmt22-cometkiwi-da")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = load_train_v1_symbols(Path(args.notebook))
    lora_config = symbols["LoraConfig"](
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=symbols["TaskType"].FEATURE_EXTRACTION,
    )
    comet_model = symbols["load_comet_base"](args.model_name, None)
    try:
        tokenizer = comet_model.tokenizer
    except AttributeError:
        from transformers import XLMRobertaTokenizer

        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/infoxlm-large")
    encoder = symbols["COMETModelWithHeads"](
        comet_model,
        lora_config,
        train_lq_mean=0.0,
        train_exp_mean=0.0,
        pooling="cls",
    )

    checkpoint_root = Path(args.checkpoint_root) / args.variant
    data_root = Path(args.data_root)
    output_root = Path(args.output_root) / args.variant
    use_delay = args.variant == "text_delay"
    fold_summaries = []

    for checkpoint_dir in sorted(checkpoint_root.glob("outer_*")):
        fold = checkpoint_dir.name
        output_dir = output_root / fold
        done = output_dir / "dev_predictions.json"
        if done.exists():
            print(f"[skip] {done}")
            continue
        checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
        model = DirectLATModel(encoder, checkpoint["lat_mean"], use_delay).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        def predict(split):
            rows = load_rows(
                str(data_root / fold / "final_outer" / f"{split}.json"),
                require_delay=use_delay,
            )
            dataset = LATDataset(
                rows,
                tokenizer,
                args.max_length,
                checkpoint["delay_mean"],
                checkpoint["delay_std"],
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers if device.type == "cuda" else 0,
                pin_memory=device.type == "cuda",
            )
            evaluated = evaluate(model, loader, device)
            return rows, evaluated

        output_dir.mkdir(parents=True, exist_ok=True)
        split_metrics = {}
        for split in ("train", "dev"):
            rows, evaluated = predict(split)
            (output_dir / f"{split}_predictions.json").write_text(
                json.dumps(prediction_rows(rows, evaluated), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            split_metrics[split] = {
                key: evaluated[key] for key in ("pearson", "spearman", "mse", "mae", "pred_std")
            }
        summary = {
            "fold": fold,
            "variant": args.variant,
            "checkpoint_best_epoch": checkpoint["best_epoch"],
            "checkpoint_seed": checkpoint["seed"],
            "calibration_source": "development predictions only",
            "metrics": split_metrics,
        }
        (output_dir / "export_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    (output_root / "export_manifest.json").write_text(
        json.dumps(fold_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
