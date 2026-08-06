#!/usr/bin/env python3
"""Warm-start professional calibration from an S1 student checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from run_train_v1 import load_train_v1_symbols, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="train_v1.ipynb", type=Path)
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--dev-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pooling", default="mean", choices=["cls", "mean"])
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--gpu-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--transfer",
        choices=["full", "lora"],
        default="full",
        help="full transfers the student head; lora keeps a separate professional head.",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")


def main() -> int:
    args = build_parser().parse_args()
    required = [args.notebook, args.student_checkpoint, args.train_data, args.dev_data]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    state = checkpoint_state(args.student_checkpoint)
    # The buffers anchor predictions to the label domain. Keep the professional
    # train-set anchors created by run_train while transferring learned weights.
    if args.transfer == "full":
        transferred = {key: value for key, value in state.items() if key not in {"lq_base", "exp_base"}}
    else:
        transferred = {key: value for key, value in state.items() if "lora_" in key}
        if not transferred:
            raise RuntimeError("S4a expected LoRA parameters in the student checkpoint")
    del state

    namespace = load_train_v1_symbols(args.notebook)
    original_model = namespace["COMETModelWithHeads"]

    class StudentWarmStartModel(original_model):
        def __init__(self, *model_args, **model_kwargs):
            super().__init__(*model_args, **model_kwargs)
            incompatible = self.load_state_dict(transferred, strict=False)
            if args.transfer == "full":
                valid = set(incompatible.missing_keys) == {"lq_base", "exp_base"}
            else:
                loaded = set(self.state_dict()) - set(incompatible.missing_keys)
                valid = set(transferred).issubset(loaded)
            if not valid or incompatible.unexpected_keys:
                raise RuntimeError(
                    "S3 checkpoint mismatch: "
                    + json.dumps({
                        "missing": incompatible.missing_keys,
                        "unexpected": incompatible.unexpected_keys,
                    })
                )
            transferred.clear()
            print(
                f"[{args.transfer.upper()} Warm Start] loaded {args.student_checkpoint}; "
                "retained professional lq_base/exp_base"
            )

    namespace["COMETModelWithHeads"] = StudentWarmStartModel
    if args.check_only:
        print("S3 check passed: paths and checkpoint payload are valid.")
        return 0

    set_seed(args.seed)
    namespace["run_train"]({
        "model_name": "Unbabel/wmt22-cometkiwi-da",
        "train_data_path": str(args.train_data),
        "dev_data_path": str(args.dev_data),
        "output_dir": str(args.output_dir),
        "pooling": args.pooling,
        "num_epochs": args.num_epochs,
        "gpu_batch_size": args.gpu_batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "lr_head": 5e-4,
        "lr_lora": 1.5e-4,
        "lora_unfreeze_epoch": 2,
        "exp_weight": 1.7,
        "max_grad_norm": 0.5,
        "variance_weight": 0.2,
        "corr_weight": 0.25,
        "variance_buffer_size": 64,
        "selection_metric": "sum",
        "selection_std_floor": 0.10,
        "selection_std_penalty": 2.0,
        "selection_exp_weight": 1.5,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
