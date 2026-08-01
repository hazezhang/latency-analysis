#!/usr/bin/env python3
"""Script-mode launcher for train_v1.ipynb.

The notebook contains the model/dataset/training definitions. This launcher
executes those definitions, then calls run_train() with platform-friendly
arguments so the same v1 training code can run in a job scheduler.
"""

import argparse
import json
import os
import random
from pathlib import Path

# Some platform images bundle old onnx/torchvision builds with newer protobuf.
# transformers may import that chain while peft initializes, so set the protobuf
# fallback before any notebook imports run.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def load_train_v1_symbols(notebook_path: Path) -> dict:
    with notebook_path.open("r", encoding="utf-8") as f:
        notebook = json.load(f)

    namespace = {"__name__": "__train_v1_notebook__"}
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    # Cell 0: setup. Cell 1: imports/classes/functions. Cell 2: run_train()
    # definition plus an auto-run block. We execute only the definition part.
    exec(code_cells[0], namespace)
    exec(code_cells[1], namespace)

    train_cell = code_cells[2]
    marker = "TRAIN_CONFIG ="
    if marker in train_cell:
        train_cell = train_cell.split(marker, 1)[0]
    exec(train_cell, namespace)
    return namespace


def set_seed(seed: int) -> None:
    """Seed all stochastic training components before model construction."""
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run train_v1.ipynb in script mode")
    parser.add_argument("--notebook", default=os.getenv("TRAIN_V1_NOTEBOOK", "train_v1.ipynb"))
    parser.add_argument("--train-data", default=os.getenv("TRAIN_DATA_PATH", "train_set.json"))
    parser.add_argument("--dev-data", default=os.getenv("DEV_DATA_PATH", "dev_set.json"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "./checkpoints2"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "Unbabel/wmt22-cometkiwi-da"))
    parser.add_argument("--pooling", default=os.getenv("POOLING", "cls"), choices=["cls", "mean"])
    parser.add_argument("--num-epochs", type=int, default=_env_int("NUM_EPOCHS", 10))
    parser.add_argument("--max-length", type=int, default=_env_int("MAX_LENGTH", 512))
    parser.add_argument("--gpu-batch-size", type=int, default=_env_int("GPU_BATCH_SIZE", 16))
    parser.add_argument("--cpu-batch-size", type=int, default=_env_int("CPU_BATCH_SIZE", 1))
    parser.add_argument("--cpu-grad-accum", type=int, default=_env_int("CPU_GRAD_ACCUM", 8))
    parser.add_argument("--num-workers", type=int, default=_env_int("NUM_WORKERS", 4))
    parser.add_argument("--seed", type=int, default=None, help="Optional global seed for reproducible training.")
    parser.add_argument("--lr-head", type=float, default=_env_float("LR_HEAD", 5e-4))
    parser.add_argument("--lr-lora", type=float, default=_env_float("LR_LORA", 1.5e-4))
    parser.add_argument("--lora-unfreeze-epoch", type=int, default=_env_int("LORA_UNFREEZE_EPOCH", 2))
    parser.add_argument("--exp-weight", type=float, default=_env_float("EXP_WEIGHT", 1.7))
    parser.add_argument("--max-grad-norm", type=float, default=_env_float("MAX_GRAD_NORM", 0.5))
    parser.add_argument("--variance-weight", type=float, default=_env_float("VARIANCE_WEIGHT", -1.0))
    parser.add_argument("--corr-weight", type=float, default=_env_float("CORR_WEIGHT", -1.0))
    parser.add_argument("--variance-buffer-size", type=int, default=_env_int("VARIANCE_BUFFER_SIZE", 64))
    parser.add_argument(
        "--selection-metric",
        default=os.getenv("SELECTION_METRIC", "sum"),
        choices=["sum", "std_penalty", "exp_weighted_std"],
        help="How to select best_model2.pt from dev metrics.",
    )
    parser.add_argument("--selection-std-floor", type=float, default=_env_float("SELECTION_STD_FLOOR", 0.10))
    parser.add_argument("--selection-std-penalty", type=float, default=_env_float("SELECTION_STD_PENALTY", 2.0))
    parser.add_argument("--selection-exp-weight", type=float, default=_env_float("SELECTION_EXP_WEIGHT", 1.5))
    parser.add_argument("--offline", action="store_true", default=_env_bool("TRAIN_V1_OFFLINE", False))
    parser.add_argument("--check-only", action="store_true", help="Validate paths/imports without starting training")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path.cwd()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    required_paths = [project_root / args.notebook, project_root / args.train_data]
    if args.dev_data:
        required_paths.append(project_root / args.dev_data)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    namespace = load_train_v1_symbols(project_root / args.notebook)
    run_train = namespace["run_train"]

    if args.seed is not None:
        set_seed(args.seed)

    variance_weight = args.variance_weight
    if variance_weight < 0:
        variance_weight = 0.2
    corr_weight = args.corr_weight
    if corr_weight < 0:
        corr_weight = 0.25

    cfg = {
        "model_name": args.model_name,
        "train_data_path": args.train_data,
        "dev_data_path": args.dev_data,
        "output_dir": args.output_dir,
        "pooling": args.pooling,
        "num_epochs": args.num_epochs,
        "max_length": args.max_length,
        "gpu_batch_size": args.gpu_batch_size,
        "cpu_batch_size": args.cpu_batch_size,
        "cpu_grad_accum": args.cpu_grad_accum,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "lr_head": args.lr_head,
        "lr_lora": args.lr_lora,
        "lora_unfreeze_epoch": args.lora_unfreeze_epoch,
        "exp_weight": args.exp_weight,
        "max_grad_norm": args.max_grad_norm,
        "variance_weight": variance_weight,
        "corr_weight": corr_weight,
        "variance_buffer_size": args.variance_buffer_size,
        "selection_metric": args.selection_metric,
        "selection_std_floor": args.selection_std_floor,
        "selection_std_penalty": args.selection_std_penalty,
        "selection_exp_weight": args.selection_exp_weight,
    }

    print("train_v1 script-mode config:")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))
    print(f"offline={args.offline}")

    if args.check_only:
        print("Check passed. Training was not started.")
        return 0

    run_train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
