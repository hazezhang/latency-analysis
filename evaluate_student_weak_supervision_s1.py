#!/usr/bin/env python3
"""Evaluate a run_train_v1 checkpoint on a fold's professional outer test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_symbols(notebook_path: Path) -> dict:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__s1_eval_notebook__"}
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    exec(cells[0], namespace)
    exec(cells[1], namespace)
    eval_cell = cells[3]
    marker = "EVAL_CONFIG ="
    if marker in eval_cell:
        eval_cell = eval_cell.split(marker, 1)[0]
    exec(eval_cell, namespace)
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default="train_v1.ipynb", type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--dev-data", required=True, type=Path)
    parser.add_argument("--test-data", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--pooling", default="mean", choices=["cls", "mean"])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    ns = load_symbols(args.notebook)
    ns["run_eval"]({
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint": "best_model2.pt",
        "pooling": args.pooling,
        "train_data": str(args.train_data),
        "dev_data": str(args.dev_data),
        "test_data": str(args.test_data),
        "export": str(args.export),
        "export_data": str(args.test_data),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
