#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv_direct_lat/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/aaai_multitask_lat}"

for seed in 20260718 20260719 20260720; do
  for variant in joint joint_delay; do
    echo "[$(date -Is)] start seed=${seed} variant=${variant}"
    PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${OUTPUT_ROOT}" SEED="${seed}" VARIANT="${variant}" \
      bash run_aaai_multitask_lat_baselines.sh
    echo "[$(date -Is)] complete seed=${seed} variant=${variant}"
  done
done

"${PYTHON_BIN}" summarize_aaai_multitask_lat.py \
  --root-template "${OUTPUT_ROOT}_seed_{}" \
  --output-dir experiments/aaai_multitask_lat_summary_20260722
touch experiments/aaai_multitask_lat.COMPLETE
