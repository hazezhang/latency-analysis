#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv_direct_lat/bin/python}"
while [[ ! -f experiments/aaai_direct_lat_calibration_export.COMPLETE ]]; do
  sleep 30
done

SMOKE_DATA="data/experiments/aaai_crossfitted_outer_quality_corrected/outer_01_speech_1/final_outer"
for variant in joint joint_delay; do
  extra=()
  [[ "${variant}" == "joint_delay" ]] && extra+=(--use-delay)
  "${PYTHON_BIN}" run_train_lat_multitask.py \
    --train-data "${SMOKE_DATA}/train.json" \
    --dev-data "${SMOKE_DATA}/dev.json" \
    --predict-data "${SMOKE_DATA}/predict.json" \
    --output-dir "experiments/aaai_multitask_smoke/${variant}" \
    --seed 20260718 --num-epochs 1 --batch-size 16 --pooling cls \
    "${extra[@]}"
  test -f "experiments/aaai_multitask_smoke/${variant}/predictions.json"
done

PYTHON_BIN="${PYTHON_BIN}" bash run_aaai_multitask_lat_full.sh
