#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv_direct_lat/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/aaai_direct_lat_corrected}"
LOG_DIR="${LOG_DIR:-experiments}"
mkdir -p "${LOG_DIR}"

for seed in 20260718 20260719 20260720; do
  for variant in text text_delay; do
    echo "[$(date -Is)] start seed=${seed} variant=${variant}"
    PYTHON_BIN="${PYTHON_BIN}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      SEED="${seed}" \
      VARIANT="${variant}" \
      bash run_aaai_direct_lat_baselines.sh
    echo "[$(date -Is)] complete seed=${seed} variant=${variant}"
  done

  "${PYTHON_BIN}" summarize_aaai_direct_lat_baselines.py \
    --seed "${seed}" \
    --direct-root "${OUTPUT_ROOT}_seed_${seed}" \
    --official-oof "experiments/aaai_crossfitted_corrected_lat_seed_${seed}_20260721/crossfitted_lat_oof_predictions.csv" \
    --output-dir "experiments/aaai_direct_lat_summary_seed_${seed}"
done

touch "${LOG_DIR}/aaai_direct_lat_corrected.COMPLETE"
echo "[$(date -Is)] all direct LAT baselines complete"
