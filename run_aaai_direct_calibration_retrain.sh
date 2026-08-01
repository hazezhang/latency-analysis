#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv_aaai/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/experiments/aaai_crossfitted_outer_quality_corrected}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/aaai_direct_lat_calibration_retrain}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-experiments/aaai_direct_lat_calibration_inputs}"
VARIANT="${VARIANT:?Set VARIANT=text or VARIANT=text_delay}"

if [[ "${VARIANT}" != "text" && "${VARIANT}" != "text_delay" ]]; then
  echo "VARIANT must be text or text_delay" >&2
  exit 2
fi

for seed in 20260718 20260719 20260720; do
  echo "[$(date -Is)] train seed=${seed} variant=${VARIANT}"
  PYTHON_BIN="${PYTHON_BIN}" \
    DATA_ROOT="${DATA_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    SEED="${seed}" \
    VARIANT="${VARIANT}" \
    bash run_aaai_direct_lat_baselines.sh

  echo "[$(date -Is)] export development predictions seed=${seed} variant=${VARIANT}"
  "${PYTHON_BIN}" export_direct_lat_calibration_inputs.py \
    --checkpoint-root "${OUTPUT_ROOT}_seed_${seed}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${CALIBRATION_ROOT}_seed_${seed}" \
    --variant "${VARIANT}"
done

touch "${CALIBRATION_ROOT}.${VARIANT}.COMPLETE"
echo "[$(date -Is)] complete variant=${VARIANT}"
