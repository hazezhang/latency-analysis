#!/usr/bin/env bash
# Validate completed LOIO predictions, run LAT models, and aggregate three formal seeds.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS_TEXT="${SEEDS:-20260718 20260719 20260720}"
QUALITY_PREFIX="${QUALITY_PREFIX:-aaai_loio_corrected_seed}"
LAT_PREFIX="${LAT_PREFIX:-aaai_loio_corrected_lat_seed}"
SUMMARY_DIR="${SUMMARY_DIR:-experiments/aaai_loio_corrected_summary_20260722}"
read -r -a SEEDS <<< "${SEEDS_TEXT}"
result_dirs=()

for seed in "${SEEDS[@]}"; do
  prediction_root="experiments/${QUALITY_PREFIX}_${seed}"
  output_dir="experiments/${LAT_PREFIX}_${seed}"
  "${PYTHON_BIN}" validate_aaai_loio_protocol.py --prediction-root "${prediction_root}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing LAT output: ${output_dir}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" run_latency_aaai_loio_bridge.py \
    --prediction-root "${prediction_root}" \
    --output-dir "${output_dir}"
  result_dirs+=("${output_dir}")
done

"${PYTHON_BIN}" summarize_aaai_loio_crossfitted.py \
  --result-dirs "${result_dirs[@]}" \
  --output-dir "${SUMMARY_DIR}"
