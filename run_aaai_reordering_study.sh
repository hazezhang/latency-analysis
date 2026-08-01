#!/usr/bin/env bash
# Run after E1, or on a separate GPU. Results are exploratory until manual validation.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="experiments/aaai_reordering_simalign_20260720"

if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install simalign
"${PYTHON_BIN}" run_reordering_mechanism_study.py \
  --input data/experiments/r027_shared_outer_quality/all_lat_segments.json \
  --output-dir "${OUT_DIR}" \
  --aligner-model bert | tee "${OUT_DIR}.log"
