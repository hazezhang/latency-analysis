#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv_aaai/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/experiments/aaai_crossfitted_outer_quality_corrected}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/aaai_multitask_lat}"
VARIANT="${VARIANT:?Set VARIANT=joint or VARIANT=joint_delay}"

if [[ "${VARIANT}" != "joint" && "${VARIANT}" != "joint_delay" ]]; then
  echo "VARIANT must be joint or joint_delay" >&2
  exit 2
fi

for seed in 20260718 20260719 20260720; do
  echo "[$(date -Is)] train seed=${seed} variant=${VARIANT}"
  PYTHON_BIN="${PYTHON_BIN}" DATA_ROOT="${DATA_ROOT}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
    SEED="${seed}" VARIANT="${VARIANT}" bash run_aaai_multitask_lat_baselines.sh
done

touch "experiments/aaai_multitask_lat.${VARIANT}.COMPLETE"
echo "[$(date -Is)] complete variant=${VARIANT}"
