#!/usr/bin/env bash
set -euo pipefail

ROOT=/122090786/process3_aaai_current
PY="$ROOT/.venv_aaai/bin/python"
SEEDS_TEXT="${SEEDS:-20260805 20260806 20260807}"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"

for seed in "${SEEDS_ARRAY[@]}"; do
  echo "[$(date -Is)] START SEED=$seed S0"
  SEED="$seed" OUT_ROOT="$ROOT/experiments/student_weak_supervision_s0_multiseed_${seed}" \
    bash run_student_weak_supervision_s0_remote.sh
  echo "[$(date -Is)] DONE SEED=$seed S0"

  echo "[$(date -Is)] START SEED=$seed S1"
  SEED="$seed" OUT_ROOT="$ROOT/experiments/student_weak_supervision_s1_multiseed_${seed}" \
    bash run_student_weak_supervision_s1_strict_remote.sh
  echo "[$(date -Is)] DONE SEED=$seed S1"

  echo "[$(date -Is)] START SEED=$seed S3"
  SEED="$seed" OUT_ROOT="$ROOT/experiments/student_weak_supervision_s3_multiseed_${seed}" \
    bash run_student_weak_supervision_s3_remote.sh
  echo "[$(date -Is)] DONE SEED=$seed S3"
done

echo "[$(date -Is)] ALL MULTISEED RUNS COMPLETE"
