#!/usr/bin/env bash
# R026-Q: three fixed-seed quality-model runs on shared two-rater labels.
set -euo pipefail

DATA_ROOT="data/experiments/lqexp_shared_20260718"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20260718 20260719 20260720)

for seed in "${SEEDS[@]}"; do
  out_dir="experiments/r026q_shared_seed_${seed}"
  if [[ -e "${out_dir}" ]]; then
    echo "Refusing to overwrite existing output: ${out_dir}" >&2
    exit 1
  fi
  mkdir -p "${out_dir}"

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" run_train_v1.py \
    --train-data "${DATA_ROOT}/professional_shared_train.json" \
    --dev-data "${DATA_ROOT}/professional_shared_dev.json" \
    --output-dir "${out_dir}" \
    --num-epochs 10 --pooling cls --gpu-batch-size 16 \
    --lr-head 5e-4 --lr-lora 0 --lora-unfreeze-epoch 999 \
    --exp-weight 1.7 --variance-weight 0.2 --corr-weight 0.25 \
    --selection-metric sum --seed "${seed}" --offline | tee "${out_dir}/train.log"

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" evaluate.py \
    --checkpoint best_model2.pt --checkpoint_dir "${out_dir}" \
    --train_data "${DATA_ROOT}/professional_shared_train.json" \
    --dev_data "${DATA_ROOT}/professional_shared_dev.json" \
    --test_data "${DATA_ROOT}/professional_shared_test.json" \
    --pooling cls --batch_size 8 \
    --export "${out_dir}/predictions_test.json" \
    --export_data "${DATA_ROOT}/professional_shared_test.json" | tee "${out_dir}/eval.log"

  echo "Completed R026-Q seed ${seed}: ${out_dir}"
done
