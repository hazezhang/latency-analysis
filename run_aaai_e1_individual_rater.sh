#!/usr/bin/env bash
# E1: professional individual-rater sensitivity, matched to the shared-label recipe.
# Run on the GPU host after synchronizing the new data/experiments directory.
set -euo pipefail

DATA_ROOT="data/experiments/aaai_rater_aware_20260720/professional_individual"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20260718 20260719 20260720)
RATERS=(R05 R06)

for rater in "${RATERS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out_dir="experiments/aaai_e1_${rater}_seed_${seed}"
    if [[ -e "${out_dir}" ]]; then
      echo "Refusing to overwrite existing output: ${out_dir}" >&2
      exit 1
    fi
    mkdir -p "${out_dir}"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" run_train_v1.py \
      --train-data "${DATA_ROOT}/${rater}/train.json" \
      --dev-data "${DATA_ROOT}/${rater}/dev.json" \
      --output-dir "${out_dir}" \
      --num-epochs 10 --pooling cls --gpu-batch-size 16 \
      --lr-head 5e-4 --lr-lora 0 --lora-unfreeze-epoch 999 \
      --exp-weight 1.7 --variance-weight 0.2 --corr-weight 0.25 \
      --selection-metric sum --seed "${seed}" --offline | tee "${out_dir}/train.log"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" evaluate.py \
      --checkpoint best_model2.pt --checkpoint_dir "${out_dir}" \
      --train_data "${DATA_ROOT}/${rater}/train.json" \
      --dev_data "${DATA_ROOT}/${rater}/dev.json" \
      --test_data "${DATA_ROOT}/${rater}/test.json" \
      --pooling cls --batch_size 8 \
      --export "${out_dir}/predictions_test.json" \
      --export_data "${DATA_ROOT}/${rater}/test.json" | tee "${out_dir}/eval.log"
  done
done
