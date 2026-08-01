#!/usr/bin/env bash
# R027: three-seed outer-speech quality predictions for the shared-label LAT bridge.
set -euo pipefail

DATA_ROOT="data/experiments/r027_shared_outer_quality"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20260718 20260719 20260720)

for seed in "${SEEDS[@]}"; do
  seed_root="experiments/r027_shared_outer_quality_seed_${seed}"
  if [[ -e "${seed_root}" ]]; then
    echo "Refusing to overwrite existing output: ${seed_root}" >&2
    exit 1
  fi
  mkdir -p "${seed_root}"

  for data_dir in "${DATA_ROOT}"/fold_*_speech_*; do
    fold="$(basename "${data_dir}")"
    out_dir="${seed_root}/${fold}"
    mkdir -p "${out_dir}"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" run_train_v1.py \
      --train-data "${data_dir}/train.json" \
      --dev-data "${data_dir}/dev.json" \
      --output-dir "${out_dir}" \
      --num-epochs 10 --pooling cls --gpu-batch-size 16 \
      --lr-head 5e-4 --lr-lora 0 --lora-unfreeze-epoch 999 \
      --exp-weight 1.7 --variance-weight 0.2 --corr-weight 0.25 \
      --selection-metric sum --seed "${seed}" --offline | tee "${out_dir}/train.log"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" evaluate.py \
      --checkpoint best_model2.pt --checkpoint_dir "${out_dir}" \
      --train_data "${data_dir}/train.json" --dev_data "${data_dir}/dev.json" \
      --test_data "${data_dir}/predict_all.json" --pooling cls --batch_size 8 \
      --export "${out_dir}/predictions_all.json" \
      --export_data "${data_dir}/predict_all.json" | tee "${out_dir}/eval.log"

    # Keep R027 disk use bounded: after predictions are exported, checkpoints are no longer needed
    # for the LAT bridge or seed summary.
    find "${out_dir}" -maxdepth 1 -type f -name "*.pt" -delete
  done
  echo "Completed R027 seed ${seed}: ${seed_root}"
done
