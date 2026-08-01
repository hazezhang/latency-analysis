#!/usr/bin/env bash
# Run on the GPU server from the project root after syncing r017_oof_quality.
set -euo pipefail

DATA_ROOT="data/experiments/r017_oof_quality"
OUT_ROOT="experiments/r017_oof_quality_20260712"
PYTHON_BIN="${PYTHON_BIN:-python}"

folds=(
  "fold_01_speech_1"
  "fold_02_speech_2"
  "fold_03_speech_3"
  "fold_04_speech_6182lawyer_3"
  "fold_05_speech_athlete_2"
)

for fold in "${folds[@]}"; do
  data_dir="${DATA_ROOT}/${fold}"
  out_dir="${OUT_ROOT}/${fold}"

  if [[ ! -f "${data_dir}/train.json" || ! -f "${data_dir}/dev.json" || ! -f "${data_dir}/test.json" ]]; then
    echo "Missing fold data: ${data_dir}" >&2
    exit 1
  fi
  if [[ -e "${out_dir}" ]]; then
    echo "Refusing to overwrite existing output: ${out_dir}" >&2
    exit 1
  fi

  mkdir -p "${out_dir}"
  echo "=== R017 ${fold}: train quality model ==="
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" run_train_v1.py \
    --train-data "${data_dir}/train.json" \
    --dev-data "${data_dir}/dev.json" \
    --output-dir "${out_dir}" \
    --num-epochs 10 \
    --pooling cls \
    --gpu-batch-size 16 \
    --lr-head 5e-4 \
    --lr-lora 0 \
    --lora-unfreeze-epoch 999 \
    --exp-weight 1.7 \
    --variance-weight 0.2 \
    --corr-weight 0.25 \
    --selection-metric sum \
    --offline | tee "${out_dir}/train.log"

  echo "=== R017 ${fold}: export held-out predictions ==="
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" evaluate.py \
    --checkpoint best_model2.pt \
    --checkpoint_dir "${out_dir}" \
    --train_data "${data_dir}/train.json" \
    --dev_data "${data_dir}/dev.json" \
    --test_data "${data_dir}/test.json" \
    --pooling cls \
    --batch_size 8 \
    --export "${out_dir}/predictions_test.json" \
    --export_data "${data_dir}/test.json" | tee "${out_dir}/eval.log"
done

echo "R017 OOF quality prediction exports complete: ${OUT_ROOT}"
