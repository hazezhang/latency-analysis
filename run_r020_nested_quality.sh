#!/usr/bin/env bash
# Run on the GPU server from the project root after syncing r020_nested_quality.
set -euo pipefail

DATA_ROOT="data/experiments/r020_nested_quality"
OUT_ROOT="experiments/r020_nested_quality_20260712"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${DATA_ROOT}/manifest.json" ]]; then
  echo "Missing R020 manifest: ${DATA_ROOT}/manifest.json" >&2
  exit 1
fi
if [[ -e "${OUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing output: ${OUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

for data_dir in "${DATA_ROOT}"/fold_*_speech_*; do
  fold="$(basename "${data_dir}")"
  out_dir="${OUT_ROOT}/${fold}"

  if [[ ! -f "${data_dir}/train.json" || ! -f "${data_dir}/dev.json" || ! -f "${data_dir}/predict_all.json" ]]; then
    echo "Missing fold data: ${data_dir}" >&2
    exit 1
  fi

  mkdir -p "${out_dir}"
  echo "=== R020 ${fold}: train outer-nested quality model ==="
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

  echo "=== R020 ${fold}: export all 150 LAT predictions ==="
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" evaluate.py \
    --checkpoint best_model2.pt \
    --checkpoint_dir "${out_dir}" \
    --train_data "${data_dir}/train.json" \
    --dev_data "${data_dir}/dev.json" \
    --test_data "${data_dir}/predict_all.json" \
    --pooling cls \
    --batch_size 8 \
    --export "${out_dir}/predictions_all.json" \
    --export_data "${data_dir}/predict_all.json" | tee "${out_dir}/eval.log"
done

echo "R020 outer-nested quality prediction exports complete: ${OUT_ROOT}"
