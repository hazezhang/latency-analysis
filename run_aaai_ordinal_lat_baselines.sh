#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_ROOT="${DATA_ROOT:-data/experiments/aaai_crossfitted_outer_quality_corrected}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/aaai_ordinal_lat}"
SEED="${SEED:-20260718}"
VARIANT="${VARIANT:-ordinal_text}"

extra_args=()
if [[ "${VARIANT}" == "ordinal_text_delay" ]]; then
  extra_args+=(--use-delay)
elif [[ "${VARIANT}" != "ordinal_text" ]]; then
  echo "VARIANT must be ordinal_text or ordinal_text_delay" >&2
  exit 2
fi

for outer_dir in "${DATA_ROOT}"/outer_*; do
  fold_name="$(basename "${outer_dir}")"
  data_dir="${outer_dir}/final_outer"
  output_dir="${OUTPUT_ROOT}_seed_${SEED}/${VARIANT}/${fold_name}"
  if [[ -f "${output_dir}/predictions.json" ]]; then
    echo "[skip] ${output_dir}"
    continue
  fi
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to reuse incomplete output: ${output_dir}" >&2
    exit 1
  fi
  echo "[run] seed=${SEED} variant=${VARIANT} fold=${fold_name}"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" run_train_lat_ordinal.py \
    --train-data "${data_dir}/train.json" \
    --dev-data "${data_dir}/dev.json" \
    --predict-data "${data_dir}/predict.json" \
    --output-dir "${output_dir}" \
    --seed "${SEED}" --pooling cls --num-epochs 10 --batch-size 16 \
    "${extra_args[@]}"
done

echo "Completed ordinal LAT baseline seed=${SEED} variant=${VARIANT}"
