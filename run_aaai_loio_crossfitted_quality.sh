#!/usr/bin/env bash
# Train interpreter-disjoint quality models for leave-one-interpreter-out LAT evaluation.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/experiments/aaai_loio_outer_quality_corrected}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-aaai_loio_corrected_seed}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
SEEDS_TEXT="${SEEDS:-20260718 20260719 20260720}"
OUTER_FILTER="${OUTER_FILTER:-}"
OFFLINE="${OFFLINE:-1}"
read -r -a SEEDS <<< "${SEEDS_TEXT}"
OFFLINE_ARGS=()
if [[ "${OFFLINE}" == "1" ]]; then
  OFFLINE_ARGS+=(--offline)
fi

run_model() {
  local data_dir="$1"
  local out_dir="$2"
  if [[ -f "${out_dir}/predictions.json" ]]; then
    echo "Keeping completed output: ${out_dir}"
    return 0
  fi
  if [[ -e "${out_dir}" ]]; then
    echo "Refusing to reuse incomplete output: ${out_dir}" >&2
    exit 1
  fi
  mkdir -p "${out_dir}"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" run_train_v1.py \
    --train-data "${data_dir}/train.json" --dev-data "${data_dir}/dev.json" \
    --output-dir "${out_dir}" --num-epochs "${NUM_EPOCHS}" --pooling cls --gpu-batch-size 16 \
    --lr-head 5e-4 --lr-lora 0 --lora-unfreeze-epoch 999 \
    --exp-weight 1.7 --variance-weight 0.2 --corr-weight 0.25 --selection-metric sum \
    --seed "${CURRENT_SEED}" "${OFFLINE_ARGS[@]}" | tee "${out_dir}/train.log"
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" evaluate.py \
    --checkpoint best_model2.pt --checkpoint_dir "${out_dir}" \
    --train_data "${data_dir}/train.json" --dev_data "${data_dir}/dev.json" \
    --test_data "${data_dir}/predict.json" --pooling cls --batch_size 8 \
    --export "${out_dir}/predictions.json" --export_data "${data_dir}/predict.json" | tee "${out_dir}/eval.log"
  find "${out_dir}" -maxdepth 1 -type f -name "*.pt" -delete
}

shopt -s nullglob
outer_dirs=("${DATA_ROOT}"/outer_*_interpreter_*)
if [[ "${#outer_dirs[@]}" -ne 7 ]]; then
  echo "Expected 7 LOIO outer folds under ${DATA_ROOT}; found ${#outer_dirs[@]}" >&2
  exit 1
fi

for CURRENT_SEED in "${SEEDS[@]}"; do
  seed_root="experiments/${OUTPUT_PREFIX}_${CURRENT_SEED}"
  mkdir -p "${seed_root}"
  for outer_dir in "${outer_dirs[@]}"; do
    outer_name="$(basename "${outer_dir}")"
    if [[ -n "${OUTER_FILTER}" && "${outer_name}" != "${OUTER_FILTER}" ]]; then
      continue
    fi
    data_dirs=("${outer_dir}"/inner_* "${outer_dir}"/final_outer)
    if [[ "${#data_dirs[@]}" -ne 5 ]]; then
      echo "Expected 4 inner folds plus final_outer in ${outer_dir}" >&2
      exit 1
    fi
    for data_dir in "${data_dirs[@]}"; do
      run_model "${data_dir}" "${seed_root}/${outer_name}/$(basename "${data_dir}")"
    done
  done
  echo "Completed LOIO quality seed ${CURRENT_SEED}: ${seed_root}"
done
