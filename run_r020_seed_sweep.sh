#!/usr/bin/env bash
# Run two independent R020 seed replications on the GPU server.
set -euo pipefail

DATA_ROOT="data/experiments/r020_nested_quality"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif [[ -x /opt/conda/bin/python ]]; then
  PYTHON_BIN="/opt/conda/bin/python"
else
  echo "No python executable found; set PYTHON_BIN explicitly." >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
SEEDS=(20260713 20260714)

for seed in "${SEEDS[@]}"; do
  out_root="experiments/r020_nested_quality_seed_${seed}"
  if [[ -e "${out_root}" ]]; then
    echo "Refusing to overwrite existing output: ${out_root}" >&2
    exit 1
  fi
  mkdir -p "${out_root}"

  for data_dir in "${DATA_ROOT}"/fold_*_speech_*; do
    fold="$(basename "${data_dir}")"
    fold_out="${out_root}/${fold}"
    mkdir -p "${fold_out}"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" run_train_v1.py \
      --train-data "${data_dir}/train.json" \
      --dev-data "${data_dir}/dev.json" \
      --output-dir "${fold_out}" \
      --num-epochs 10 --pooling cls --gpu-batch-size 16 \
      --lr-head 5e-4 --lr-lora 0 --lora-unfreeze-epoch 999 \
      --exp-weight 1.7 --variance-weight 0.2 --corr-weight 0.25 \
      --selection-metric sum --seed "${seed}" --offline | tee "${fold_out}/train.log"

    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "${PYTHON_BIN}" evaluate.py \
      --checkpoint best_model2.pt --checkpoint_dir "${fold_out}" \
      --train_data "${data_dir}/train.json" --dev_data "${data_dir}/dev.json" \
      --test_data "${data_dir}/predict_all.json" --pooling cls --batch_size 8 \
      --export "${fold_out}/predictions_all.json" \
      --export_data "${data_dir}/predict_all.json" | tee "${fold_out}/eval.log"
  done

  echo "Completed R020 seed ${seed}: ${out_root}"
done
