#!/usr/bin/env bash
set -euo pipefail

ROOT=/122090786/process3_aaai_current
STUDENT_DATA=${STUDENT_DATA_ROOT:-data/experiments/student_weak_supervision_strict_20260806}
PROFESSIONAL_DATA=data/experiments/aaai_crossfitted_outer_quality_corrected
OUT="${OUT_ROOT:-$ROOT/experiments/student_weak_supervision_s1_strict_20260824}"
PY="$ROOT/.venv_aaai/bin/python"
SEED="${SEED:-20260805}"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

run_fold() {
  local fold="$1"
  local student_fold="$STUDENT_DATA/$fold"
  local professional_fold="$PROFESSIONAL_DATA/$fold/final_outer"
  local model_dir="$OUT/$fold/model"
  mkdir -p "$model_dir"
  "$PY" run_train_v1.py --notebook train_v1.ipynb \
    --train-data "$student_fold/student_fit.json" --dev-data "$student_fold/student_dev.json" \
    --output-dir "$model_dir" --pooling mean --num-epochs 10 --gpu-batch-size 16 --num-workers 2 \
    --seed "$SEED" --offline
  "$PY" evaluate_student_weak_supervision_s1.py --notebook train_v1.ipynb \
    --checkpoint-dir "$model_dir" --train-data "$student_fold/student_fit.json" \
    --dev-data "$student_fold/student_dev.json" --test-data "$professional_fold/predict.json" \
    --export "$OUT/$fold/predictions_outer_test.json" --pooling mean --offline
  find "$model_dir" -maxdepth 1 -type f \( -name 'checkpoint_epoch_*.pt' -o -name 'final_model2.pt' -o -name 'best_model2.pt' \) -delete
}

run_worker() {
  local gpu="$1"; shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  for fold in "$@"; do
    echo "[$(date -Is)] START gpu=$gpu fold=$fold"
    run_fold "$fold"
    echo "[$(date -Is)] DONE gpu=$gpu fold=$fold"
  done
}

mkdir -p "$OUT"
run_worker 0 outer_01_speech_1 outer_03_speech_3 outer_05_speech_6182lawyer_1 outer_07_speech_6182lawyer_3 outer_09_speech_Orphanages outer_11_speech_Petranek_2 outer_13_speech_Petranek_4 outer_15_speech_athlete_1 > "$OUT/worker_gpu0.log" 2>&1 &
pid0=$!
run_worker 1 outer_02_speech_2 outer_04_speech_5737 outer_06_speech_6182lawyer_2 outer_08_speech_6182lawyer_4 outer_10_speech_Petranek_1 outer_12_speech_Petranek_3 outer_14_speech_Petranek_5 outer_16_speech_athlete_2 > "$OUT/worker_gpu1.log" 2>&1 &
pid1=$!
echo "worker_gpu0_pid=$pid0"
echo "worker_gpu1_pid=$pid1"
wait "$pid0"
wait "$pid1"
"$PY" run_student_weak_supervision_s1.py summarize --predictions "$OUT" --protocol "S1 strict raw student-only; professional outer speech test" --output-name s1_outer_summary.json
