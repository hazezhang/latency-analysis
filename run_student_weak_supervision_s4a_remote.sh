#!/usr/bin/env bash
set -euo pipefail

ROOT=/122090786/process3_aaai_current
STUDENT_DATA=data/experiments/student_weak_supervision_s1_20260805
PROFESSIONAL_DATA=data/experiments/aaai_crossfitted_outer_quality_corrected
OUT="$ROOT/experiments/student_weak_supervision_s4a_20260806"
PY="$ROOT/.venv_aaai/bin/python"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

run_fold() {
  local fold="$1"
  local student_fold="$STUDENT_DATA/$fold"
  local professional_fold="$PROFESSIONAL_DATA/$fold/final_outer"
  local student_model="$OUT/$fold/student_model"
  local professional_model="$OUT/$fold/professional_model"
  mkdir -p "$student_model" "$professional_model"
  "$PY" run_train_v1.py --notebook train_v1.ipynb \
    --train-data "$student_fold/student_raw_fit.json" --dev-data "$student_fold/student_raw_dev.json" \
    --output-dir "$student_model" --pooling mean --num-epochs 10 --gpu-batch-size 16 --num-workers 2 \
    --seed 20260805 --offline
  "$PY" run_student_weak_supervision_s3.py --notebook train_v1.ipynb \
    --student-checkpoint "$student_model/best_model2.pt" --transfer lora \
    --train-data "$professional_fold/train.json" --dev-data "$professional_fold/dev.json" \
    --output-dir "$professional_model" --pooling mean --num-epochs 10 --gpu-batch-size 16 --num-workers 2 \
    --seed 20260805 --offline
  "$PY" evaluate_student_weak_supervision_s1.py --notebook train_v1.ipynb \
    --checkpoint-dir "$professional_model" --train-data "$professional_fold/train.json" \
    --dev-data "$professional_fold/dev.json" --test-data "$professional_fold/predict.json" \
    --export "$OUT/$fold/predictions_outer_test.json" --pooling mean --offline
}

run_worker() {
  local gpu="$1"; shift; export CUDA_VISIBLE_DEVICES="$gpu"
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
"$PY" run_student_weak_supervision_s1.py summarize --predictions "$OUT" \
  --protocol "S4a student LoRA pretrain then fresh professional head; professional outer speech test" \
  --output-name s4a_outer_summary.json
