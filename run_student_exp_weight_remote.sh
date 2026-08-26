#!/usr/bin/env bash
set -euo pipefail

ROOT=/122090786/process3_aaai_current
PY="${PYTHON_BIN:-$ROOT/.venv_aaai/bin/python}"
SEED="${SEED:-20260825}"
EXP_WEIGHT="${EXP_WEIGHT:?set EXP_WEIGHT}"
TAG="${TAG:?set TAG}"
OUT="$ROOT/experiments/student_exp_weight_${TAG}_${SEED}"
STUDENT_ROOT="$ROOT/data/experiments/student_weak_supervision_strict_20260806"
PROF_ROOT="$ROOT/data/experiments/aaai_crossfitted_outer_quality_corrected"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
folds=(outer_01_speech_1 outer_02_speech_2 outer_03_speech_3 outer_04_speech_5737 outer_05_speech_6182lawyer_1 outer_06_speech_6182lawyer_2 outer_07_speech_6182lawyer_3 outer_08_speech_6182lawyer_4 outer_09_speech_Orphanages outer_10_speech_Petranek_1 outer_11_speech_Petranek_2 outer_12_speech_Petranek_3 outer_13_speech_Petranek_4 outer_14_speech_Petranek_5 outer_15_speech_athlete_1 outer_16_speech_athlete_2)

for fold in "${folds[@]}"; do
  student_out="$OUT/student_pretrain/$fold/student_model"
  if [ ! -f "$student_out/best_model2.pt" ] && [ ! -f "$student_out/final_model2.pt" ]; then
    mkdir -p "$student_out"
    "$PY" run_train_v1.py --notebook train_v1.ipynb \
      --train-data "$STUDENT_ROOT/$fold/student_fit.json" \
      --dev-data "$STUDENT_ROOT/$fold/student_dev.json" \
      --output-dir "$student_out" --pooling mean --num-epochs 10 \
      --gpu-batch-size 16 --num-workers 2 --seed "$SEED" \
      --exp-weight "$EXP_WEIGHT" --offline
  fi
  model="$student_out/best_model2.pt"
  [ -f "$model" ] || model="$student_out/final_model2.pt"
  out="$OUT/s3/$fold"
  cal="$out/calibrated_model"
  if [ ! -f "$out/predictions_outer_test.json" ]; then
    mkdir -p "$cal"
    "$PY" run_student_weak_supervision_s3.py --notebook train_v1.ipynb \
      --student-checkpoint "$model" \
      --train-data "$PROF_ROOT/$fold/final_outer/train.json" \
      --dev-data "$PROF_ROOT/$fold/final_outer/dev.json" \
      --output-dir "$cal" --pooling mean --num-epochs 10 \
      --gpu-batch-size 16 --num-workers 2 --seed "$SEED" --offline
    "$PY" evaluate_student_weak_supervision_s1.py --notebook train_v1.ipynb \
      --checkpoint-dir "$cal" \
      --train-data "$PROF_ROOT/$fold/final_outer/train.json" \
      --dev-data "$PROF_ROOT/$fold/final_outer/dev.json" \
      --test-data "$PROF_ROOT/$fold/final_outer/predict.json" \
      --export "$out/predictions_outer_test.json" --pooling mean --offline
    find "$student_out" "$cal" -maxdepth 1 -type f \( -name 'checkpoint_epoch_*.pt' -o -name 'final_model2.pt' -o -name 'best_model2.pt' \) -delete
  fi
done

"$PY" run_student_weak_supervision_s1.py summarize --predictions "$OUT/s3" \
  --protocol "Student pretrain EXP weight ${EXP_WEIGHT} then professional calibration" \
  --output-name "s3_exp_weight_${TAG}_outer_summary.json"
