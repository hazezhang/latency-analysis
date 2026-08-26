#!/usr/bin/env bash
set -euo pipefail

ROOT=/122090786/process3_aaai_current
PROF_ROOT="$ROOT/data/experiments/aaai_crossfitted_outer_quality_corrected"
STUDENT_ROOT="$ROOT/data/experiments/student_weak_supervision_strict_20260806"
OUT="${OUT_ROOT:-$ROOT/experiments/professional_learning_curve_20260825}"
PY="${PYTHON_BIN:-/122090786/process3_aaai_py311/bin/python}"
SEED="${SEED:-20260825}"
LEVELS="${LEVELS:-10 25 50 75 100}"
export HF_HOME="$ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

folds=(outer_01_speech_1 outer_02_speech_2 outer_03_speech_3 outer_04_speech_5737 outer_05_speech_6182lawyer_1 outer_06_speech_6182lawyer_2 outer_07_speech_6182lawyer_3 outer_08_speech_6182lawyer_4 outer_09_speech_Orphanages outer_10_speech_Petranek_1 outer_11_speech_Petranek_2 outer_12_speech_Petranek_3 outer_13_speech_Petranek_4 outer_14_speech_Petranek_5 outer_15_speech_athlete_1 outer_16_speech_athlete_2)

make_subset() {
  local src="$1" dst="$2" pct="$3" fold="$4"
  "$PY" - "$src" "$dst" "$pct" "$SEED" "$fold" <<'PY'
import json, random, sys
src, dst, pct, seed, fold = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rows = json.load(open(src, encoding='utf-8'))
n = max(1, round(len(rows) * pct / 100))
rng = random.Random(seed * 1009 + pct * 9176 + sum(map(ord, fold)))
idx = sorted(rng.sample(range(len(rows)), n))
json.dump([rows[i] for i in idx], open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
PY
}

run_s0() {
  local pct="$1" fold="$2"
  local fold_dir="$PROF_ROOT/$fold/final_outer"
  local out="$OUT/s0_${pct}pct/$fold"
  local model="$out/model"
  mkdir -p "$model"
  make_subset "$fold_dir/train.json" "$out/train_subset.json" "$pct" "$fold"
  "$PY" run_train_v1.py --notebook train_v1.ipynb --train-data "$out/train_subset.json" --dev-data "$fold_dir/dev.json" --output-dir "$model" --pooling mean --num-epochs 10 --gpu-batch-size 16 --num-workers 2 --seed "$SEED" --offline
  "$PY" evaluate_student_weak_supervision_s1.py --notebook train_v1.ipynb --checkpoint-dir "$model" --train-data "$out/train_subset.json" --dev-data "$fold_dir/dev.json" --test-data "$fold_dir/predict.json" --export "$out/predictions_outer_test.json" --pooling mean --offline
  find "$model" -maxdepth 1 -type f \( -name 'checkpoint_epoch_*.pt' -o -name 'final_model2.pt' -o -name 'best_model2.pt' \) -delete
}

run_s3_pretrain() {
  local fold="$1" student_out="$OUT/student_pretrain/$fold/student_model"
  mkdir -p "$student_out"
  "$PY" run_train_v1.py --notebook train_v1.ipynb --train-data "$STUDENT_ROOT/$fold/student_fit.json" --dev-data "$STUDENT_ROOT/$fold/student_dev.json" --output-dir "$student_out" --pooling mean --num-epochs 10 --gpu-batch-size 16 --num-workers 2 --seed "$SEED" --offline
}

run_s3_calibration() {
  local pct="$1" fold="$2"
  local fold_dir="$PROF_ROOT/$fold/final_outer"
  local student_model="$OUT/student_pretrain/$fold/student_model"
  local out="$OUT/s3_${pct}pct/$fold"
  local model="$out/calibrated_model"
  mkdir -p "$model"
  make_subset "$fold_dir/train.json" "$out/train_subset.json" "$pct" "$fold"
  "$PY" run_student_weak_supervision_s3.py --notebook train_v1.ipynb --student-checkpoint "$student_model/best_model2.pt" --train-data "$out/train_subset.json" --dev-data "$fold_dir/dev.json" --output-dir "$model" --pooling mean --num-epochs 10 --gpu-batch-size 16 --num-workers 2 --seed "$SEED" --offline
  "$PY" evaluate_student_weak_supervision_s1.py --notebook train_v1.ipynb --checkpoint-dir "$model" --train-data "$out/train_subset.json" --dev-data "$fold_dir/dev.json" --test-data "$fold_dir/predict.json" --export "$out/predictions_outer_test.json" --pooling mean --offline
  find "$model" -maxdepth 1 -type f \( -name 'checkpoint_epoch_*.pt' -o -name 'final_model2.pt' -o -name 'best_model2.pt' \) -delete
}

mkdir -p "$OUT"
for fold in "${folds[@]}"; do
  echo "[$(date -Is)] PRETRAIN fold=$fold"
  if [ -f "$OUT/student_pretrain/$fold/student_model/best_model2.pt" ] || [ -f "$OUT/student_pretrain/$fold/student_model/final_model2.pt" ]; then
    echo "[$(date -Is)] PRETRAIN_SKIP fold=$fold existing_checkpoint"
  else
    run_s3_pretrain "$fold"
  fi
  find "$OUT/student_pretrain/$fold/student_model" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pt' -delete
done
for pct in $LEVELS; do
  for fold in "${folds[@]}"; do
    echo "[$(date -Is)] S0 pct=$pct fold=$fold"
    run_s0 "$pct" "$fold"
  done
  for fold in "${folds[@]}"; do
    echo "[$(date -Is)] S3 pct=$pct fold=$fold"
    run_s3_calibration "$pct" "$fold"
  done
done
for pct in $LEVELS; do
  "$PY" run_student_weak_supervision_s1.py summarize --predictions "$OUT/s0_${pct}pct" --protocol "Professional-only learning curve ${pct}%" --output-name "s0_${pct}pct_outer_summary.json"
  "$PY" run_student_weak_supervision_s1.py summarize --predictions "$OUT/s3_${pct}pct" --protocol "Student pretrain then professional calibration learning curve ${pct}%" --output-name "s3_${pct}pct_outer_summary.json"
done
echo "[$(date -Is)] LEARNING_CURVE_COMPLETE"
