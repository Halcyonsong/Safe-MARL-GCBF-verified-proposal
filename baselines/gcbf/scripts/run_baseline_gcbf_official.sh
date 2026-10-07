#!/usr/bin/env bash
# Official-scale GCBF (non-plus) training — MIT-REALM README / paper scale.
# Env+eval protocol identical to project baselines (obs=4, n=8, area=4).
# Usage: run_baseline_gcbf_official.sh [steps] [n_env_train] [batch_size]
#   Defaults: steps=1000, n_env_train=16, batch=256
# Fallback if OOM: e.g. bash scripts/run_baseline_gcbf_official.sh 1000 8 128
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
STEPS="${1:-1000}"
NTRAIN="${2:-16}"
BATCH="${3:-256}"
# n_env_test: keep 2x train envs like official default 16→32
NTEST=$(( NTRAIN * 2 ))
# buffer: at least 2*batch, cap-ish at official 512 when batch<=256
BUF=512
if (( BATCH > 256 )); then BUF=$BATCH; fi
echo "[run_baseline_gcbf_official] steps=$STEPS n_env_train=$NTRAIN n_env_test=$NTEST batch=$BATCH buffer=$BUF lr=1e-5"
"$PY" baselines/gcbf/code/train.py \
  --algo gcbf \
  --env DoubleIntegrator \
  --area-size 4.0 \
  --num-agents 8 \
  --obs 4 \
  --n-rays 32 \
  --seed 0 \
  --steps "$STEPS" \
  --n-env-train "$NTRAIN" \
  --n-env-test "$NTEST" \
  --buffer-size "$BUF" \
  --batch-size "$BATCH" \
  --inner-epoch 8 \
  --horizon 32 \
  --lr-actor 1e-5 \
  --lr-cbf 1e-5 \
  --loss-action-coef 1e-4 \
  --loss-unsafe-coef 1.0 \
  --loss-safe-coef 1.0 \
  --loss-h-dot-coef 0.01 \
  --eval-interval 1 \
  --save-interval 10
