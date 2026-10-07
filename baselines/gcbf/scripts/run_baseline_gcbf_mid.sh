#!/usr/bin/env bash
# Mid-scale GCBF (non-plus): n_env=8, batch=128, inner=8, lr=1e-5.
# Same eval protocol as other project baselines.
# Usage: run_baseline_gcbf_mid.sh [steps] [n_env_train] [batch_size]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
STEPS="${1:-1000}"
NTRAIN="${2:-8}"
BATCH="${3:-128}"
NTEST=$(( NTRAIN * 2 ))
BUF=512
echo "[run_baseline_gcbf_mid] steps=$STEPS n_env_train=$NTRAIN n_env_test=$NTEST batch=$BATCH buffer=$BUF lr=1e-5"
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
