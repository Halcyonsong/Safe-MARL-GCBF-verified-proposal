#!/usr/bin/env bash
# Official GCBF (non-plus) baseline training — same env/hparams as GCBF+ script.
# Usage: run_baseline_gcbf.sh [steps] [train_envs] [test_envs] [batch_size] [inner_epoch]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
"$PY" baselines/gcbf/code/train.py \
  --algo gcbf \
  --env DoubleIntegrator \
  --area-size 4.0 \
  --num-agents 8 \
  --obs 4 \
  --n-rays 32 \
  --seed 0 \
  --steps "${1:-1000}" \
  --n-env-train "${2:-2}" \
  --n-env-test "${3:-4}" \
  --buffer-size 256 \
  --batch-size "${4:-32}" \
  --inner-epoch "${5:-4}" \
  --lr-actor 3e-5 \
  --lr-cbf 3e-5 \
  --loss-action-coef 1e-4 \
  --loss-unsafe-coef 1.0 \
  --loss-safe-coef 1.0 \
  --loss-h-dot-coef 0.01 \
  --eval-interval 1 \
  --save-interval 10
