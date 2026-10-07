#!/usr/bin/env bash
# Train MAPPO-Lagrangian baseline (same env protocol as other baselines).
# Usage: run_baseline_mappo_lag.sh [steps] [seed] [cost_limit]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
STEPS="${1:-400}"
SEED="${2:-0}"
CLIMIT="${3:-0.5}"
"$PY" baselines/mappo_lag/code/train.py \
  --steps "$STEPS" \
  --seed "$SEED" \
  --n-envs 8 \
  --cost-limit "$CLIMIT" \
  --eval-every 50 \
  --eval-epi 16
