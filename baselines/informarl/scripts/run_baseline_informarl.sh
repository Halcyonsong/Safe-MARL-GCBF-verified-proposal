#!/usr/bin/env bash
# Train InforMARL baseline (GNN+PPO, soft cost; same env protocol).
# Usage: run_baseline_informarl.sh [steps] [seed] [cost_weight]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
STEPS="${1:-400}"
SEED="${2:-0}"
CW="${3:-1.0}"
"$PY" baselines/informarl/code/train.py \
  --steps "$STEPS" \
  --seed "$SEED" \
  --n-envs 8 \
  --cost-weight "$CW" \
  --gnn-layers 2 \
  --eval-every 50 \
  --eval-epi 16
