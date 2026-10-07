#!/usr/bin/env bash
# Validate official GCBF+ baseline teacher.
# Usage: validate_baseline_gcbfplus.sh RUN_DIR [step] [episodes]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
ARGS=(
  --path "${1:?usage: validate_baseline_gcbfplus.sh RUN_DIR [step] [episodes]}"
  --env DoubleIntegrator
  --area-size 4.0
  --obs 4
  --n-rays 32
  --seed 1234
  --epi "${3:-32}"
  --no-video
)
if [[ "${2:-}" != "" ]]; then
  ARGS+=(--step "$2")
fi
"$PY" baselines/gcbfplus/code/validate.py "${ARGS[@]}"
