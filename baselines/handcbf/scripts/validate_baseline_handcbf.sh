#!/usr/bin/env bash
# Validate HandCBF = official DecShareCBF (no training).
# Usage: validate_baseline_handcbf.sh [episodes] [out_json]
# Protocol: DoubleIntegrator · 8 · area4 · obs4 · rays32 · seed=1234 · alpha=1.0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
EPI="${1:-32}"
OUT="${2:-baselines/handcbf/output/handcbf/validate_${EPI}epi.json}"
mkdir -p "$(dirname "$OUT")"
"$PY" baselines/handcbf/code/validate.py \
  --epi "$EPI" \
  --seed 1234 \
  --alpha 1.0 \
  --out-json "$OUT"
