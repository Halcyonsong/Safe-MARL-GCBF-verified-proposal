#!/usr/bin/env bash
# Validate InforMARL under project eval protocol.
# Usage: validate_baseline_informarl.sh [episodes] [checkpoint] [out_json]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
EPI="${1:-32}"
CKPT="${2:-}"
OUT="${3:-baselines/informarl/output/informarl/validate_${EPI}epi.json}"
mkdir -p "$(dirname "$OUT")"
ARGS=(--epi "$EPI" --seed 1234 --out-json "$OUT")
if [[ -n "$CKPT" ]]; then
  ARGS+=(--checkpoint "$CKPT")
fi
"$PY" baselines/informarl/code/validate.py "${ARGS[@]}"
