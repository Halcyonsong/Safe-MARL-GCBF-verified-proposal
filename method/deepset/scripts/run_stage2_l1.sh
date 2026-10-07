#!/usr/bin/env bash
# Stage2 first-version formal checks: L1 action bounds on DeepSet buckets.
# Usage: run_stage2_l1.sh [timeout_s] [samples] [bucket_csv]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
# Marabou is CPU-bound; avoid JAX grabbing the GPU.
export CUDA_VISIBLE_DEVICES=""
ARGS=(
  --checkpoint method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl
  --timeout "${1:-30}"
  --samples "${2:-2000}"
  --out method/deepset/output/stage2/l1_action_bounds_report.json
  --onnx-dir method/deepset/output/stage2/onnx
)
if [[ "${3:-}" != "" ]]; then
  ARGS+=(--buckets "$3")
fi
"$PY" method/deepset/stage2/verify_l1.py "${ARGS[@]}"
