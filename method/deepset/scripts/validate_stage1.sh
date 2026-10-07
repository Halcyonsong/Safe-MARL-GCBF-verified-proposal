#!/usr/bin/env bash
# Empirical rollout validation for Stage1 DeepSet student.
# Usage: validate_stage1.sh STUDENT_CHECKPOINT [episodes]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"
"$PY" method/deepset/stage1/validate.py \
  --checkpoint "${1:?usage: validate_stage1.sh STUDENT_CHECKPOINT [episodes]}" \
  --episodes "${2:-32}" \
  --obs 4 \
  --area-size 4.0
