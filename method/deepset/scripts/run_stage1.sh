#!/usr/bin/env bash
# Method Stage1: Actor-distilled DeepSet (hidden=512).
# For multi-seed re-distill with full logging, prefer:
#   bash scripts/run_stage1_redistill.sh [RUN_ID] [SEEDS] [STEPS]
#
# Usage: run_stage1.sh OFFICIAL_RUN_DIR [steps] [batch_size] [teacher_step] [hidden] [seed]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

TEACHER="${1:?usage: run_stage1.sh OFFICIAL_RUN_DIR [steps] [batch_size] [teacher_step] [hidden] [seed]}"
STEPS="${2:-12000}"
BATCH="${3:-16}"
TSTEP="${4:-1000}"
HIDDEN="${5:-512}"
SEED="${6:-0}"
OUT_DIR="method/deepset/output/stage1/actor"
mkdir -p "$OUT_DIR"
CKPT="$OUT_DIR/deepset_actor512_student.pkl"

export PYTHONUNBUFFERED=1
if [ "${FORCE_CPU:-0}" = "1" ]; then
  export JAX_PLATFORMS=cpu
  export CUDA_VISIBLE_DEVICES=""
else
  unset JAX_PLATFORMS 2>/dev/null || true
fi
"$PY" -c "import jax; print('[stage1] jax devices:', jax.devices())" || true

"$PY" method/deepset/stage1/train.py \
  --teacher-run "$TEACHER" \
  --teacher-step "$TSTEP" \
  --steps "$STEPS" \
  --batch-size "$BATCH" \
  --minibatch-size 4096 \
  --buffer-capacity 131072 \
  --refresh-every 5 \
  --lr 1e-4 \
  --hidden "$HIDDEN" \
  --seed "$SEED" \
  --eval-seed 1234 \
  --eval-every 250 \
  --eval-episodes 8 \
  --save-every 500 \
  --log-train-every 50 \
  --danger-radius 0.3 \
  --danger-coef 4.0 \
  --checkpoint "$CKPT"

echo "[stage1] done → $CKPT (+ _best.pkl, .json, .csv)"
echo "  validate: bash scripts/validate_stage1.sh ${CKPT%.pkl}_best.pkl 32"
