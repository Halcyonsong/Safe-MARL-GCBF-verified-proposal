#!/usr/bin/env bash
# Stage3 R1-style full-net repair (AA+AO). Prefer Phase-2 isolation via:
#   bash scripts/run_phase2_seed.sh R1 <seed> <run_id>
#
# Default writes a *non-canon* path so historical
#   method/deepset/output/stage3/repair/deepset_l2_safe_repair*.pkl
# is not overwritten.
#
# Usage: run_stage3_l2_repair.sh [tries] [keep] [steps]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

TRIES="${1:-100000}"
KEEP="${2:-2048}"
STEPS="${3:-600}"
BASE="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl"
CE_OUT="method/deepset/output/stage3/ce/l2_counterexamples_r1_rerun.json"
CKPT="method/deepset/output/stage3/repair/deepset_l2_safe_repair_rerun.pkl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

"$PY" method/deepset/stage3/collect_counterexamples.py \
  --checkpoint "$BASE" \
  --buckets l2_aa_near,l2_ao_near \
  --label-mode max_clearance \
  --tries "$TRIES" \
  --keep "$KEEP" \
  --seed 0 \
  --out "$CE_OUT"

"$PY" method/deepset/stage3/retrain_l2_safe.py \
  --init-from "$BASE" \
  --ce-json "$CE_OUT" \
  --normal-source frozen \
  --steps "$STEPS" \
  --ce-weight 4.0 \
  --lr 3e-5 \
  --eval-every 100 \
  --eval-episodes 8 \
  --eval-seed 1234 \
  --resample 2000 \
  --eval-buckets l2_aa_near,l2_ao_near,l2_11_near \
  --save-eval-checkpoints \
  --method-tag R1 \
  --checkpoint "$CKPT"

echo "[r1-repair] done → $CKPT (+ _best, .json, .csv)"
echo "  (canon Phase-1 r1 remains method/deepset/output/stage3/repair/deepset_l2_safe_repair*.pkl)"
