#!/usr/bin/env bash
# Residual-gate V2 (AA+AO ablation): freeze Stage1 base, dual_ce_weight=0.
# Prefer Phase-2 isolation: bash scripts/run_phase2_seed.sh V2 <seed> <run_id>
#
# Usage: run_stage3_l2_residual_gate_v2.sh [steps] [ce_weight] [ao_ce_weight]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

STEPS="${1:-800}"
CE_W="${2:-2.5}"
AO_W="${3:-1.75}"
BASE="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl"
CE="method/deepset/output/stage3/ce/l2_counterexamples.json"
OUT="method/deepset/output/stage3/repair/deepset_residual_gate_v2.pkl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# WARNING: default OUT overwrites Phase-1 v2 slot. Prefer run_phase2_seed.sh.
echo "[residual-v2] WARNING: writing $OUT (Phase-1 slot). Prefer method/deepset/output/phase2/." >&2

"$PY" method/deepset/stage3/retrain_residual_gate.py \
  --base-checkpoint "$BASE" \
  --ce-json "$CE" \
  --gate-mode hybrid \
  --open-mult 3.0 \
  --close-mult 1.0 \
  --gate-hidden 64 \
  --correction-hidden 128 \
  --steps "$STEPS" \
  --ce-batch 320 \
  --normal-batch 768 \
  --ce-weight "$CE_W" \
  --ao-ce-weight "$AO_W" \
  --dual-ce-weight 0.0 \
  --idle-weight 1.5 \
  --delta-l2 1e-3 \
  --gate-sparse 5e-3 \
  --gate-ce-open 0.0 \
  --lr 8e-5 \
  --eval-buckets l2_aa_near,l2_ao_near,l2_11_near \
  --eval-every 100 \
  --eval-episodes 8 \
  --eval-seed 1234 \
  --min-success 0.80 \
  --min-safe 0.92 \
  --max-aa-hit 0.20 \
  --max-ao-hit 0.35 \
  --save-eval-checkpoints \
  --resample 2000 \
  --method-tag V2 \
  --checkpoint "$OUT"

echo "[residual-v2] done. CSV curves: ${OUT%.pkl}.csv"
echo "  $PY method/deepset/stage3/validate_residual.py --checkpoint ${OUT%.pkl}_best.pkl --episodes 32"
