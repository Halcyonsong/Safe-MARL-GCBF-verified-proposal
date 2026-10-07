#!/usr/bin/env bash
# Residual-gate V2d (joint): freeze Stage1 base, CE on AA+AO+(1,1).
# Main Phase-2 acceptance path. Does NOT overwrite Phase-1 v2 canon unless
# you pass the default checkpoint path (prefer method/deepset/output/phase2/ via run_phase2_seed.sh).
#
# Usage: run_stage3_l2_residual_gate_v2d.sh [steps] [ce_weight] [ao_w] [dual_w] [out_pkl]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

STEPS="${1:-800}"
CE_W="${2:-2.5}"
AO_W="${3:-1.75}"
DUAL_W="${4:-1.75}"
OUT="${5:-method/deepset/output/stage3/repair/deepset_residual_gate_v2d.pkl}"
BASE="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl"
CE_OUT="${OUT%.pkl}_ce.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

mkdir -p "$(dirname "$OUT")" "$(dirname "$CE_OUT")"

echo "[v2d] collect joint CE → $CE_OUT"
"$PY" method/deepset/stage3/collect_counterexamples.py \
  --checkpoint "$BASE" \
  --buckets "l2_aa_near,l2_ao_near,l2_11_near" \
  --label-mode min_intervention \
  --tries 100000 \
  --keep 2048 \
  --seed 0 \
  --out "$CE_OUT"

echo "[v2d] residual repair dual_w=$DUAL_W → $OUT"
"$PY" method/deepset/stage3/retrain_residual_gate.py \
  --base-checkpoint "$BASE" \
  --ce-json "$CE_OUT" \
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
  --dual-ce-weight "$DUAL_W" \
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
  --method-tag V2d \
  --checkpoint "$OUT"

echo "[v2d] done. Validate:"
echo "  $PY method/deepset/stage3/validate_residual.py --checkpoint ${OUT%.pkl}_best.pkl --episodes 32"
echo "  curves: ${OUT%.pkl}.csv"
