#!/usr/bin/env bash
# Side-branch: finish-oriented residual fine-tune from frozen r2.
# - Loads base + r2 delta from method/meanfield/output/ (NOT acceptance/mf_best_20261006/1_primary/)
# - Writes ONLY under method/meanfield/output/repair/<RUN_ID>/
# - Does NOT touch ACTIVE_MF_* / acceptance/mf_best_20261006/1_primary/ / DeepSet ACTIVE_*
#
# Recipe (lightweight weight tweak vs r2):
#   ce_weight 3.0 → 1.5   (less collision-avoidance pressure)
#   idle_weight 2.5 → 4.0 (stronger far-zone anchor to high-finish base)
#   lr 5e-5 → 3e-5, steps 400 (gentle)
#   gate hyperparams identical to r2 (open=3, close=1, mix=0.5, inv_ttc=5, clip=1)
#   best-ckpt score = 2*suc + finish
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
unset JAX_PLATFORMS 2>/dev/null || true
export PYTHONUNBUFFERED=1

BASE="${1:-method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl}"
INIT="${2:-method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl}"
CE="${3:-method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/ce_train_merged.json}"
RUN_ID="${4:-mf_residual_v5_r3_finish_$(date +%Y%m%d)}"
OUT="method/meanfield/output/repair/${RUN_ID}/seed0"
mkdir -p "$OUT"

for p in "$BASE" "$INIT" "$CE"; do
  if [ ! -f "$p" ]; then
    echo "missing: $p" >&2
    exit 1
  fi
done

# Snapshot ACTIVE pointers before run (must stay identical after)
ACTIVE_RES_BEFORE="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
ACTIVE_BASE_BEFORE="$(cat method/meanfield/output/ACTIVE_MF_BASE.txt 2>/dev/null || true)"

{
  echo "[mf-r3-finish] SIDE BRANCH — not acceptance"
  echo "  base=$BASE"
  echo "  init_r2=$INIT"
  echo "  ce=$CE"
  echo "  out=$OUT"
  echo "  ACTIVE_MF_RESIDUAL(before)=$ACTIVE_RES_BEFORE"
  echo "  ACTIVE_MF_BASE(before)=$ACTIVE_BASE_BEFORE"
  echo "  recipe: ce_w=1.5 idle_w=4.0 lr=3e-5 steps=400 score=success_finish --no-update-active"
} | tee "$OUT/launch.log"

"$PY" -c "import jax; print('devices', jax.devices())" 2>&1 | tee -a "$OUT/launch.log"

"$PY" method/meanfield/stage_mf/retrain_residual.py \
  --base-checkpoint "$BASE" \
  --init-residual "$INIT" \
  --ce-json "$CE" \
  --prior-summary "method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/ce_post_summary.json" \
  --steps "${STEPS:-400}" \
  --lr 3e-5 \
  --correction-hidden 128 \
  --ce-batch 256 \
  --normal-batch 512 \
  --ce-weight 1.5 \
  --ao-ce-weight 1.5 \
  --both-ce-weight 2.0 \
  --idle-weight 4.0 \
  --delta-l2 2e-3 \
  --open-mult 3.0 \
  --close-mult 1.0 \
  --inv-ttc-ref 5.0 \
  --gate-mix 0.5 \
  --delta-clip 1.0 \
  --eval-every 50 \
  --eval-episodes 8 \
  --eval-seed 1234 \
  --resample-episodes 24 \
  --resample-seed 123 \
  --score-mode success_finish \
  --no-update-active \
  --checkpoint "$OUT/repair.pkl" \
  2>&1 | tee -a "$OUT/train.log"

ACTIVE_RES_AFTER="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
ACTIVE_BASE_AFTER="$(cat method/meanfield/output/ACTIVE_MF_BASE.txt 2>/dev/null || true)"
{
  echo "[mf-r3-finish] done → $OUT"
  echo "  ACTIVE_MF_RESIDUAL(after)=$ACTIVE_RES_AFTER"
  echo "  ACTIVE_MF_BASE(after)=$ACTIVE_BASE_AFTER"
  if [ "$ACTIVE_RES_BEFORE" != "$ACTIVE_RES_AFTER" ] || [ "$ACTIVE_BASE_BEFORE" != "$ACTIVE_BASE_AFTER" ]; then
    echo "  ERROR: ACTIVE_* changed — restore from before!" >&2
    exit 2
  fi
  echo "  ACTIVE_* unchanged ✓"
  echo "  acceptance/mf_best_20261006/1_primary/ untouched (by design)"
} | tee -a "$OUT/launch.log"
