#!/usr/bin/env bash
# Phase-2 on redistill seed0 DeepSet base (provisional stronger Stage1).
# Does NOT touch Phase-1 frozen or pilot repair slots.
#
# Usage:
#   # recommended smoke: formal L2 + CE collect only (fast-ish)
#   bash scripts/run_phase2_from_seed0_base.sh smoke
#
#   # full V2d repair train seed 0
#   bash scripts/run_phase2_from_seed0_base.sh V2d 0
#
#   # multi-seed later:
#   bash scripts/run_phase2_from_seed0_base.sh V2d 1
#   bash scripts/run_phase2_from_seed0_base.sh V2d 2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

SEED0_BASE="${SEED0_BASE:-method/deepset/output/stage1/redistill/stage1_redistill_20261004_183322/seed0/deepset_actor512_student_best.pkl}"
RUN_ID="${RUN_ID:-phase2_seed0base_$(date +%Y%m%d)}"
MODE="${1:-smoke}"

if [ ! -f "$SEED0_BASE" ]; then
  echo "missing seed0 base: $SEED0_BASE" >&2
  exit 1
fi

export BASE_CHECKPOINT="$SEED0_BASE"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONUNBUFFERED=1

OUT_ROOT="method/deepset/output/phase2/${RUN_ID}"
mkdir -p "$OUT_ROOT"
echo "[seed0base] base=$SEED0_BASE run_id=$RUN_ID mode=$MODE" | tee "$OUT_ROOT/launch.log"

case "$MODE" in
smoke)
  # 1) Stage2 L2 formal/sampling report on seed0 base (writes under phase2)
  echo "[seed0base] Stage2 L2 verify (timeout smoke + samples) ..." | tee -a "$OUT_ROOT/launch.log"
  mkdir -p "$OUT_ROOT/stage2"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
  "$PY" method/deepset/stage2/verify_l2.py \
    --checkpoint "$SEED0_BASE" \
    --timeout "${TIMEOUT:-20}" \
    --samples "${SAMPLES:-2000}" \
    --buckets "l2_aa_near,l2_ao_near,l2_11_near" \
    --out "$OUT_ROOT/stage2/l2_collision_report.json" \
    --onnx-dir "$OUT_ROOT/stage2/onnx" \
    2>&1 | tee -a "$OUT_ROOT/stage2/verify_l2.log" | tee -a "$OUT_ROOT/launch.log"

  # 2) CE collect only (joint V2d buckets) — builds residual repair inputs
  echo "[seed0base] collect joint CE (AA+AO+(1,1)) ..." | tee -a "$OUT_ROOT/launch.log"
  mkdir -p "$OUT_ROOT/ce"
  "$PY" method/deepset/stage3/collect_counterexamples.py \
    --checkpoint "$SEED0_BASE" \
    --buckets "l2_aa_near,l2_ao_near,l2_11_near" \
    --label-mode min_intervention \
    --tries "${TRIES:-100000}" \
    --keep "${KEEP:-2048}" \
    --seed 0 \
    --out "$OUT_ROOT/ce/l2_counterexamples_v2d_seed0.json" \
    2>&1 | tee -a "$OUT_ROOT/ce/collect.log" | tee -a "$OUT_ROOT/launch.log"

  cat > "$OUT_ROOT/SMOKE_MANIFEST.json" <<EOF
{
  "run_id": "$RUN_ID",
  "base_checkpoint": "$SEED0_BASE",
  "stage2_report": "$OUT_ROOT/stage2/l2_collision_report.json",
  "ce_json": "$OUT_ROOT/ce/l2_counterexamples_v2d_seed0.json",
  "next": "BASE_CHECKPOINT=$SEED0_BASE bash scripts/run_phase2_seed.sh V2d 0 $RUN_ID"
}
EOF
  echo "[seed0base] smoke done → $OUT_ROOT/SMOKE_MANIFEST.json"
  echo "  Next full V2d: BASE_CHECKPOINT=$SEED0_BASE bash scripts/run_phase2_seed.sh V2d 0 $RUN_ID"
  ;;
V2d|V2|R1)
  SEED="${2:-0}"
  export STEPS="${STEPS:-1400}"
  export EVAL_EVERY="${EVAL_EVERY:-50}"
  bash "$ROOT/scripts/run_phase2_seed.sh" "$MODE" "$SEED" "$RUN_ID"
  ;;
*)
  echo "usage: $0 smoke | V2d [seed] | V2 [seed] | R1 [seed]" >&2
  exit 1
  ;;
esac
