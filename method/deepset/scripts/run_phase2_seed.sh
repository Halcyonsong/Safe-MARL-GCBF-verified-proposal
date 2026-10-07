#!/usr/bin/env bash
# Phase-2 single-seed repair (writes only under method/deepset/output/phase2/).
#
# Usage:
#   bash scripts/run_phase2_seed.sh METHOD SEED RUN_ID [extra args...]
#
# METHOD:
#   R1   full-net strong safety (AA+AO, max_clearance, ce_weight=4)
#   V2   residual AA+AO only   (dual_ce_weight=0)
#   V2d  residual AA+AO+(1,1)  (joint main; dual_ce_weight=1.75)
#
# Example:
#   bash scripts/run_phase2_seed.sh V2d 0 phase2_20261005
#   BASE_CHECKPOINT=method/deepset/output/stage1/redistill/.../seed0/..._best.pkl \
#     bash scripts/run_phase2_seed.sh V2d 0 phase2_seed0base
#
# Env overrides:
#   BASE_CHECKPOINT  DeepSet base (default: Phase-1 frozen)
#   STEPS            repair steps (default per method below)
#   JAX_PLATFORMS    default cpu (stable); set empty/cuda for GPU
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

METHOD="${1:?METHOD=R1|V2|V2d}"
SEED="${2:?train seed}"
RUN_ID="${3:?phase2 run id}"
shift 3 || true

BASE="${BASE_CHECKPOINT:-method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl}"
OUT_DIR="method/deepset/output/phase2/${RUN_ID}/seed${SEED}/${METHOD}"
mkdir -p "$OUT_DIR"
CE_OUT="$OUT_DIR/ce.json"
REPAIR_OUT="$OUT_DIR/repair.pkl"
LOG="$OUT_DIR/run.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

if [ ! -f "$BASE" ]; then
  echo "missing base checkpoint: $BASE" >&2
  exit 1
fi

echo "[phase2] method=$METHOD seed=$SEED run_id=$RUN_ID out=$OUT_DIR" | tee "$LOG"
echo "[phase2] base=$BASE" | tee -a "$LOG"

case "$METHOD" in
R1)
  BUCKETS="l2_aa_near,l2_ao_near"
  LABEL="max_clearance"
  STEPS="${STEPS:-1000}"
  CE_W=4.0
  ;;
V2)
  BUCKETS="l2_aa_near,l2_ao_near"
  LABEL="min_intervention"
  STEPS="${STEPS:-1400}"
  CE_W=2.5
  AO_W=1.75
  DUAL_W=0.0
  ;;
V2d)
  BUCKETS="l2_aa_near,l2_ao_near,l2_11_near"
  LABEL="min_intervention"
  # Route A: longer budget + dense eval (eval_every set below)
  STEPS="${STEPS:-1400}"
  CE_W=2.5
  AO_W=1.75
  DUAL_W=1.75
  ;;
*)
  echo "unknown METHOD=$METHOD (want R1|V2|V2d)" >&2
  exit 1
  ;;
esac

# --- CE collect (or reuse precomputed, e.g. smoke CE) ---
if [ -n "${CE_JSON:-}" ] && [ -f "$CE_JSON" ]; then
  echo "[phase2] reuse CE_JSON=$CE_JSON → $CE_OUT" | tee -a "$LOG"
  cp -f "$CE_JSON" "$CE_OUT"
elif [ "${SKIP_CE_COLLECT:-0}" = "1" ] && [ -f "$CE_OUT" ]; then
  echo "[phase2] SKIP_CE_COLLECT=1 using existing $CE_OUT" | tee -a "$LOG"
else
  echo "[phase2] CE collect buckets=$BUCKETS label=$LABEL seed=$SEED" | tee -a "$LOG"
  "$PY" method/deepset/stage3/collect_counterexamples.py \
    --checkpoint "$BASE" \
    --buckets "$BUCKETS" \
    --label-mode "$LABEL" \
    --tries "${TRIES:-100000}" \
    --keep "${KEEP:-2048}" \
    --seed "$SEED" \
    --out "$CE_OUT" 2>&1 | tee -a "$LOG"
fi

# --- repair ---
case "$METHOD" in
R1)
  echo "[phase2] R1 full-net repair steps=$STEPS ce_w=$CE_W" | tee -a "$LOG"
  "$PY" method/deepset/stage3/retrain_l2_safe.py \
    --init-from "$BASE" \
    --ce-json "$CE_OUT" \
    --normal-source frozen \
    --steps "$STEPS" \
    --ce-weight "$CE_W" \
    --lr 3e-5 \
    --seed "$SEED" \
    --eval-every 100 \
    --eval-episodes 8 \
    --eval-seed 1234 \
    --resample 2000 \
    --eval-buckets "l2_aa_near,l2_ao_near,l2_11_near" \
    --min-success 0.0 \
    --min-safe 0.0 \
    --max-aa-hit 1.0 \
    --max-ao-hit 1.0 \
    --save-eval-checkpoints \
    --method-tag R1 \
    --phase2-run-id "$RUN_ID" \
    --checkpoint "$REPAIR_OUT" \
    "$@" 2>&1 | tee -a "$LOG"
  BEST="${REPAIR_OUT%.pkl}_best.pkl"
  if [ ! -f "$BEST" ]; then BEST="$REPAIR_OUT"; fi
  "$PY" method/deepset/stage1/validate.py --checkpoint "$BEST" --episodes 32 --seed 1234 --obs 4 --area-size 4.0 \
    2>&1 | tee -a "$OUT_DIR/validate_32epi.txt" | tee -a "$LOG"
  ;;
V2|V2d)
  echo "[phase2] residual $METHOD steps=$STEPS ce=$CE_W ao=$AO_W dual=$DUAL_W" | tee -a "$LOG"
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
    --seed "$SEED" \
    --eval-buckets "l2_aa_near,l2_ao_near,l2_11_near" \
    --eval-every "${EVAL_EVERY:-50}" \
    --eval-episodes 8 \
    --eval-seed 1234 \
    --min-success 0.80 \
    --min-safe 0.92 \
    --max-aa-hit 0.20 \
    --max-ao-hit 0.35 \
    --resample 2000 \
    --save-eval-checkpoints \
    --method-tag "$METHOD" \
    --phase2-run-id "$RUN_ID" \
    --checkpoint "$REPAIR_OUT" \
    "$@" 2>&1 | tee -a "$LOG"
  BEST="${REPAIR_OUT%.pkl}_best.pkl"
  if [ ! -f "$BEST" ]; then BEST="$REPAIR_OUT"; fi
  "$PY" method/deepset/stage3/validate_residual.py --checkpoint "$BEST" --episodes 32 --seed 1234 \
    2>&1 | tee -a "$OUT_DIR/validate_32epi.txt" | tee -a "$LOG"
  ;;
esac

# --- same-std D-hit (seed 123) ---
echo "[phase2] D-hit resample seed=123" | tee -a "$LOG"
DHIT_ARGS=(
  --buckets "l2_aa_near,l2_ao_near,l2_11_near"
  --samples 2000
  --seed 123
  --skip-gcbf
  --skip-deepset
  --out "$OUT_DIR/d_hit_seed123.json"
)
if [ "$METHOD" = "R1" ]; then
  DHIT_ARGS+=(--skip-residual --r1 "$BEST")
else
  DHIT_ARGS+=(--skip-r1 --residual "$BEST")
fi
"$PY" method/deepset/stage2/eval_l2_gcbfplus.py "${DHIT_ARGS[@]}" 2>&1 | tee -a "$LOG"

# --- nest intermediate step checkpoints (keep *_best / final at top) ---
"$PY" - <<PY
from pathlib import Path
out = Path("$OUT_DIR")
steps = out / "steps"
steps.mkdir(exist_ok=True)
n = 0
for pat in ("repair_step*.pkl", "repair_r2_step*.pkl", "*_step*.pkl"):
    for f in sorted(out.glob(pat)):
        # never move best/final bare names
        if f.name.endswith("_best.pkl") or f.name in {"repair.pkl", "repair_r2.pkl"}:
            continue
        if "step" not in f.name:
            continue
        dest = steps / f.name
        if not dest.exists():
            f.rename(dest)
            n += 1
print(f"[phase2] nested {n} intermediate checkpoints under {steps}")
PY

# --- manifest ---
cat > "$OUT_DIR/MANIFEST.json" <<EOF
{
  "phase2_run_id": "$RUN_ID",
  "method": "$METHOD",
  "train_seed": $SEED,
  "base_checkpoint": "$BASE",
  "ce_json": "$CE_OUT",
  "repair_checkpoint": "$REPAIR_OUT",
  "best_checkpoint": "$BEST",
  "eval_csv": "${REPAIR_OUT%.pkl}.csv",
  "summary_json": "${REPAIR_OUT%.pkl}.json",
  "d_hit_report": "$OUT_DIR/d_hit_seed123.json",
  "validate_32epi": "$OUT_DIR/validate_32epi.txt",
  "steps_dir": "$OUT_DIR/steps",
  "buckets": "$BUCKETS",
  "label_mode": "$LABEL"
}
EOF

echo "[phase2] done → $OUT_DIR" | tee -a "$LOG"
ls -la "$OUT_DIR" | tee -a "$LOG"
