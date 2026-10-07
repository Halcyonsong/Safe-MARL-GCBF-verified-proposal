#!/usr/bin/env bash
# Mean-field MLP distillation from frozen GCBF+ teacher (parallel track).
# Does NOT touch DeepSet paths, phase2 ACTIVE_*, or stage1/frozen.
#
# Usage:
#   bash scripts/run_mf_distill.sh [RUN_ID] [SEEDS] [STEPS]
#
# Outputs:
#   method/meanfield/output/distill/<RUN_ID>/seed*/meanfield_mlp${HIDDEN}_student{,_best}.pkl
# Env overrides: HIDDEN (default 512), TEACHER_RUN, STEPS via $3, etc.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

RUN_ID="${1:-mf_distill_$(date +%Y%m%d)}"
SEEDS_CSV="${2:-0}"
STEPS="${3:-10000}"
TEACHER="${TEACHER_RUN:-baselines/gcbfplus/log/gcbf+/seed0_20261001232611}"
TEACHER_STEP="${TEACHER_STEP:-1000}"
HIDDEN="${HIDDEN:-512}"
EVAL_EVERY="${EVAL_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-500}"
BATCH="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
LOG_TRAIN_EVERY="${LOG_TRAIN_EVERY:-50}"
CKPT_STEM="${CKPT_STEM:-meanfield_mlp${HIDDEN}_student}"

if [ ! -d "$TEACHER/models" ]; then
  echo "missing teacher models: $TEACHER/models" >&2
  exit 1
fi

ROOT_OUT="method/meanfield/output/distill/${RUN_ID}"
mkdir -p "$ROOT_OUT"
echo "[mf-distill] run_id=$RUN_ID feature_v5=4quad_wmean+invttc teacher=$TEACHER step=$TEACHER_STEP steps=$STEPS seeds=$SEEDS_CSV hidden=$HIDDEN" \
  | tee "$ROOT_OUT/launch.log"

export PYTHONUNBUFFERED=1
if [ "${FORCE_CPU:-0}" = "1" ]; then
  export JAX_PLATFORMS=cpu
  export CUDA_VISIBLE_DEVICES=""
  echo "[mf-distill] FORCE_CPU=1 → CPU" | tee -a "$ROOT_OUT/launch.log"
else
  unset JAX_PLATFORMS 2>/dev/null || true
  echo "[mf-distill] GPU preferred" | tee -a "$ROOT_OUT/launch.log"
fi
"$PY" -c "import jax; print('[mf-distill] jax devices:', jax.devices())" 2>&1 | tee -a "$ROOT_OUT/launch.log" || true

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for SEED in "${SEEDS[@]}"; do
  SEED="$(echo "$SEED" | tr -d ' ')"
  OUT_DIR="$ROOT_OUT/seed${SEED}"
  mkdir -p "$OUT_DIR"
  CKPT="$OUT_DIR/${CKPT_STEM}.pkl"
  LOG="$OUT_DIR/train.log"
  echo "[mf-distill] === seed=$SEED hidden=$HIDDEN → $OUT_DIR ===" | tee -a "$ROOT_OUT/launch.log"
  "$PY" method/meanfield/stage_mf/train.py \
    --teacher-run "$TEACHER" \
    --teacher-step "$TEACHER_STEP" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --minibatch-size 4096 \
    --buffer-capacity 131072 \
    --refresh-every 5 \
    --lr "$LR" \
    --hidden "$HIDDEN" \
    --seed "$SEED" \
    --eval-seed 1234 \
    --eval-every "$EVAL_EVERY" \
    --eval-episodes 8 \
    --save-every "$SAVE_EVERY" \
    --log-train-every "$LOG_TRAIN_EVERY" \
    --danger-radius 0.3 \
    --danger-coef 4.0 \
    --run-id "$RUN_ID" \
    --checkpoint "$CKPT" \
    2>&1 | tee "$LOG"

  BEST="${CKPT%.pkl}_best.pkl"
  if [ ! -f "$BEST" ]; then BEST="$CKPT"; fi

  echo "[mf-distill] 32-epi validate seed=$SEED" | tee -a "$LOG"
  "$PY" method/meanfield/stage_mf/validate.py \
    --checkpoint "$BEST" \
    --episodes 32 \
    --seed 1234 \
    --obs 4 \
    --area-size 4.0 \
    --out-json "$OUT_DIR/validate_32epi.json" \
    2>&1 | tee "$OUT_DIR/validate_32epi.txt" | tee -a "$LOG"

  cat > "$OUT_DIR/MANIFEST.json" <<EOF
{
  "track": "meanfield",
  "architecture": "meanfield_mlp",
  "feature_version": 5,
  "feature_spec": "4quad_agent(n,min_d,cmax,cwmean,invttc)+obs(n,min_d,cmax,invttc)",
  "run_id": "$RUN_ID",
  "train_seed": $SEED,
  "teacher_run": "$TEACHER",
  "teacher_step": $TEACHER_STEP,
  "steps": $STEPS,
  "hidden": $HIDDEN,
  "checkpoint": "$CKPT",
  "best_checkpoint": "$BEST",
  "summary_json": "${CKPT%.pkl}.json",
  "eval_csv": "${CKPT%.pkl}.csv",
  "validate_32epi": "$OUT_DIR/validate_32epi.txt",
  "validate_32epi_json": "$OUT_DIR/validate_32epi.json",
  "deepset_paths_untouched": true
}
EOF
done

# LATEST pointer (meanfield only)
ln -sfn "$RUN_ID" "method/meanfield/output/distill/LATEST" 2>/dev/null || true
echo "$RUN_ID" > "method/meanfield/output/LATEST_DISTILL.txt"
# Prefer seed0 best if present
if [ -f "$ROOT_OUT/seed0/${CKPT_STEM}_best.pkl" ]; then
  echo "method/meanfield/output/distill/${RUN_ID}/seed0/${CKPT_STEM}_best.pkl" \
    > "method/meanfield/output/ACTIVE_MF_BASE.txt"
elif [ -f "$ROOT_OUT/seed0/${CKPT_STEM}.pkl" ]; then
  echo "method/meanfield/output/distill/${RUN_ID}/seed0/${CKPT_STEM}.pkl" \
    > "method/meanfield/output/ACTIVE_MF_BASE.txt"
fi

# Summarize success if json exists
"$PY" - <<PY 2>/dev/null | tee -a "$ROOT_OUT/launch.log" || true
import json, pathlib
root = pathlib.Path("$ROOT_OUT")
rows = []
for p in sorted(root.glob("seed*/validate_32epi.json")):
    d = json.loads(p.read_text())
    rows.append((p.parent.name, d.get("success_rate"), d.get("safe_rate"), d.get("finish_rate")))
print("[mf-distill] 32-epi summary:")
for name, suc, saf, fin in rows:
    print(f"  {name}: success={100*suc:.2f}% safe={100*saf:.2f}% finish={100*fin:.2f}%")
sel = root / "SELECTION.md"
lines = ["# Mean-field distill selection — $RUN_ID", "",
         "Track: **meanfield** (isolated from DeepSet).", "",
         "| seed | success_32 | safe_32 | finish_32 |",
         "|---|---:|---:|---:|"]
for name, suc, saf, fin in rows:
    lines.append(f"| {name} | {100*suc:.2f}% | {100*saf:.2f}% | {100*fin:.2f}% |")
lines += ["", f"Outputs: \`{root}\`",
          "", "DeepSet ACTIVE_* pointers were **not** modified.", ""]
sel.write_text("\n".join(lines))
print(f"[mf-distill] wrote {sel}")
PY

echo "[mf-distill] done → $ROOT_OUT"
echo "  DeepSet frozen unchanged: method/deepset/output/stage1/frozen/..."
echo "  DeepSet phase2 ACTIVE_* unchanged"
echo "  MF base pointer: method/meanfield/output/ACTIVE_MF_BASE.txt"
