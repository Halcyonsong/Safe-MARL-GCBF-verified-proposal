#!/usr/bin/env bash
# Stage1 re-distill from frozen GCBF+ teacher → stronger/more stable DeepSet base.
# Does NOT overwrite Phase-1 canon frozen baseline.
#
# Usage:
#   bash scripts/run_stage1_redistill.sh [RUN_ID] [SEEDS] [STEPS]
#
# Examples:
#   bash scripts/run_stage1_redistill.sh
#   bash scripts/run_stage1_redistill.sh stage1_redistill_20261005 0,1,2 12000
#
# Outputs under:
#   method/deepset/output/stage1/redistill/<RUN_ID>/seed*/ 
# After all seeds: compare + optional promote (manual):
#   python scripts/select_stage1_base.py --run-id <RUN_ID>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

RUN_ID="${1:-stage1_redistill_$(date +%Y%m%d)}"
SEEDS_CSV="${2:-0,1,2}"
STEPS="${3:-12000}"
TEACHER="${TEACHER_RUN:-baselines/gcbfplus/log/gcbf+/seed0_20261001232611}"
TEACHER_STEP="${TEACHER_STEP:-1000}"
HIDDEN="${HIDDEN:-512}"
EVAL_EVERY="${EVAL_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-500}"
BATCH="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
# denser train loss for curves (every 50 steps)
LOG_TRAIN_EVERY="${LOG_TRAIN_EVERY:-50}"

if [ ! -d "$TEACHER/models" ]; then
  echo "missing teacher models: $TEACHER/models" >&2
  exit 1
fi

ROOT_OUT="method/deepset/output/stage1/redistill/${RUN_ID}"
mkdir -p "$ROOT_OUT"
echo "[redistill] run_id=$RUN_ID teacher=$TEACHER step=$TEACHER_STEP steps=$STEPS seeds=$SEEDS_CSV" \
  | tee "$ROOT_OUT/launch.log"

export PYTHONUNBUFFERED=1
# GPU by default. Force CPU: FORCE_CPU=1
if [ "${FORCE_CPU:-0}" = "1" ]; then
  export JAX_PLATFORMS=cpu
  export CUDA_VISIBLE_DEVICES=""
  echo "[redistill] FORCE_CPU=1 → CPU" | tee -a "$ROOT_OUT/launch.log"
else
  unset JAX_PLATFORMS 2>/dev/null || true
  echo "[redistill] GPU preferred (JAX auto-select)" | tee -a "$ROOT_OUT/launch.log"
fi
"$PY" -c "import jax; print('[redistill] jax devices:', jax.devices())" 2>&1 | tee -a "$ROOT_OUT/launch.log" || true

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for SEED in "${SEEDS[@]}"; do
  SEED="$(echo "$SEED" | tr -d ' ')"
  OUT_DIR="$ROOT_OUT/seed${SEED}"
  mkdir -p "$OUT_DIR"
  CKPT="$OUT_DIR/deepset_actor512_student.pkl"
  LOG="$OUT_DIR/train.log"
  echo "[redistill] === seed=$SEED → $OUT_DIR ===" | tee -a "$ROOT_OUT/launch.log"
  "$PY" method/deepset/stage1/train.py \
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

  echo "[redistill] 32-epi validate seed=$SEED" | tee -a "$LOG"
  "$PY" method/deepset/stage1/validate.py \
    --checkpoint "$BEST" \
    --episodes 32 \
    --seed 1234 \
    --obs 4 \
    --area-size 4.0 \
    2>&1 | tee "$OUT_DIR/validate_32epi.txt" | tee -a "$LOG"

  # Optional same-std L2-D hit for base quality (vs frozen later)
  if [ "${SKIP_D_HIT:-0}" != "1" ]; then
    echo "[redistill] D-hit seed=123" | tee -a "$LOG"
    "$PY" method/deepset/stage2/eval_l2_gcbfplus.py \
      --deepset "$BEST" \
      --buckets "l2_aa_near,l2_ao_near,l2_11_near" \
      --samples "${D_HIT_SAMPLES:-2000}" \
      --seed 123 \
      --skip-gcbf --skip-residual --skip-r1 \
      --out "$OUT_DIR/d_hit_seed123.json" \
      2>&1 | tee -a "$LOG" || echo "[redistill] D-hit failed (non-fatal)" | tee -a "$LOG"
  fi

  cat > "$OUT_DIR/MANIFEST.json" <<EOF
{
  "run_id": "$RUN_ID",
  "train_seed": $SEED,
  "teacher_run": "$TEACHER",
  "teacher_step": $TEACHER_STEP,
  "steps": $STEPS,
  "checkpoint": "$CKPT",
  "best_checkpoint": "$BEST",
  "summary_json": "${CKPT%.pkl}.json",
  "eval_csv": "${CKPT%.pkl}.csv",
  "validate_32epi": "$OUT_DIR/validate_32epi.txt",
  "d_hit_report": "$OUT_DIR/d_hit_seed123.json"
}
EOF
done

echo "[redistill] selecting best base among seeds..." | tee -a "$ROOT_OUT/launch.log"
"$PY" scripts/select_stage1_base.py --run-dir "$ROOT_OUT" 2>&1 | tee -a "$ROOT_OUT/launch.log"

echo "[redistill] plot curves..."
"$PY" scripts/plot_train_curves.py \
  --glob "$ROOT_OUT/seed*/deepset_actor512_student.csv" \
  --out-dir "$ROOT_OUT/figures" \
  2>&1 | tee -a "$ROOT_OUT/launch.log" || true

echo "[redistill] done → $ROOT_OUT"
echo "  Phase-1 frozen unchanged: method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl"
echo "  To promote new base (manual copy after review):"
echo "    see $ROOT_OUT/SELECTION.md"
