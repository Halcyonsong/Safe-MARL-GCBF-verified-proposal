#!/usr/bin/env bash
# Compare GCBF+ (GNN teacher) vs Phase-1 frozen vs redistill seed0.
# Accumulates JSONL under method/deepset/output/stage1/compare/<id>/ for later multi-seed plots.
#
# Usage:
#   bash scripts/run_stage1_compare_eval.sh [COMPARE_ID] [EPISODES_CSV]
# Examples:
#   bash scripts/run_stage1_compare_eval.sh gcbf_vs_seed0 32
#   bash scripts/run_stage1_compare_eval.sh gcbf_vs_seed0 32,64,128
#   WITH_D_HIT=1 bash scripts/run_stage1_compare_eval.sh gcbf_vs_seed0 32
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

COMPARE_ID="${1:-gcbf_vs_seed0_$(date +%Y%m%d)}"
EPIS="${2:-32,64}"
COMPARE_DIR="method/deepset/output/stage1/compare/${COMPARE_ID}"
SEED0="${SEED0_CKPT:-method/deepset/output/stage1/redistill/stage1_redistill_20261004_183322/seed0/deepset_actor512_student_best.pkl}"
FROZEN="${FROZEN_CKPT:-method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl}"
TEACHER="${TEACHER_RUN:-baselines/gcbfplus/log/gcbf+/seed0_20261001232611}"

export PYTHONUNBUFFERED=1
# eval can use GPU; FORCE_CPU=1 to force
if [ "${FORCE_CPU:-0}" = "1" ]; then
  export JAX_PLATFORMS=cpu
  export CUDA_VISIBLE_DEVICES=""
else
  unset JAX_PLATFORMS 2>/dev/null || true
fi

mkdir -p "$COMPARE_DIR"
ARGS=(
  --compare-dir "$COMPARE_DIR"
  --teacher-run "$TEACHER"
  --teacher-step 1000
  --seed0 "$SEED0"
  --frozen "$FROZEN"
  --episodes "$EPIS"
  --eval-seed 1234
)
if [ "${WITH_D_HIT:-0}" = "1" ]; then
  ARGS+=(--with-d-hit --d-hit-samples "${D_HIT_SAMPLES:-2000}" --d-hit-seed 123)
fi

echo "[compare-eval] → $COMPARE_DIR episodes=$EPIS"
"$PY" scripts/eval_stage1_compare.py "${ARGS[@]}" 2>&1 | tee -a "$COMPARE_DIR/run.log"
"$PY" scripts/plot_stage1_compare.py --compare-dir "$COMPARE_DIR" 2>&1 | tee -a "$COMPARE_DIR/run.log"
echo "[compare-eval] done"
echo "  $COMPARE_DIR/SNAPSHOT.md"
echo "  $COMPARE_DIR/figures/"
echo "  $COMPARE_DIR/rows.jsonl   # append more later with same --compare-dir"
