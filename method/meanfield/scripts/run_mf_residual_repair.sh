#!/usr/bin/env bash
# MF track: collect two-step CEs on frozen base → physics-gated residual repair.
# Does NOT touch DeepSet ACTIVE_* or phase2 paths.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
unset JAX_PLATFORMS 2>/dev/null || true
export PYTHONUNBUFFERED=1

BASE="${1:-$(cat method/meanfield/output/ACTIVE_MF_BASE.txt 2>/dev/null | tr -d '[:space:]')}"
RUN_ID="${2:-mf_residual_v5_$(date +%Y%m%d)}"
OUT="method/meanfield/output/repair/${RUN_ID}/seed0"
mkdir -p "$OUT"

if [ -z "$BASE" ] || [ ! -f "$BASE" ]; then
  echo "missing base checkpoint: '$BASE'" >&2
  exit 1
fi

echo "[mf-repair] base=$BASE out=$OUT" | tee "$OUT/launch.log"
"$PY" -c "import jax; print(jax.devices())" 2>&1 | tee -a "$OUT/launch.log"

"$PY" method/meanfield/stage_mf/retrain_residual.py \
  --base-checkpoint "$BASE" \
  --collect-episodes "${COLLECT_EPISODES:-48}" \
  --collect-seed 123 \
  --danger-radius 0.35 \
  --max-ce 4000 \
  --steps "${STEPS:-600}" \
  --lr 1e-4 \
  --correction-hidden 128 \
  --ce-batch 256 \
  --normal-batch 512 \
  --ce-weight 3.0 \
  --eval-every 100 \
  --eval-episodes 8 \
  --resample-episodes 32 \
  --checkpoint "$OUT/repair.pkl" \
  2>&1 | tee -a "$OUT/train.log"

echo "[mf-repair] done → $OUT"
echo "  ACTIVE_MF_RESIDUAL=$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
echo "  DeepSet ACTIVE_* untouched"
