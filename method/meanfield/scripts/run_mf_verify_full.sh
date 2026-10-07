#!/usr/bin/env bash
# MF verification: 44-D feature-box full coverage + two-step L2 property.
# Not DeepSet (k,m) buckets. Not primary AA/AO claim split.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONUNBUFFERED=1

CKPT="${1:-$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null | tr -d '[:space:]')}"
MODE="${2:-smoke}"   # smoke | near | near_plus | full
OUT="${3:-}"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "missing residual checkpoint: '$CKPT'" >&2
  exit 1
fi

ARGS=(
  --checkpoint "$CKPT"
  --mode "$MODE"
  --max-splits "${MAX_SPLITS:-800}"
  --max-depth "${MAX_DEPTH:-14}"
  --marabou-timeout "${MARABOU_TIMEOUT:-20}"
  --wall-limit "${WALL_LIMIT:-900}"
  --sample-per-unknown "${SAMPLE_UNK:-48}"
  --radius "${RADIUS:-0.10}"
)
# optional overrides (omit → mode presets in domain.py)
if [ -n "${D_MIN:-}" ]; then ARGS+=(--d-min "$D_MIN"); fi
if [ -n "${D_MAX:-}" ]; then ARGS+=(--d-max "$D_MAX"); fi
if [ -n "${CLOSE_HI:-}" ]; then ARGS+=(--close-hi "$CLOSE_HI"); fi
if [ -n "$OUT" ]; then
  ARGS+=(--out-dir "$OUT")
fi

echo "[mf-verify] 44-D feature box + two-step L2 | ckpt=$CKPT mode=$MODE"
"$PY" method/meanfield/stage_mf/verify_full.py "${ARGS[@]}"
echo "[mf-verify] DeepSet ACTIVE_* untouched"
