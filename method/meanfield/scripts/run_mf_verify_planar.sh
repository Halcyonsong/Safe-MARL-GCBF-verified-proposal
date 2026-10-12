#!/usr/bin/env bash
# Side-branch PLANAR SafeSet verify. Never writes ACTIVE_*.
# Pipeline: geometry SafeSet(a) → prove Deploy∈SafeSet (IBP then Marabou).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/method/meanfield:$ROOT/method/deepset:${PYTHONPATH:-}"

CKPT="${1:-method/meanfield/output/repair/mf_residual_v5_netdep_r3_20261010/seed0/repair_manual_pick.pkl}"
OUT="${2:-method/meanfield/output/verify/planar_safeset_v1/netdep_r3}"
if [[ ! -f "$CKPT" ]]; then
  CKPT="method/meanfield/output/repair/mf_residual_v5_netdep_r3_20261010/seed0/repair_best.pkl"
fi
mkdir -p "$OUT"
RES_B="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
VER_B="$(cat method/meanfield/output/ACTIVE_MF_VERIFY.txt 2>/dev/null || true)"

echo "[planar-safeset] ckpt=$CKPT out=$OUT"
echo "[planar-safeset] pipeline: SafeSet(a) from rel → Deploy(x)∈SafeSet (IBP/Marabou)"
echo "[planar-safeset] X_rel netdep dx∈[0.118,0.122] close∈[0.25,0.30] lat≤0.04 R=0.10"
echo "[planar-safeset] X_feat reachable_v1 goal_xy≤comm goal_dv=-v"
echo "[planar-safeset] Marabou gate: pose_w≤${MARABOU_POSE_MAX:-0.50} vg_w≤${MARABOU_VG_MAX:-0.20}"
SCOPE_ARGS=()
if [[ "${FREEZE_POSE:-0}" == "1" || "${FREEZE_POSE:-}" == "true" ]]; then
  SCOPE_ARGS+=(--freeze-pose)
  echo "[planar-safeset] FREEZE_POSE=1 (px/py mid slice; still netdep on rel/v/goal)"
fi
if [[ -n "${VG_ABS:-}" ]]; then
  SCOPE_ARGS+=(--vg-abs "${VG_ABS}")
  echo "[planar-safeset] VG_ABS=${VG_ABS} (vx,vy,goal_xy ∈ [-vg,vg])"
fi
if [[ "${THREAT_MID:-0}" == "1" || "${THREAT_MID:-}" == "true" ]]; then
  SCOPE_ARGS+=(--threat-mid)
  echo "[planar-safeset] THREAT_MID=1 (threat feats pinned to midpoint)"
fi

"$PY" -m stage_mf.verify_planar \
  --checkpoint "$CKPT" \
  --out-dir "$OUT" \
  --radius 0.10 \
  --d-min 0.118 --d-max 0.122 \
  --close-lo 0.25 --close-hi 0.30 \
  --lat-max 0.04 \
  --max-splits "${MAX_SPLITS:-4000}" \
  --max-depth "${MAX_DEPTH:-28}" \
  --wall-limit "${WALL_LIMIT:-1800}" \
  --sample-per-unknown "${SAMPLE_PER_UNKNOWN:-16}" \
  --marabou-timeout "${MARABOU_TIMEOUT:-35}" \
  --marabou-max-width "${MARABOU_MAX_WIDTH:-0.20}" \
  --marabou-pose-max "${MARABOU_POSE_MAX:-0.50}" \
  --marabou-vg-max "${MARABOU_VG_MAX:-0.20}" \
  --marabou-threat-max "${MARABOU_THREAT_MAX:-0.15}" \
  "${SCOPE_ARGS[@]}" \
  --no-update-active \
  2>&1 | tee "$OUT/verify.log"

RES_A="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
VER_A="$(cat method/meanfield/output/ACTIVE_MF_VERIFY.txt 2>/dev/null || true)"
if [[ "$RES_B" != "$RES_A" || "$VER_B" != "$VER_A" ]]; then
  echo "ERROR: ACTIVE changed" >&2
  exit 2
fi
echo "[planar-safeset] ACTIVE_* unchanged ✓"
echo "[planar-safeset] REPORT → $OUT/REPORT.md"
