#!/usr/bin/env bash
# One finish-loop round: train residual → 32-epi already in train → near_plus verify.
# Side-branch only: --no-update-active everywhere. Does NOT touch acceptance pack.
#
# Usage:
#   run_mf_finish_loop_round.sh RUN_ID INIT_RESIDUAL [CE_JSON] [EXTRA_TRAIN_ARGS...]
# Example:
#   bash scripts/run_mf_finish_loop_round.sh r4_gate \
#     method/meanfield/output/repair/mf_residual_v5_r3_finish_20261006/seed0/repair_best.pkl \
#     method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/ce_train_merged.json \
#     --ce-weight 1.0 --idle-weight 5.0 --open-mult 4.0 --gate-mix 0.3 --lr 3e-5 --steps 400
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
unset JAX_PLATFORMS 2>/dev/null || true
export PYTHONUNBUFFERED=1

RUN_TAG="${1:?usage: RUN_ID INIT_RESIDUAL [CE] [extra args...]}"
INIT="${2:?init residual pkl}"
shift 2
CE="method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/ce_train_merged.json"
if [[ "${1:-}" == *.json ]]; then
  CE="$1"
  shift
fi
BASE="method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl"
RUN_ID="mf_residual_v5_${RUN_TAG}_$(date +%Y%m%d)"
OUT="method/meanfield/output/repair/${RUN_ID}/seed0"
VOUT="method/meanfield/output/verify/mf_${RUN_TAG}_near_plus_sidebranch"
mkdir -p "$OUT"

for p in "$BASE" "$INIT" "$CE"; do
  [[ -f "$p" ]] || { echo "missing $p" >&2; exit 1; }
done

RES_B="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
VER_B="$(cat method/meanfield/output/ACTIVE_MF_VERIFY.txt 2>/dev/null || true)"

{
  echo "[loop] ROUND=$RUN_TAG SIDE-BRANCH"
  echo "  base=$BASE"
  echo "  init=$INIT"
  echo "  ce=$CE"
  echo "  out=$OUT"
  echo "  verify→$VOUT"
  echo "  ACTIVE_RES(before)=$RES_B"
  echo "  ACTIVE_VER(before)=$VER_B"
  echo "  extra: $*"
} | tee "$OUT/launch.log"

"$PY" method/meanfield/stage_mf/retrain_residual.py \
  --base-checkpoint "$BASE" \
  --init-residual "$INIT" \
  --ce-json "$CE" \
  --prior-summary "method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/ce_post_summary.json" \
  --correction-hidden 128 \
  --ce-batch 256 \
  --normal-batch 512 \
  --ao-ce-weight 1.5 \
  --both-ce-weight 2.0 \
  --delta-l2 2e-3 \
  --close-mult 1.0 \
  --inv-ttc-ref 5.0 \
  --delta-clip 1.0 \
  --eval-every 50 \
  --eval-episodes 8 \
  --eval-seed 1234 \
  --resample-episodes 24 \
  --resample-seed 123 \
  --score-mode success_finish \
  --no-update-active \
  --checkpoint "$OUT/repair.pkl" \
  "$@" \
  2>&1 | tee -a "$OUT/train.log"

# Formal same-domain verify (CPU)
export JAX_PLATFORMS=cpu
mkdir -p "$VOUT"
"$PY" method/meanfield/stage_mf/verify_full.py \
  --checkpoint "$OUT/repair_best.pkl" \
  --mode near_plus \
  --radius 0.10 \
  --no-update-active \
  --out-dir "$VOUT" \
  2>&1 | tee "$VOUT/verify.log"

RES_A="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
VER_A="$(cat method/meanfield/output/ACTIVE_MF_VERIFY.txt 2>/dev/null || true)"
{
  echo "[loop] done $RUN_TAG"
  echo "  task: $(cat "$OUT/validate_32epi.json" 2>/dev/null | tr -d '\n' | head -c 200)"
  echo "  verify: $(grep -E 'aggregate=|PROVEN|VIOLATED|PARTIAL' "$VOUT/verify.log" | tail -3)"
  echo "  ACTIVE_RES(after)=$RES_A"
  echo "  ACTIVE_VER(after)=$VER_A"
  if [[ "$RES_B" != "$RES_A" ]] || [[ "$VER_B" != "$VER_A" ]]; then
    echo "  ERROR: ACTIVE_* changed" >&2
    exit 2
  fi
  echo "  ACTIVE_* unchanged ✓"
} | tee -a "$OUT/launch.log"

# Gate check vs r2
"$PY" - <<PY
import json, pathlib
r2 = dict(success_rate=0.8203125, safe_rate=0.96484375, finish_rate=0.83984375, cost=0.3046875)
m = json.loads(pathlib.Path("$OUT/validate_32epi.json").read_text())
rep = pathlib.Path("$VOUT/report.json")
agg = "?"
if rep.exists():
    agg = json.loads(rep.read_text()).get("aggregate_status", "?")
else:
    # fallback from REPORT.md / verify.log
    t = pathlib.Path("$VOUT/verify.log").read_text()
    if "PROVEN_FULL" in t: agg = "PROVEN_FULL"
    elif "VIOLATED" in t: agg = "VIOLATED"
    elif "PARTIAL" in t: agg = "PARTIAL"
df = 100*(m["finish_rate"]-r2["finish_rate"])
ds = 100*(m["success_rate"]-r2["success_rate"])
safe_ok = m["safe_rate"] >= 0.94
e1 = m["finish_rate"] >= r2["finish_rate"] + 0.03
e2 = m["success_rate"] >= r2["success_rate"] + 0.02
f1 = agg == "PROVEN_FULL"
print(f"[gate] vs r2: Δfin={df:+.2f}pp Δsuc={ds:+.2f}pp safe={100*m['safe_rate']:.2f}% cost={m['cost']:.3f} L2={agg}")
print(f"[gate] E1_finish+3={e1} E2_suc+2={e2} E3_safe≥94={safe_ok} F1_PROVEN={f1}")
print(f"[gate] PROMOTE_CANDIDATE={e1 and e2 and safe_ok and f1}")
pathlib.Path("$OUT/GATE.json").write_text(json.dumps({
    "vs_r2": {"d_finish_pp": df, "d_success_pp": ds},
    "metrics": m,
    "aggregate_status": agg,
    "E1_finish": e1, "E2_success": e2, "E3_safe": safe_ok, "F1_formal": f1,
    "PROMOTE_CANDIDATE": bool(e1 and e2 and safe_ok and f1),
}, indent=2)+"\n")
PY
