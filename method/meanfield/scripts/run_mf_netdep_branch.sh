#!/usr/bin/env bash
# Side-branch: base → sample CE → residual repair → task eval → (gate) netdep formal.
# Does NOT touch acceptance/mf_best_20261006/ or ACTIVE_MF_*.
#
# Usage:
#   bash method/meanfield/scripts/run_mf_netdep_branch.sh [TAG] [extra retrain args...]
# Env overrides:
#   COLLECT_EPISODES STEPS LR CE_WEIGHT SAFE_GATE SKIP_FORMAL=1 VERIFY_ONLY=1
#   INIT_RESIDUAL  CE_JSON  (optional warm-start / external CE)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
unset JAX_PLATFORMS 2>/dev/null || true
export PYTHONUNBUFFERED=1

TAG="${1:-netdep_r1}"
if [[ "${1:-}" == -* ]]; then TAG="netdep_r1"; else shift || true; fi

BRANCH="method/meanfield/output/sidebranch_netdep"
BASE="method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl"
RUN_ID="mf_residual_v5_${TAG}_$(date +%Y%m%d)"
OUT="method/meanfield/output/repair/${RUN_ID}/seed0"
VOUT="method/meanfield/output/verify/mf_${TAG}_netdep_sidebranch"
VOUT_L2="method/meanfield/output/verify/mf_${TAG}_near_plus_sidebranch_check"
SAFE_GATE="${SAFE_GATE:-0.94}"
COLLECT_EPISODES="${COLLECT_EPISODES:-48}"
STEPS="${STEPS:-800}"
LR="${LR:-5e-5}"
CE_WEIGHT="${CE_WEIGHT:-4.0}"

mkdir -p "$BRANCH" "$OUT"
# freeze ACTIVE snapshot if missing
for f in ACTIVE_MF_BASE.txt ACTIVE_MF_RESIDUAL.txt ACTIVE_MF_VERIFY.txt; do
  if [[ ! -f "$BRANCH/$f" && -f "method/meanfield/output/$f" ]]; then
    cp -a "method/meanfield/output/$f" "$BRANCH/$f"
  fi
done

RES_B="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
VER_B="$(cat method/meanfield/output/ACTIVE_MF_VERIFY.txt 2>/dev/null || true)"
BASE_B="$(cat method/meanfield/output/ACTIVE_MF_BASE.txt 2>/dev/null || true)"

[[ -f "$BASE" ]] || { echo "missing base $BASE" >&2; exit 1; }

{
  echo "[netdep] TAG=$TAG SIDE-BRANCH from BASE"
  echo "  base=$BASE"
  echo "  out=$OUT"
  echo "  verify_netdep→$VOUT"
  echo "  SAFE_GATE=$SAFE_GATE collect=$COLLECT_EPISODES steps=$STEPS lr=$LR"
  echo "  extra: $*"
  echo "  ACTIVE_RES(before)=$RES_B"
  echo "  ACTIVE_VER(before)=$VER_B"
  echo "  ACTIVE_BASE(before)=$BASE_B"
  echo "  claim_box: netdep min_d∈[0.118,0.122] close∈[0.25,0.30] R=0.10"
  echo "  physics: d2_lo(|a|=1)=0.091<R ; coast_hard=0.100≥R (NOT envelope)"
} | tee "$OUT/launch.log"

if [[ "${VERIFY_ONLY:-0}" != "1" ]]; then
  TRAIN_ARGS=(
    --base-checkpoint "$BASE"
    --collect-episodes "$COLLECT_EPISODES"
    --collect-seed 123
    --danger-radius 0.35
    --max-ce 4000
    --steps "$STEPS"
    --lr "$LR"
    --correction-hidden 128
    --ce-batch 256
    --normal-batch 512
    --ce-weight "$CE_WEIGHT"
    --ao-ce-weight "${AO_CE_WEIGHT:-2.5}"
    --both-ce-weight 2.0
    --idle-weight "${IDLE_WEIGHT:-2.5}"
    --delta-l2 "${DELTA_L2:-2e-3}"
    --open-mult 3.0
    --close-mult 1.0
    --inv-ttc-ref 5.0
    --gate-mix 0.5
    --delta-clip 1.0
    --eval-every "${EVAL_EVERY:-50}"
    --eval-episodes "${EVAL_EPISODES:-16}"
    --eval-seed 1234
    --resample-episodes 32
    --resample-seed 123
    --score-mode "${SCORE_MODE:-safe_success}"
    --no-update-active
    --checkpoint "$OUT/repair.pkl"
  )
  if [[ -n "${INIT_RESIDUAL:-}" ]]; then
    TRAIN_ARGS+=(--init-residual "$INIT_RESIDUAL")
  fi
  if [[ -n "${CE_JSON:-}" ]]; then
    TRAIN_ARGS+=(--ce-json "$CE_JSON")
  fi
  # user extras last
  TRAIN_ARGS+=("$@")

  echo "[netdep] phase1: sample CE (if no CE_JSON) + repair residual" | tee -a "$OUT/launch.log"
  "$PY" method/meanfield/stage_mf/retrain_residual.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$OUT/train.log"
else
  echo "[netdep] VERIFY_ONLY=1 skip train" | tee -a "$OUT/launch.log"
  if [[ -z "${CKPT:-}" ]]; then
    echo "VERIFY_ONLY needs CKPT=..." >&2
    exit 1
  fi
  # copy pointer for gate
  mkdir -p "$OUT"
  echo "$CKPT" > "$OUT/repair_best_path.txt"
fi

CKPT_BEST="$OUT/repair_best.pkl"
if [[ ! -f "$CKPT_BEST" && -n "${CKPT:-}" ]]; then
  CKPT_BEST="$CKPT"
fi
[[ -f "$CKPT_BEST" ]] || { echo "missing best ckpt $CKPT_BEST" >&2; exit 1; }

# ensure 32-epi metrics exist
if [[ ! -f "$OUT/validate_32epi.json" ]]; then
  echo "[netdep] phase2: task eval @32" | tee -a "$OUT/launch.log"
  "$PY" method/meanfield/stage_mf/validate_residual.py \
    --checkpoint "$CKPT_BEST" \
    --episodes 32 --seed 1234 \
    --out-json "$OUT/validate_32epi.json" \
    2>&1 | tee -a "$OUT/launch.log"
fi

# Gate on task safe
SAFE="$("$PY" - <<PY
import json
m=json.load(open("$OUT/validate_32epi.json"))
print(float(m.get("safe_rate", 0.0)))
PY
)"
SUCC="$("$PY" - <<PY
import json
m=json.load(open("$OUT/validate_32epi.json"))
print(float(m.get("success_rate", 0.0)))
PY
)"
HIT="null"
if [[ -f "$OUT/ce_post_summary.json" ]]; then
  HIT="$("$PY" - <<PY
import json
print(json.load(open("$OUT/ce_post_summary.json")).get("hit_rate_on_danger"))
PY
)"
fi

echo "[netdep] metrics: safe=$SAFE success=$SUCC hit_on_danger=$HIT gate=$SAFE_GATE" | tee -a "$OUT/launch.log"

FORMAL_RAN=0
FORMAL_STATUS="skipped"
if [[ "${SKIP_FORMAL:-0}" == "1" ]]; then
  echo "[netdep] SKIP_FORMAL=1" | tee -a "$OUT/launch.log"
elif "$PY" -c "import sys; sys.exit(0 if float('$SAFE') >= float('$SAFE_GATE') else 1)"; then
  echo "[netdep] phase3: formal netdep (non-envelope) + near_plus sanity" | tee -a "$OUT/launch.log"
  FORMAL_RAN=1
  export JAX_PLATFORMS=cpu
  export PYTHONPATH="$ROOT:$ROOT/method/meanfield:$ROOT/method/deepset:${PYTHONPATH:-}"
  mkdir -p "$VOUT" "$VOUT_L2"
  "$PY" -m stage_mf.verify_full \
    --checkpoint "$CKPT_BEST" \
    --mode netdep \
    --radius 0.10 \
    --max-splits "${MAX_SPLITS:-1200}" \
    --max-depth "${MAX_DEPTH:-16}" \
    --wall-limit "${WALL_LIMIT:-900}" \
    --out-dir "$VOUT" \
    --no-update-active \
    2>&1 | tee "$VOUT/verify.log"
  # sanity: must not break old near_plus if we care; still side-branch only
  "$PY" -m stage_mf.verify_full \
    --checkpoint "$CKPT_BEST" \
    --mode near_plus \
    --radius 0.10 \
    --out-dir "$VOUT_L2" \
    --no-update-active \
    2>&1 | tee "$VOUT_L2/verify.log" || true
else
  echo "[netdep] gate FAIL: safe=$SAFE < $SAFE_GATE → no formal yet (sample/repair more)" | tee -a "$OUT/launch.log"
fi

RES_A="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
VER_A="$(cat method/meanfield/output/ACTIVE_MF_VERIFY.txt 2>/dev/null || true)"
BASE_A="$(cat method/meanfield/output/ACTIVE_MF_BASE.txt 2>/dev/null || true)"

"$PY" - <<PY
import json, pathlib, re
out = pathlib.Path("$OUT")
vout = pathlib.Path("$VOUT")
vout_l2 = pathlib.Path("$VOUT_L2")
branch = pathlib.Path("$BRANCH")
m = json.loads((out/"validate_32epi.json").read_text()) if (out/"validate_32epi.json").exists() else {}
post = json.loads((out/"ce_post_summary.json").read_text()) if (out/"ce_post_summary.json").exists() else {}
base_ce = json.loads((out/"ce_baseline_summary.json").read_text()) if (out/"ce_baseline_summary.json").exists() else {}

def agg(p: pathlib.Path):
    rj = p/"REPORT.json"
    if rj.exists():
        return json.loads(rj.read_text()).get("aggregate_status", "?")
    t = (p/"verify.log").read_text() if (p/"verify.log").exists() else ""
    for key in ("PROVEN_FULL", "VIOLATED", "PARTIAL"):
        if key in t:
            return key
    return "skipped" if int("$FORMAL_RAN")==0 else "?"

def nce(p: pathlib.Path):
    f = p/"counterexamples_all.json"
    if not f.exists():
        return 0
    j = json.loads(f.read_text())
    return int(j.get("n", j.get("n_ce", 0)) or 0)

safe = float(m.get("safe_rate", 0.0))
gate_ok = safe >= float("$SAFE_GATE")
st_net = agg(vout) if int("$FORMAL_RAN") else "skipped_gate"
st_l2 = agg(vout_l2) if int("$FORMAL_RAN") else "skipped_gate"
gate = {
  "tag": "$TAG",
  "branch": "sidebranch_netdep",
  "base": "$BASE",
  "ckpt": str(pathlib.Path("$CKPT_BEST")),
  "out": str(out),
  "pipeline": "sample_ce → repair → task_eval → [safe_gate] → formal_netdep",
  "claim_box": {
    "mode": "netdep",
    "min_d": [0.118, 0.122],
    "close": [0.25, 0.30],
    "R": 0.10,
    "d2_lo_abs_a1": 0.091,
    "coast_hard": 0.100,
    "envelope": False,
    "note": "NOT near_plus; bad |a|=1 can fail; coast preventable",
  },
  "metrics_32": m,
  "ce_baseline_hit": base_ce.get("hit_rate_on_danger"),
  "ce_post_hit": post.get("hit_rate_on_danger"),
  "safe_gate": float("$SAFE_GATE"),
  "safe_ok": gate_ok,
  "formal_ran": bool(int("$FORMAL_RAN")),
  "netdep_status": st_net,
  "netdep_n_ce": nce(vout) if int("$FORMAL_RAN") else None,
  "near_plus_check_status": st_l2,
  "ACTIVE_unchanged": ("$RES_B" == "$RES_A" and "$VER_B" == "$VER_A" and "$BASE_B" == "$BASE_A"),
  "promote": False,
  "note": "Side-branch only. acceptance pack + ACTIVE_* must stay frozen.",
}
(out/"GATE_NETDEP.json").write_text(json.dumps(gate, indent=2) + "\n")
(branch/"GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
# pointer file for humans
(branch/"LATEST_RUN.txt").write_text(str(out) + "\n")
print("[gate]", json.dumps(gate, indent=2))
if not gate["ACTIVE_unchanged"]:
    raise SystemExit("ERROR: ACTIVE_* changed — restore from sidebranch_netdep snapshot")
print("[netdep] ACTIVE_* unchanged ✓")
PY

{
  echo "[netdep] done TAG=$TAG"
  echo "  GATE → $OUT/GATE_NETDEP.json  and  $BRANCH/GATE.json"
  echo "  acceptance/mf_best_20261006 untouched; ACTIVE_* frozen"
} | tee -a "$OUT/launch.log"
