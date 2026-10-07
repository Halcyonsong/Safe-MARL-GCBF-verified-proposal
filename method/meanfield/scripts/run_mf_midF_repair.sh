#!/usr/bin/env bash
# Side-branch: repair toward mid-F PROVEN_FULL.
# Does NOT touch acceptance/mf_best_20261006/1_primary/ or ACTIVE_MF_*.
#
# mid-F box: d_min=0.125, d_max=0.40, close_hi=0.30, R=0.10
#
# Usage:
#   bash scripts/run_mf_midF_repair.sh TAG [INIT_PKL] [FORMAL_CE_JSON] [extra train args...]
# Examples:
#   bash scripts/run_mf_midF_repair.sh midF_r1
#   bash scripts/run_mf_midF_repair.sh midF_r2 \
#     method/meanfield/output/repair/mf_residual_v5_midF_r1_20261006/seed0/repair_best.pkl \
#     method/meanfield/output/verify/mf_midF_r1_midF_sidebranch/counterexamples_all.json \
#     --ce-weight 6.0 --steps 700 --lr 3e-5
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
unset JAX_PLATFORMS 2>/dev/null || true
export PYTHONUNBUFFERED=1

TAG="${1:?TAG required}"
shift
BASE="method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl"
INIT="method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl"
FORMAL_CE="method/meanfield/output/verify/mf_r2_midF_probe_20261006/counterexamples_all.json"
PRIOR_CE="method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/ce_train_merged.json"

if [[ "${1:-}" == *.pkl ]]; then
  INIT="$1"; shift
fi
if [[ "${1:-}" == *.json ]]; then
  FORMAL_CE="$1"; shift
fi

# defaults; extras from "$@" override via being appended after fixed args... 
# We pass fixed hyperparams then "$@" so user can override by putting later flags
# (argparse last-wins only if duplicated — so put stronger defaults here for r2+)
CE_WEIGHT="${CE_WEIGHT:-5.0}"
STEPS="${STEPS:-600}"
LR="${LR:-4e-5}"
IDLE_WEIGHT="${IDLE_WEIGHT:-1.5}"

RUN_ID="mf_residual_v5_${TAG}_$(date +%Y%m%d)"
OUT="method/meanfield/output/repair/${RUN_ID}/seed0"
CE_OUT="${OUT}/ce_midF_merged.json"
VOUT="method/meanfield/output/verify/mf_${TAG}_midF_sidebranch"
VOUT_L2="method/meanfield/output/verify/mf_${TAG}_near_plus_sidebranch"
mkdir -p "$OUT"

for p in "$BASE" "$INIT" "$FORMAL_CE" "$PRIOR_CE"; do
  [[ -f "$p" ]] || { echo "missing $p" >&2; exit 1; }
done

RES_B="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
VER_B="$(cat method/meanfield/output/ACTIVE_MF_VERIFY.txt 2>/dev/null || true)"

{
  echo "[midF-repair] TAG=$TAG SIDE-BRANCH"
  echo "  base=$BASE"
  echo "  init=$INIT"
  echo "  formal_ce=$FORMAL_CE"
  echo "  prior_ce=$PRIOR_CE"
  echo "  ce_weight=$CE_WEIGHT steps=$STEPS lr=$LR"
  echo "  extra: $*"
  echo "  out=$OUT"
  echo "  verify→$VOUT (mid-F)"
  echo "  ACTIVE_RES(before)=$RES_B"
  echo "  ACTIVE_VER(before)=$VER_B"
} | tee "$OUT/launch.log"

"$PY" scripts/make_midF_ce_labels.py \
  --formal-ce "$FORMAL_CE" \
  --merge "$PRIOR_CE" \
  --radius 0.10 \
  --out "$CE_OUT" \
  2>&1 | tee -a "$OUT/launch.log"

"$PY" method/meanfield/stage_mf/retrain_residual.py \
  --base-checkpoint "$BASE" \
  --init-residual "$INIT" \
  --ce-json "$CE_OUT" \
  --prior-summary "method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/ce_post_summary.json" \
  --correction-hidden 128 \
  --ce-batch 256 \
  --normal-batch 512 \
  --ce-weight "$CE_WEIGHT" \
  --ao-ce-weight 1.5 \
  --both-ce-weight 2.0 \
  --idle-weight "$IDLE_WEIGHT" \
  --delta-l2 2e-3 \
  --open-mult 3.0 \
  --close-mult 1.0 \
  --inv-ttc-ref 5.0 \
  --gate-mix 0.5 \
  --delta-clip 1.0 \
  --lr "$LR" \
  --steps "$STEPS" \
  --eval-every 50 \
  --eval-episodes 8 \
  --eval-seed 1234 \
  --resample-episodes 24 \
  --resample-seed 123 \
  --score-mode success_finish \
  --no-update-active \
  --checkpoint "$OUT/repair.pkl" \
  "$@" \
  2>&1 | tee "$OUT/train.log"

export JAX_PLATFORMS=cpu
mkdir -p "$VOUT"
"$PY" -m stage_mf.verify_full \
  --checkpoint "$OUT/repair_best.pkl" \
  --mode near_plus \
  --d-min 0.125 --d-max 0.40 --close-hi 0.30 \
  --radius 0.10 \
  --max-splits 1200 --max-depth 16 \
  --wall-limit 600 \
  --out-dir "$VOUT" \
  --no-update-active \
  2>&1 | tee "$VOUT/verify.log"

mkdir -p "$VOUT_L2"
"$PY" -m stage_mf.verify_full \
  --checkpoint "$OUT/repair_best.pkl" \
  --mode near_plus \
  --radius 0.10 \
  --out-dir "$VOUT_L2" \
  --no-update-active \
  2>&1 | tee "$VOUT_L2/verify.log"

RES_A="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
VER_A="$(cat method/meanfield/output/ACTIVE_MF_VERIFY.txt 2>/dev/null || true)"
{
  echo "[midF-repair] done $TAG"
  echo "  task: $(python3 -c "import json;print(json.load(open('$OUT/validate_32epi.json')))" 2>/dev/null | head -c 240)"
  echo "  midF: $(grep -E 'aggregate=|PROVEN|VIOLATED|PARTIAL' "$VOUT/verify.log" | tail -2)"
  echo "  L2:   $(grep -E 'aggregate=|PROVEN|VIOLATED|PARTIAL' "$VOUT_L2/verify.log" | tail -2)"
  echo "  ACTIVE_RES(after)=$RES_A"
  echo "  ACTIVE_VER(after)=$VER_A"
  if [[ "$RES_B" != "$RES_A" ]] || [[ "$VER_B" != "$VER_A" ]]; then
    echo "  ERROR: ACTIVE_* changed" >&2
    exit 2
  fi
  echo "  ACTIVE_* unchanged ✓"
} | tee -a "$OUT/launch.log"

"$PY" - <<PY
import json, pathlib
out = pathlib.Path("$OUT")
vout = pathlib.Path("$VOUT")
vout_l2 = pathlib.Path("$VOUT_L2")
m = json.loads((out/"validate_32epi.json").read_text()) if (out/"validate_32epi.json").exists() else {}
def agg(p):
    rj = p/"REPORT.json"
    if rj.exists():
        return json.loads(rj.read_text()).get("aggregate_status","?")
    t = (p/"verify.log").read_text() if (p/"verify.log").exists() else ""
    if "PROVEN_FULL" in t: return "PROVEN_FULL"
    if "VIOLATED" in t: return "VIOLATED"
    if "PARTIAL" in t: return "PARTIAL"
    return "?"
a_mid, a_l2 = agg(vout), agg(vout_l2)
# CE counts
def nce(p):
    f = p/"counterexamples_all.json"
    if not f.exists(): return 0
    return int(json.loads(f.read_text()).get("n", 0))
gate = {
  "tag": "$TAG",
  "init": "$INIT",
  "formal_ce": "$FORMAL_CE",
  "midF_status": a_mid,
  "midF_n_ce": nce(vout),
  "near_plus_status": a_l2,
  "metrics_32": m,
  "midF_PROVEN": a_mid == "PROVEN_FULL",
  "near_plus_ok": a_l2 == "PROVEN_FULL",
  "safe_ok": float(m.get("safe_rate",0)) >= 0.94 if m else False,
  "ACCEPT_MIDF_CANDIDATE": (a_mid == "PROVEN_FULL" and a_l2 == "PROVEN_FULL"),
  "note": "Side-branch only. acceptance/mf_best_20261006/1_primary / ACTIVE_* untouched.",
}
(out/"GATE_MIDF.json").write_text(json.dumps(gate, indent=2)+"\n")
print("[gate]", json.dumps(gate, indent=2))
PY
