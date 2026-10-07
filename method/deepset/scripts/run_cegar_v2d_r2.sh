#!/usr/bin/env bash
# CEGAR round-2 on residual V2d best (writes next to parent V2d by default).
#
# Day0: dense sampling verify on declared L2 claim
# Day1: collect CE under residual π → warm-start short V2d repair
# Day2: re-verify + 32-epi + D-hit
#
# Usage:
#   bash scripts/run_cegar_v2d_r2.sh [PARENT_V2D_DIR] [OUT_DIR]
#
# Defaults:
#   PARENT = method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d
#   OUT    = <PARENT>/../V2d_r2   (same seed lineage, sibling of V2d)
#
# Env:
#   RESIDUAL_BEST, BASE_CHECKPOINT, SAMPLES, TRIES, KEEP, STEPS, ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true

PARENT="${1:-method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d}"
# Default: sibling V2d_r2 under the same seed dir (single Phase-2 lineage)
OUT="${2:-$(dirname "$PARENT")/V2d_r2}"
RESIDUAL_BEST="${RESIDUAL_BEST:-$PARENT/repair_best.pkl}"
BASE="${BASE_CHECKPOINT:-method/deepset/output/stage1/redistill/stage1_redistill_20261004_183322/seed0/deepset_actor512_student_best.pkl}"
# method/run tags for summary JSON only
RUN_TAG="${PHASE2_RUN_ID:-$(basename "$(dirname "$(dirname "$PARENT")")")}"
METHOD_TAG="${METHOD_TAG:-V2d_r2}"

mkdir -p "$OUT" "$OUT/steps"
LOG="$OUT/cegar.log"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONUNBUFFERED=1

if [ ! -f "$RESIDUAL_BEST" ]; then
  echo "missing residual best: $RESIDUAL_BEST" >&2
  exit 1
fi
if [ ! -f "$BASE" ]; then
  echo "missing base: $BASE" >&2
  exit 1
fi

{
  echo "[cegar-r2] parent=$PARENT"
  echo "[cegar-r2] residual_best=$RESIDUAL_BEST"
  echo "[cegar-r2] base=$BASE"
  echo "[cegar-r2] out=$OUT"
} | tee "$LOG"

# --- Day 0: post-repair sampling verify ---
echo "[cegar-r2] Day0 verify residual on D (samples=${SAMPLES:-4000}) ..." | tee -a "$LOG"
"$PY" method/deepset/stage3/verify_residual_l2_sampling.py \
  --checkpoint "$RESIDUAL_BEST" \
  --buckets "l2_aa_near,l2_ao_near,l2_11_near" \
  --samples "${SAMPLES:-4000}" \
  --seed 0 \
  --out "$OUT/verify_r1_residual_sampling.json" \
  2>&1 | tee -a "$OUT/verify_r1.log" | tee -a "$LOG"

VIOL=$("$PY" - <<PY
import json
from pathlib import Path
d=json.loads(Path("$OUT/verify_r1_residual_sampling.json").read_text())
vc=d.get("summary",{}).get("verdict_counts",{})
print(int(vc.get("violated",0)))
PY
)

if [ "$VIOL" = "0" ] && [ "${FORCE_CEGAR:-0}" != "1" ]; then
  echo "[cegar-r2] Day0: all sampling_clear — skip repair (set FORCE_CEGAR=1 to force)." | tee -a "$LOG"
  cat > "$OUT/MANIFEST.json" <<EOF
{
  "stage": "cegar_r2_verify_only",
  "parent_v2d": "$PARENT",
  "residual_in": "$RESIDUAL_BEST",
  "verify_r1": "$OUT/verify_r1_residual_sampling.json",
  "violated_buckets": 0,
  "note": "sampling_clear on all buckets — not Marabou proven"
}
EOF
  exit 0
fi

echo "[cegar-r2] Day0 violated_buckets=$VIOL → collect residual CE ..." | tee -a "$LOG"
"$PY" method/deepset/stage3/collect_residual_counterexamples.py \
  --checkpoint "$RESIDUAL_BEST" \
  --buckets "l2_aa_near,l2_ao_near,l2_11_near" \
  --tries "${TRIES:-80000}" \
  --keep "${KEEP:-1536}" \
  --seed 1 \
  --out "$OUT/ce_r2.json" \
  2>&1 | tee -a "$OUT/collect_r2.log" | tee -a "$LOG"

# --- Day 1: warm-start short residual repair ---
STEPS="${STEPS:-600}"
EVAL_EVERY="${EVAL_EVERY:-50}"
echo "[cegar-r2] Day1 residual repair steps=$STEPS warm-start from r1 best ..." | tee -a "$LOG"
"$PY" method/deepset/stage3/retrain_residual_gate.py \
  --base-checkpoint "$BASE" \
  --ce-json "$OUT/ce_r2.json" \
  --init-residual-from "$RESIDUAL_BEST" \
  --gate-mode hybrid \
  --open-mult 3.0 \
  --close-mult 1.0 \
  --gate-hidden 64 \
  --correction-hidden 128 \
  --steps "$STEPS" \
  --ce-batch 320 \
  --normal-batch 768 \
  --ce-weight 2.5 \
  --ao-ce-weight 1.75 \
  --dual-ce-weight "${DUAL_W:-2.0}" \
  --idle-weight 1.5 \
  --delta-l2 1e-3 \
  --gate-sparse 5e-3 \
  --gate-ce-open 0.0 \
  --lr "${LR:-6e-5}" \
  --seed 1 \
  --eval-buckets "l2_aa_near,l2_ao_near,l2_11_near" \
  --eval-every "$EVAL_EVERY" \
  --eval-episodes 8 \
  --eval-seed 1234 \
  --min-success "${MIN_SUCCESS:-0.85}" \
  --min-safe "${MIN_SAFE:-0.92}" \
  --max-aa-hit "${MAX_AA:-0.18}" \
  --max-ao-hit "${MAX_AO:-0.32}" \
  --resample 2000 \
  --save-eval-checkpoints \
  --method-tag "$METHOD_TAG" \
  --phase2-run-id "$RUN_TAG" \
  --checkpoint "$OUT/repair_r2.pkl" \
  2>&1 | tee -a "$OUT/train_r2.log" | tee -a "$LOG"

# Nest intermediate step checkpoints if trainer wrote them next to OUT
"$PY" - <<PY
from pathlib import Path
out = Path("$OUT")
steps = out / "steps"
steps.mkdir(exist_ok=True)
n = 0
for f in sorted(out.glob("repair_r2_step*.pkl")):
    f.rename(steps / f.name)
    n += 1
print(f"[cegar-r2] nested {n} step checkpoints under {steps}")
PY

BEST="$OUT/repair_r2_best.pkl"
if [ ! -f "$BEST" ]; then BEST="$OUT/repair_r2.pkl"; fi

# --- Day 2: re-verify + task + d-hit ---
echo "[cegar-r2] Day2 verify r2 ..." | tee -a "$LOG"
"$PY" method/deepset/stage3/verify_residual_l2_sampling.py \
  --checkpoint "$BEST" \
  --buckets "l2_aa_near,l2_ao_near,l2_11_near" \
  --samples "${SAMPLES:-4000}" \
  --seed 0 \
  --out "$OUT/verify_r2_residual_sampling.json" \
  2>&1 | tee -a "$OUT/verify_r2.log" | tee -a "$LOG"

"$PY" method/deepset/stage3/validate_residual.py \
  --checkpoint "$BEST" --episodes 32 --seed 1234 \
  --out-json "$OUT/validate_32epi.json" \
  2>&1 | tee "$OUT/validate_32epi.txt" | tee -a "$LOG"

"$PY" method/deepset/stage2/eval_l2_gcbfplus.py \
  --residual "$BEST" \
  --buckets "l2_aa_near,l2_ao_near,l2_11_near" \
  --samples 2000 --seed 123 \
  --skip-gcbf --skip-deepset --skip-r1 \
  --out "$OUT/d_hit_seed123.json" \
  2>&1 | tee -a "$OUT/d_hit.log" | tee -a "$LOG"

cat > "$OUT/MANIFEST.json" <<EOF
{
  "stage": "cegar_r2",
  "phase2_run_id": "$RUN_TAG",
  "method": "$METHOD_TAG",
  "parent_v2d": "$PARENT",
  "residual_in": "$RESIDUAL_BEST",
  "base_checkpoint": "$BASE",
  "verify_r1": "$OUT/verify_r1_residual_sampling.json",
  "ce_r2": "$OUT/ce_r2.json",
  "repair_best": "$BEST",
  "repair_last": "$OUT/repair_r2.pkl",
  "verify_r2": "$OUT/verify_r2_residual_sampling.json",
  "validate_32epi": "$OUT/validate_32epi.txt",
  "d_hit": "$OUT/d_hit_seed123.json",
  "steps_dir": "$OUT/steps",
  "steps": $STEPS,
  "min_success": ${MIN_SUCCESS:-0.85},
  "dual_ce_weight": ${DUAL_W:-2.0}
}
EOF

echo "[cegar-r2] done → $OUT/MANIFEST.json" | tee -a "$LOG"
"$PY" - <<PY
import json
from pathlib import Path
root=Path("$OUT")
print("=== CEGAR R2 SUMMARY ===")
for name in ("verify_r1_residual_sampling.json","verify_r2_residual_sampling.json"):
    p=root/name
    if not p.exists():
        continue
    d=json.loads(p.read_text())
    print(name, d.get("summary",{}).get("verdict_counts"))
    for b in d.get("buckets") or []:
        print(f"  {b['bucket']}: {b['verdict']} hit={b['hit_rate_on_D']*100:.2f}%")
v=root/"validate_32epi.txt"
if v.exists():
    print(v.read_text().strip().splitlines()[0])
dh=root/"d_hit_seed123.json"
if dh.exists():
    d=json.loads(dh.read_text())
    for n,pol in (d.get("policies") or {}).items():
        hits={k:v.get("hit_rate_on_D") for k,v in (pol.get("buckets") or {}).items() if isinstance(v,dict)}
        print("D-hit", n, hits)
PY
