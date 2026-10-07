#!/usr/bin/env bash
# Zero-shot transfer: train @ N=8 → eval @ N=16/32 (area4 · obs4 · rays32 · seed=1234 · 32-epi).
# Usage: eval_transfer_nagents.sh [N_LIST]
#   e.g. eval_transfer_nagents.sh "16 32"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="/home/xiang/miniconda3/envs/safe_marl_verify/bin/python"
cd "$ROOT"
source "$ROOT/scripts/cuda_env.sh"

N_LIST="${1:-16 32}"
EPI=32
SEED=1234
OUT_ROOT="baselines/compare/output/transfer_n"
mkdir -p "$OUT_ROOT"

MF_CKPT="acceptance/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl"
DS_CKPT="$(cat method/deepset/output/phase2/ACTIVE_RESIDUAL.txt)"
GCBFPLUS_RUN="baselines/gcbfplus/log/gcbf+/seed0_20261001232611"
MAPPO_CKPT="$(cat baselines/mappo_lag/output/mappo_lag/ACTIVE.txt)"
INFOR_CKPT="$(cat baselines/informarl/output/informarl/ACTIVE.txt)"

echo "=== Transfer N-agents · epi=$EPI seed=$SEED ==="
echo "MF=$MF_CKPT"
echo "DS=$DS_CKPT"
echo "GCBF+=$GCBFPLUS_RUN"
echo "MAPPO=$MAPPO_CKPT"
echo "Infor=$INFOR_CKPT"

for N in $N_LIST; do
  DIR="$OUT_ROOT/N${N}"
  mkdir -p "$DIR"
  echo ""
  echo "######## N=$N ########"

  echo "[MF-r2]"
  "$PY" method/meanfield/stage_mf/validate_residual.py \
    --checkpoint "$MF_CKPT" --episodes "$EPI" --seed "$SEED" \
    --num-agents "$N" --obs 4 --area-size 4.0 \
    --out-json "$DIR/mf_r2.json"

  echo "[DS-r2]"
  "$PY" method/deepset/stage3/validate_residual.py \
    --checkpoint "$DS_CKPT" --episodes "$EPI" --seed "$SEED" \
    --num-agents "$N" --obs 4 --area-size 4.0 \
    --out-json "$DIR/ds_r2.json"

  echo "[GCBF+]"
  # test_official prints summary; capture + parse into JSON via handcbf-style wrapper below
  "$PY" baselines/gcbfplus/code/test_official.py \
    --path "$GCBFPLUS_RUN" --step 1000 \
    --env DoubleIntegrator -n "$N" --area-size 4.0 --obs 4 --n-rays 32 \
    --seed "$SEED" --epi "$EPI" --no-video --log \
    > "$DIR/gcbfplus_stdout.txt" 2>"$DIR/gcbfplus_stderr.txt" || true
  "$PY" - <<PY
import json, re, pathlib
text = pathlib.Path("$DIR/gcbfplus_stdout.txt").read_text() + "\n" + pathlib.Path("$DIR/gcbfplus_stderr.txt").read_text()
# Final summary from test_official.py
pat = re.compile(
    r"reward:\s*([-\d.]+),\s*min/max reward:.*?cost:\s*([-\d.]+),\s*min/max cost:.*?"
    r"safe_rate:\s*([\d.]+)%,\s*finish_rate:\s*([\d.]+)%,\s*success_rate:\s*([\d.]+)%",
    re.S,
)
ms = list(pat.finditer(text))
if not ms:
    pat2 = re.compile(
        r"safe_rate:\s*([\d.]+)%,\s*finish_rate:\s*([\d.]+)%,\s*success_rate:\s*([\d.]+)%"
    )
    ms2 = list(pat2.finditer(text))
    if not ms2:
        print("WARN: could not parse GCBF+ summary; tail:")
        print(text[-800:])
        raise SystemExit(0)
    m = ms2[-1]
    d = {
        "policy_id": "gcbfplus",
        "num_agents": int("$N"),
        "episodes": int("$EPI"),
        "eval_seed": int("$SEED"),
        "safe_rate": float(m.group(1))/100.0,
        "finish_rate": float(m.group(2))/100.0,
        "success_rate": float(m.group(3))/100.0,
        "reward": None,
        "cost": None,
    }
else:
    m = ms[-1]
    d = {
        "policy_id": "gcbfplus",
        "num_agents": int("$N"),
        "episodes": int("$EPI"),
        "eval_seed": int("$SEED"),
        "reward": float(m.group(1)),
        "cost": float(m.group(2)),
        "safe_rate": float(m.group(3))/100.0,
        "finish_rate": float(m.group(4))/100.0,
        "success_rate": float(m.group(5))/100.0,
    }
pathlib.Path("$DIR/gcbfplus.json").write_text(json.dumps(d, indent=2)+"\n")
c = d.get("cost")
cstr = f"{c:.3f}" if isinstance(c, (int, float)) else "—"
print(f"GCBF+ N=$N suc={d['success_rate']*100:.1f}% safe={d['safe_rate']*100:.1f}% fin={d['finish_rate']*100:.1f}% cost={cstr}")
PY

  echo "[HandCBF]"
  "$PY" baselines/handcbf/code/validate.py \
    --epi "$EPI" --seed "$SEED" --num-agents "$N" --obs 4 --area-size 4.0 \
    --out-json "$DIR/handcbf.json"

  echo "[MAPPO-Lag]"
  "$PY" baselines/mappo_lag/code/validate.py \
    --checkpoint "$MAPPO_CKPT" --epi "$EPI" --seed "$SEED" \
    --num-agents "$N" --obs 4 --area-size 4.0 \
    --out-json "$DIR/mappo_lag.json"

  echo "[InforMARL]"
  "$PY" baselines/informarl/code/validate.py \
    --checkpoint "$INFOR_CKPT" --epi "$EPI" --seed "$SEED" \
    --num-agents "$N" --obs 4 --area-size 4.0 \
    --out-json "$DIR/informarl.json"
done

echo ""
echo "=== SUMMARY ==="
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("baselines/compare/output/transfer_n")
methods = ["mf_r2","ds_r2","gcbfplus","handcbf","mappo_lag","informarl"]
print(f"{'N':>4} {'method':<12} {'suc%':>7} {'safe%':>7} {'fin%':>7} {'cost':>8}")
for ndir in sorted(root.glob("N*")):
    N = ndir.name[1:]
    for m in methods:
        p = ndir / f"{m}.json"
        if not p.exists():
            print(f"{N:>4} {m:<12} {'—':>7}")
            continue
        d = json.loads(p.read_text())
        suc = d.get("success_rate")
        saf = d.get("safe_rate")
        fin = d.get("finish_rate")
        cost = d.get("cost")
        def pct(x):
            return f"{100*x:.1f}" if x is not None else "—"
        cstr = f"{cost:.3f}" if isinstance(cost,(int,float)) else "—"
        print(f"{N:>4} {m:<12} {pct(suc):>7} {pct(saf):>7} {pct(fin):>7} {cstr:>8}")
PY
