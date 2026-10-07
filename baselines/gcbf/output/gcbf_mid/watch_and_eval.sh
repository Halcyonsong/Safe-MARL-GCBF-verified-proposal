#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
LOG=baselines/gcbf/output/gcbf_mid/train_outer.log
PID_FILE=baselines/gcbf/output/gcbf_mid/train.pid
RUN_FILE=baselines/gcbf/output/gcbf_mid/ACTIVE_RUN.txt
PID=$(cat "$PID_FILE")
RUN=$(cat "$RUN_FILE")
echo "[watch $(date -Iseconds)] waiting pid=$PID run=$RUN"
while kill -0 "$PID" 2>/dev/null || pgrep -f 'baselines/gcbf/code/train.py --algo gcbf' >/dev/null; do
  sleep 120
  PROG=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -E '[0-9]+%\|' | tail -1 || true)
  echo "[watch $(date +%H:%M:%S)] $PROG"
done
echo "[watch $(date -Iseconds)] train exited"
RUN=$(ls -dt baselines/gcbf/log/gcbf/seed0_* 2>/dev/null | head -1)
# Prefer run that has n_env_train=8 in config if multiple
for cand in $(ls -dt baselines/gcbf/log/gcbf/seed0_* 2>/dev/null); do
  if grep -q 'n_env_train: 8' "$cand/config.yaml" 2>/dev/null || grep -q 'n_env_train: 8' "$cand/config.yaml" 2>/dev/null; then
    RUN=$cand; break
  fi
done
# fallback: parse Namespace from config
for cand in $(ls -dt baselines/gcbf/log/gcbf/seed0_* 2>/dev/null); do
  if grep -q "n_env_train: 8\|n_env_train: 8" "$cand/config.yaml" 2>/dev/null || \
     grep -q "n_env_train: 8" "$cand/config.yaml" 2>/dev/null; then RUN=$cand; break; fi
  if grep -aq "n_env_train: 8" "$cand/config.yaml" 2>/dev/null; then RUN=$cand; break; fi
  if python3 -c "import pathlib; t=pathlib.Path('$cand/config.yaml').read_text(); exit(0 if 'n_env_train: 8' in t or 'n_env_train: 8' in t else 1)" 2>/dev/null; then
    RUN=$cand; break
  fi
done
# simpler: newest run started after mid launch — use ACTIVE if models/1000 exists else newest with 1000
if [[ ! -f "$RUN/models/1000/actor.pkl" ]]; then
  for cand in $(ls -dt baselines/gcbf/log/gcbf/seed0_*); do
    [[ -f "$cand/models/1000/actor.pkl" ]] || continue
    if grep -q 'n_env_train: 8' "$cand/config.yaml" 2>/dev/null; then RUN=$cand; break; fi
  done
fi
echo "$RUN" > "$RUN_FILE"
echo "eval run=$RUN"
if [[ ! -f "$RUN/models/1000/actor.pkl" ]]; then
  echo "ERROR missing 1000"; ls "$RUN/models" | sort -n | tail; exit 1
fi
cp -a "$RUN/models/1000/actor.pkl" "$RUN/models/1000/cbf.pkl" "$RUN/config.yaml" baselines/gcbf/output/gcbf_mid/
echo 1000 > baselines/gcbf/output/gcbf_mid/ACTIVE_STEP.txt
VAL=baselines/gcbf/output/gcbf_mid/validate_32epi.log
echo "[$(date -Iseconds)] validate step1000 epi32 seed1234" | tee "$VAL"
bash scripts/validate_baseline_gcbf.sh "$RUN" 1000 32 2>&1 | tee -a "$VAL"
python3 - <<'PY'
import json, re
from pathlib import Path
from datetime import datetime, timezone
text = Path("baselines/gcbf/output/gcbf_mid/validate_32epi.log").read_text()
m = re.search(
    r"AVG over\s+(\d+)\s+episodes:.*?reward:\s*([-\d.]+).*?cost:\s*([-\d.]+).*?safe rate:\s*([\d.]+)%.*?finish rate:\s*([\d.]+)%.*?success rate:\s*([\d.]+)%",
    text, re.S)
run = Path("baselines/gcbf/output/gcbf_mid/ACTIVE_RUN.txt").read_text().strip()
out = {
  "policy_id": "gcbf_baseline_mid_scale",
  "policy_kind": "gcbf",
  "label": "gcbf_mid_scale",
  "checkpoint": f"{run}/models/1000",
  "artifacts": "baselines/gcbf/output/gcbf_mid/",
  "algo": "gcbf",
  "env": "DoubleIntegrator",
  "num_agents": 8, "area_size": 4.0, "obs": 4, "n_rays": 32,
  "train_seed": 0, "train_steps": 1000,
  "train_hparams": {
    "n_env_train": 8, "n_env_test": 16, "batch_size": 128, "buffer_size": 512,
    "inner_epoch": 8, "lr_actor": 1e-5, "lr_cbf": 1e-5,
    "loss_action_coef": 1e-4, "loss_h_dot_coef": 0.01,
    "note": "Mid-scale stable-enough. Same eval protocol as GCBF+/MF. Official 16-env aborted.",
  },
  "episodes": 32, "eval_seed": 1234, "teacher_step": 1000,
  "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "claim": "Empirical only. Mid-scale GCBF ablation. Not formal verification.",
  "compare_ref_gcbf_weak_32epi": {"success_rate": 0.515625, "safe_rate": 0.5390625, "finish_rate": 0.9140625},
  "compare_ref_gcbfplus_32epi": {"success_rate": 0.9140625, "safe_rate": 0.95703125, "finish_rate": 0.953125},
}
if m:
  out.update({
    "episodes": int(m.group(1)), "reward": float(m.group(2)), "cost": float(m.group(3)),
    "safe_rate": float(m.group(4))/100.0, "finish_rate": float(m.group(5))/100.0,
    "success_rate": float(m.group(6))/100.0,
  })
else:
  out["parse_error"] = "AVG line not found"
Path("baselines/gcbf/output/gcbf_mid/validate_32epi.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
suc, safe, fin = out.get("success_rate"), out.get("safe_rate"), out.get("finish_rate")
def pct(x):
  return f"{x*100:.1f}%" if x is not None else "n/a"
Path("baselines/gcbf/output/gcbf_mid/README.md").write_text(f"""# Mid-scale GCBF baseline

Train: `n_env_train=8`, `batch=128`, `inner_epoch=8`, `lr=1e-5`, `steps=1000`.
Eval: **same** as GCBF+/MF — 8 agents · obs4 · area4 · **32-epi · seed=1234**.

| 策略 | success↑ | safe↑ | finish↑ |
|---|---:|---:|---:|
| GCBF weak (2-env) | 51.6% | 53.9% | 91.4% |
| **GCBF mid-scale** | **{pct(suc)}** | **{pct(safe)}** | **{pct(fin)}** |
| GCBF+ teacher | 91.4% | 95.7% | 95.3% |

Run: `{run}`
""")
print("DONE_EVAL")
PY
