#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
# script is at baselines/gcbf/output/gcbf_official/ → parents: gcbf_official, baseline, runs, ROOT
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
LOG=baselines/gcbf/output/gcbf_official/train_outer.log
PID_FILE=baselines/gcbf/output/gcbf_official/train.pid
RUN_FILE=baselines/gcbf/output/gcbf_official/ACTIVE_RUN.txt
PID=$(cat "$PID_FILE")
RUN=$(cat "$RUN_FILE")
echo "[watch $(date -Iseconds)] waiting pid=$PID run=$RUN"
while kill -0 "$PID" 2>/dev/null || pgrep -f 'baselines/gcbf/code/train.py --algo gcbf' >/dev/null; do
  sleep 120
  PROG=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -E '[0-9]+%\|' | tail -1 || true)
  echo "[watch $(date +%H:%M:%S)] $PROG"
done
echo "[watch $(date -Iseconds)] train exited"
# refresh run dir (newest official after weak)
RUN=$(ls -dt baselines/gcbf/log/gcbf/seed0_* 2>/dev/null | head -1)
echo "$RUN" > "$RUN_FILE"
# ensure model 1000 exists
if [[ ! -f "$RUN/models/1000/actor.pkl" ]]; then
  echo "ERROR: missing $RUN/models/1000/actor.pkl" | tee -a "$LOG"
  ls "$RUN/models" | sort -n | tail -10
  exit 1
fi
# mirror weights
cp -a "$RUN/models/1000/actor.pkl" "$RUN/models/1000/cbf.pkl" "$RUN/config.yaml" baselines/gcbf/output/gcbf_official/
echo 1000 > baselines/gcbf/output/gcbf_official/ACTIVE_STEP.txt
# same eval protocol
VAL=baselines/gcbf/output/gcbf_official/validate_32epi.log
echo "[$(date -Iseconds)] validate step1000 epi32 seed1234" | tee "$VAL"
bash scripts/validate_baseline_gcbf.sh "$RUN" 1000 32 2>&1 | tee -a "$VAL"
# parse rates from last AVG line
python3 - <<'PY'
import json, re
from pathlib import Path
from datetime import datetime, timezone
text = Path("baselines/gcbf/output/gcbf_official/validate_32epi.log").read_text()
# AVG line: AVG over 32 episodes: reward: ..., cost: ..., safe rate: xx.xxx%, finish rate: xx.xxx%, success rate: xx.xxx%
m = re.search(
    r"AVG over\s+(\d+)\s+episodes:.*?reward:\s*([-\d.]+).*?cost:\s*([-\d.]+).*?safe rate:\s*([\d.]+)%.*?finish rate:\s*([\d.]+)%.*?success rate:\s*([\d.]+)%",
    text, re.S)
run = Path("baselines/gcbf/output/gcbf_official/ACTIVE_RUN.txt").read_text().strip()
out = {
  "policy_id": "gcbf_baseline_official_scale",
  "policy_kind": "gcbf",
  "label": "gcbf_official_scale",
  "checkpoint": f"{run}/models/1000",
  "artifacts": "baselines/gcbf/output/gcbf_official/",
  "algo": "gcbf",
  "env": "DoubleIntegrator",
  "num_agents": 8, "area_size": 4.0, "obs": 4, "n_rays": 32,
  "train_seed": 0, "train_steps": 1000,
  "train_hparams": {
    "n_env_train": 16, "n_env_test": 32, "batch_size": 256, "buffer_size": 512,
    "inner_epoch": 8, "lr_actor": 1e-5, "lr_cbf": 1e-5,
    "loss_action_coef": 1e-4, "loss_h_dot_coef": 0.01,
    "note": "Official-scale (README/paper). Same eval protocol as GCBF+/MF.",
  },
  "episodes": 32, "eval_seed": 1234, "teacher_step": 1000,
  "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "claim": "Empirical only. Official-scale GCBF ablation. Not formal verification.",
  "compare_ref_gcbf_weak_32epi": {"success_rate": 0.515625, "safe_rate": 0.5390625, "finish_rate": 0.9140625},
  "compare_ref_gcbfplus_32epi": {"success_rate": 0.9140625, "safe_rate": 0.95703125, "finish_rate": 0.953125},
}
if m:
  out.update({
    "episodes": int(m.group(1)),
    "reward": float(m.group(2)),
    "cost": float(m.group(3)),
    "safe_rate": float(m.group(4))/100.0,
    "finish_rate": float(m.group(5))/100.0,
    "success_rate": float(m.group(6))/100.0,
  })
else:
  out["parse_error"] = "AVG line not found"
Path("baselines/gcbf/output/gcbf_official/validate_32epi.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
# README
suc = out.get("success_rate")
safe = out.get("safe_rate")
fin = out.get("finish_rate")
readme = f"""# Official-scale GCBF baseline

Train: `n_env_train=16`, `batch=256`, `inner_epoch=8`, `lr=1e-5`, `steps=1000` (README/paper).
Eval protocol **same** as GCBF+/DeepSet/MF: 8 agents · obs4 · area4 · **32-epi · seed=1234**.

| 策略 | success↑ | safe↑ | finish↑ |
|---|---:|---:|---:|
| GCBF weak (2-env) | 51.6% | 53.9% | 91.4% |
| **GCBF official-scale** | **{suc*100 if suc else float('nan'):.1f}%** | **{safe*100 if safe else float('nan'):.1f}%** | **{fin*100 if fin else float('nan'):.1f}%** |
| GCBF+ teacher | 91.4% | 95.7% | 95.3% |

Run: `{run}`
"""
Path("baselines/gcbf/output/gcbf_official/README.md").write_text(readme)
print("DONE_EVAL")
PY
