#!/usr/bin/env bash
# Re-rank residual repair checkpoints with full 32-epi eval (manual selection helper).
# Does NOT write ACTIVE_* or touch acceptance pack.
#
# Usage:
#   bash method/meanfield/scripts/rerank_repair_ckpts.sh RUN_DIR [episodes=32] [seed=1234]
# Example:
#   bash method/meanfield/scripts/rerank_repair_ckpts.sh \
#     method/meanfield/output/repair/mf_residual_v5_netdep_r2_20261010/seed0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="${PY:-/home/xiang/miniconda3/envs/safe_marl_verify/bin/python}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/cuda_env.sh" 2>/dev/null || true
unset JAX_PLATFORMS 2>/dev/null || true
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/method/meanfield:$ROOT/method/deepset:${PYTHONPATH:-}"

RUN="${1:?run dir with repair*.pkl}"
EP="${2:-32}"
SEED="${3:-1234}"
RUN="$(cd "$RUN" && pwd)"
OUT="$RUN/manual_rerank_${EP}epi_seed${SEED}"
mkdir -p "$OUT"

mapfile -t CKPTS < <(ls -1 "$RUN"/repair_best.pkl "$RUN"/repair.pkl "$RUN"/repair_last.pkl "$RUN"/repair_step*.pkl 2>/dev/null | sort -u)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "no repair*.pkl in $RUN" >&2
  exit 1
fi

echo "[rerank] run=$RUN episodes=$EP seed=$SEED"
echo "[rerank] candidates: ${CKPTS[*]}"

RES_B="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"

"$PY" - <<PY
import json, pathlib, subprocess, os, sys
run = pathlib.Path("$RUN")
out = pathlib.Path("$OUT")
py = "$PY"
ep, seed = int("$EP"), int("$SEED")
ckpts = [pathlib.Path(p) for p in """${CKPTS[@]}""".split() if p]
rows = []
for ck in ckpts:
    if not ck.is_file():
        continue
    jpath = out / f"{ck.stem}_validate_{ep}epi.json"
    cmd = [
        py, "method/meanfield/stage_mf/validate_residual.py",
        "--checkpoint", str(ck),
        "--episodes", str(ep),
        "--seed", str(seed),
        "--out-json", str(jpath),
    ]
    print("[rerank] eval", ck.name, flush=True)
    subprocess.check_call(cmd)
    m = json.loads(jpath.read_text())
    rows.append({
        "name": ck.name,
        "path": str(ck),
        "success": float(m["success_rate"]),
        "safe": float(m["safe_rate"]),
        "finish": float(m["finish_rate"]),
        "cost": float(m["cost"]),
        "score_safe_success": 2.0 * float(m["safe_rate"]) + float(m["success_rate"]),
        "score_legacy": 2.0 * float(m["success_rate"]) + float(m["safe_rate"]),
    })

# rank by safe then success then low cost
rows.sort(key=lambda r: (-r["safe"], -r["success"], r["cost"]))
report = {
    "run": str(run),
    "episodes": ep,
    "seed": seed,
    "ranking_key": "safe desc, success desc, cost asc",
    "candidates": rows,
    "pick": rows[0] if rows else None,
    "note": "Manual helper only. Does not write ACTIVE_* / acceptance.",
}
(out / "RERANK.json").write_text(json.dumps(report, indent=2) + "\n")
# human table
lines = [
    f"# Manual re-rank @ {ep}-epi seed={seed}",
    "",
    "| rank | ckpt | suc | safe | fin | cost | score_safe_success |",
    "|---:|---|---:|---:|---:|---:|---:|",
]
for i, r in enumerate(rows, 1):
    mark = " ← pick" if i == 1 else ""
    lines.append(
        f"| {i} | `{r['name']}` | {100*r['success']:.1f}% | {100*r['safe']:.1f}% | "
        f"{100*r['finish']:.1f}% | {r['cost']:.3f} | {r['score_safe_success']:.3f} |{mark}"
    )
lines += [
    "",
    f"Pick path: `{rows[0]['path'] if rows else ''}`",
    "",
    "Copy to `repair_manual_pick.pkl` if you want a stable name (optional).",
]
(out / "RERANK.md").write_text("\n".join(lines) + "\n")
print((out / "RERANK.md").read_text())
PY

RES_A="$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt 2>/dev/null || true)"
if [[ "$RES_B" != "$RES_A" ]]; then
  echo "ERROR: ACTIVE_MF_RESIDUAL changed" >&2
  exit 2
fi
echo "[rerank] ACTIVE_* unchanged ✓ → $OUT/RERANK.md"
