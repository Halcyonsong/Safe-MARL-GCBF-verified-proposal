"""Validate HandCBF = official DecShareCBF (hand-crafted CBF + online QP).

No checkpoint. Same eval protocol as project baselines:
  DoubleIntegrator · 8 agents · area4 · obs4 · rays32 · seed=1234

Usage:
  python baselines/handcbf/code/validate.py --epi 32 [--out-json PATH] [--alpha 1.0]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

import pathlib as _pathlib
import sys as _sys
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "core" / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from core.paths import project_root, ensure_import_path
_ROOT = project_root()
ensure_import_path()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epi", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--num-agents", type=int, default=8)
    ap.add_argument("--obs", type=int, default=4)
    ap.add_argument("--area-size", type=float, default=4.0)
    ap.add_argument("--n-rays", type=int, default=32)
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--cpu", action="store_true", default=False)
    args = ap.parse_args()

    py = sys.executable
    cmd = [
        py,
        str(_ROOT / "baselines/gcbfplus/code/test_official.py"),
        "--env",
        "DoubleIntegrator",
        "-n",
        str(args.num_agents),
        "--area-size",
        str(args.area_size),
        "--obs",
        str(args.obs),
        "--n-rays",
        str(args.n_rays),
        "--algo",
        "dec_share_cbf",
        "--alpha",
        str(args.alpha),
        "--seed",
        str(args.seed),
        "--epi",
        str(args.epi),
        "--no-video",
        "--log",
    ]
    if args.cpu:
        cmd.append("--cpu")

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    wall = time.time() - t0
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    # Parse final summary line (has min/max reward) — not per-epi lines
    text = proc.stdout
    ms = list(
        re.finditer(
            r"reward:\s*([-\d.]+),\s*min/max reward:.*?cost:\s*([-\d.]+),\s*min/max cost:.*?"
            r"safe_rate:\s*([\d.]+)%,\s*finish_rate:\s*([\d.]+)%,\s*success_rate:\s*([\d.]+)%",
            text,
            re.S,
        )
    )
    if not ms:
        raise RuntimeError("failed to parse HandCBF summary from test_official output")
    m = ms[-1]
    result = {
        "policy_id": "handcbf_dec_share_cbf",
        "policy_kind": "handcrafted_cbf_qp",
        "label": "HandCBF (DecShareCBF)",
        "algo": "dec_share_cbf",
        "alpha": args.alpha,
        "episodes": args.epi,
        "eval_seed": args.seed,
        "reward": float(m.group(1)),
        "cost": float(m.group(2)),
        "safe_rate": round(float(m.group(3)) / 100.0, 6),
        "finish_rate": round(float(m.group(4)) / 100.0, 6),
        "success_rate": round(float(m.group(5)) / 100.0, 6),
        "wall_time_s": round(wall, 3),
        "claim": "Empirical rollout only. Online hand-crafted CBF-QP. Not formal verification.",
    }
    print(json.dumps({k: result[k] for k in (
        "success_rate", "safe_rate", "finish_rate", "cost", "reward", "wall_time_s"
    )}, indent=2))

    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
