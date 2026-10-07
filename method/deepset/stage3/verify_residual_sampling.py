"""Joint sampling verification of a residual-gate policy on L2 preventable D.

Reports hit-rate on D for requested buckets. This is empirical evidence only:
not IBP/Marabou proven for the residual (base+g·δ) network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

from stage2.bounds import default_l2_buckets
from stage3.retrain_residual_gate import resample_hit_rate_residual
from stage3.residual_policy import load_residual_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--buckets", default="l2_aa_near,l2_ao_near,l2_11_near")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="method/deepset/output/stage2/l2_residual_sampling_report.json")
    args = parser.parse_args()

    ckpt, modules, base_params, train_params = load_residual_checkpoint(args.checkpoint)
    want = {b.strip() for b in args.buckets.split(",") if b.strip()}
    specs = [s for s in default_l2_buckets() if s.name in want]
    if not specs:
        raise SystemExit(f"no specs for {want}")

    hit = resample_hit_rate_residual(
        modules,
        base_params,
        train_params,
        specs,
        n_samples=args.samples,
        seed=args.seed,
        gate_mode=str(ckpt.get("gate_mode", "hybrid")),
        open_mult=float(ckpt.get("open_mult", 2.5)),
        close_mult=float(ckpt.get("close_mult", 1.0)),
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "architecture": ckpt.get("architecture"),
        "base_checkpoint": ckpt.get("base_checkpoint"),
        "gate_mode": ckpt.get("gate_mode"),
        "samples_per_bucket": args.samples,
        "seed": args.seed,
        "buckets": hit,
        "claim": (
            "Sampling hit-rate on preventable set D for residual policy. "
            "NOT a formal safety proof. Marabou/IBP on full residual not in this report."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {"hit": v["hit_rate_on_D"], "gate": v.get("mean_gate_on_D")} for k, v in hit.items()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
