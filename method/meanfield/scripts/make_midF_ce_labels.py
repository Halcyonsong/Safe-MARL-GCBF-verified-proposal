#!/usr/bin/env python3
"""Convert formal mid-F product-box CEs → residual training CE json.

Formal CEs only carry (x, a_bad, d0, c0, d2). For the *radial* two-step
property we certify, a sound min-intervention label is to throttle ‖a‖∞:

    a_max_ok = max(0, (d0 - α c0 - R) / β)
    a_safe   = a_bad * (a_max_ok / ‖a_bad‖∞)   if ‖a_bad‖∞ > a_max_ok
             = a_bad                            otherwise (should not happen)

All mid-F CEs from the r2 probe were coast-safe and throttle-preventable.

Optionally merge with an existing CE pool (e.g. r2 ce_train_merged.json).

Usage:
  python scripts/make_midF_ce_labels.py \\
    --formal-ce method/meanfield/output/verify/mf_r2_midF_probe_20261006/counterexamples_all.json \\
    --merge method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/ce_train_merged.json \\
    --out method/meanfield/output/repair/midF_ce_train.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

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

from stage2.dynamics import DT, HORIZON_STEPS, MASS
from stage_mf.property_l2 import d0_c0_from_x, two_step_radial_d2

R_DEFAULT = 0.10
ALPHA = HORIZON_STEPS * DT  # 0.06
BETA = (DT**2 / MASS) * HORIZON_STEPS * (HORIZON_STEPS - 1) / 2.0  # 0.009


def throttle_label(a_bad: np.ndarray, d0: float, c0: float, *, radius: float) -> tuple[np.ndarray, bool, float]:
    a = np.asarray(a_bad, dtype=np.float64).reshape(-1)[:2]
    a_inf = float(np.max(np.abs(a)))
    a_max_ok = (d0 - ALPHA * c0 - radius) / max(BETA, 1e-12)
    a_max_ok = float(max(0.0, a_max_ok))
    if a_inf <= a_max_ok + 1e-12:
        # already radial-safe under this a — should not be a CE
        return a.copy(), False, a_max_ok
    scale = a_max_ok / a_inf
    a_safe = a * scale
    d2 = two_step_radial_d2(d0, c0, a_safe)
    ok = bool(d2 >= radius - 1e-9)
    return a_safe, ok, a_max_ok


def convert_formal(path: pathlib.Path, *, radius: float) -> list[dict]:
    raw = json.loads(path.read_text())
    ces = raw["counterexamples"] if isinstance(raw, dict) else raw
    out = []
    n_ok = 0
    for e in ces:
        x = np.asarray(e["x"], dtype=np.float64)
        a_bad = np.asarray(e.get("a") or e.get("action"), dtype=np.float64).reshape(-1)[:2]
        d0 = float(e["d0"]) if "d0" in e else d0_c0_from_x(x)[0]
        c0 = float(e["c0"]) if "c0" in e else d0_c0_from_x(x)[1]
        a_safe, ok, a_max = throttle_label(a_bad, d0, c0, radius=radius)
        if ok:
            n_ok += 1
        out.append(
            {
                "x": x.tolist(),
                "a": a_bad.tolist(),  # violating action (debug)
                "a_l2_safe": a_safe.tolist(),
                "a_l2_safe_dist": float(two_step_radial_d2(d0, c0, a_safe) - radius),
                "a_l2_label_ok": bool(ok),
                "a_l2_label_mode": "radial_throttle_min_intervention",
                "a_l2_delta": float(np.linalg.norm(a_safe - a_bad)),
                "a_max_ok": a_max,
                "kind": "aa",  # radial; treat as aa-weight bucket
                "d0": d0,
                "c0": c0,
                "d2_bad": float(e.get("d2", two_step_radial_d2(d0, c0, a_bad))),
                "source": "formal_midF_probe",
                "claim": e.get("claim", "mf_midF"),
            }
        )
    print(f"[convert] formal {path}: n={len(out)} label_ok={n_ok}")
    return out


def load_merge(path: pathlib.Path) -> list[dict]:
    raw = json.loads(path.read_text())
    ces = raw.get("counterexamples", raw if isinstance(raw, list) else [])
    print(f"[merge] {path}: n={len(ces)} label_ok={sum(1 for c in ces if c.get('a_l2_label_ok'))}")
    return list(ces)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-ce", type=str, required=True)
    ap.add_argument("--merge", type=str, default="", help="optional existing CE pool")
    ap.add_argument("--radius", type=float, default=R_DEFAULT)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    ces = convert_formal(pathlib.Path(args.formal_ce), radius=args.radius)
    sources = [args.formal_ce]
    if args.merge.strip():
        ces = load_merge(pathlib.Path(args.merge)) + ces
        sources.append(args.merge)

    n_ok = sum(1 for c in ces if c.get("a_l2_label_ok"))
    out = {
        "track": "meanfield",
        "merged": True,
        "sources": sources,
        "counterexamples": ces,
        "n_kept": len(ces),
        "n_labels_safe": n_ok,
        "radius": args.radius,
        "alpha": ALPHA,
        "beta": BETA,
        "note": (
            "mid-F formal CEs labeled by radial throttle min-intervention; "
            "optionally merged with prior CE pool. Side-branch repair only."
        ),
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out) + "\n")
    print(f"wrote {out_path} n={len(ces)} label_ok={n_ok}")


if __name__ == "__main__":
    main()
