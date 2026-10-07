#!/usr/bin/env python3
"""Empirical MC on a *custom* near_plus-style product box (feature policies).

Used after a side-branch formal PROVEN on (d_min, close_hi) to check whether
baselines still show nonzero two-step radial L2 fails on the *same* declared X.

Graph policies (GCBF+ / InforMARL / DS) → N/A here; use sample_midF_env_baselines.py
with matching --d0-lo/--c0-hi semantic band.

Usage:
  python scripts/sample_candidate_box_baselines.py \
    --d-min 0.126 --close-hi 0.28 --n 50000 \
    --mf-ckpt method/meanfield/output/repair/mf_residual_v5_midF_r3_20261006/seed0/repair_best.pkl \
    --out-dir baselines/compare/output/cand_d126_c28
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time

import jax
import jax.numpy as jnp
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



from baseline.mappo_lag.nets import GaussianActor
from stage_mf.domain import feature_box
from stage_mf.encode import load_mf_residual_weights, physics_gate_np
from stage_mf.student import FEATURE_VERSION, INPUT_DIM

R = 0.10


def score_batch(xs: np.ndarray, acts: np.ndarray, radius: float = R) -> dict:
    from stage2.dynamics import DT, HORIZON_STEPS, MASS
    from stage_mf.property_l2 import D_EMPTY_DEFAULT, N_OCCUPIED, occupied_threat_pairs

    xs = np.asarray(xs, dtype=np.float64)
    acts = np.asarray(acts, dtype=np.float64)
    n = xs.shape[0]
    pairs = occupied_threat_pairs()
    d0 = np.full(n, D_EMPTY_DEFAULT, dtype=np.float64)
    c0 = np.zeros(n, dtype=np.float64)
    any_occ = np.zeros(n, dtype=bool)
    for n_i, d_i, c_i in pairs:
        occ = xs[:, n_i] >= N_OCCUPIED
        any_occ |= occ
        d0 = np.where(occ, np.minimum(d0, xs[:, d_i]), d0)
        c0 = np.where(occ, np.maximum(c0, xs[:, c_i]), c0)
    d0 = np.where(any_occ, d0, D_EMPTY_DEFAULT)
    c0 = np.maximum(c0, 0.0)
    a_mag = np.max(np.abs(acts), axis=1)
    d, c = d0.copy(), c0.copy()
    acc = a_mag / max(MASS, 1e-9)
    for _ in range(HORIZON_STEPS):
        d = d - c * DT
        c = c + acc * DT
        c = np.maximum(c, 0.0)
    d2 = d
    safe = d2 >= radius - 1e-12
    margin = d2 - radius
    return {
        "n": int(n),
        "n_l2_safe": int(safe.sum()),
        "n_fail": int((~safe).sum()),
        "l2_safe_rate": float(safe.mean()),
        "fail_rate": float((~safe).mean()),
        "margin_mean": float(margin.mean()),
        "margin_p10": float(np.percentile(margin, 10)),
        "margin_min": float(margin.min()),
        "mean_d2": float(d2.mean()),
        "mean_abs_a_inf": float(a_mag.mean()),
        "p90_abs_a_inf": float(np.percentile(a_mag, 90)),
        "mean_d0": float(d0.mean()),
        "mean_c0": float(c0.mean()),
    }


def mf_actions(w, xs, batch: int = 2048) -> np.ndarray:
    """Per-row numpy forward (deploy_action_np is single-vector)."""
    outs = []
    for i in range(0, xs.shape[0], batch):
        chunk = xs[i : i + batch]
        a0 = np.stack([w.base.forward(x) for x in chunk], axis=0)
        gs = np.array(
            [
                physics_gate_np(
                    x,
                    open_mult=w.open_mult,
                    close_mult=w.close_mult,
                    r_ref=w.r_ref,
                    inv_ttc_ref=w.inv_ttc_ref,
                    mix=w.gate_mix,
                )
                for x in chunk
            ]
        )
        ds = np.stack(
            [np.clip(w.delta.forward(x), -w.delta_clip, w.delta_clip) for x in chunk],
            axis=0,
        )
        outs.append(np.clip(a0 + gs[:, None] * ds, -1.0, 1.0))
        if (i // batch) % 5 == 0:
            print(f"  mf fwd {min(i + batch, xs.shape[0])}/{xs.shape[0]}", flush=True)
    return np.concatenate(outs, axis=0)


def mappo_actions(actor, params, xs: np.ndarray, batch: int = 8192) -> np.ndarray:
    @jax.jit
    def fwd(x):
        mean, _ = actor.apply(params, x)
        return jnp.clip(jnp.tanh(mean), -1.0, 1.0)

    outs = []
    for i in range(0, xs.shape[0], batch):
        outs.append(np.asarray(fwd(jnp.asarray(xs[i : i + batch]))))
    return np.concatenate(outs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--d-min", type=float, required=True)
    ap.add_argument("--d-max", type=float, default=0.40)
    ap.add_argument("--close-hi", type=float, required=True)
    ap.add_argument("--radius", type=float, default=R)
    ap.add_argument(
        "--mf-ckpt",
        type=str,
        default="method/meanfield/output/repair/mf_residual_v5_midF_r3_20261006/seed0/repair_best.pkl",
    )
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--hard-fringe", action="store_true",
                    help="also score conditioned hard fringe: d0∈[d_min,d_min+0.01], c0∈[close_hi-0.05,close_hi]")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    box = feature_box(
        d_min=args.d_min,
        d_max=args.d_max,
        close_lo=-args.close_hi,
        close_hi=args.close_hi,
        force_occupied=True,
    )
    assert box.dim == INPUT_DIM
    rng = np.random.default_rng(args.seed)
    d2_lo = args.d_min - 0.06 * args.close_hi - 0.009
    print(
        f"[box] n={args.n} d∈[{args.d_min},{args.d_max}] |c|≤{args.close_hi} "
        f"d2_lo={d2_lo:.4f} net_dep={d2_lo < args.radius}",
        flush=True,
    )
    xs = box.sample(rng, args.n).astype(np.float64)

    rows = []

    def add(name, acts, note=""):
        st = score_batch(xs, acts, radius=args.radius)
        st["policy"] = name
        st["note"] = note
        rows.append(st)
        print(
            f"[score] {name}: fail={st['n_fail']} ({100*st['fail_rate']:.3f}%) "
            f"safe={100*st['l2_safe_rate']:.2f}% ‖a‖∞={st['mean_abs_a_inf']:.3f} "
            f"margin_min={st['margin_min']:.4f}",
            flush=True,
        )

    add("coast_a0", np.zeros((args.n, 2)), "a=0 envelope reference")

    w = load_mf_residual_weights(args.mf_ckpt)
    add("mf_cand", mf_actions(w, xs), f"ckpt={args.mf_ckpt}")

    # also acceptance r2 for contrast
    r2 = "method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl"
    if pathlib.Path(r2).exists() and pathlib.Path(r2).resolve() != pathlib.Path(args.mf_ckpt).resolve():
        add("mf_r2", mf_actions(load_mf_residual_weights(r2), xs), "acceptance r2")

    mpath = pathlib.Path("baselines/mappo_lag/output/mappo_lag/ACTIVE.txt").read_text().strip()
    with open(mpath, "rb") as f:
        mck = pickle.load(f)
    actor = GaussianActor(hidden=int(mck["hidden"]), action_dim=2)
    add(
        "mappo_lag",
        mappo_actions(actor, mck["actor_params"], xs.astype(np.float32)),
        "a=tanh(mean); uref omitted",
    )

    for name, reason in [
        ("gcbfplus", "needs GraphsTuple — use env-band script"),
        ("informarl", "needs GraphsTuple — use env-band script"),
    ]:
        rows.append({"policy": name, "n": 0, "n_fail": None, "fail_rate": None, "note": f"N/A — {reason}"})

    report = {
        "method": "direct_uniform_sample_feature_box",
        "domain": {
            "d_min": args.d_min,
            "d_max": args.d_max,
            "close_hi": args.close_hi,
            "radius": args.radius,
            "d2_lo": d2_lo,
            "net_dep": bool(d2_lo < args.radius),
        },
        "mf_ckpt": args.mf_ckpt,
        "n": args.n,
        "seed": args.seed,
        "rows": rows,
        "note": (
            "Empirical only. If d2_lo≥R, envelope predicts ~0 fail for all clipped policies; "
            "nonzero MAPPO fail still useful. Graph baselines need env-band pack."
        ),
    }
    (out / "box_mc.json").write_text(json.dumps(report, indent=2) + "\n")

    # optional hard fringe (rejection on semantic d0,c0)
    if args.hard_fringe:
        from stage_mf.property_l2 import d0_c0_from_x

        d_hi = args.d_min + 0.010
        c_lo = max(0.0, args.close_hi - 0.05)
        keep = []
        # draw until enough
        need = min(10000, args.n)
        drawn = 0
        while len(keep) < need and drawn < args.n * 20:
            batch = box.sample(rng, 8192).astype(np.float64)
            drawn += len(batch)
            for x in batch:
                d0, c0 = d0_c0_from_x(x)
                if args.d_min <= d0 <= d_hi and c_lo <= abs(c0) <= args.close_hi:
                    keep.append(x)
                    if len(keep) >= need:
                        break
        xs_h = np.asarray(keep, dtype=np.float64)
        print(f"[fringe] kept {len(xs_h)} / drawn≈{drawn} in d0∈[{args.d_min},{d_hi}] |c|∈[{c_lo},{args.close_hi}]", flush=True)
        frows = []
        for name, acts in [
            ("coast_a0", np.zeros((len(xs_h), 2))),
            ("mf_cand", mf_actions(w, xs_h)),
            ("mappo_lag", mappo_actions(actor, mck["actor_params"], xs_h.astype(np.float32))),
        ]:
            st = score_batch(xs_h, acts, radius=args.radius)
            st["policy"] = name
            frows.append(st)
            print(f"[fringe] {name}: fail={st['n_fail']} ({100*st['fail_rate']:.3f}%)", flush=True)
        report["hard_fringe"] = {
            "d0": [args.d_min, d_hi],
            "c0_abs": [c_lo, args.close_hi],
            "n": len(xs_h),
            "rows": frows,
        }
        (out / "box_mc.json").write_text(json.dumps(report, indent=2) + "\n")

    md = [
        f"# Candidate box MC · d≥{args.d_min}, |c|≤{args.close_hi}",
        "",
        f"> d2_lo={d2_lo:.4f} · net_dep={d2_lo < args.radius} · n={args.n} · seed={args.seed}",
        f"> mf_ckpt=`{args.mf_ckpt}`",
        "",
        "| policy | n | fail% | n_fail | L2-safe% | margin min | ‖a‖∞ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("fail_rate") is None:
            md.append(f"| {r['policy']} | — | N/A | — | — | — | — |")
        else:
            md.append(
                f"| {r['policy']} | {r['n']} | {100*r['fail_rate']:.3f} | {r['n_fail']} | "
                f"{100*r['l2_safe_rate']:.2f} | {r['margin_min']:.4f} | {r['mean_abs_a_inf']:.3f} |"
            )
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/REPORT.md", flush=True)


if __name__ == "__main__":
    main()
