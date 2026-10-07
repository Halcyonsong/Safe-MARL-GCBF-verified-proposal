#!/usr/bin/env python3
"""Discriminative sampling inside the MF near_plus acceptance box.

Binary L2-safe% is ~100% for any |a|_∞≤1 (physics envelope d2_lo=0.113>R).
This script measures *margin* and *hard-corner* behavior instead:

  margin_i = d2(x_i, a_i) - R
  hard corner = low d0 + high c0 slice of the same box

Feature-input policies only (MF-r2, MAPPO-Lag, coast). Graph policies = N/A.

Usage:
  python scripts/sample_near_plus_margin.py --n 50000 --n-hard 20000
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
from stage2.dynamics import DT, HORIZON_STEPS, MASS
from stage_mf.domain import default_mf_claims
from stage_mf.encode import load_mf_residual_weights, physics_gate_np
from stage_mf.property_l2 import D_EMPTY_DEFAULT, N_OCCUPIED, occupied_threat_pairs
from stage_mf.student import FEATURE_VERSION, INPUT_DIM

R = 0.10


def d0_c0_batch(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(xs, dtype=np.float64)
    n = xs.shape[0]
    d0 = np.full(n, D_EMPTY_DEFAULT, dtype=np.float64)
    c0 = np.zeros(n, dtype=np.float64)
    any_occ = np.zeros(n, dtype=bool)
    for n_i, d_i, c_i in occupied_threat_pairs():
        occ = xs[:, n_i] >= N_OCCUPIED
        any_occ |= occ
        d0 = np.where(occ, np.minimum(d0, xs[:, d_i]), d0)
        c0 = np.where(occ, np.maximum(c0, xs[:, c_i]), c0)
    d0 = np.where(any_occ, d0, D_EMPTY_DEFAULT)
    c0 = np.maximum(c0, 0.0)
    return d0, c0


def d2_batch(d0: np.ndarray, c0: np.ndarray, acts: np.ndarray) -> np.ndarray:
    a_mag = np.max(np.abs(acts), axis=1)
    d, c = d0.astype(np.float64).copy(), c0.astype(np.float64).copy()
    acc = a_mag / max(MASS, 1e-9)
    for _ in range(HORIZON_STEPS):
        d = d - c * DT
        c = np.maximum(c + acc * DT, 0.0)
    return d


def summarize(name: str, xs: np.ndarray, acts: np.ndarray, tag: str) -> dict:
    d0, c0 = d0_c0_batch(xs)
    d2 = d2_batch(d0, c0, acts)
    margin = d2 - R
    a_inf = np.max(np.abs(acts), axis=1)
    return {
        "policy": name,
        "slice": tag,
        "n": int(xs.shape[0]),
        "l2_safe_rate": float(np.mean(d2 >= R - 1e-9)),
        "margin_mean": float(margin.mean()),
        "margin_p10": float(np.percentile(margin, 10)),
        "margin_p50": float(np.percentile(margin, 50)),
        "margin_p5": float(np.percentile(margin, 5)),
        "margin_min": float(margin.min()),
        "frac_margin_lt_0.02": float(np.mean(margin < 0.02)),
        "frac_margin_lt_0.05": float(np.mean(margin < 0.05)),
        "mean_abs_a_inf": float(a_inf.mean()),
        "p90_abs_a_inf": float(np.percentile(a_inf, 90)),
        "mean_d0": float(d0.mean()),
        "mean_c0": float(c0.mean()),
        "mean_d2": float(d2.mean()),
    }


def sample_uniform(box, rng, n: int) -> np.ndarray:
    return box.sample(rng, n).astype(np.float64)


def sample_hard_corner(box, rng, n: int, *, d0_hi: float = 0.18, c0_lo: float = 0.22) -> np.ndarray:
    """Rejection / conditioned draw: semantic (d0,c0) in hard band, x still in product box.

    Approximate: draw from box, keep if d0≤d0_hi and c0≥c0_lo; refill until n.
    """
    kept = []
    tries = 0
    max_tries = n * 80
    while sum(k.shape[0] for k in kept) < n and tries < max_tries:
        need = n - sum(k.shape[0] for k in kept)
        batch = max(need * 8, 4096)
        xs = box.sample(rng, batch).astype(np.float64)
        d0, c0 = d0_c0_batch(xs)
        m = (d0 <= d0_hi) & (c0 >= c0_lo)
        if m.any():
            kept.append(xs[m])
        tries += batch
    if not kept:
        raise RuntimeError("hard-corner sampling got 0 hits — loosen d0_hi/c0_lo")
    xs = np.concatenate(kept, axis=0)[:n]
    return xs


def mf_actions(w, xs: np.ndarray, batch: int = 2048) -> np.ndarray:
    outs = []
    for i in range(0, xs.shape[0], batch):
        chunk = xs[i : i + batch]
        a0 = np.stack([w.base.forward(x) for x in chunk], axis=0)
        gs = np.array([
            physics_gate_np(
                x,
                open_mult=w.open_mult,
                close_mult=w.close_mult,
                r_ref=w.r_ref,
                inv_ttc_ref=w.inv_ttc_ref,
                mix=w.gate_mix,
            )
            for x in chunk
        ])
        ds = np.stack([
            np.clip(w.delta.forward(x), -w.delta_clip, w.delta_clip) for x in chunk
        ], axis=0)
        outs.append(np.clip(a0 + gs[:, None] * ds, -1.0, 1.0))
    return np.concatenate(outs, axis=0)


def mappo_actions(actor, params, xs: np.ndarray, batch: int = 8192) -> np.ndarray:
    @jax.jit
    def fwd(x):
        mean, _ = actor.apply(params, x)
        return jnp.clip(jnp.tanh(mean), -1.0, 1.0)

    outs = []
    for i in range(0, xs.shape[0], batch):
        outs.append(np.asarray(fwd(jnp.asarray(xs[i : i + batch], dtype=np.float32))))
    return np.concatenate(outs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--n-hard", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--d0-hi", type=float, default=0.18)
    ap.add_argument("--c0-lo", type=float, default=0.22)
    ap.add_argument("--out-dir", type=str, default="baselines/compare/output/near_plus_margin")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    claim = default_mf_claims(mode="near_plus", radius=R)[0]
    box = claim.feature
    assert box.dim == INPUT_DIM
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print(f"[1] uniform n={args.n}", flush=True)
    xs_u = sample_uniform(box, rng, args.n)
    print(f"[2] hard-corner n={args.n_hard} (d0≤{args.d0_hi}, c0≥{args.c0_lo})", flush=True)
    xs_h = sample_hard_corner(box, rng, args.n_hard, d0_hi=args.d0_hi, c0_lo=args.c0_lo)
    d0h, c0h = d0_c0_batch(xs_h)
    print(
        f"    hard realized mean_d0={d0h.mean():.3f} mean_c0={c0h.mean():.3f} "
        f"d0∈[{d0h.min():.3f},{d0h.max():.3f}] c0∈[{c0h.min():.3f},{c0h.max():.3f}]",
        flush=True,
    )

    # load policies
    w = load_mf_residual_weights(
        "method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl"
    )
    mpath = pathlib.Path("baselines/mappo_lag/output/mappo_lag/ACTIVE.txt").read_text().strip()
    with open(mpath, "rb") as f:
        mck = pickle.load(f)
    actor = GaussianActor(hidden=int(mck["hidden"]), action_dim=2)
    mparams = mck["actor_params"]

    rows = []
    for tag, xs in (("uniform", xs_u), ("hard_corner", xs_h)):
        print(f"[score] slice={tag} …", flush=True)
        a_coast = np.zeros((xs.shape[0], 2))
        rows.append(summarize("coast_a0", xs, a_coast, tag))

        a_mf = mf_actions(w, xs)
        rows.append(summarize("mf_r2", xs, a_mf, tag))

        a_mp = mappo_actions(actor, mparams, xs)
        rows.append(summarize("mappo_lag", xs, a_mp, tag))

        print(
            f"  coast margin_mean={rows[-3]['margin_mean']:.4f}  "
            f"mf={rows[-2]['margin_mean']:.4f}  mappo={rows[-1]['margin_mean']:.4f}",
            flush=True,
        )

    wall = time.time() - t0
    report = {
        "claim": (
            "Margin + hard-corner sampling inside near_plus box. "
            "NOT formal verification. Binary L2% is uninformative here (envelope)."
        ),
        "why": (
            "Envelope ⇒ d2_lo=0.113>R for all |a|_∞≤1. "
            "Discrimination via margin=d2-R and hard (low-d0, high-c0) slice."
        ),
        "domain": {
            "mode": "near_plus",
            "R": R,
            "hard_corner": {"d0_hi": args.d0_hi, "c0_lo": args.c0_lo},
        },
        "n_uniform": args.n,
        "n_hard": args.n_hard,
        "seed": args.seed,
        "wall_time_s": round(wall, 2),
        "rows": rows,
        "graph_policies": "N/A (need GraphsTuple)",
    }
    (out / "margin_sample.json").write_text(json.dumps(report, indent=2) + "\n")

    def fmt_row(r):
        return (
            f"| {r['policy']} | {r['slice']} | {r['n']} | "
            f"{100*r['l2_safe_rate']:.1f} | {r['margin_mean']:.4f} | "
            f"{r['margin_p10']:.4f} | {r['margin_min']:.4f} | "
            f"{100*r['frac_margin_lt_0.02']:.1f} | {r['mean_abs_a_inf']:.3f} |"
        )

    md = [
        "# near_plus · margin + hard-corner (discriminative)",
        "",
        "> Binary L2-safe% ≈100% for any clipped policy (physics envelope).  ",
        "> **Use margin \(d_2-R\)** and **hard corner** (low \(d_0\), high \(c_0\)) to compare.",
        "",
        f"- uniform n={args.n} · hard n={args.n_hard} (d0≤{args.d0_hi}, c0≥{args.c0_lo})",
        f"- seed={args.seed} · wall={wall:.1f}s · R={R}",
        "",
        "| policy | slice | n | L2% | margin mean | p10 | min | % margin<0.02 | ‖a‖∞ mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(fmt_row(r))
    md += [
        "",
        "## How to read",
        "",
        "1. **L2%** near 100% on both slices → expected; not a ranking signal.",
        "2. **Higher margin** = more clearance under the same radial two-step model.",
        "3. **Hard corner** stresses the preventable band edge (still inside X).",
        "4. **coast_a0** is the physics reference (often largest margin).",
        "5. MF formal claim remains PROVEN_FULL on the whole box; this table is empirical contrast only.",
        "6. DS / GCBF+ / HandCBF need graphs → not in this table.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(md))
    print(json.dumps(rows, indent=2))
    print(f"wrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
