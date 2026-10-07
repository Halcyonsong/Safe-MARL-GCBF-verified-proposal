#!/usr/bin/env python3
"""Empirical two-step L2 sampling by *direct construction* of the near_plus feature box.

NOT formal verification — but the sample domain matches the acceptance claim X.

Why not rollout filtering?
  Formal X is an abstract 44-D product box. Real trajectories almost never land
  inside it (empty quads have min_d≈0.5 ∉ [0.14,0.40]). Collecting via rollout
  is slow and statistically misaligned with the certificate domain.

This script:
  x ~ Uniform(near_plus box)   # same Box as stage_mf/verify_full.py --mode near_plus
  a = π(x)                     # feature-input policies only
  score TwoStepRadialL2Safe(x, a)

Policies that need a GraphsTuple (DS / GCBF+ / HandCBF) cannot be queried from
abstract features alone → listed as N/A here (use task rollouts for them).

Usage:
  python scripts/sample_near_plus_l2_baselines.py --n 50000
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
from stage_mf.domain import default_mf_claims
from stage_mf.encode import deploy_action_np, load_mf_residual_weights
from stage_mf.property_l2 import d0_c0_from_x, two_step_radial_d2
from stage_mf.student import FEATURE_VERSION, INPUT_DIM

R = 0.10


def score_batch(xs: np.ndarray, acts: np.ndarray, radius: float = R) -> dict:
    """xs [N,44], acts [N,2] → aggregate L2 stats (vectorized-ish over rows)."""
    from stage_mf.property_l2 import occupied_threat_pairs, D_EMPTY_DEFAULT, N_OCCUPIED
    from stage2.dynamics import DT, HORIZON_STEPS, MASS

    xs = np.asarray(xs, dtype=np.float64)
    acts = np.asarray(acts, dtype=np.float64)
    n = xs.shape[0]
    pairs = occupied_threat_pairs()
    # d0 = min occupied min_d; c0 = max(0, max occupied close)
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
    safe = d2 >= radius - 1e-9
    return {
        "n": int(n),
        "n_l2_safe": int(safe.sum()),
        "l2_safe_rate": float(safe.mean()),
        "mean_d2": float(d2.mean()),
        "mean_abs_a_inf": float(a_mag.mean()),
    }


def mf_actions(w, xs: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Batched numpy forward of MF residual deploy."""
    from stage_mf.encode import physics_gate_np

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
        if (i // batch) % 5 == 0:
            print(f"  mf fwd {min(i+batch, xs.shape[0])}/{xs.shape[0]}", flush=True)
    return np.concatenate(outs, axis=0)


def mappo_actions(actor, params, xs: np.ndarray, batch: int = 8192) -> np.ndarray:
    """MAPPO mean head on features; uref unavailable without graph → a=tanh(mean)."""

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
    ap.add_argument("--n", type=int, default=50000, help="samples drawn from near_plus box")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", type=str, default="baselines/compare/output/near_plus_sample")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    claim = default_mf_claims(mode="near_plus", radius=R)[0]
    box = claim.feature
    assert box.dim == INPUT_DIM
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print(f"[box] sampling n={args.n} from near_plus product box (dim={INPUT_DIM})…", flush=True)
    xs = box.sample(rng, args.n).astype(np.float64)
    print(f"[box] drawn in {time.time()-t0:.2f}s", flush=True)

    rows = []

    # coast / a=0 reference (physics envelope sanity)
    a0 = np.zeros((args.n, 2), dtype=np.float64)
    st = score_batch(xs, a0)
    st["policy"] = "coast_a0"
    st["note"] = "a=0 reference; envelope expects mostly safe in this box"
    rows.append(st)
    print(f"[score] coast_a0 L2-safe={100*st['l2_safe_rate']:.2f}%", flush=True)

    # MF-r2
    mf_ckpt = "method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl"
    w = load_mf_residual_weights(mf_ckpt)
    t1 = time.time()
    a_mf = mf_actions(w, xs)
    st = score_batch(xs, a_mf)
    st["policy"] = "mf_r2"
    st["note"] = "acceptance policy; formal PROVEN_FULL on this box"
    st["fwd_s"] = round(time.time() - t1, 2)
    rows.append(st)
    print(f"[score] mf_r2 L2-safe={100*st['l2_safe_rate']:.2f}% fwd={st['fwd_s']}s", flush=True)

    # MAPPO-Lag (feature MLP; no uref)
    mpath = pathlib.Path("baselines/mappo_lag/output/mappo_lag/ACTIVE.txt").read_text().strip()
    with open(mpath, "rb") as f:
        mck = pickle.load(f)
    assert int(mck.get("feature_version", FEATURE_VERSION)) == FEATURE_VERSION
    actor = GaussianActor(hidden=int(mck["hidden"]), action_dim=2)
    params = mck["actor_params"]
    t1 = time.time()
    a_mp = mappo_actions(actor, params, xs.astype(np.float32))
    st = score_batch(xs, a_mp)
    st["policy"] = "mappo_lag"
    st["note"] = "a=tanh(mean) only; uref omitted (needs graph). Approximate vs deploy."
    st["fwd_s"] = round(time.time() - t1, 2)
    rows.append(st)
    print(f"[score] mappo_lag L2-safe={100*st['l2_safe_rate']:.2f}% fwd={st['fwd_s']}s", flush=True)

    # Graph policies: cannot invert x→graph
    for name, reason in [
        ("ds_r2", "DeepSet needs ego/entity local sets (graph)"),
        ("gcbfplus", "GNN actor needs GraphsTuple"),
        ("handcbf", "CBF-QP needs GraphsTuple"),
    ]:
        rows.append({
            "policy": name,
            "n": 0,
            "n_l2_safe": 0,
            "l2_safe_rate": None,
            "mean_d2": None,
            "mean_abs_a_inf": None,
            "note": f"N/A under direct box sampling — {reason}",
        })

    wall = time.time() - t0
    report = {
        "claim": (
            "Empirical Monte-Carlo on the same near_plus product box as MF L2 acceptance. "
            "NOT a formal proof (formal proof already exists for mf_r2)."
        ),
        "method": "direct_uniform_sample_feature_box",
        "domain": {
            "mode": "near_plus",
            "min_d": [0.14, 0.40],
            "close_abs": 0.30,
            "inv_ttc_max": 8.0,
            "force_Q0_occupied": True,
            "R": R,
            "dim": INPUT_DIM,
        },
        "n": args.n,
        "seed": args.seed,
        "wall_time_s": round(wall, 2),
        "graph_policies": "N/A — cannot build GraphsTuple from abstract features",
        "rows": rows,
    }
    (out / "near_plus_l2_sample.json").write_text(json.dumps(report, indent=2) + "\n")

    md = [
        "# near_plus box · direct Monte-Carlo (empirical)",
        "",
        "> **Not formal.** Samples \(x\) uniformly from the **same 44-D product box** as MF-r2 L2 acceptance.",
        "> No env rollout. Graph policies (DS / GCBF+ / HandCBF) = **N/A** here.",
        "",
        f"- n={args.n} · seed={args.seed} · wall={wall:.1f}s · R={R}",
        "",
        "| policy | n | L2-safe% | mean d2 | mean ‖a‖∞ | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        rate = "—" if r["l2_safe_rate"] is None else f"{100*r['l2_safe_rate']:.2f}"
        d2 = "—" if r["mean_d2"] is None else f"{r['mean_d2']:.4f}"
        aa = "—" if r["mean_abs_a_inf"] is None else f"{r['mean_abs_a_inf']:.3f}"
        md.append(
            f"| {r['policy']} | {r['n']} | {rate} | {d2} | {aa} | {r.get('note','')} |"
        )
    md += [
        "",
        "## Why this matches acceptance better than rollout filtering",
        "",
        "Certificate quantifies ∀x∈ product box. Empty-quad `min_d≈0.5` makes real states miss the box;",
        "direct sampling evaluates policies **on the declared X**.",
        "",
        "MF-r2 already has **PROVEN_FULL** on this box; high empirical L2-safe% here is a sanity check,",
        "not a new proof.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(md))
    print(json.dumps(rows, indent=2))
    print(f"wrote {out}/REPORT.md in {wall:.1f}s")


if __name__ == "__main__":
    main()
