#!/usr/bin/env python3
"""Post-repair L2 claim diagnosis for residual-gate policies (sampling on D).

Same claim as stage2/verify_l2 sampling leg:
  preventable-D, two Euler steps, AA/AO/(1,1) near buckets.

This is **not** a Marabou/IBP full proof of the residual graph (ONNX path is
base-DeepSet only today). It is a dense sampling attack + hit-rate report so we
know whether the declared two-step claim still has concrete CEs after repair.

Verdict per bucket:
  violated       — found ≥1 in-D two-step collision under residual π
  sampling_clear — no collision in n_samples in-D draws (NOT formal proven)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

import numpy as np



from stage2.bounds import default_l2_buckets
from stage2.l2_domain import in_preventable_domain_x
from stage2.l2_query import eval_collision_dual, next_rels_for_spec, rels_from_x
from stage3.retrain_l2_safe import bucket_x_to_student_inputs
from stage3.residual_policy import apply_residual, load_residual_checkpoint
import jax.numpy as jnp


def _action_fn(modules, base_params, train_params, gate_mode, open_mult, close_mult):
    def fn(x, spec):
        ego_np, ent_np = bucket_x_to_student_inputs(np.asarray(x), spec.k, spec.m)
        a, g, _ = apply_residual(
            modules,
            base_params,
            train_params,
            jnp.asarray(ego_np[None, ...]),
            jnp.asarray(ent_np[None, ...]),
            gate_mode=gate_mode,
            open_mult=open_mult,
            close_mult=close_mult,
        )
        return np.asarray(a[0], dtype=np.float64), float(np.asarray(g[0]))

    return fn


def attack_bucket(action_fn, spec, *, n_samples: int, seed: int) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n_eval = n_draw = n_skip = n_hit = 0
    first_ce = None
    min_dist = 1e9
    gate_sum = 0.0
    budget = max(n_samples * 40, 20000)
    while n_eval < n_samples and n_draw < budget:
        x = spec.box.sample(rng)
        n_draw += 1
        if not in_preventable_domain_x(x, spec, grid_n=7):
            n_skip += 1
            continue
        n_eval += 1
        a, g = action_fn(x, spec)
        gate_sum += g
        ra0, ro0 = rels_from_x(x, spec)
        ran, ron = next_rels_for_spec(ra0, ro0, a, spec)
        ev = eval_collision_dual(ran, ron, spec)
        if ev.dist is not None:
            min_dist = min(min_dist, float(ev.dist))
        if ev.collided:
            n_hit += 1
            if first_ce is None:
                first_ce = {
                    "x": x.tolist(),
                    "a": a.tolist(),
                    "gate": g,
                    "distn": float(ev.dist) if ev.dist is not None else None,
                    "kind": ev.kind,
                    "threshold": float(ev.threshold) if ev.threshold is not None else None,
                }
            # keep scanning for hit_rate; still count full n_samples
    hit_rate = n_hit / max(n_eval, 1)
    verdict = "violated" if n_hit > 0 else "sampling_clear"
    return {
        "bucket": spec.name,
        "k": spec.k,
        "m": spec.m,
        "verdict": verdict,
        "method": "sampling_on_D",
        "n_eval_on_D": n_eval,
        "n_draw": n_draw,
        "n_skip_out_of_D": n_skip,
        "n_hit": n_hit,
        "hit_rate_on_D": hit_rate,
        "min_distn": None if min_dist >= 1e8 else float(min_dist),
        "mean_gate_on_D": gate_sum / max(n_eval, 1),
        "first_counterexample": first_ce,
        "elapsed_s": round(time.time() - t0, 3),
        "note": (
            "sampling_clear ≠ formal proven; only no CE in this sample budget. "
            "violated = concrete two-step collision on D under residual π."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="residual-gate pkl")
    parser.add_argument("--buckets", default="l2_aa_near,l2_ao_near,l2_11_near")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt, modules, base_params, train_params = load_residual_checkpoint(args.checkpoint)
    gate_mode = str(ckpt.get("gate_mode", "hybrid"))
    open_mult = float(ckpt.get("open_mult", 2.5))
    close_mult = float(ckpt.get("close_mult", 1.0))
    action_fn = _action_fn(modules, base_params, train_params, gate_mode, open_mult, close_mult)

    want = {b.strip() for b in args.buckets.split(",") if b.strip()}
    specs = [s for s in default_l2_buckets() if s.name in want]
    if not specs:
        raise SystemExit(f"no buckets in {want}")

    print(f"[verify-residual] ckpt={args.checkpoint}")
    print(f"  base={ckpt.get('base_checkpoint')} gate_mode={gate_mode}")
    print(f"  samples/bucket={args.samples} seed={args.seed}")
    print(
        "  claim: two-step collision on preventable-D (AA coast / AO static). "
        "Sampling diagnosis only — not Marabou proven."
    )

    buckets = []
    counts: dict[str, int] = {}
    for i, spec in enumerate(specs):
        print(f"\n=== {spec.name} k={spec.k} m={spec.m} ===")
        row = attack_bucket(action_fn, spec, n_samples=args.samples, seed=args.seed + i * 97)
        buckets.append(row)
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        print(
            f"  [{row['verdict']}] hits={row['n_hit']}/{row['n_eval_on_D']} "
            f"rate={row['hit_rate_on_D']*100:.2f}% min_dist={row['min_distn']} "
            f"gate={row['mean_gate_on_D']:.3f} ({row['elapsed_s']}s)"
        )

    report = {
        "property": "l2_two_step_collision_on_D_residual_sampling",
        "claim": (
            "On preventable-D: after two Euler steps with a=clip(base+g·δ) held, "
            "relative pose does not enter collision proxy. Dense sampling diagnosis; "
            "NOT IBP/Marabou complete proof of residual ONNX."
        ),
        "checkpoint": str(args.checkpoint),
        "base_checkpoint": ckpt.get("base_checkpoint"),
        "architecture": "deepset_residual_gate",
        "gate_mode": gate_mode,
        "samples_per_bucket": args.samples,
        "seed": args.seed,
        "buckets": buckets,
        "summary": {
            "verdict_counts": counts,
            "all_sampling_clear": counts.get("violated", 0) == 0,
            "interpretation": {
                "violated": "Found in-D two-step collision under residual — claim false on sample",
                "sampling_clear": "No CE in budget — necessary but not sufficient for proven",
            },
            "scope": (
                "Only listed (k,m) near boxes + preventable-D + two steps. "
                "Does not cover k>=2, long horizon, or Marabou encoding of residual."
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n==== SUMMARY verdict_counts={counts} ====")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
