#!/usr/bin/env python3
"""Collect L2 CEs under a residual-gate policy (for CEGAR round-2+).

Same domain/label protocol as collect_counterexamples.py, but π is
a = clip(base + g·δ) instead of plain DeepSet.
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

import jax.numpy as jnp
import numpy as np



from stage2.bounds import default_l2_buckets
from stage2.l2_domain import (
    dist0,
    in_preventable_domain_x,
    l2_min_intervention_action_x,
)
from stage2.l2_query import eval_collision_dual, next_rels_for_spec, rels_from_x
from stage3.retrain_l2_safe import bucket_x_to_student_inputs
from stage3.residual_policy import apply_residual, load_residual_checkpoint


def collect_for_spec(
    action_fn,
    spec,
    *,
    n_tries: int,
    max_keep: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    ces = []
    n_draw = n_in_d = n_hit = 0
    sum_da = 0.0
    while n_draw < n_tries and len(ces) < max_keep:
        x = spec.box.sample(rng)
        n_draw += 1
        if not in_preventable_domain_x(x, spec, grid_n=7):
            continue
        n_in_d += 1
        a, g = action_fn(x, spec)
        ra0, ro0 = rels_from_x(x, spec)
        ran, ron = next_rels_for_spec(ra0, ro0, a, spec)
        ev = eval_collision_dual(ran, ron, spec)
        if not ev.collided:
            continue
        n_hit += 1
        a_safe, dist_safe, is_safe, da = l2_min_intervention_action_x(
            x, spec, a, grid_n=11, margin=0.0
        )
        sum_da += da
        entry = {
            "x": x.tolist(),
            "a": a.tolist(),
            "gate": float(g),
            "a_l2_safe": a_safe.tolist(),
            "a_l2_safe_dist": dist_safe,
            "a_l2_label_ok": bool(is_safe),
            "a_l2_label_mode": "min_intervention",
            "a_l2_label_submode": "min_intervention",
            "a_l2_delta": da,
            "distn": ev.dist,
            "threshold": ev.threshold,
            "kind": ev.kind,
        }
        if ra0 is not None:
            entry["rel0_aa"] = [ra0.dx, ra0.dy, ra0.dvx, ra0.dvy]
            entry["dist0_aa"] = dist0(ra0)
            entry["reln_aa"] = [ran.dx, ran.dy, ran.dvx, ran.dvy]
        if ro0 is not None:
            entry["rel0_ao"] = [ro0.dx, ro0.dy, ro0.dvx, ro0.dvy]
            entry["dist0_ao"] = dist0(ro0)
            entry["reln_ao"] = [ron.dx, ron.dy, ron.dvx, ron.dvy]
        if ra0 is not None and ro0 is None:
            entry["rel0"] = entry["rel0_aa"]
            entry["dist0"] = entry["dist0_aa"]
            entry["reln"] = entry["reln_aa"]
        elif ro0 is not None and ra0 is None:
            entry["rel0"] = entry["rel0_ao"]
            entry["dist0"] = entry["dist0_ao"]
            entry["reln"] = entry["reln_ao"]
        else:
            entry["dist0"] = min(entry.get("dist0_aa", 1e9), entry.get("dist0_ao", 1e9))
        ces.append(entry)

    dists = [c["distn"] for c in ces if c.get("distn") is not None]
    return {
        "bucket": spec.name,
        "k": spec.k,
        "m": spec.m,
        "property": spec.property,
        "domain": "preventable_D",
        "label_mode": "min_intervention",
        "n_draws": n_draw,
        "n_in_D": n_in_d,
        "n_hit_on_D": n_hit,
        "hit_rate_on_D": n_hit / max(n_in_d, 1),
        "n_kept": len(ces),
        "min_distn": min(dists) if dists else None,
        "mean_a_l2_delta": float(sum_da / max(len(ces), 1)),
        "n_labels_safe": sum(1 for c in ces if c["a_l2_label_ok"]),
        "counterexamples": ces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="residual-gate checkpoint")
    parser.add_argument("--tries", type=int, default=80000)
    parser.add_argument("--keep", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--buckets", default="l2_aa_near,l2_ao_near,l2_11_near")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt, modules, base_params, train_params = load_residual_checkpoint(args.checkpoint)
    gate_mode = str(ckpt.get("gate_mode", "hybrid"))
    open_mult = float(ckpt.get("open_mult", 2.5))
    close_mult = float(ckpt.get("close_mult", 1.0))

    def action_fn(x, spec):
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

    want = {b.strip() for b in args.buckets.split(",") if b.strip()}
    specs = [s for s in default_l2_buckets() if s.name in want]
    report = {
        "checkpoint": {
            "path": str(args.checkpoint),
            "architecture": "deepset_residual_gate",
            "base_checkpoint": ckpt.get("base_checkpoint"),
            "gate_mode": gate_mode,
        },
        "domain": "D preventable; π = residual gate on frozen base",
        "label_mode": "min_intervention",
        "tries_per_bucket": args.tries,
        "keep_per_bucket": args.keep,
        "seed": args.seed,
        "buckets": [],
    }
    for i, spec in enumerate(specs):
        print(
            f"collecting residual CE {spec.name} draws<={args.tries} keep={args.keep} ..."
        )
        block = collect_for_spec(
            action_fn,
            spec,
            n_tries=args.tries,
            max_keep=args.keep,
            seed=args.seed + i * 17,
        )
        report["buckets"].append(block)
        print(
            f"  in_D={block['n_in_D']}/{block['n_draws']} hits={block['n_hit_on_D']} "
            f"rate_on_D={block['hit_rate_on_D']*100:.2f}% kept={block['n_kept']} "
            f"mean_Δa={block['mean_a_l2_delta']:.4f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
