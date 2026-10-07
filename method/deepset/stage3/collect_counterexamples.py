"""Stage3: sample L2 two-step collision CEs from preventable claim set D.

Supports (1,0), (0,1), and (1,1) dual-threat buckets. Uses Stage2 domain /
query helpers; writes CE json under ``method/deepset/output/stage3/ce/``.
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

import numpy as np

from stage2.bounds import default_l2_buckets
from stage2.encode import load_weights
from stage2.l2_domain import (
    dist0,
    in_preventable_domain_x,
    l2_hybrid_safe_action_x,
    l2_min_intervention_action_x,
    l2_safe_action_x,
)
from stage2.l2_query import (
    action_at,
    eval_collision_dual,
    next_rels_for_spec,
    rels_from_x,
)


def collect_for_spec(
    weights,
    spec,
    *,
    n_tries: int,
    max_keep: int,
    seed: int,
    label_mode: str = "min_intervention",
    hybrid_lam: float = 0.5,
    hybrid_mu: float = 0.15,
    hybrid_near: float = 1.5,
) -> dict:
    rng = np.random.default_rng(seed)
    ces = []
    n_draw = 0
    n_in_d = 0
    n_hit = 0
    sum_da = 0.0
    submode_counts: dict[str, int] = {}
    while n_draw < n_tries and len(ces) < max_keep:
        x = spec.box.sample(rng)
        n_draw += 1
        if not in_preventable_domain_x(x, spec, grid_n=7):
            continue
        n_in_d += 1
        a = action_at(weights, x, spec)
        rel_aa0, rel_ao0 = rels_from_x(x, spec)
        rel_aan, rel_aon = next_rels_for_spec(rel_aa0, rel_ao0, a, spec)
        ev = eval_collision_dual(rel_aan, rel_aon, spec)
        if not ev.collided:
            continue
        n_hit += 1
        submode = label_mode
        if label_mode == "max_clearance":
            a_safe, dist_safe, is_safe = l2_safe_action_x(x, spec, grid_n=11, margin=0.0)
            da = float(np.linalg.norm(a_safe - a))
        elif label_mode == "hybrid":
            a_safe, dist_safe, is_safe, da, submode = l2_hybrid_safe_action_x(
                x,
                spec,
                a,
                lam=hybrid_lam,
                mu=hybrid_mu,
                near_clearance=hybrid_near,
                grid_n=11,
                margin=0.0,
            )
        else:
            a_safe, dist_safe, is_safe, da = l2_min_intervention_action_x(
                x, spec, a, grid_n=11, margin=0.0
            )
        sum_da += da
        submode_counts[submode] = submode_counts.get(submode, 0) + 1
        entry = {
            "x": x.tolist(),
            "a": a.tolist(),
            "a_l2_safe": a_safe.tolist(),
            "a_l2_safe_dist": dist_safe,
            "a_l2_label_ok": bool(is_safe),
            "a_l2_label_mode": label_mode,
            "a_l2_label_submode": submode,
            "a_l2_delta": da,
            "distn": ev.dist,
            "threshold": ev.threshold,
            "kind": ev.kind,
        }
        if rel_aa0 is not None:
            entry["rel0_aa"] = [rel_aa0.dx, rel_aa0.dy, rel_aa0.dvx, rel_aa0.dvy]
            entry["dist0_aa"] = dist0(rel_aa0)
            entry["reln_aa"] = [rel_aan.dx, rel_aan.dy, rel_aan.dvx, rel_aan.dvy]
        if rel_ao0 is not None:
            entry["rel0_ao"] = [rel_ao0.dx, rel_ao0.dy, rel_ao0.dvx, rel_ao0.dvy]
            entry["dist0_ao"] = dist0(rel_ao0)
            entry["reln_ao"] = [rel_aon.dx, rel_aon.dy, rel_aon.dvx, rel_aon.dvy]
        # legacy single fields for older loaders
        if rel_aa0 is not None and rel_ao0 is None:
            entry["rel0"] = entry["rel0_aa"]
            entry["dist0"] = entry["dist0_aa"]
            entry["reln"] = entry["reln_aa"]
        elif rel_ao0 is not None and rel_aa0 is None:
            entry["rel0"] = entry["rel0_ao"]
            entry["dist0"] = entry["dist0_ao"]
            entry["reln"] = entry["reln_ao"]
        else:
            entry["dist0"] = min(entry.get("dist0_aa", 1e9), entry.get("dist0_ao", 1e9))
        ces.append(entry)

    dists = [c["distn"] for c in ces]
    deltas = [c["a_l2_delta"] for c in ces]
    return {
        "bucket": spec.name,
        "k": spec.k,
        "m": spec.m,
        "property": spec.property,
        "domain": "preventable_D",
        "label_mode": label_mode,
        "n_draws": n_draw,
        "n_in_D": n_in_d,
        "n_hit_on_D": n_hit,
        "hit_rate_on_D": n_hit / max(n_in_d, 1),
        "n_kept": len(ces),
        "min_distn": min(dists) if dists else None,
        "median_distn": float(np.median(dists)) if dists else None,
        "mean_a_l2_delta": float(np.mean(deltas)) if deltas else None,
        "n_labels_safe": sum(1 for c in ces if c["a_l2_label_ok"]),
        "submode_counts": submode_counts,
        "hybrid_lam": hybrid_lam if label_mode == "hybrid" else None,
        "hybrid_mu": hybrid_mu if label_mode == "hybrid" else None,
        "hybrid_near": hybrid_near if label_mode == "hybrid" else None,
        "counterexamples": ces,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl",
    )
    parser.add_argument("--tries", type=int, default=100000)
    parser.add_argument("--keep", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="method/deepset/output/stage3/ce/l2_counterexamples.json")
    parser.add_argument("--buckets", default="l2_aa_near,l2_ao_near")
    parser.add_argument(
        "--label-mode",
        default="min_intervention",
        choices=["min_intervention", "max_clearance", "hybrid"],
        help=(
            "CE label: min Δa (default), max clearance, or hybrid "
            "(near→clearance; else goal-aware + frozen anchor; never crashing a)"
        ),
    )
    parser.add_argument(
        "--hybrid-lam",
        type=float,
        default=0.5,
        help="hybrid mid/far: weight on clearance vs goal_progress",
    )
    parser.add_argument(
        "--hybrid-mu",
        type=float,
        default=0.15,
        help="hybrid mid/far: penalty on ||a - π(x)||",
    )
    parser.add_argument(
        "--hybrid-near",
        type=float,
        default=1.5,
        help="hybrid: initial clearance_norm below this → near_max_clearance",
    )
    args = parser.parse_args()

    weights, meta = load_weights(args.checkpoint)
    want = {b.strip() for b in args.buckets.split(",") if b.strip()}
    specs = [s for s in default_l2_buckets() if s.name in want]
    if not specs:
        raise SystemExit(f"no matching buckets in {want}")

    report = {
        "checkpoint": meta,
        "domain": (
            "D = initially safe AND exists crashing a AND exists avoiding a "
            "(dual for l2_aa_ao). Labels a_l2_safe: min-intervention / "
            "max_clearance / hybrid (safe-set only; no Zone-B)."
        ),
        "label_mode": args.label_mode,
        "hybrid_lam": args.hybrid_lam if args.label_mode == "hybrid" else None,
        "hybrid_mu": args.hybrid_mu if args.label_mode == "hybrid" else None,
        "hybrid_near": args.hybrid_near if args.label_mode == "hybrid" else None,
        "tries_per_bucket": args.tries,
        "keep_per_bucket": args.keep,
        "seed": args.seed,
        "buckets": [],
    }
    for i, spec in enumerate(specs):
        print(
            f"collecting {spec.name} draws<={args.tries} keep={args.keep} on D "
            f"label={args.label_mode}"
            + (
                f" lam={args.hybrid_lam} mu={args.hybrid_mu} near={args.hybrid_near}"
                if args.label_mode == "hybrid"
                else ""
            )
            + "..."
        )
        entry = collect_for_spec(
            weights,
            spec,
            n_tries=args.tries,
            max_keep=args.keep,
            seed=args.seed + 17 * i,
            label_mode=args.label_mode,
            hybrid_lam=args.hybrid_lam,
            hybrid_mu=args.hybrid_mu,
            hybrid_near=args.hybrid_near,
        )
        print(
            f"  in_D={entry['n_in_D']}/{entry['n_draws']} "
            f"hits={entry['n_hit_on_D']} rate_on_D={entry['hit_rate_on_D']*100:.2f}% "
            f"kept={entry['n_kept']} safe_labels={entry['n_labels_safe']} "
            f"mean_Δa={entry['mean_a_l2_delta']} min_distn={entry['min_distn']} "
            f"submodes={entry.get('submode_counts')}"
        )
        report["buckets"].append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
