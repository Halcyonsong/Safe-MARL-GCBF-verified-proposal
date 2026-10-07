"""Stage2 first-version verification: L1 action bounds on DeepSet buckets.

Checks whether the frozen Stage1 DeepSet Actor satisfies
    a(x) ∈ [-1, 1]^2
for declared (k,m) presence modes and continuous input boxes.

Methods (in order per bucket):
  1. sampling   — empirical attack (not formal)
  2. IBP        — sound incomplete proof via interval bound propagation
  3. Marabou    — complete SAT/UNSAT on ONNX encoding (may timeout)

Usage:
  python stage2/verify_l1.py \
    --checkpoint method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl \
    --timeout 30
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

from stage2.bounds import BucketSpec, default_l1_buckets
from stage2.encode import load_weights, relu_count_for_bucket
from stage2.export_onnx import check_torch_matches_numpy
from stage2.ibp import ibp_from_flat_box
from stage2.marabou_query import (
    QueryResult,
    marabou_l1_on_bucket,
    random_attack,
    summarize_marabou_results,
)


def run_ibp(weights, spec: BucketSpec) -> QueryResult:
    t0 = time.time()
    lo, hi = ibp_from_flat_box(weights, spec.k, spec.m, spec.box.lo, spec.box.hi)
    lim = spec.action_limit
    elapsed = time.time() - t0
    if float(np_max_excess(lo, hi, lim)) <= 0.0:
        return QueryResult(
            spec.name,
            "ibp",
            "proven",
            "IBP output box inside [-1,1]^2 (sound)",
            elapsed,
            action_lo=lo.tolist(),
            action_hi=hi.tolist(),
        )
    return QueryResult(
        spec.name,
        "ibp",
        "unknown",
        "IBP inconclusive (over-approx exits [-1,1]^2); not a refutation",
        elapsed,
        action_lo=lo.tolist(),
        action_hi=hi.tolist(),
    )


def np_max_excess(lo, hi, lim: float) -> float:
    import numpy as np

    lo = np.asarray(lo)
    hi = np.asarray(hi)
    # excess above lim or below -lim
    return max(float(np.max(hi - lim)), float(np.max(-lim - lo)))


def verify_bucket(
    weights,
    spec: BucketSpec,
    *,
    onnx_dir: Path,
    timeout_s: int,
    n_samples: int,
    skip_marabou: bool,
) -> dict:
    print(f"\n=== {spec.describe()} ===")
    print(f"  notes: {spec.notes}")
    print(f"  approx ReLU neurons (unrolled φ): {relu_count_for_bucket(weights.hidden, spec.k, spec.m)}")

    # Sanity: numpy forward at box center
    try:
        from stage2.encode import forward_bucket, unpack_inputs

        ego, agents, obs = unpack_inputs(spec.box.center(), spec.k, spec.m)
        a0 = forward_bucket(weights, ego, agents, obs)
        print(f"  encode check: center action={a0.tolist()}")
        # Optional torch parity (used for ONNX path; not required for Marabou API)
        try:
            check_torch_matches_numpy(weights, spec.k, spec.m, spec.box.center())
            print("  encode check: torch == numpy at box center")
        except Exception as e:
            print(f"  torch parity skipped/failed (ok for Marabou API path): {e}")
    except Exception as e:
        print(f"  encode check FAILED: {e}")
        return {
            "bucket": spec.name,
            "k": spec.k,
            "m": spec.m,
            "verdict": "error",
            "results": [
                QueryResult(spec.name, "encode", "error", str(e), 0.0).as_dict()
            ],
        }

    results: list[QueryResult] = []

    # 1) sampling attack
    samp = random_attack(weights, spec, n_samples=n_samples)
    results.append(samp)
    print(f"  [sampling] {samp.status}: {samp.detail} ({samp.elapsed_s:.2f}s)")
    if samp.status == "violated":
        return _pack(spec, "violated", results)

    # 2) IBP
    ibp_res = run_ibp(weights, spec)
    results.append(ibp_res)
    print(
        f"  [ibp] {ibp_res.status}: {ibp_res.detail} "
        f"lo={ibp_res.action_lo} hi={ibp_res.action_hi} ({ibp_res.elapsed_s:.2f}s)"
    )
    if ibp_res.status == "proven":
        return _pack(spec, "proven_ibp", results)

    # 3) Marabou complete query
    if skip_marabou:
        results.append(
            QueryResult(spec.name, "marabou", "skipped", "skipped by flag", 0.0)
        )
        return _pack(spec, "unknown", results)

    print(f"  [marabou] running one-sided queries (timeout={timeout_s}s each)...")
    mres = marabou_l1_on_bucket(weights, spec, onnx_dir, timeout_s=timeout_s)
    for r in mres:
        print(f"    - {r.status}: {r.detail} ({r.elapsed_s:.2f}s)")
        results.append(r)
    summary = summarize_marabou_results(mres)
    print(f"  [marabou summary] {summary.status}: {summary.detail}")
    results.append(summary)
    verdict = {
        "proven": "proven_marabou",
        "violated": "violated",
        "timeout": "timeout",
        "error": "error",
        "unknown": "unknown",
    }.get(summary.status, summary.status)
    return _pack(spec, verdict, results)


def _pack(spec: BucketSpec, verdict: str, results: list[QueryResult]) -> dict:
    return {
        "bucket": spec.name,
        "k": spec.k,
        "m": spec.m,
        "property": spec.property,
        "action_limit": spec.action_limit,
        "input_dim": spec.input_dim(),
        "box": spec.box.as_dict(),
        "notes": spec.notes,
        "verdict": verdict,
        "results": [r.as_dict() for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2 L1 action-bound verification")
    parser.add_argument(
        "--checkpoint",
        default="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Marabou timeout per one-sided query (s)")
    parser.add_argument("--samples", type=int, default=2000, help="random attack samples per bucket")
    parser.add_argument("--skip-marabou", action="store_true")
    parser.add_argument(
        "--out",
        default="method/deepset/output/stage2/l1_action_bounds_report.json",
    )
    parser.add_argument(
        "--onnx-dir",
        default="method/deepset/output/stage2/onnx",
    )
    parser.add_argument(
        "--buckets",
        default="",
        help="comma-separated bucket names to run (default: all v1 buckets)",
    )
    args = parser.parse_args()

    weights, meta = load_weights(args.checkpoint)
    buckets = default_l1_buckets()
    if args.buckets.strip():
        want = {b.strip() for b in args.buckets.split(",") if b.strip()}
        buckets = [b for b in buckets if b.name in want]
        missing = want - {b.name for b in buckets}
        if missing:
            raise SystemExit(f"unknown buckets: {sorted(missing)}")

    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Stage2 L1 action-bound verification")
    print(f"  checkpoint: {meta['checkpoint']}")
    print(f"  hidden={meta['hidden']} entity_slots={meta['entity_slots']}")
    print(f"  buckets: {[b.name for b in buckets]}")
    print(
        "  NOTE: results are only for declared (k,m) boxes; "
        "not a global safety certificate."
    )

    report = {
        "property": "action_bounds_L1",
        "claim": "a(x) in [-1,1]^2 on each declared bucket box",
        "checkpoint": meta,
        "timeout_s_per_onesided": args.timeout,
        "samples_per_bucket": args.samples,
        "buckets": [],
        "summary": {},
    }

    t_all = time.time()
    for spec in buckets:
        entry = verify_bucket(
            weights,
            spec,
            onnx_dir=onnx_dir,
            timeout_s=args.timeout,
            n_samples=args.samples,
            skip_marabou=args.skip_marabou,
        )
        report["buckets"].append(entry)

    # Summary counts
    counts: dict[str, int] = {}
    for b in report["buckets"]:
        counts[b["verdict"]] = counts.get(b["verdict"], 0) + 1
    report["summary"] = {
        "verdict_counts": counts,
        "elapsed_s": time.time() - t_all,
        "interpretation": {
            "proven_ibp": "Sound IBP proof of action bounds on that box",
            "proven_marabou": "Complete Marabou proof of action bounds on that box",
            "violated": "Found counterexample exiting [-1,1]^2",
            "timeout": "Marabou incomplete; NOT a proof",
            "unknown": "No proof and no counterexample yet",
            "error": "Tooling/encoding failure",
        },
        "scope": (
            "Formal claims apply only to listed buckets and input boxes. "
            "Does not prove multi-agent collision freedom or infinite-horizon safety."
        ),
    }

    out_path.write_text(json.dumps(report, indent=2))
    print("\n==== SUMMARY ====")
    print(json.dumps(report["summary"], indent=2))
    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()
