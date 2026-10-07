"""Stage2 L2: local two-step collision verification on DeepSet (k,m) buckets.

Deployment contract:
    a = clip(DeepSet(x), -1, 1)
Two official Euler steps with held action (needed because one Euler step
does not move position with the current action).

Buckets: (1,0) AA neighbor-coast, (0,1) AO static hit. No (0,0).

Usage:
  python stage2/verify_l2.py --timeout 30 --samples 4000
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

from stage2.bounds import BucketSpec, default_l2_buckets
from stage2.encode import load_weights, relu_count_for_bucket
from stage2.export_onnx import check_torch_matches_numpy
from stage2.l2_query import l2_ibp, l2_sampling_attack, marabou_l2_square_crash
from stage2.marabou_query import QueryResult


def verify_bucket(
    weights,
    spec: BucketSpec,
    *,
    onnx_dir: Path,
    timeout_s: int,
    n_samples: int,
    skip_marabou: bool,
    continue_formal: bool = False,
) -> dict:
    print(f"\n=== {spec.describe()} ===")
    print(f"  notes: {spec.notes}")
    print(f"  approx ReLU neurons (unrolled φ): {relu_count_for_bucket(weights.hidden, spec.k, spec.m)}")

    try:
        from stage2.encode import clip_action, forward_bucket, unpack_inputs

        ego, agents, obs = unpack_inputs(spec.box.center(), spec.k, spec.m)
        a0 = clip_action(forward_bucket(weights, ego, agents, obs))
        print(f"  encode check: center clip(action)={a0.tolist()}")
        try:
            check_torch_matches_numpy(weights, spec.k, spec.m, spec.box.center(), with_clip=True)
            print("  encode check: torch clip == numpy clip at box center")
        except Exception as e:
            print(f"  torch parity skipped/failed: {e}")
    except Exception as e:
        print(f"  encode check FAILED: {e}")
        return {
            "bucket": spec.name,
            "k": spec.k,
            "m": spec.m,
            "verdict": "error",
            "results": [QueryResult(spec.name, "encode", "error", str(e), 0.0).as_dict()],
        }

    results: list[QueryResult] = []

    # Smoke boxes are tiny toolchain checks (may lie outside D); main buckets use D.
    use_D = not spec.name.endswith("_smoke")
    samp = l2_sampling_attack(
        weights, spec, n_samples=n_samples, domain_filter=use_D
    )
    results.append(samp)
    print(f"  [sampling] {samp.status}: {samp.detail} ({samp.elapsed_s:.2f}s)")
    if samp.status == "violated" and not continue_formal:
        return _pack(spec, "violated", results)

    ibp_res = l2_ibp(weights, spec)
    results.append(ibp_res)
    print(f"  [ibp] {ibp_res.status}: {ibp_res.detail} ({ibp_res.elapsed_s:.2f}s)")
    if ibp_res.status == "proven":
        if samp.status == "violated":
            return _pack(spec, "violated", results)
        packed = _pack(spec, "proven_ibp", results)
        if spec.name.endswith("_smoke") and not skip_marabou:
            print(f"  [marabou] forcing smoke query (timeout={timeout_s}s)...")
            mres = marabou_l2_square_crash(weights, spec, onnx_dir, timeout_s=timeout_s)
            print(f"  [marabou] {mres.status}: {mres.detail} ({mres.elapsed_s:.2f}s)")
            results.append(mres)
            packed = _pack(spec, "proven_ibp", results)
            packed["marabou_smoke"] = mres.status
        return packed

    if skip_marabou:
        results.append(QueryResult(spec.name, "marabou", "skipped", "skipped by flag", 0.0))
        return _pack(spec, "violated" if samp.status == "violated" else "unknown", results)

    print(f"  [marabou] L2 square crash query (timeout={timeout_s}s per threat)...")
    mres = marabou_l2_square_crash(weights, spec, onnx_dir, timeout_s=timeout_s)
    print(f"  [marabou] {mres.status}: {mres.detail} ({mres.elapsed_s:.2f}s)")
    results.append(mres)
    verdict = {
        "violated": "violated",
        "proven_square": "proven_marabou_square",
        "proven_outer": "proven_marabou_outer",
        "outer_counterexample": "unknown",
        "timeout": "timeout",
        "error": "error",
        "unknown": "unknown",
    }.get(mres.status, mres.status)
    if samp.status == "violated":
        verdict = "violated"
    return _pack(spec, verdict, results)


def _pack(spec: BucketSpec, verdict: str, results: list[QueryResult]) -> dict:
    return {
        "bucket": spec.name,
        "k": spec.k,
        "m": spec.m,
        "property": spec.property,
        "input_dim": spec.input_dim(),
        "box": spec.box.as_dict(),
        "notes": spec.notes,
        "horizon_euler_steps": 2,
        "claim_set": "preventable_D" if not spec.name.endswith("_smoke") else "smoke_outer",
        "verdict": verdict,
        "results": [r.as_dict() for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2 L2 two-step collision verification")
    parser.add_argument(
        "--checkpoint",
        default="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--skip-marabou", action="store_true")
    parser.add_argument(
        "--continue-formal",
        action="store_true",
        help="run IBP/Marabou even after sampling finds a counterexample",
    )
    parser.add_argument("--out", default="method/deepset/output/stage2/l2_collision_report.json")
    parser.add_argument("--onnx-dir", default="method/deepset/output/stage2/onnx")
    parser.add_argument("--buckets", default="", help="comma-separated bucket names")
    args = parser.parse_args()

    weights, meta = load_weights(args.checkpoint)
    buckets = default_l2_buckets()
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

    print("Stage2 L2 two-step collision verification")
    print(f"  checkpoint: {meta['checkpoint']}")
    print(f"  hidden={meta['hidden']}")
    print(f"  buckets: {[b.name for b in buckets]}")
    print("  NOTE: (0,0) omitted; claims only for declared near boxes + assumptions.")

    report = {
        "property": "l2_two_step_collision_on_D",
        "claim": (
            "On claim set D = {dist0>=r AND exists a that crashes in two steps "
            "AND exists a that stays safe}: after two Euler steps with "
            "a=clip(DeepSet(x)) held, relative pose does not enter the collision "
            "proxy (AA coast / AO static hit). Farther / hard-unavoidable / "
            "already-colliding states are excluded from this prevention claim."
        ),
        "checkpoint": meta,
        "timeout_s": args.timeout,
        "samples_per_bucket": args.samples,
        "assumptions": {
            "horizon_euler_steps": 2,
            "reason": "official agent_step_euler does not apply current accel to position in one step",
            "claim_set_D": "dist0>=r AND min_a dist'<r AND max_a dist'>=r",
            "outer_box": "axis-aligned ||rel||_inf <= 0.2 is a search envelope; sampling filters to D",
            "l2_aa": "neighbor acceleration = 0",
            "l2_ao": "lidar hit static; observed-hit ball radius=car_radius (not full rectangle)",
            "marabou_proxy": (
                "inscribed square on the outer box (not yet D-restricted in Marabou bounds); "
                "single-threat: inscribed square (UNSAT proves only square avoidance); "
                "dual: circumscribed squares (both UNSAT imply disk avoidance on outer box). "
                "SAT witnesses are replayed and filtered to D before claiming a D violation"
            ),
            "clip": "encoded as -1+ReLU(a+1)-ReLU(a-1) in ONNX",
        },
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
            continue_formal=args.continue_formal,
        )
        report["buckets"].append(entry)

    counts: dict[str, int] = {}
    for b in report["buckets"]:
        counts[b["verdict"]] = counts.get(b["verdict"], 0) + 1
    report["summary"] = {
        "verdict_counts": counts,
        "elapsed_s": time.time() - t_all,
        "interpretation": {
            "proven_ibp": "Sound IBP: next-rel box misses the collision ball",
            "proven_marabou_square": "UNSAT for inscribed-square crash only; NOT disk avoidance",
            "proven_marabou_outer": "Both outer-square queries UNSAT; both disks avoided under encoded dynamics",
            "violated": "Found two-step collision (sampling or Marabou SAT)",
            "timeout": "Marabou incomplete; NOT a proof",
            "unknown": "No proof and no counterexample yet",
            "error": "Tooling/encoding failure",
        },
        "scope": (
            "Only listed (k,m) boxes. Does not cover k>=2, m>=2, (0,0), "
            "full rectangle obstacles, arbitrary neighbor policies, or infinite horizon."
        ),
    }
    out_path.write_text(json.dumps(report, indent=2))
    print("\n==== SUMMARY ====")
    print(json.dumps(report["summary"], indent=2))
    print(f"report written: {out_path}")


if __name__ == "__main__":
    main()
