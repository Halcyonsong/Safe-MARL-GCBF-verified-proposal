"""Marabou L1 action-bound queries via ONNX encodings."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from maraboupy import Marabou

from stage2.bounds import BucketSpec
from stage2.encode import DeepSetWeights, forward_bucket, unpack_inputs
from stage2.export_onnx import export_bucket_onnx


@dataclass
class QueryResult:
    bucket: str
    method: str
    status: str
    detail: str
    elapsed_s: float
    action_lo: Optional[List[float]] = None
    action_hi: Optional[List[float]] = None
    counterexample: Optional[List[float]] = None
    counterexample_action: Optional[List[float]] = None

    next_rel: Optional[List[float]] = None
    next_dist: Optional[float] = None
    subqueries: Optional[List[dict]] = None

    def as_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "method": self.method,
            "status": self.status,
            "detail": self.detail,
            "elapsed_s": self.elapsed_s,
            "action_lo": self.action_lo,
            "action_hi": self.action_hi,
            "counterexample": self.counterexample,
            "counterexample_action": self.counterexample_action,
            "next_rel": self.next_rel,
            "next_dist": self.next_dist,
            "subqueries": self.subqueries,
        }


def _load_onnx_with_box(onnx_path: Path, box_lo: np.ndarray, box_hi: np.ndarray):
    net = Marabou.read_onnx(str(onnx_path))
    ins = net.inputVars[0].flatten()
    if len(ins) != len(box_lo):
        raise ValueError(f"ONNX input dim {len(ins)} != box dim {len(box_lo)}")
    for v, lo, hi in zip(ins, box_lo, box_hi):
        net.setLowerBound(int(v), float(lo))
        net.setUpperBound(int(v), float(hi))
    # Finite bounds on auxiliaries help some Marabou builds.
    big = 1e4
    for v in range(net.numVars):
        if not net.lowerBoundExists(v):
            net.setLowerBound(v, -big)
        if not net.upperBoundExists(v):
            net.setUpperBound(v, big)
    outs = net.outputVars[0].flatten().tolist()
    return net, [int(v) for v in ins], [int(v) for v in outs]


def _solve(net, timeout_s: int):
    # verbosity=0 keeps Marabou from flooding stdout (still honors timeout).
    options = Marabou.createOptions(timeoutInSeconds=int(timeout_s), verbosity=0)
    ret = net.solve(options=options, verbose=False)
    exit_code, vals = ret[0], ret[1]
    stats = ret[2] if len(ret) > 2 else None
    return str(exit_code), vals, stats


def marabou_action_bound_query(
    onnx_path: Path,
    spec: BucketSpec,
    *,
    timeout_s: int = 60,
    channel: int = 0,
    exceed_upper: bool = True,
    weights: Optional[DeepSetWeights] = None,
) -> QueryResult:
    t0 = time.time()
    lim = float(spec.action_limit)
    eps = 1e-4
    try:
        net, in_vars, out_vars = _load_onnx_with_box(onnx_path, spec.box.lo, spec.box.hi)
        y = out_vars[channel]
        if exceed_upper:
            net.setLowerBound(y, lim + eps)
            side = f"a[{channel}] >= {lim + eps}"
        else:
            net.setUpperBound(y, -lim - eps)
            side = f"a[{channel}] <= {-lim - eps}"

        exit_code, vals, _stats = _solve(net, timeout_s)
        elapsed = time.time() - t0
        code = exit_code.lower()

        if code == "sat":
            x = [float(vals[v]) for v in in_vars]
            a_m = [float(vals[v]) for v in out_vars]
            a_np = None
            if weights is not None:
                ego, agents, obs = unpack_inputs(np.asarray(x), spec.k, spec.m)
                a_np = forward_bucket(weights, ego, agents, obs).tolist()
            return QueryResult(
                spec.name,
                "marabou",
                "violated",
                f"SAT for negation ({side}); marabou_a={a_m}; numpy_a={a_np}",
                elapsed,
                counterexample=x,
                counterexample_action=a_np if a_np is not None else a_m,
            )
        if code == "unsat":
            return QueryResult(
                spec.name, "marabou", "proven", f"UNSAT for negation ({side})", elapsed
            )
        if code == "timeout":
            return QueryResult(
                spec.name, "marabou", "timeout", f"TIMEOUT on ({side})", elapsed
            )
        return QueryResult(
            spec.name, "marabou", "unknown", f"exit={exit_code} on ({side})", elapsed
        )
    except Exception as e:
        return QueryResult(
            spec.name,
            "marabou",
            "error",
            f"{type(e).__name__}: {e}",
            time.time() - t0,
        )


def marabou_l1_on_bucket(
    weights: DeepSetWeights,
    spec: BucketSpec,
    onnx_dir: Path,
    *,
    timeout_s: int = 60,
) -> List[QueryResult]:
    onnx_path = onnx_dir / f"{spec.name}_k{spec.k}_m{spec.m}.onnx"
    export_bucket_onnx(weights, spec.k, spec.m, onnx_path)
    results: List[QueryResult] = []
    for ch, up in [(0, True), (0, False), (1, True), (1, False)]:
        r = marabou_action_bound_query(
            onnx_path,
            spec,
            timeout_s=timeout_s,
            channel=ch,
            exceed_upper=up,
            weights=weights,
        )
        r.detail = f"ch={ch} upper={up}: " + r.detail
        results.append(r)
        if r.status in ("violated", "error", "timeout"):
            # Keep going on timeout only for first side if we want completeness;
            # for v1 stop early on timeout to save wall time.
            if r.status != "timeout":
                break
            # still continue other sides? for v1 break to avoid 4x cost
            break
    return results


def summarize_marabou_results(results: Sequence[QueryResult]) -> QueryResult:
    if not results:
        return QueryResult("?", "marabou", "unknown", "no results", 0.0)
    bucket = results[0].bucket
    elapsed = sum(r.elapsed_s for r in results)
    if any(r.status == "violated" for r in results):
        bad = next(r for r in results if r.status == "violated")
        return QueryResult(
            bucket,
            "marabou",
            "violated",
            f"action bounds violated; {bad.detail}",
            elapsed,
            counterexample=bad.counterexample,
            counterexample_action=bad.counterexample_action,
        )
    if any(r.status == "error" for r in results):
        err = next(r for r in results if r.status == "error")
        return QueryResult(bucket, "marabou", "error", err.detail, elapsed)
    if any(r.status in ("timeout", "unknown") for r in results):
        return QueryResult(
            bucket,
            "marabou",
            "timeout",
            "incomplete: some one-sided queries timeout/unknown; NOT a proof",
            elapsed,
        )
    if all(r.status == "proven" for r in results):
        return QueryResult(
            bucket,
            "marabou",
            "proven",
            "all one-sided negation queries UNSAT => a in [-1,1]^2 on this box",
            elapsed,
        )
    return QueryResult(
        bucket, "marabou", "unknown", str([r.status for r in results]), elapsed
    )


def random_attack(
    weights: DeepSetWeights,
    spec: BucketSpec,
    *,
    n_samples: int = 2000,
    seed: int = 0,
) -> QueryResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    lim = spec.action_limit
    worst = None
    worst_score = -1.0
    for _ in range(n_samples):
        x = spec.box.sample(rng)
        ego, agents, obs = unpack_inputs(x, spec.k, spec.m)
        a = forward_bucket(weights, ego, agents, obs)
        score = float(np.max(np.abs(a) - lim))
        if score > worst_score:
            worst_score = score
            worst = (x.tolist(), a.tolist())
        if score > 1e-6:
            return QueryResult(
                spec.name,
                "sampling",
                "violated",
                f"sample |a|_max exceeded by {score:.4f}",
                time.time() - t0,
                counterexample=worst[0],
                counterexample_action=worst[1],
            )
    return QueryResult(
        spec.name,
        "sampling",
        "unknown",
        f"no violation in {n_samples} samples (not a proof); worst_excess={worst_score:.4e}",
        time.time() - t0,
        counterexample_action=None if worst is None else worst[1],
    )
