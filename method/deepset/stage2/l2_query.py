"""L2 collision queries: sampling, IBP next-rel box, Marabou SAT on a sufficient crash proxy.

Collision (Euclidean) is nonconvex. Marabou v1 uses a **sufficient** crash
condition: after two Euler steps, relative position falls into the axis-aligned
square |dx'| < r/√2 and |dy'| < r/√2, which implies ||Δp'|| < r.
SAT on that query is a true collision (under L2 dynamics/assumptions).
UNSAT only proves the square is avoided, not the full disk.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from maraboupy import Marabou
from maraboupy.MarabouUtils import Equation

from stage2.bounds import BucketSpec
from stage2.collision import eval_aa, eval_aa_ao, eval_ao
from stage2.dynamics import (
    COLLISION_AA,
    COLLISION_AO,
    DT,
    HORIZON_STEPS,
    MASS,
    RelState,
    next_rel_aa_coast,
    next_rel_ao_static,
)
from stage2.encode import DeepSetWeights, clip_action, forward_bucket, unpack_inputs
from stage2.export_onnx import export_bucket_onnx
from stage2.ibp import ibp_clipped_action_from_flat_box
from stage2.marabou_query import QueryResult, _load_onnx_with_box, _solve


def rel_from_x(x: np.ndarray, spec: BucketSpec) -> RelState:
    """Primary relative state (AA if present, else AO). For (1,1) prefer AA."""
    _, agents, obs = unpack_inputs(x, spec.k, spec.m)
    if spec.k >= 1:
        cont = agents[0]
    else:
        cont = obs[0]
    return RelState(float(cont[0]), float(cont[1]), float(cont[2]), float(cont[3]))


def rels_from_x(x: np.ndarray, spec: BucketSpec) -> Tuple[RelState | None, RelState | None]:
    """Return (rel_aa, rel_ao); either may be None depending on (k,m)."""
    _, agents, obs = unpack_inputs(x, spec.k, spec.m)
    rel_aa = None
    rel_ao = None
    if spec.k >= 1:
        c = agents[0]
        rel_aa = RelState(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    if spec.m >= 1:
        c = obs[0]
        rel_ao = RelState(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    return rel_aa, rel_ao


def action_at(weights: DeepSetWeights, x: np.ndarray, spec: BucketSpec) -> np.ndarray:
    ego, agents, obs = unpack_inputs(x, spec.k, spec.m)
    return clip_action(forward_bucket(weights, ego, agents, obs))


def next_rel_for_spec(rel: RelState, a: np.ndarray, spec: BucketSpec) -> RelState:
    if spec.property == "l2_aa":
        return next_rel_aa_coast(rel, a, n=HORIZON_STEPS)
    if spec.property == "l2_ao":
        return next_rel_ao_static(rel, a, n=HORIZON_STEPS)
    raise ValueError(f"unsupported single-rel property {spec.property}")


def next_rels_for_spec(
    rel_aa: RelState | None,
    rel_ao: RelState | None,
    a: np.ndarray,
    spec: BucketSpec,
) -> Tuple[RelState | None, RelState | None]:
    na = next_rel_aa_coast(rel_aa, a, n=HORIZON_STEPS) if rel_aa is not None else None
    no = next_rel_ao_static(rel_ao, a, n=HORIZON_STEPS) if rel_ao is not None else None
    return na, no


def eval_collision(rel: RelState, spec: BucketSpec):
    if spec.property == "l2_aa":
        return eval_aa(rel)
    if spec.property == "l2_ao":
        return eval_ao(rel)
    raise ValueError(spec.property)


def eval_collision_dual(
    rel_aa: RelState | None,
    rel_ao: RelState | None,
    spec: BucketSpec,
):
    if spec.property == "l2_aa":
        assert rel_aa is not None
        return eval_aa(rel_aa)
    if spec.property == "l2_ao":
        assert rel_ao is not None
        return eval_ao(rel_ao)
    if spec.property == "l2_aa_ao":
        assert rel_aa is not None and rel_ao is not None
        return eval_aa_ao(rel_aa, rel_ao)
    raise ValueError(spec.property)


def threshold_for(spec: BucketSpec) -> float:
    if spec.property == "l2_aa":
        return COLLISION_AA
    if spec.property == "l2_ao":
        return COLLISION_AO
    if spec.property == "l2_aa_ao":
        # dual: use AA radius only for legacy single-thr displays; dual eval has both
        return COLLISION_AA
    raise ValueError(spec.property)


def l2_sampling_attack(
    weights: DeepSetWeights,
    spec: BucketSpec,
    *,
    n_samples: int = 4000,
    seed: int = 0,
    domain_filter: bool = True,
    max_draw: int | None = None,
) -> QueryResult:
    """Sample for two-step collisions under π.

    When ``domain_filter`` is True (default), only evaluate states in the
    preventable claim set D: initially safe AND some held action crashes in
    two steps. Outer-box samples outside D are skipped.
    """
    from stage2.l2_domain import in_preventable_domain_x

    t0 = time.time()
    rng = np.random.default_rng(seed)
    worst = None
    worst_score = -1.0
    n_eval = 0
    n_draw = 0
    n_skip = 0
    budget = max_draw if max_draw is not None else max(n_samples * 40, 20000)
    while n_eval < n_samples and n_draw < budget:
        x = spec.box.sample(rng)
        n_draw += 1
        if domain_filter and not in_preventable_domain_x(x, spec, grid_n=7):
            n_skip += 1
            continue
        n_eval += 1
        a = action_at(weights, x, spec)
        rel_aa0, rel_ao0 = rels_from_x(x, spec)
        rel_aan, rel_aon = next_rels_for_spec(rel_aa0, rel_ao0, a, spec)
        ev = eval_collision_dual(rel_aan, rel_aon, spec)
        # score: how deep into collision (normalized); higher worse
        score = 1.0 - (
            min(
                (np.linalg.norm(rel_aan.d_pos) / COLLISION_AA) if rel_aan is not None else 1e9,
                (np.linalg.norm(rel_aon.d_pos) / COLLISION_AO) if rel_aon is not None else 1e9,
            )
        )
        primary = rel_aan if rel_aan is not None else rel_aon
        if score > worst_score:
            worst_score = score
            worst = (x.tolist(), a.tolist(), primary, ev.dist)
        if ev.collided:
            return QueryResult(
                spec.name,
                "sampling",
                "violated",
                (
                    f"two-step {spec.property} collision on D: dist'={ev.dist:.4f} "
                    f"thr~{ev.threshold} (evaluated {n_eval}/{n_samples} in-domain, "
                    f"skipped {n_skip})"
                ),
                time.time() - t0,
                counterexample=worst[0],
                counterexample_action=worst[1],
                next_rel=[
                    v for rel in (rel_aan, rel_aon) if rel is not None
                    for v in (rel.dx, rel.dy, rel.dvx, rel.dvy)
                ],
                next_dist=ev.dist,
            )
    detail = (
        f"no collision in {n_eval} in-domain samples "
        f"(draws={n_draw}, skipped_out_of_D={n_skip}; not a proof); "
        f"min_dist={worst[3] if worst else None}"
    )
    if n_eval == 0:
        return QueryResult(
            spec.name,
            "sampling",
            "unknown",
            f"could not sample claim set D (draws={n_draw}); outer box may be too tight",
            time.time() - t0,
        )
    return QueryResult(
        spec.name,
        "sampling",
        "unknown",
        detail,
        time.time() - t0,
        counterexample_action=None if worst is None else worst[1],
        next_dist=None if worst is None else worst[3],
    )


def _interval_next_rel(
    dpos_lo: np.ndarray,
    dpos_hi: np.ndarray,
    dvel_lo: np.ndarray,
    dvel_hi: np.ndarray,
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    *,
    n: int = HORIZON_STEPS,
    other_static: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """IBP of relative position after n Euler steps, neighbor accel=0 or static hit.

    d_p' = d_p + d_v dt
    d_v' = d_v + (a_ego / m) dt     (other accel 0)
    """
    acc_lo = a_lo / MASS
    acc_hi = a_hi / MASS
    p_lo, p_hi = np.array(dpos_lo, dtype=np.float64), np.array(dpos_hi, dtype=np.float64)
    v_lo, v_hi = np.array(dvel_lo, dtype=np.float64), np.array(dvel_hi, dtype=np.float64)
    for _ in range(n):
        p_lo = p_lo + v_lo * DT
        p_hi = p_hi + v_hi * DT
        v_lo = v_lo + acc_lo * DT
        v_hi = v_hi + acc_hi * DT
    return p_lo, p_hi


def l2_ibp(weights: DeepSetWeights, spec: BucketSpec) -> QueryResult:
    t0 = time.time()
    a_lo, a_hi = ibp_clipped_action_from_flat_box(
        weights, spec.k, spec.m, spec.box.lo, spec.box.hi
    )
    # For (1,1), independently over-approximate the two threat outputs using
    # the same action interval.  This is sound for proving avoidance of either
    # threat, although it loses correlation between the shared action channels.
    threat_specs = (
        [("aa", COLLISION_AA), ("ao", COLLISION_AO)]
        if spec.property == "l2_aa_ao"
        else [(spec.property.removeprefix("l2_"), threshold_for(spec))]
    )
    threat_results = []
    for threat, thr in threat_specs:
        off = 8 if threat == "aa" else 8 + 4 * spec.k
        d_lo = spec.box.lo[off : off + 4]
        d_hi = spec.box.hi[off : off + 4]
        p_lo, p_hi = _interval_next_rel(
            d_lo[:2],
            d_hi[:2],
            d_lo[2:],
            d_hi[2:],
            a_lo,
            a_hi,
            n=HORIZON_STEPS,
            other_static=threat == "ao",
        )
        min_abs = np.maximum(0.0, np.maximum(p_lo, -p_hi))
        min_dist = float(np.linalg.norm(min_abs))
        threat_results.append((threat, thr, min_dist, p_lo, p_hi))

    if spec.property == "l2_aa_ao":
        detail = "; ".join(
            f"{threat}: min_dist={min_dist:.4f} {'outside' if min_dist >= thr else 'intersects'} radius {thr}"
            for threat, thr, min_dist, _lo, _hi in threat_results
        )
        elapsed = time.time() - t0
        if all(min_dist >= thr for _threat, thr, min_dist, _lo, _hi in threat_results):
            return QueryResult(
                spec.name,
                "ibp",
                "proven",
                f"dual IBP next-rel boxes avoid both collision balls: {detail}",
                elapsed,
                action_lo=a_lo.tolist(),
                action_hi=a_hi.tolist(),
                next_rel=[v for _threat, _thr, _d, lo, hi in threat_results for v in (lo.tolist() + hi.tolist())],
                next_dist=min(d for _t, _thr, d, _lo, _hi in threat_results),
            )
        return QueryResult(
            spec.name,
            "ibp",
            "unknown",
            f"dual IBP intersects at least one collision ball: {detail}; not a refutation",
            elapsed,
            action_lo=a_lo.tolist(),
            action_hi=a_hi.tolist(),
            next_rel=[v for _threat, _thr, _d, lo, hi in threat_results for v in (lo.tolist() + hi.tolist())],
            next_dist=min(d for _t, _thr, d, _lo, _hi in threat_results),
        )

    _threat, thr, min_dist, p_lo, p_hi = threat_results[0]
    elapsed = time.time() - t0
    if min_dist >= thr:
        return QueryResult(
            spec.name,
            "ibp",
            "proven",
            f"IBP next-rel box stays outside radius {thr} (min_dist>={min_dist:.4f})",
            elapsed,
            action_lo=a_lo.tolist(),
            action_hi=a_hi.tolist(),
            next_rel=p_lo.tolist() + p_hi.tolist(),
            next_dist=min_dist,
        )
    return QueryResult(
        spec.name,
        "ibp",
        "unknown",
        f"IBP next-rel box intersects ball (min_dist={min_dist:.4f} < {thr}); not a refutation",
        elapsed,
        action_lo=a_lo.tolist(),
        action_hi=a_hi.tolist(),
        next_rel=p_lo.tolist() + p_hi.tolist(),
        next_dist=min_dist,
    )


def _inscribed_square_half(thr: float) -> float:
    """|dx|<s and |dy|<s ⇒ ||Δp|| < thr, with s = thr/√2."""
    return float(thr / np.sqrt(2.0))


def marabou_l2_square_crash(
    weights: DeepSetWeights,
    spec: BucketSpec,
    onnx_dir: Path,
    *,
    timeout_s: int = 60,
) -> QueryResult:
    """Single-threat inscribed-square search; dual uses circumscribed squares.

    All queries use the outer input box and the held-action relative map.
    For dual, replay SAT witnesses and check D before claiming a D violation.
    """
    t0 = time.time()
    tag = "clip"
    onnx_path = onnx_dir / f"{spec.name}_k{spec.k}_m{spec.m}_{tag}.onnx"

    def query_one(threat: str, threshold: float) -> QueryResult:
        """Query one disjunct of the dual unsafe condition."""
        q0 = time.time()
        try:
            export_bucket_onnx(weights, spec.k, spec.m, onnx_path, with_clip=True)
            net, in_vars, out_vars = _load_onnx_with_box(
                onnx_path, spec.box.lo, spec.box.hi
            )
            ax, ay = int(out_vars[0]), int(out_vars[1])
            off = 8 if threat == "aa" else 8 + 4 * spec.k
            dx, dy = int(in_vars[off]), int(in_vars[off + 1])
            dvx, dvy = int(in_vars[off + 2]), int(in_vars[off + 3])

            n = HORIZON_STEPS
            c_v = n * DT
            c_a = (n * (n - 1) / 2.0) * (DT * DT) / MASS
            pxn = net.getNewVariable()
            pyn = net.getNewVariable()
            for pvar, dvar, dvvar, avar in (
                (pxn, dx, dvx, ax),
                (pyn, dy, dvy, ay),
            ):
                e = Equation()
                e.addAddend(1.0, dvar)
                e.addAddend(c_v, dvvar)
                e.addAddend(c_a, avar)
                e.addAddend(-1.0, pvar)
                e.setScalar(0.0)
                net.addEquation(e)

            # The full collision disk lies inside this square. UNSAT is thus
            # sufficient for disk avoidance; SAT must be replayed on the disk.
            s = threshold
            net.setLowerBound(pxn, -s)
            net.setUpperBound(pxn, s)
            net.setLowerBound(pyn, -s)
            net.setUpperBound(pyn, s)

            exit_code, vals, _stats = _solve(net, timeout_s)
            elapsed = time.time() - q0
            code = exit_code.lower()
            if code == "sat":
                x = [float(vals[v]) for v in in_vars]
                a = [float(vals[v]) for v in out_vars]
                rel_aa, rel_ao = rels_from_x(np.asarray(x), spec)
                rel_aan, rel_aon = next_rels_for_spec(rel_aa, rel_ao, np.asarray(a), spec)
                ev = eval_collision_dual(rel_aan, rel_aon, spec)
                from stage2.l2_domain import in_preventable_domain_x

                a_numpy = action_at(weights, np.asarray(x), spec)
                parity = bool(np.allclose(a_numpy, a, atol=1e-4, rtol=1e-4))
                in_d = in_preventable_domain_x(np.asarray(x), spec, grid_n=7)
                confirmed = parity and ev.collided and in_d
                return QueryResult(
                    spec.name,
                    "marabou",
                    "violated" if confirmed else "outer_counterexample",
                    f"SAT {threat.upper()} outer-square query; disk_collision={ev.collided}, "
                    f"in_sampled_D={in_d}, action_parity={parity}; numpy dist={ev.dist:.4f}",
                    elapsed,
                    counterexample=x,
                    counterexample_action=a,
                    next_rel=[
                        v
                        for rel in (rel_aan, rel_aon)
                        if rel is not None
                        for v in (rel.dx, rel.dy, rel.dvx, rel.dvy)
                    ],
                    next_dist=ev.dist,
                )
            if code == "unsat":
                return QueryResult(
                    spec.name,
                    "marabou",
                    "proven_outer",
                    f"UNSAT: {threat.upper()} two-step rel pos avoids circumscribed square and disk",
                    elapsed,
                )
            if code == "timeout":
                return QueryResult(
                    spec.name, "marabou", "timeout", f"TIMEOUT on {threat.upper()} square proxy", elapsed
                )
            return QueryResult(spec.name, "marabou", "unknown", f"{threat.upper()} exit={exit_code}", elapsed)
        except Exception as e:
            return QueryResult(
                spec.name,
                "marabou",
                "error",
                f"{threat.upper()} {type(e).__name__}: {e}",
                time.time() - q0,
            )

    if spec.property == "l2_aa_ao":
        aa = query_one("aa", COLLISION_AA)
        if aa.status == "violated":
            return aa
        ao = query_one("ao", COLLISION_AO)
        if ao.status == "violated":
            return ao
        elapsed = time.time() - t0
        if aa.status == "proven_outer" and ao.status == "proven_outer":
            status = "proven_outer"
        elif "timeout" in (aa.status, ao.status):
            status = "timeout"
        elif "error" in (aa.status, ao.status):
            status = "error"
        else:
            status = "unknown"
        return QueryResult(
            spec.name,
            "marabou",
            status,
            f"dual OR query: AA={aa.status}, AO={ao.status}; "
            f"AA detail: {aa.detail}; AO detail: {ao.detail}; "
            "both outer-square UNSAT are required to prove disk avoidance",
            elapsed,
            subqueries=[aa.as_dict(), ao.as_dict()],
        )

    try:
        export_bucket_onnx(weights, spec.k, spec.m, onnx_path, with_clip=True)
        net, in_vars, out_vars = _load_onnx_with_box(onnx_path, spec.box.lo, spec.box.hi)
        # out_vars = clip(a) = a_deploy
        ax, ay = int(out_vars[0]), int(out_vars[1])
        dx, dy = int(in_vars[8]), int(in_vars[9])
        dvx, dvy = int(in_vars[10]), int(in_vars[11])

        n = HORIZON_STEPS
        # d_p_n = d_p + n dt d_v + (n(n-1)/2) dt^2 (a/m)
        # for n=2: d_p'' = d_p + 2 dt d_v + dt^2 (a/m)
        c_v = n * DT
        c_a = (n * (n - 1) / 2.0) * (DT * DT) / MASS
        pxn = net.getNewVariable()
        pyn = net.getNewVariable()
        for pvar, dvar, dvvar, avar in (
            (pxn, dx, dvx, ax),
            (pyn, dy, dvy, ay),
        ):
            e = Equation()
            e.addAddend(1.0, dvar)
            e.addAddend(c_v, dvvar)
            e.addAddend(c_a, avar)
            e.addAddend(-1.0, pvar)
            e.setScalar(0.0)
            net.addEquation(e)

        s = _inscribed_square_half(threshold_for(spec))
        net.setLowerBound(pxn, -s)
        net.setUpperBound(pxn, s)
        net.setLowerBound(pyn, -s)
        net.setUpperBound(pyn, s)

        exit_code, vals, _stats = _solve(net, timeout_s)
        elapsed = time.time() - t0
        code = exit_code.lower()
        if code == "sat":
            x = [float(vals[v]) for v in in_vars]
            a = [float(vals[v]) for v in out_vars]
            rel0 = rel_from_x(np.asarray(x), spec)
            reln = next_rel_for_spec(rel0, np.asarray(a), spec)
            ev = eval_collision(reln, spec)
            return QueryResult(
                spec.name,
                "marabou",
                "violated",
                f"SAT inscribed-square crash proxy; numpy dist={ev.dist:.4f}",
                elapsed,
                counterexample=x,
                counterexample_action=a,
                next_rel=[reln.dx, reln.dy, reln.dvx, reln.dvy],
                next_dist=ev.dist,
            )
        if code == "unsat":
            return QueryResult(
                spec.name,
                "marabou",
                "proven_square",
                "UNSAT: two-step rel pos cannot enter inscribed square; does NOT prove full disk avoidance",
                elapsed,
            )
        if code == "timeout":
            return QueryResult(spec.name, "marabou", "timeout", "TIMEOUT on L2 square proxy", elapsed)
        return QueryResult(spec.name, "marabou", "unknown", f"exit={exit_code}", elapsed)
    except Exception as e:
        return QueryResult(
            spec.name,
            "marabou",
            "error",
            f"{type(e).__name__}: {e}",
            time.time() - t0,
        )
