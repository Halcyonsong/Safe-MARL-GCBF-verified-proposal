"""L2 verification / repair domain: initially safe ∩ two-step controllable danger.

Single-threat D ((1,0) or (0,1)):

    dist0 >= r
    AND  min_a dist'(a) < r
    AND  max_a dist'(a) >= r

Dual-threat D ((1,1) / l2_aa_ao):

    initially safe vs BOTH threats
    AND  exists a that collides with EITHER after two steps
    AND  exists a that avoids BOTH after two steps

Label modes for repair (Stage3):
  - max_clearance: argmax two-step clearance
  - min_intervention: min ||a - π(x)|| among two-step-safe actions
  - hybrid: near-danger → high clearance; else goal-aware + frozen anchor
    (candidates restricted to the two-step-safe set; never returns a crashing a)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from stage2.bounds import BucketSpec
from stage2.collision import eval_aa, eval_aa_ao, eval_ao
from stage2.dynamics import (
    COLLISION_AA,
    COLLISION_AO,
    DT,
    HORIZON_STEPS,
    MASS,
    RelState,
    V_LIM,
    euler_n,
    next_rel_aa_coast,
    next_rel_ao_static,
)
from stage2.encode import EGO_DIM


def collision_radius(spec: BucketSpec) -> float:
    if spec.property == "l2_aa":
        return COLLISION_AA
    if spec.property == "l2_ao":
        return COLLISION_AO
    if spec.property == "l2_aa_ao":
        return COLLISION_AA  # display only; dual uses both radii
    raise ValueError(spec.property)


def dist0(rel: RelState) -> float:
    return float(np.linalg.norm(rel.d_pos))


def action_grid(n: int = 9) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, n)
    return np.array([(ax, ay) for ax in xs for ay in xs], dtype=np.float64)


def candidate_actions(
    rel_aa: RelState | None,
    rel_ao: RelState | None,
    grid_n: int = 9,
) -> np.ndarray:
    """Grid + heuristics from whichever threats are present."""
    grid = action_grid(grid_n)
    extras: List[np.ndarray] = []
    for rel in (rel_aa, rel_ao):
        if rel is None:
            continue
        p = rel.d_pos
        v = rel.d_vel
        pn = np.linalg.norm(p)
        vn = np.linalg.norm(v)
        if pn > 1e-8:
            u = p / pn
            extras.extend([u, -u])
        if vn > 1e-8:
            u = v / vn
            lat = np.array([-u[1], u[0]])
            extras.extend([-u, u, lat, -lat])
    extras.extend(
        [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2),
            np.array([1.0, -1.0]) / np.sqrt(2),
            np.array([-1.0, 1.0]) / np.sqrt(2),
            np.array([-1.0, -1.0]) / np.sqrt(2),
        ]
    )
    extras_a = np.clip(np.stack(extras), -1.0, 1.0)
    return np.unique(np.vstack([grid, extras_a]), axis=0)


def _next_pair(
    rel_aa: RelState | None,
    rel_ao: RelState | None,
    a: np.ndarray,
) -> Tuple[RelState | None, RelState | None]:
    na = next_rel_aa_coast(rel_aa, a, n=HORIZON_STEPS) if rel_aa is not None else None
    no = next_rel_ao_static(rel_ao, a, n=HORIZON_STEPS) if rel_ao is not None else None
    return na, no


def _eval_pair(rel_aa: RelState | None, rel_ao: RelState | None, spec: BucketSpec):
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


def _clearance_score(rel_aa: RelState | None, rel_ao: RelState | None, spec: BucketSpec) -> float:
    """Higher = safer. Dual uses min normalized clearance."""
    if spec.property == "l2_aa":
        assert rel_aa is not None
        return dist0(rel_aa) / COLLISION_AA
    if spec.property == "l2_ao":
        assert rel_ao is not None
        return dist0(rel_ao) / COLLISION_AO
    if spec.property == "l2_aa_ao":
        assert rel_aa is not None and rel_ao is not None
        return min(dist0(rel_aa) / COLLISION_AA, dist0(rel_ao) / COLLISION_AO)
    raise ValueError(spec.property)


def initially_safe(
    rel_aa: RelState | None,
    rel_ao: RelState | None,
    spec: BucketSpec,
    *,
    margin: float = 0.0,
) -> bool:
    ev = _eval_pair(rel_aa, rel_ao, spec)
    # For dual, eval_aa_ao already ORs collisions; also enforce both clearances.
    if spec.property == "l2_aa_ao":
        assert rel_aa is not None and rel_ao is not None
        return (
            dist0(rel_aa) >= COLLISION_AA + margin
            and dist0(rel_ao) >= COLLISION_AO + margin
        )
    return (not ev.collided) and (ev.dist >= ev.threshold + margin)


def sweep_two_step(
    rel_aa: RelState | None,
    rel_ao: RelState | None,
    spec: BucketSpec,
    *,
    grid_n: int = 9,
) -> Tuple[float, float, np.ndarray, np.ndarray, bool, bool]:
    """Return (min_score, max_score, a_min, a_max, any_crash, any_safe).

    Scores are normalized clearances (dual: min of AA/AO ratios).
    """
    acts = candidate_actions(rel_aa, rel_ao, grid_n=grid_n)
    scores = []
    crashes = []
    for a in acts:
        na, no = _next_pair(rel_aa, rel_ao, a)
        scores.append(_clearance_score(na, no, spec))
        crashes.append(_eval_pair(na, no, spec).collided)
    scores_a = np.asarray(scores, dtype=np.float64)
    i_min = int(np.argmin(scores_a))
    i_max = int(np.argmax(scores_a))
    any_crash = bool(any(crashes))
    any_safe = bool(any(not c for c in crashes))
    return (
        float(scores_a[i_min]),
        float(scores_a[i_max]),
        acts[i_min],
        acts[i_max],
        any_crash,
        any_safe,
    )


def in_preventable_domain_x(
    x: np.ndarray,
    spec: BucketSpec,
    *,
    grid_n: int = 9,
    margin: float = 0.0,
) -> bool:
    from stage2.l2_query import rels_from_x

    rel_aa, rel_ao = rels_from_x(x, spec)
    if not initially_safe(rel_aa, rel_ao, spec, margin=margin):
        return False
    _smin, _smax, _alo, _ahi, any_crash, any_safe = sweep_two_step(
        rel_aa, rel_ao, spec, grid_n=grid_n
    )
    return any_crash and any_safe


def in_preventable_domain(
    rel: RelState,
    spec: BucketSpec,
    *,
    grid_n: int = 9,
    margin: float = 0.0,
) -> bool:
    """Backward-compatible single-rel API for (1,0)/(0,1) only."""
    if spec.property == "l2_aa_ao":
        raise ValueError("dual-threat buckets must use in_preventable_domain_x(x, spec)")
    rel_aa = rel if spec.property == "l2_aa" else None
    rel_ao = rel if spec.property == "l2_ao" else None
    if not initially_safe(rel_aa, rel_ao, spec, margin=margin):
        return False
    _smin, _smax, _alo, _ahi, any_crash, any_safe = sweep_two_step(
        rel_aa, rel_ao, spec, grid_n=grid_n
    )
    return any_crash and any_safe


def l2_safe_action_x(
    x: np.ndarray,
    spec: BucketSpec,
    *,
    grid_n: int = 11,
    margin: float = 0.0,
) -> Tuple[np.ndarray, float, bool]:
    """a* maximizing normalized clearance; is_safe if avoids all threats."""
    from stage2.l2_query import rels_from_x

    rel_aa, rel_ao = rels_from_x(x, spec)
    _smin, smax, _alo, a_hi, _ac, any_safe = sweep_two_step(
        rel_aa, rel_ao, spec, grid_n=grid_n
    )
    # absolute min distance among threats under a_hi (for logging)
    na, no = _next_pair(rel_aa, rel_ao, a_hi)
    if spec.property == "l2_aa_ao":
        d_abs = min(dist0(na), dist0(no))  # type: ignore
    elif spec.property == "l2_aa":
        d_abs = dist0(na)  # type: ignore
    else:
        d_abs = dist0(no)  # type: ignore
    safe = bool(any_safe) and smax >= 1.0 + margin
    return a_hi.astype(np.float64), float(d_abs), safe


def _abs_min_dist(
    na: RelState | None, no: RelState | None, spec: BucketSpec
) -> float:
    if spec.property == "l2_aa_ao":
        return min(dist0(na), dist0(no))  # type: ignore
    if spec.property == "l2_aa":
        return dist0(na)  # type: ignore
    return dist0(no)  # type: ignore


def _safe_candidate_set(
    rel_aa: RelState | None,
    rel_ao: RelState | None,
    spec: BucketSpec,
    a_ref: np.ndarray,
    *,
    grid_n: int = 11,
    margin: float = 0.0,
) -> List[Tuple[np.ndarray, float, float]]:
    """Return list of (a, clearance_score, dist_abs) in the two-step-safe set S."""
    acts = candidate_actions(rel_aa, rel_ao, grid_n=grid_n)
    extras = [a_ref]
    for scale in (0.25, 0.5, 0.75):
        extras.append(np.clip((1.0 - scale) * a_ref, -1.0, 1.0))
    # Blend a_ref toward a few repulsive directions for denser S near danger.
    for rel in (rel_aa, rel_ao):
        if rel is None:
            continue
        pn = np.linalg.norm(rel.d_pos)
        if pn > 1e-8:
            u = rel.d_pos / pn
            for alpha in (0.3, 0.6):
                extras.append(np.clip((1.0 - alpha) * a_ref + alpha * u, -1.0, 1.0))
                extras.append(np.clip((1.0 - alpha) * a_ref - alpha * u, -1.0, 1.0))
    acts = np.unique(np.vstack([acts, np.stack(extras)]), axis=0)

    safe: List[Tuple[np.ndarray, float, float]] = []
    for a in acts:
        na, no = _next_pair(rel_aa, rel_ao, a)
        score = _clearance_score(na, no, spec)
        crashed = _eval_pair(na, no, spec).collided
        if crashed or score < 1.0 + margin:
            continue
        safe.append((a.astype(np.float64), float(score), _abs_min_dist(na, no, spec)))
    return safe


def l2_min_intervention_action_x(
    x: np.ndarray,
    spec: BucketSpec,
    a_ref: np.ndarray,
    *,
    grid_n: int = 11,
    margin: float = 0.0,
) -> Tuple[np.ndarray, float, bool, float]:
    """Smallest action change from ``a_ref`` that stays two-step safe.

    Among candidate actions that avoid all threats after two held Euler steps
    (clearance score >= 1 + margin), pick the one minimizing ||a - a_ref||.
    Tie-break with higher clearance. Returns (a*, dist_abs, is_safe, ||Δa||).
    """
    from stage2.l2_query import rels_from_x

    a_ref = np.clip(np.asarray(a_ref, dtype=np.float64).reshape(-1), -1.0, 1.0)
    rel_aa, rel_ao = rels_from_x(x, spec)
    safe = _safe_candidate_set(
        rel_aa, rel_ao, spec, a_ref, grid_n=grid_n, margin=margin
    )
    if not safe:
        a_hi, d_abs, ok = l2_safe_action_x(x, spec, grid_n=grid_n, margin=margin)
        return a_hi, d_abs, ok, float(np.linalg.norm(a_hi - a_ref))
    best = None  # (dist_to_ref, -score, a, d_abs)
    for a, score, d_abs in safe:
        d_ref = float(np.linalg.norm(a - a_ref))
        cand = (d_ref, -score, a, float(d_abs))
        if best is None or cand < best:
            best = cand
    assert best is not None
    d_ref, _neg_s, a_star, d_abs = best
    return a_star, d_abs, True, float(d_ref)


def _goal_features_reliable(ego: np.ndarray, *, eps: float = 1e-4) -> bool:
    """Synthetic-box CEs with near-zero goal rel → no usable task signal."""
    g = np.asarray(ego, dtype=np.float64).reshape(-1)
    if g.shape[0] < EGO_DIM:
        return False
    return float(np.linalg.norm(g[4:6])) >= eps


def _goal_progress(ego: np.ndarray, a: np.ndarray) -> float:
    """Normalized reduction in distance-to-goal after two held Euler steps.

    Ego layout: [px, py, vx, vy, goal_dx, goal_dy, goal_dvx, goal_dvy].
    Goal is treated as static in absolute frame (g_abs = p + goal_xy).
    Returns value roughly in [-1, 1] (positive = closer to goal).
    """
    ego = np.asarray(ego, dtype=np.float64).reshape(-1)
    p0 = ego[0:2]
    v0 = ego[2:4]
    g_rel = ego[4:6]
    d0 = float(np.linalg.norm(g_rel))
    if d0 < 1e-8:
        return 0.0
    g_abs = p0 + g_rel
    p2, _v2 = euler_n(p0, v0, a, n=HORIZON_STEPS, dt=DT, mass=MASS, v_lim=V_LIM)
    d2 = float(np.linalg.norm(g_abs - p2))
    # Scale: ~one short DI hop (~0.05 m) → unit progress.
    return float(np.clip((d0 - d2) / 0.05, -1.0, 1.0))


def _initial_clearance_norm(
    rel_aa: RelState | None, rel_ao: RelState | None, spec: BucketSpec
) -> float:
    return float(_clearance_score(rel_aa, rel_ao, spec))


def l2_hybrid_safe_action_x(
    x: np.ndarray,
    spec: BucketSpec,
    a_ref: np.ndarray,
    *,
    lam: float = 0.5,
    mu: float = 0.15,
    near_clearance: float = 1.5,
    grid_n: int = 11,
    margin: float = 0.0,
) -> Tuple[np.ndarray, float, bool, float, str]:
    """Hybrid / goal-aware label restricted to the two-step-safe set S.

    Policy (corrected hybrid; rejects original Zone-B):
      1. Enumerate only actions that are two-step safe (clearance >= 1+margin).
         Never return a crashing actor action.
      2. If initially near-danger (clearance_norm < near_clearance):
         pick max clearance in S (tie-break: smaller ||a-a_ref||).
      3. Else if goal features look usable:
         score = lam * clearance + (1-lam) * goal_progress - mu * ||a-a_ref||
      4. Else (dirty / missing goal): fall back to min-intervention on S.

    Returns (a*, dist_abs, is_safe, ||Δa||, submode) where submode is one of
    ``near_max_clearance`` / ``goal_aware`` / ``min_intervention_fallback`` /
    ``max_clearance_fallback``.
    """
    from stage2.l2_query import rels_from_x
    from stage2.encode import unpack_inputs

    a_ref = np.clip(np.asarray(a_ref, dtype=np.float64).reshape(-1), -1.0, 1.0)
    lam = float(np.clip(lam, 0.0, 1.0))
    mu = float(max(0.0, mu))
    ego, _agents, _obs = unpack_inputs(x, spec.k, spec.m)
    rel_aa, rel_ao = rels_from_x(x, spec)
    safe = _safe_candidate_set(
        rel_aa, rel_ao, spec, a_ref, grid_n=grid_n, margin=margin
    )
    if not safe:
        a_hi, d_abs, ok = l2_safe_action_x(x, spec, grid_n=grid_n, margin=margin)
        return (
            a_hi,
            d_abs,
            ok,
            float(np.linalg.norm(a_hi - a_ref)),
            "max_clearance_fallback",
        )

    d0_norm = _initial_clearance_norm(rel_aa, rel_ao, spec)
    near = d0_norm < float(near_clearance)

    if near:
        # High-safety zone: maximize clearance; prefer small Δa on ties.
        best = None  # (-score, dist_to_ref, a, d_abs)
        for a, score, d_abs in safe:
            d_ref = float(np.linalg.norm(a - a_ref))
            cand = (-score, d_ref, a, float(d_abs))
            if best is None or cand < best:
                best = cand
        assert best is not None
        _ns, d_ref, a_star, d_abs = best
        return a_star, d_abs, True, float(d_ref), "near_max_clearance"

    if not _goal_features_reliable(ego):
        a_mi, d_abs, ok, da = l2_min_intervention_action_x(
            x, spec, a_ref, grid_n=grid_n, margin=margin
        )
        return a_mi, d_abs, ok, da, "min_intervention_fallback"

    # Mid / far: goal-aware + frozen anchor, still inside S only.
    best = None  # (-score, a, d_abs, d_ref)
    for a, clearance, d_abs in safe:
        prog = _goal_progress(ego, a)
        d_ref = float(np.linalg.norm(a - a_ref))
        score = lam * clearance + (1.0 - lam) * prog - mu * d_ref
        cand = (-score, a, float(d_abs), d_ref)
        if best is None or cand < best:
            best = cand
    assert best is not None
    _ns, a_star, d_abs, d_ref = best
    return a_star, d_abs, True, float(d_ref), "goal_aware"


def l2_safe_action(
    rel: RelState,
    spec: BucketSpec,
    *,
    grid_n: int = 11,
    margin: float = 0.01,
) -> Tuple[np.ndarray, float, bool]:
    """Single-threat label API used by older collectors."""
    rel_aa = rel if spec.property == "l2_aa" else None
    rel_ao = rel if spec.property == "l2_ao" else None
    if spec.property == "l2_aa_ao":
        raise ValueError("use l2_safe_action_x for dual-threat buckets")
    _smin, smax, _alo, a_hi, _ac, any_safe = sweep_two_step(
        rel_aa, rel_ao, spec, grid_n=grid_n
    )
    thr = collision_radius(spec)
    d_abs = smax * thr
    safe = bool(any_safe) and d_abs >= thr + margin
    return a_hi.astype(np.float64), float(d_abs), safe


def domain_stats(
    spec: BucketSpec,
    rng: np.random.Generator,
    *,
    n: int = 2000,
    grid_n: int = 7,
) -> dict:
    from stage2.l2_query import rels_from_x

    n_already = n_safe_triv = n_preventable = n_hard = 0
    for _ in range(n):
        x = spec.box.sample(rng)
        rel_aa, rel_ao = rels_from_x(x, spec)
        if not initially_safe(rel_aa, rel_ao, spec):
            n_already += 1
            continue
        _smin, _smax, _alo, _ahi, any_crash, any_safe = sweep_two_step(
            rel_aa, rel_ao, spec, grid_n=grid_n
        )
        if not any_crash:
            n_safe_triv += 1
        elif not any_safe:
            n_hard += 1
        else:
            n_preventable += 1
    return {
        "n": n,
        "already_colliding": n_already / n,
        "trivial_safe_two_step": n_safe_triv / n,
        "preventable_D": n_preventable / n,
        "hard_unavoidable": n_hard / n,
        "property": spec.property,
    }
