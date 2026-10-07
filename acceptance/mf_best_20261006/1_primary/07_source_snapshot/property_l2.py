"""Two-step L2 property on MF features only (no DeepSet entity slots).

Specification (horizon simplification of closed-loop safety):
  From feature x, read nearest-threat summaries over **occupied** quadrants
  (n ≥ 0.5); empty quads use d_empty and are ignored for d0/c0:

      d0 = min { min_d_q : n_q ≥ 0.5 }   (or d_empty if none occupied)
      c0 = max(0, max { close_max_q : n_q ≥ 0.5 })

  Deploy a = π(x). Radial over-approx with held action (neighbor coast /
  hit static → relative accel can feed approach up to |a|_∞ / mass):

      d2 = d0 - α c0 - β |a|_∞
      α = n_steps * dt
      β = (dt² / mass) * n_steps*(n_steps-1)/2

  Safe ⇔ d2 ≥ R.

Soundness
---------
- Radial head-on over-approx: if radial-safe ⇒ planar-safe under same d0/c0
  (sufficient). Radial-unsafe may be false alarm without geometry.
- Occupied-only d0/c0 matches feature extractor (empty → d_empty, close=0).
- Domain X should be a **preventable-ish** product box (see domain.py):
  exclude points already doomed under a=0 coast (d0 - α c0 < R).
- Still **two-step** (not infinite horizon). Not DeepSet (k,m) buckets.
"""

from __future__ import annotations

import numpy as np

from stage2.dynamics import COLLISION_AA, DT, HORIZON_STEPS, MASS
from stage_mf.encode import (
    MFResidualWeights,
    deploy_action_np,
    min_d_indices,
)
from stage_mf.student import AGENT_QUAD_FEAT, EGO_DIM, N_QUAD, OBS_QUAD_FEAT

# Empty-quad distance fill (matches student d_empty = comm_radius; typical 0.5)
D_EMPTY_DEFAULT = 0.5
N_OCCUPIED = 0.5


def _agent_quad_base(q: int) -> int:
    return EGO_DIM + q * AGENT_QUAD_FEAT


def _obs_quad_base(q: int) -> int:
    return EGO_DIM + N_QUAD * AGENT_QUAD_FEAT + q * OBS_QUAD_FEAT


def occupied_threat_pairs() -> list[tuple[int, int, int]]:
    """List of (n_idx, min_d_idx, close_max_idx) for every quadrant."""
    pairs = []
    for q in range(N_QUAD):
        b = _agent_quad_base(q)
        pairs.append((b + 0, b + 1, b + 2))  # n, min_d, close_max
    for q in range(N_QUAD):
        b = _obs_quad_base(q)
        pairs.append((b + 0, b + 1, b + 2))
    return pairs


def close_max_indices() -> list[int]:
    return [c for _n, _d, c in occupied_threat_pairs()]


def d0_c0_from_x(
    x: np.ndarray, *, d_empty: float = D_EMPTY_DEFAULT
) -> tuple[float, float]:
    """Nearest occupied threat distance and nonnegative closing."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d_occ: list[float] = []
    c_occ: list[float] = []
    for n_i, d_i, c_i in occupied_threat_pairs():
        if float(x[n_i]) >= N_OCCUPIED:
            d_occ.append(float(x[d_i]))
            c_occ.append(float(x[c_i]))
    if not d_occ:
        return float(d_empty), 0.0
    return float(min(d_occ)), float(max(0.0, max(c_occ)))


def d0_from_x(x: np.ndarray, *, d_empty: float = D_EMPTY_DEFAULT) -> float:
    return d0_c0_from_x(x, d_empty=d_empty)[0]


def c0_from_x(x: np.ndarray, *, d_empty: float = D_EMPTY_DEFAULT) -> float:
    return d0_c0_from_x(x, d_empty=d_empty)[1]


def two_step_radial_d2(
    d0: float,
    c0: float,
    a: np.ndarray,
    *,
    dt: float = DT,
    mass: float = MASS,
    n: int = HORIZON_STEPS,
) -> float:
    """Radial distance after n held steps (scalar head-on model).

    Uses |a|_∞ so each component can fully feed approach (conservative for
    proving safety; may false-alarm for CE search).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a_mag = float(np.max(np.abs(a)))
    d, c = float(d0), float(max(0.0, c0))
    acc = a_mag / max(mass, 1e-9)
    for _ in range(n):
        d = d - c * dt
        c = c + acc * dt
        c = max(0.0, c)
    return float(d)


def is_two_step_safe_point(
    w: MFResidualWeights,
    x: np.ndarray,
    *,
    radius: float = COLLISION_AA,
    d_empty: float = D_EMPTY_DEFAULT,
) -> tuple[bool, dict]:
    """Evaluate property at a single feature vector."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    a = deploy_action_np(w, x)
    d0, c0 = d0_c0_from_x(x, d_empty=d_empty)
    d2 = two_step_radial_d2(d0, c0, a)
    safe = d2 >= radius - 1e-9
    return safe, {
        "a": a.tolist(),
        "d0": d0,
        "c0": c0,
        "d2": d2,
        "radius": float(radius),
        "safe": bool(safe),
        "initially_safe": d0 >= radius - 1e-9,
        "coast_safe": (d0 - d2_formula_coeffs()[0] * c0) >= radius - 1e-9,
    }


def d2_formula_coeffs(n: int = HORIZON_STEPS, dt: float = DT, mass: float = MASS):
    """d2 = d0 - alpha*c0 - beta*|a|_∞  (c stays on approach path).

    n=2: d2 = d0 - 2*dt*c0 - (dt²/mass)*|a|_∞
    """
    alpha = n * dt
    beta = (dt * dt / mass) * (n * (n - 1) / 2.0)
    return float(alpha), float(beta)


def coast_safe_margin(d0: float, c0: float, radius: float) -> float:
    """d0 - α c0 - R; ≥0 means two-step safe under a=0 radial coast."""
    alpha, _ = d2_formula_coeffs()
    return float(d0 - alpha * max(0.0, c0) - radius)


# ---- IBP on property ----


def ibp_d0_c0(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    d_empty: float = D_EMPTY_DEFAULT,
) -> tuple[float, float, float, float]:
    """Interval bounds on (d0, c0) over a feature box (occupied-aware).

    d0_lo: lower bound of min_{occupied} min_d.
      Sound: min over quads that *might* be occupied (n_hi ≥ 0.5) of min_d_lo;
      if none might be occupied → d_empty.
    d0_hi: upper bound of that min (looser; for reporting).
    c0_lo / c0_hi: bounds on max(0, max occupied close_max).
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    d_cands_lo = []
    d_cands_hi = []
    c_cands_lo = []
    c_cands_hi = []
    for n_i, d_i, c_i in occupied_threat_pairs():
        if float(hi[n_i]) >= N_OCCUPIED:
            d_cands_lo.append(float(lo[d_i]))
            d_cands_hi.append(float(hi[d_i]))
            # close contributes to max only if quad can be occupied
            c_cands_lo.append(float(lo[c_i]))
            c_cands_hi.append(float(hi[c_i]))
    if not d_cands_lo:
        return float(d_empty), float(d_empty), 0.0, 0.0
    d0_lo = float(min(d_cands_lo))
    d0_hi = float(min(d_cands_hi))  # not tight upper of min, but ok for hi report
    # max of closes: lo bound = max of lo's; hi bound = max of hi's; then clip ≥0
    c0_lo = float(max(0.0, max(c_cands_lo)))
    c0_hi = float(max(0.0, max(c_cands_hi)))
    return d0_lo, d0_hi, c0_lo, c0_hi


def ibp_min_d(lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    d0_lo, d0_hi, _, _ = ibp_d0_c0(lo, hi)
    return d0_lo, d0_hi


def ibp_c0(lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    _, _, c0_lo, c0_hi = ibp_d0_c0(lo, hi)
    return c0_lo, c0_hi


def ibp_property_status(
    w: MFResidualWeights,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
    *,
    radius: float,
    d_empty: float = D_EMPTY_DEFAULT,
) -> dict:
    """Prove two-step radial safety on a feature sub-box via IBP if possible.

    Proven if worst-case d2_lo = d0_lo - α c0_hi - β |a|_∞_hi ≥ R.
    """
    from stage_mf.ibp_mf import ibp_residual_action

    a_lo, a_hi, info = ibp_residual_action(w, feat_lo, feat_hi)
    a_abs_hi = float(np.max(np.maximum(np.abs(a_lo), np.abs(a_hi))))
    d0_lo, d0_hi, c0_lo, c0_hi = ibp_d0_c0(feat_lo, feat_hi, d_empty=d_empty)
    alpha, beta = d2_formula_coeffs()
    d2_lo = d0_lo - alpha * c0_hi - beta * a_abs_hi
    d2_hi = d0_hi - alpha * c0_lo  # a=0 best-case rough
    proven = d2_lo >= radius - 1e-12
    return {
        "status": "proven" if proven else "unknown",
        "method": "ibp_radial_two_step",
        "d0_lo": d0_lo,
        "d0_hi": d0_hi,
        "c0_lo": c0_lo,
        "c0_hi": c0_hi,
        "a_abs_hi": a_abs_hi,
        "d2_lo": float(d2_lo),
        "d2_hi": float(d2_hi),
        "alpha": alpha,
        "beta": beta,
        "radius": float(radius),
        "a_lo": a_lo.tolist(),
        "a_hi": a_hi.tolist(),
        **{k: info[k] for k in ("g_lo", "g_hi") if k in info},
    }
