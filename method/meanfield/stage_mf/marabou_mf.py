"""Marabou encoding: MF residual on 44-D features + two-step radial L2 property.

Query (SAT = violation exists):
  x ∈ [feat_lo, feat_hi]
  a = clip(base(x) + g_phys(x)*clip(δ(x)))
  d0 = min_i min_d_i(x),  c0 = max(0, max_j close_max_j(x))
  d2 = d0 - alpha*c0 - beta*|a|_∞
  SAT if d2 < R   (exists x with two-step radial collision)

UNSAT ⇒ every x in the box is two-step radially safe (sound for this property).
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
from maraboupy import Marabou
from maraboupy.MarabouUtils import Equation

from stage2.marabou_query import _solve
from stage_mf.encode import (
    MLPWeights,
    MFResidualWeights,
    min_d_indices,
)
from stage_mf.ibp_mf import ibp_mlp
from stage_mf.property_l2 import close_max_indices, d2_formula_coeffs


def _linear(net, xs: Sequence[int], w: np.ndarray, b: np.ndarray) -> list[int]:
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ys = [net.getNewVariable() for _ in range(len(b))]
    for j, y in enumerate(ys):
        e = Equation()
        for i, x in enumerate(xs):
            c = float(w[i, j])
            if c != 0.0:
                e.addAddend(c, x)
        e.addAddend(-1.0, y)
        e.setScalar(float(-b[j]))
        net.addEquation(e)
    return ys


def _relu(net, xs: Sequence[int]) -> list[int]:
    ys = [net.getNewVariable() for _ in xs]
    for x, y in zip(xs, ys):
        net.addRelu(int(x), int(y))
    return ys


def _eq(net, terms: Iterable[tuple[float, int]], scalar: float) -> None:
    e = Equation()
    for coeff, var in terms:
        e.addAddend(float(coeff), int(var))
    e.setScalar(float(scalar))
    net.addEquation(e)


def _le(net, terms: Iterable[tuple[float, int]], scalar: float) -> None:
    terms = list(terms)
    net.addInequality(
        [int(v) for _c, v in terms],
        [float(c) for c, _v in terms],
        float(scalar),
    )


def _add_mlp(net, x_vars: Sequence[int], w: MLPWeights) -> list[int]:
    h = list(x_vars)
    for i, (kern, bias) in enumerate(zip(w.kernels, w.biases)):
        pre = _linear(net, h, kern, bias)
        h = _relu(net, pre) if i < len(w.kernels) - 1 else pre
    return h


def _add_clip_pm1(net, raw: Sequence[int]) -> list[int]:
    out = []
    for r in raw:
        rp = _linear(net, [r], np.array([[1.0]]), np.array([1.0]))[0]
        rm = _linear(net, [r], np.array([[1.0]]), np.array([-1.0]))[0]
        hp, hm = _relu(net, [rp])[0], _relu(net, [rm])[0]
        a = net.getNewVariable()
        _eq(net, [(1.0, hp), (-1.0, hm), (-1.0, a)], 1.0)
        net.setLowerBound(a, -1.0)
        net.setUpperBound(a, 1.0)
        out.append(a)
    return out


def _add_clip_delta(net, raw: Sequence[int], clip: float) -> list[int]:
    c = float(clip)
    scaled = []
    for r in raw:
        s = net.getNewVariable()
        _eq(net, [(1.0 / c, r), (-1.0, s)], 0.0)
        scaled.append(s)
    unit = _add_clip_pm1(net, scaled)
    out = []
    for u in unit:
        y = net.getNewVariable()
        _eq(net, [(c, u), (-1.0, y)], 0.0)
        net.setLowerBound(y, -c)
        net.setUpperBound(y, c)
        out.append(y)
    return out


def _pairwise_min(net, a: int, b: int) -> int:
    diff = net.getNewVariable()
    _eq(net, [(1.0, b), (-1.0, a), (-1.0, diff)], 0.0)
    rd = _relu(net, [diff])[0]
    m = net.getNewVariable()
    _eq(net, [(1.0, b), (-1.0, rd), (-1.0, m)], 0.0)
    return m


def _pairwise_max(net, a: int, b: int) -> int:
    diff = net.getNewVariable()
    _eq(net, [(1.0, b), (-1.0, a), (-1.0, diff)], 0.0)
    rd = _relu(net, [diff])[0]
    M = net.getNewVariable()
    _eq(net, [(1.0, a), (1.0, rd), (-1.0, M)], 0.0)
    return M


def _fold_min(net, vars_: Sequence[int]) -> int:
    m = vars_[0]
    for v in vars_[1:]:
        m = _pairwise_min(net, m, v)
    return m


def _fold_max(net, vars_: Sequence[int]) -> int:
    m = vars_[0]
    for v in vars_[1:]:
        m = _pairwise_max(net, m, v)
    return m


def _clip01(net, v: int) -> int:
    r0 = _relu(net, [v])[0]
    vm = _linear(net, [v], np.array([[1.0]]), np.array([-1.0]))[0]
    r1 = _relu(net, [vm])[0]
    y = net.getNewVariable()
    _eq(net, [(1.0, r0), (-1.0, r1), (-1.0, y)], 0.0)
    net.setLowerBound(y, 0.0)
    net.setUpperBound(y, 1.0)
    return y


def _add_physics_gate(net, x_vars: Sequence[int], w: MFResidualWeights) -> int:
    from stage_mf.encode import inv_ttc_indices

    md = _fold_min(net, [x_vars[i] for i in min_d_indices()])
    it = _fold_max(net, [x_vars[i] for i in inv_ttc_indices()])
    denom = max(w.open_mult - w.close_mult, 1e-6)
    coef = -1.0 / (max(w.r_ref, 1e-6) * denom)
    bias = w.open_mult / denom
    gd_raw = net.getNewVariable()
    _eq(net, [(coef, md), (-1.0, gd_raw)], -bias)
    gd = _clip01(net, gd_raw)
    gt_raw = net.getNewVariable()
    _eq(net, [(1.0 / max(w.inv_ttc_ref, 1e-6), it), (-1.0, gt_raw)], 0.0)
    gt = _clip01(net, gt_raw)
    g = net.getNewVariable()
    _eq(net, [(1.0 - w.gate_mix, gd), (w.gate_mix, gt), (-1.0, g)], 0.0)
    net.setLowerBound(g, 0.0)
    net.setUpperBound(g, 1.0)
    return g


def _add_mccormick(net, g: int, d: int, z: int, dlo: float, dhi: float) -> None:
    """McCormick envelope for z = g * d with g ∈ [0, 1], d ∈ [dlo, dhi].

    Standard four inequalities (sound outer approx of the bilinear product):
      z ≥ dlo * g
      z ≥ d + dhi * g - dhi
      z ≤ d + dlo * g - dlo
      z ≤ dhi * g

    NOTE: a previous sign error on the last inequality (z ≥ dhi*g) made any
    positive-width box encode as globally UNSAT, so SafeSet "proven" was a
    false certificate. Keep the last line as z - dhi*g ≤ 0.
    """
    # z >= dlo * g    <=>  dlo*g - z <= 0
    _le(net, [(dlo, g), (-1.0, z)], 0.0)
    # z >= d + dhi*g - dhi  <=>  d + dhi*g - z <= dhi
    _le(net, [(1.0, d), (dhi, g), (-1.0, z)], dhi)
    # z <= d + dlo*g - dlo  <=>  z - d - dlo*g <= -dlo
    _le(net, [(1.0, z), (-1.0, d), (-dlo, g)], -dlo)
    # z <= dhi * g    <=>  z - dhi*g <= 0
    _le(net, [(1.0, z), (-dhi, g)], 0.0)


def _abs_upper(net, v: int) -> int:
    """a_abs >= |v| via a_abs >= v, a_abs >= -v (and we minimize implicitly by use)."""
    # For upper-bounding |a| in a linear constraint d2 = ... - beta*|a|,
    # we need |a| as a variable. Encode a_abs = relu(v) + relu(-v).
    rp = _relu(net, [v])[0]
    vn = net.getNewVariable()
    _eq(net, [(-1.0, v), (-1.0, vn)], 0.0)  # vn = -v
    rn = _relu(net, [vn])[0]
    a_abs = net.getNewVariable()
    _eq(net, [(1.0, rp), (1.0, rn), (-1.0, a_abs)], 0.0)
    return a_abs


def build_violation_query(
    w: MFResidualWeights,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
    *,
    radius: float,
) -> tuple:
    """Build Marabou net that is SAT iff some x has d2 < radius."""
    feat_lo = np.asarray(feat_lo, dtype=np.float64)
    feat_hi = np.asarray(feat_hi, dtype=np.float64)
    net = Marabou.MarabouNetwork()
    x_vars = [net.getNewVariable() for _ in range(len(feat_lo))]
    for v, lo, hi in zip(x_vars, feat_lo, feat_hi):
        net.setLowerBound(v, float(lo))
        net.setUpperBound(v, float(hi))

    base_raw = _add_mlp(net, x_vars, w.base)
    delta_raw = _add_mlp(net, x_vars, w.delta)
    delta_c = _add_clip_delta(net, delta_raw, w.delta_clip)
    g = _add_physics_gate(net, x_vars, w)

    d_iv = ibp_mlp(w.delta, feat_lo, feat_hi)
    dlo = np.clip(d_iv.lo, -w.delta_clip, w.delta_clip)
    dhi = np.clip(d_iv.hi, -w.delta_clip, w.delta_clip)
    zvars = []
    for j in range(2):
        z = net.getNewVariable()
        lo_j, hi_j = float(min(dlo[j], dhi[j])), float(max(dlo[j], dhi[j]))
        net.setLowerBound(z, min(0.0, lo_j, hi_j))
        net.setUpperBound(z, max(0.0, lo_j, hi_j))
        _add_mccormick(net, g, delta_c[j], z, lo_j, hi_j)
        zvars.append(z)
    raw = []
    for b, z in zip(base_raw, zvars):
        a = net.getNewVariable()
        _eq(net, [(1.0, b), (1.0, z), (-1.0, a)], 0.0)
        raw.append(a)
    action = _add_clip_pm1(net, raw)

    # d0/c0 over-approx: min/max over ALL quads (incl. empty).
    # min_all ≤ d0_occupied, max_all ≥ c0_occupied → d2_mar ≤ d2_true.
    # UNSAT remains sound for the occupied-aware property; SAT may false-alarm
    # and is re-checked with true deploy + occupied-aware d0/c0.
    d0 = _fold_min(net, [x_vars[i] for i in min_d_indices()])
    c_raw = _fold_max(net, [x_vars[i] for i in close_max_indices()])
    c0 = _relu(net, [c_raw])[0]  # max(c_raw, 0)

    # |a|_∞ = max(|a0|, |a1|)
    abs0 = _abs_upper(net, action[0])
    abs1 = _abs_upper(net, action[1])
    a_inf = _pairwise_max(net, abs0, abs1)

    alpha, beta = d2_formula_coeffs()
    # d2 = d0 - alpha*c0 - beta*a_inf
    d2 = net.getNewVariable()
    _eq(net, [(1.0, d0), (-alpha, c0), (-beta, a_inf), (-1.0, d2)], 0.0)

    # violation: d2 < R  ⇔  d2 <= R - eps
    eps = 1e-5
    net.setUpperBound(d2, float(radius) - eps)

    big = 1e4
    nvars = int(net.numVars) if not callable(net.numVars) else int(net.numVars())
    for v in range(nvars):
        if not net.lowerBoundExists(v):
            net.setLowerBound(v, -big)
        if not net.upperBoundExists(v):
            net.setUpperBound(v, big)

    return net, x_vars, action, d2


def marabou_subbox(
    w: MFResidualWeights,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
    *,
    radius: float,
    timeout_s: int = 30,
) -> dict:
    t0 = time.time()
    try:
        net, x_vars, action, d2 = build_violation_query(
            w, feat_lo, feat_hi, radius=radius
        )
        code, vals, _stats = _solve(net, timeout_s)
    except Exception as e:
        return {
            "status": "error",
            "method": "marabou_radial",
            "detail": str(e),
            "elapsed_s": round(time.time() - t0, 3),
        }

    elapsed = round(time.time() - t0, 3)
    code_l = str(code).lower()
    if "unsat" in code_l:
        return {
            "status": "proven",
            "method": "marabou_radial",
            "detail": str(code),
            "elapsed_s": elapsed,
        }
    if "sat" in code_l and "unsat" not in code_l:
        x_ce = [float(vals.get(v, 0.0)) for v in x_vars] if vals else None
        a_ce = [float(vals.get(v, 0.0)) for v in action] if vals else None
        d2_ce = float(vals.get(d2, 0.0)) if vals else None
        return {
            "status": "sat_relaxed",
            "method": "marabou_radial",
            "detail": str(code),
            "elapsed_s": elapsed,
            "x": x_ce,
            "a_relaxed": a_ce,
            "d2_relaxed": d2_ce,
        }
    return {
        "status": "timeout" if "timeout" in code_l or "unknown" in code_l else "unknown",
        "method": "marabou_radial",
        "detail": str(code),
        "elapsed_s": elapsed,
    }


def _encode_deploy_action(
    w: MFResidualWeights,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
):
    """Shared Deploy encoding: returns (net, x_vars, action_vars).

    Also enforces reachable contract goal_dv ≡ -v (eqs on dims 6,7).
    """
    feat_lo = np.asarray(feat_lo, dtype=np.float64)
    feat_hi = np.asarray(feat_hi, dtype=np.float64)
    net = Marabou.MarabouNetwork()
    x_vars = [net.getNewVariable() for _ in range(len(feat_lo))]
    for v, lo, hi in zip(x_vars, feat_lo, feat_hi):
        net.setLowerBound(v, float(lo))
        net.setUpperBound(v, float(hi))
    # goal_dv = -v  (reachable_v1 contract; dims: v=2,3 goal_dv=6,7)
    if len(x_vars) >= 8:
        _eq(net, [(1.0, x_vars[6]), (1.0, x_vars[2])], 0.0)  # gdvx + vx = 0
        _eq(net, [(1.0, x_vars[7]), (1.0, x_vars[3])], 0.0)  # gdvy + vy = 0

    base_raw = _add_mlp(net, x_vars, w.base)
    delta_raw = _add_mlp(net, x_vars, w.delta)
    delta_c = _add_clip_delta(net, delta_raw, w.delta_clip)
    g = _add_physics_gate(net, x_vars, w)

    d_iv = ibp_mlp(w.delta, feat_lo, feat_hi)
    dlo = np.clip(d_iv.lo, -w.delta_clip, w.delta_clip)
    dhi = np.clip(d_iv.hi, -w.delta_clip, w.delta_clip)
    zvars = []
    for j in range(2):
        z = net.getNewVariable()
        lo_j, hi_j = float(min(dlo[j], dhi[j])), float(max(dlo[j], dhi[j]))
        net.setLowerBound(z, min(0.0, lo_j, hi_j))
        net.setUpperBound(z, max(0.0, lo_j, hi_j))
        _add_mccormick(net, g, delta_c[j], z, lo_j, hi_j)
        zvars.append(z)
    raw = []
    for b, z in zip(base_raw, zvars):
        a = net.getNewVariable()
        _eq(net, [(1.0, b), (1.0, z), (-1.0, a)], 0.0)
        raw.append(a)
    action = _add_clip_pm1(net, raw)
    return net, x_vars, action


def build_safeset_violation_query(
    w: MFResidualWeights,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
    *,
    a_min: float,
    ay_abs: float = 1.0,
) -> tuple:
    """SAT iff some x has Deploy(x) outside SafeSet {ax≥a_min, |ay|≤ay_abs}.

    Encoding: ax ≤ a_min - eps  (violation of ax≥a_min).
    |ay|≤ay_abs is usually 1.0 (= action clip), so only ax matters.
    """
    net, x_vars, action = _encode_deploy_action(w, feat_lo, feat_hi)
    eps = 1e-5
    # violation of ax ≥ a_min  ⇔  ax ≤ a_min - eps
    net.setUpperBound(action[0], float(a_min) - eps)
    # optional: force |ay| already in [-1,1] via clip; if ay_abs < 1 could add
    # but default ay_abs=1 matches clip.

    big = 1e4
    nvars = int(net.numVars) if not callable(net.numVars) else int(net.numVars())
    for v in range(nvars):
        if not net.lowerBoundExists(v):
            net.setLowerBound(v, -big)
        if not net.upperBoundExists(v):
            net.setUpperBound(v, big)
    return net, x_vars, action


def marabou_safeset_subbox(
    w: MFResidualWeights,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
    *,
    a_min: float,
    ay_abs: float = 1.0,
    timeout_s: int = 30,
) -> dict:
    """UNSAT ⇒ every x in feat box has Deploy.ax ≥ a_min (SafeSet membership)."""
    t0 = time.time()
    try:
        net, x_vars, action = build_safeset_violation_query(
            w, feat_lo, feat_hi, a_min=a_min, ay_abs=ay_abs
        )
        code, vals, _stats = _solve(net, timeout_s)
    except Exception as e:
        return {
            "status": "error",
            "method": "marabou_safeset_ax_ge",
            "detail": str(e),
            "elapsed_s": round(time.time() - t0, 3),
            "a_min": float(a_min),
        }

    elapsed = round(time.time() - t0, 3)
    code_l = str(code).lower()
    if "unsat" in code_l:
        return {
            "status": "proven",
            "method": "marabou_safeset_ax_ge",
            "detail": str(code),
            "elapsed_s": elapsed,
            "a_min": float(a_min),
        }
    if "sat" in code_l and "unsat" not in code_l:
        x_ce = [float(vals.get(v, 0.0)) for v in x_vars] if vals else None
        a_ce = [float(vals.get(v, 0.0)) for v in action] if vals else None
        return {
            "status": "sat_candidate",
            "method": "marabou_safeset_ax_ge",
            "detail": str(code),
            "elapsed_s": elapsed,
            "a_min": float(a_min),
            "x": x_ce,
            "a_relaxed": a_ce,
        }
    return {
        "status": "timeout" if "timeout" in code_l or "unknown" in code_l else "unknown",
        "method": "marabou_safeset_ax_ge",
        "detail": str(code),
        "elapsed_s": elapsed,
        "a_min": float(a_min),
    }
