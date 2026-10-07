"""Interval Bound Propagation for MF pure-ReLU MLPs + residual (sound)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage_mf.encode import (
    MLPWeights,
    MFResidualWeights,
    inv_ttc_indices,
    min_d_indices,
)


@dataclass
class Interval:
    lo: np.ndarray
    hi: np.ndarray

    def relu(self) -> "Interval":
        return Interval(np.maximum(self.lo, 0.0), np.maximum(self.hi, 0.0))

    def clip(self, lo: float, hi: float) -> "Interval":
        return Interval(np.clip(self.lo, lo, hi), np.clip(self.hi, lo, hi))


def affine(x: Interval, w: np.ndarray, b: np.ndarray) -> Interval:
    """y = x @ W + b, sound interval image."""
    c = 0.5 * (x.lo + x.hi)
    r = 0.5 * (x.hi - x.lo)
    yc = c @ w + b
    yr = r @ np.abs(w)
    return Interval(yc - yr, yc + yr)


def ibp_mlp(weights: MLPWeights, lo: np.ndarray, hi: np.ndarray) -> Interval:
    h = Interval(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
    for i, (w, b) in enumerate(zip(weights.kernels, weights.biases)):
        h = affine(h, w, b)
        if i < len(weights.kernels) - 1:
            h = h.relu()
    return h


def ibp_min_coords(lo: np.ndarray, hi: np.ndarray, idxs: list[int]) -> tuple[float, float]:
    """Interval bounds on min_i x[i] for i in idxs."""
    # min of intervals: lo = min(lo_i), hi = min(hi_i)
    return float(min(lo[i] for i in idxs)), float(min(hi[i] for i in idxs))


def ibp_max_coords(lo: np.ndarray, hi: np.ndarray, idxs: list[int]) -> tuple[float, float]:
    return float(max(lo[i] for i in idxs)), float(max(hi[i] for i in idxs))


def ibp_physics_gate(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    open_mult: float,
    close_mult: float,
    r_ref: float,
    inv_ttc_ref: float,
    mix: float,
) -> tuple[float, float]:
    """Sound interval for g_phys(x) ∈ [0,1]."""
    d_lo, d_hi = ibp_min_coords(lo, hi, min_d_indices())
    # g_d = clip((open - d/r) / (open-close), 0, 1)
    denom = max(open_mult - close_mult, 1e-6)
    # n = d/r_ref → n_lo = d_lo/r, n_hi = d_hi/r
    n_lo, n_hi = d_lo / max(r_ref, 1e-6), d_hi / max(r_ref, 1e-6)
    # g_d decreases in n
    gd_lo = (open_mult - n_hi) / denom
    gd_hi = (open_mult - n_lo) / denom
    gd_lo, gd_hi = float(np.clip(gd_lo, 0.0, 1.0)), float(np.clip(gd_hi, 0.0, 1.0))
    if gd_lo > gd_hi:
        gd_lo, gd_hi = gd_hi, gd_lo

    it_lo, it_hi = ibp_max_coords(lo, hi, inv_ttc_indices())
    gt_lo = float(np.clip(it_lo / max(inv_ttc_ref, 1e-6), 0.0, 1.0))
    gt_hi = float(np.clip(it_hi / max(inv_ttc_ref, 1e-6), 0.0, 1.0))

    # g = (1-mix)*gd + mix*gt
    g_lo = (1.0 - mix) * gd_lo + mix * gt_lo
    g_hi = (1.0 - mix) * gd_hi + mix * gt_hi
    return float(np.clip(min(g_lo, g_hi), 0.0, 1.0)), float(np.clip(max(g_lo, g_hi), 0.0, 1.0))


def ibp_product(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> tuple[float, float]:
    """Interval product [a]*[b]."""
    corners = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
    return float(min(corners)), float(max(corners))


def ibp_residual_action(
    w: MFResidualWeights,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sound action bounds for a = clip(base + g*clip(delta)).

    Returns a_lo, a_hi, info.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    b = ibp_mlp(w.base, lo, hi)
    d = ibp_mlp(w.delta, lo, hi)
    # clip delta
    d = d.clip(-w.delta_clip, w.delta_clip)
    g_lo, g_hi = ibp_physics_gate(
        lo,
        hi,
        open_mult=w.open_mult,
        close_mult=w.close_mult,
        r_ref=w.r_ref,
        inv_ttc_ref=w.inv_ttc_ref,
        mix=w.gate_mix,
    )
    # g * delta componentwise
    z_lo = np.zeros(2)
    z_hi = np.zeros(2)
    for j in range(2):
        z_lo[j], z_hi[j] = ibp_product(g_lo, g_hi, float(d.lo[j]), float(d.hi[j]))
    raw_lo = b.lo + z_lo
    raw_hi = b.hi + z_hi
    a_lo = np.clip(raw_lo, -1.0, 1.0)
    a_hi = np.clip(raw_hi, -1.0, 1.0)
    # clip is monotonic componentwise on intervals already clipped above
    info = {
        "base_lo": b.lo.tolist(),
        "base_hi": b.hi.tolist(),
        "delta_lo": d.lo.tolist(),
        "delta_hi": d.hi.tolist(),
        "g_lo": g_lo,
        "g_hi": g_hi,
    }
    return a_lo, a_hi, info


def ibp_two_step_next_rel(
    rel_lo: np.ndarray,
    rel_hi: np.ndarray,
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    *,
    dt: float = 0.03,
    mass: float = 0.1,
    n: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sound interval for next relative pos after n held-action Euler steps.

    d_p' = d_p + n*dt*d_v + (n(n-1)/2)*dt^2*(a/mass)
    (coast neighbor / static hit → acc_other=0)
    """
    rel_lo = np.asarray(rel_lo, dtype=np.float64)
    rel_hi = np.asarray(rel_hi, dtype=np.float64)
    a_lo = np.asarray(a_lo, dtype=np.float64)
    a_hi = np.asarray(a_hi, dtype=np.float64)
    cv = n * dt
    ca = (n * (n - 1) / 2.0) * (dt * dt) / mass
    # next dx = dx + cv*dvx + ca*ax
    nlo = np.zeros(2)
    nhi = np.zeros(2)
    for i in range(2):
        # affine combination of intervals
        terms_lo = [rel_lo[i], cv * rel_lo[2 + i], ca * a_lo[i]]
        terms_hi = [rel_hi[i], cv * rel_hi[2 + i], ca * a_hi[i]]
        # each term interval; sum
        s_lo = 0.0
        s_hi = 0.0
        for tl, th in (
            (rel_lo[i], rel_hi[i]),
            (cv * rel_lo[2 + i], cv * rel_hi[2 + i]),
            (ca * a_lo[i], ca * a_hi[i]),
        ):
            s_lo += min(tl, th)
            s_hi += max(tl, th)
        nlo[i], nhi[i] = s_lo, s_hi
        _ = (terms_lo, terms_hi)
    return nlo, nhi


def ibp_square_safe(
    next_lo: np.ndarray,
    next_hi: np.ndarray,
    *,
    threshold: float,
) -> bool:
    """True if next pos interval is **disjoint** from [-thr, thr]^2 (safe proven).

    If the entire next box lies outside the square, collision square-proxy is
    impossible. If next box intersects the square, IBP is inconclusive.
    """
    thr = float(threshold)
    # intersect next box with danger square?
    ix_lo = max(float(next_lo[0]), -thr)
    ix_hi = min(float(next_hi[0]), thr)
    iy_lo = max(float(next_lo[1]), -thr)
    iy_hi = min(float(next_hi[1]), thr)
    intersects = (ix_lo <= ix_hi) and (iy_lo <= iy_hi)
    return not intersects


def ibp_claim_status(
    w: MFResidualWeights,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
    rel_lo: np.ndarray,
    rel_hi: np.ndarray,
    *,
    radius: float,
) -> dict:
    """IBP attempt to prove square-proxy safety on a sub-box."""
    a_lo, a_hi, info = ibp_residual_action(w, feat_lo, feat_hi)
    nlo, nhi = ibp_two_step_next_rel(rel_lo, rel_hi, a_lo, a_hi)
    thr = radius / np.sqrt(2.0)  # square inside circle of radius
    proven = ibp_square_safe(nlo, nhi, threshold=thr)
    return {
        "status": "proven" if proven else "unknown",
        "method": "ibp_square",
        "a_lo": a_lo.tolist(),
        "a_hi": a_hi.tolist(),
        "next_lo": nlo.tolist(),
        "next_hi": nhi.tolist(),
        "square_thr": float(thr),
        "radius": float(radius),
        **info,
    }
