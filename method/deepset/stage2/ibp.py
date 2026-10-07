"""Interval Bound Propagation (IBP) for DeepSet buckets.

IBP is incomplete but sound for proving action bounds: if the over-approx
output box lies inside [-1,1]^2, the true network also respects the bounds
on that input box. If IBP is inconclusive, a complete solver (Marabou) may
still prove or refute the property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from stage2.encode import (
    EGO_DIM,
    ENTITY_CONT_DIM,
    DeepSetWeights,
    build_entity_row,
)


@dataclass
class Interval:
    lo: np.ndarray
    hi: np.ndarray

    def relu(self) -> "Interval":
        return Interval(np.maximum(self.lo, 0.0), np.maximum(self.hi, 0.0))


def _affine(x: Interval, w: np.ndarray, b: np.ndarray) -> Interval:
    """Sound interval image of y = x @ W + b (row-vector convention)."""
    c = 0.5 * (x.lo + x.hi)
    r = 0.5 * (x.hi - x.lo)
    yc = c @ w + b
    yr = r @ np.abs(w)
    return Interval(yc - yr, yc + yr)


def _entity_interval(
    cont_lo: np.ndarray,
    cont_hi: np.ndarray,
    *,
    is_agent: bool,
) -> Interval:
    """Interval over one 7-D entity with fixed type/presence."""
    # Continuous part varies; type/presence are constants.
    ta, to, pr = (1.0, 0.0, 1.0) if is_agent else (0.0, 1.0, 1.0)
    lo = np.array(
        [cont_lo[0], cont_lo[1], cont_lo[2], cont_lo[3], ta, to, pr],
        dtype=np.float64,
    )
    hi = np.array(
        [cont_hi[0], cont_hi[1], cont_hi[2], cont_hi[3], ta, to, pr],
        dtype=np.float64,
    )
    return Interval(lo, hi)


def ibp_action_bounds(
    weights: DeepSetWeights,
    ego_lo: np.ndarray,
    ego_hi: np.ndarray,
    agent_cont_bounds: Sequence[Tuple[np.ndarray, np.ndarray]],
    obs_cont_bounds: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (action_lo, action_hi) over-approx via IBP."""
    # φ pooled sum of intervals
    pooled_lo = np.zeros((weights.hidden,), dtype=np.float64)
    pooled_hi = np.zeros((weights.hidden,), dtype=np.float64)

    def add_entity(cont_lo: np.ndarray, cont_hi: np.ndarray, is_agent: bool) -> None:
        nonlocal pooled_lo, pooled_hi
        e = _entity_interval(cont_lo, cont_hi, is_agent=is_agent)
        h = _affine(e, weights.phi0_w, weights.phi0_b).relu()
        h = _affine(h, weights.phi1_w, weights.phi1_b).relu()
        h = _affine(h, weights.phi2_w, weights.phi2_b).relu()
        pooled_lo = pooled_lo + h.lo
        pooled_hi = pooled_hi + h.hi

    for lo, hi in agent_cont_bounds:
        add_entity(np.asarray(lo), np.asarray(hi), True)
    for lo, hi in obs_cont_bounds:
        add_entity(np.asarray(lo), np.asarray(hi), False)

    ego = Interval(np.asarray(ego_lo, dtype=np.float64), np.asarray(ego_hi, dtype=np.float64))
    y = Interval(
        np.concatenate([ego.lo, pooled_lo]),
        np.concatenate([ego.hi, pooled_hi]),
    )
    y = _affine(y, weights.rho0_w, weights.rho0_b).relu()
    y = _affine(y, weights.rho1_w, weights.rho1_b).relu()
    y = _affine(y, weights.rho2_w, weights.rho2_b).relu()
    out = _affine(y, weights.act_w, weights.act_b)
    return out.lo, out.hi


def clip_interval(lo: np.ndarray, hi: np.ndarray, limit: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Interval image of clip(·, -limit, limit)."""
    return np.clip(lo, -limit, limit), np.clip(hi, -limit, limit)


def ibp_clipped_action_from_flat_box(
    weights: DeepSetWeights,
    k: int,
    m: int,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    limit: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = ibp_from_flat_box(weights, k, m, box_lo, box_hi)
    return clip_interval(lo, hi, limit)


def ibp_from_flat_box(
    weights: DeepSetWeights,
    k: int,
    m: int,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ego_lo = box_lo[:EGO_DIM]
    ego_hi = box_hi[:EGO_DIM]
    off = EGO_DIM
    agents: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(k):
        agents.append((box_lo[off : off + ENTITY_CONT_DIM], box_hi[off : off + ENTITY_CONT_DIM]))
        off += ENTITY_CONT_DIM
    obs: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(m):
        obs.append((box_lo[off : off + ENTITY_CONT_DIM], box_hi[off : off + ENTITY_CONT_DIM]))
        off += ENTITY_CONT_DIM
    return ibp_action_bounds(weights, ego_lo, ego_hi, agents, obs)


# keep build_entity_row imported for symmetry / future use
_ = build_entity_row
