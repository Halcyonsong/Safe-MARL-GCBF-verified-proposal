"""Bounded input domains for Stage2 L1 / L2 queries.

First-version buckets intentionally use tight boxes so Marabou/IBP have a
chance on the large DeepSet Actor (hidden=512).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from stage2.encode import EGO_DIM, ENTITY_CONT_DIM, input_dim_for_bucket


@dataclass(frozen=True)
class Box:
    """Axis-aligned input box: lo[i] <= x[i] <= hi[i]."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        lo = np.asarray(self.lo, dtype=np.float64)
        hi = np.asarray(self.hi, dtype=np.float64)
        if lo.shape != hi.shape:
            raise ValueError(f"lo/hi shape mismatch: {lo.shape} vs {hi.shape}")
        if np.any(lo > hi):
            raise ValueError("lo > hi in box")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def dim(self) -> int:
        return int(self.lo.shape[0])

    def center(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lo, self.hi)

    def as_dict(self) -> Dict[str, List[float]]:
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}


@dataclass(frozen=True)
class BucketSpec:
    """One verification bucket: fixed presence counts + continuous box."""

    name: str
    k: int  # active agent-neighbor slots
    m: int  # active obstacle-hit slots
    box: Box
    property: str = "action_bounds"  # L1
    action_limit: float = 1.0
    notes: str = ""

    def input_dim(self) -> int:
        return input_dim_for_bucket(self.k, self.m)

    def describe(self) -> str:
        return (
            f"{self.name}: k={self.k}, m={self.m}, dim={self.input_dim()}, "
            f"prop={self.property}, |a|<={self.action_limit}"
        )


def _ego_bounds(
    *,
    area: float = 4.0,
    v_abs: float = 1.5,
    goal_abs: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    # [px, py, vx, vy, goal_dx, goal_dy, goal_dvx, goal_dvy]
    lo = np.array(
        [0.0, 0.0, -v_abs, -v_abs, -goal_abs, -goal_abs, -v_abs, -v_abs],
        dtype=np.float64,
    )
    hi = np.array(
        [area, area, v_abs, v_abs, goal_abs, goal_abs, v_abs, v_abs],
        dtype=np.float64,
    )
    return lo, hi


def _entity_cont_bounds(
    *,
    dist_abs: float,
    dv_abs: float,
    dist_min: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bounds on [dx, dy, dvx, dvy].

    Note: axis-aligned box over-approximates the disk ||(dx,dy)|| <= dist_abs.
    dist_min is recorded for documentation; first version does not encode the
    nonlinear annulus constraint in the linear box (sound over-approx).
    """
    del dist_min  # reserved for future nonlinear constraints
    lo = np.array([-dist_abs, -dist_abs, -dv_abs, -dv_abs], dtype=np.float64)
    hi = np.array([dist_abs, dist_abs, dv_abs, dv_abs], dtype=np.float64)
    return lo, hi


def make_box(k: int, m: int, *, v_abs: float, dist_abs: float, dv_abs: float) -> Box:
    ego_lo, ego_hi = _ego_bounds(v_abs=v_abs)
    e_lo, e_hi = _entity_cont_bounds(dist_abs=dist_abs, dv_abs=dv_abs)
    lo_parts = [ego_lo]
    hi_parts = [ego_hi]
    for _ in range(k + m):
        lo_parts.append(e_lo)
        hi_parts.append(e_hi)
    return Box(np.concatenate(lo_parts), np.concatenate(hi_parts))


def tiny_center_box(k: int, m: int, *, radius: float = 1e-3) -> Box:
    """Tiny box around a nominal center — for Marabou toolchain smoke proofs."""
    base = make_box(k, m, v_abs=1.5, dist_abs=0.5, dv_abs=3.0)
    c = base.center()
    # Keep px,py away from boundary artifacts: use area center already.
    lo = c - radius
    hi = c + radius
    return Box(lo, hi)


def tiny_l2_box(k: int, m: int, *, radius: float = 1e-3, dist: float = 0.15) -> Box:
    """Tiny L2 smoke box: area center, small velocity, neighbor/hit along +x at `dist`."""
    base = make_box(k, m, v_abs=0.5, dist_abs=0.2, dv_abs=1.0)
    c = base.center()
    # ego: (px,py)=(2,2), v=0, goal rel=0
    c[:8] = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    off = 8
    for _ in range(k + m):
        c[off : off + 4] = np.array([dist, 0.0, 0.0, 0.0], dtype=np.float64)
        off += 4
    return Box(c - radius, c + radius)


def default_l1_buckets() -> List[BucketSpec]:
    """First-version L1 action-bound buckets.

    Tight communication-scale distances (0.5) and moderate speeds keep the
    continuous domain manageable for the large Actor512 network.
    """
    # Shared continuous scales for v1
    v_abs = 1.5
    dist_abs = 0.5  # within comm_radius
    dv_abs = 3.0

    specs = [
        BucketSpec(
            name="km_0_0_smoke",
            k=0,
            m=0,
            box=tiny_center_box(0, 0, radius=1e-3),
            notes=(
                "Toolchain smoke: tiny box around empty-local center. "
                "Used to confirm Marabou can return UNSAT on Actor512."
            ),
        ),
        BucketSpec(
            name="km_0_0",
            k=0,
            m=0,
            box=make_box(0, 0, v_abs=v_abs, dist_abs=dist_abs, dv_abs=dv_abs),
            notes="Empty local set (no neighbors / obstacle hits).",
        ),
        BucketSpec(
            name="km_1_0",
            k=1,
            m=0,
            box=make_box(1, 0, v_abs=v_abs, dist_abs=dist_abs, dv_abs=dv_abs),
            notes="Single agent neighbor in communication range.",
        ),
        BucketSpec(
            name="km_0_1",
            k=0,
            m=1,
            box=make_box(0, 1, v_abs=v_abs, dist_abs=dist_abs, dv_abs=dv_abs),
            notes="Single obstacle lidar hit in communication range.",
        ),
        BucketSpec(
            name="km_1_1",
            k=1,
            m=1,
            box=make_box(1, 1, v_abs=v_abs, dist_abs=dist_abs, dv_abs=dv_abs),
            notes="One neighbor + one obstacle hit.",
        ),
        # Tighter near-danger boxes (same counts, smaller distance)
        BucketSpec(
            name="km_1_0_near",
            k=1,
            m=0,
            box=make_box(1, 0, v_abs=v_abs, dist_abs=0.2, dv_abs=dv_abs),
            notes="Single neighbor in near-danger distance box (<=0.2).",
        ),
        BucketSpec(
            name="km_0_1_near",
            k=0,
            m=1,
            box=make_box(0, 1, v_abs=v_abs, dist_abs=0.2, dv_abs=dv_abs),
            notes="Single obstacle hit in near-danger distance box (<=0.2).",
        ),
    ]
    for s in specs:
        if s.box.dim != s.input_dim():
            raise RuntimeError(f"box dim mismatch for {s.name}")
    return specs


def default_l2_buckets() -> List[BucketSpec]:
    """L2 preventable near-collision buckets: k,m in {(1,0),(0,1)}.

    (0,0) omitted. Outer axis-aligned box is a search envelope; the *claim set*
    D is the nonlinear filter (see stage2/l2_domain.py):

        ||Δp|| >= collision_radius
        AND  exists a with two-step dist'(a) < radius   (danger)
        AND  exists a with two-step dist'(a) >= radius  (controllable)

    Sampling / CE / repair operate on D. Farther / hard-unavoidable / already
    colliding states are excluded from the prevention claim.
    """
    v_abs = 0.5
    dv_abs = 1.0
    dist_abs = 0.20
    specs = [
        BucketSpec(
            name="l2_aa_near_smoke",
            k=1,
            m=0,
            box=tiny_l2_box(1, 0, radius=1e-3, dist=0.15),
            property="l2_aa",
            notes=(
                "Toolchain smoke for L2-AA. Center at dist=0.15 > r=0.1 "
                "(initially safe); neighbor coasts; two Euler steps."
            ),
        ),
        BucketSpec(
            name="l2_aa_near",
            k=1,
            m=0,
            box=make_box(1, 0, v_abs=v_abs, dist_abs=dist_abs, dv_abs=dv_abs),
            property="l2_aa",
            notes=(
                "Outer box ||rel_xy||_∞<=0.2. Claim set D = dist>=0.1 AND "
                "min_a dist'<r AND max_a dist'>=r (controllable danger). "
                "Neighbor coasts; a=clip(DeepSet); two Euler steps."
            ),
        ),
        BucketSpec(
            name="l2_ao_near_smoke",
            k=0,
            m=1,
            box=tiny_l2_box(0, 1, radius=1e-3, dist=0.15),
            property="l2_ao",
            notes="Toolchain smoke for L2-AO (static hit ball); center dist=0.15>r.",
        ),
        BucketSpec(
            name="l2_ao_near",
            k=0,
            m=1,
            box=make_box(0, 1, v_abs=v_abs, dist_abs=dist_abs, dv_abs=dv_abs),
            property="l2_ao",
            notes=(
                "Outer box ||rel_xy||_∞<=0.2. Claim set D = dist>=0.05 AND "
                "min_a dist'<r AND max_a dist'>=r. Hit static; hit-ball=car_radius."
            ),
        ),
        BucketSpec(
            name="l2_11_near",
            k=1,
            m=1,
            box=make_box(1, 1, v_abs=v_abs, dist_abs=dist_abs, dv_abs=dv_abs),
            property="l2_aa_ao",
            notes=(
                "Outer box ||rel||_∞<=0.2 for neighbor AND hit. Claim set D = "
                "initially safe vs both (dist_aa>=0.1, dist_ao>=0.05) AND "
                "exists a that hits either AND exists a that avoids both. "
                "Neighbor coasts; hit static; a=clip(DeepSet); two Euler steps."
            ),
        ),
    ]
    for s in specs:
        if s.box.dim != s.input_dim():
            raise RuntimeError(f"box dim mismatch for {s.name}")
    return specs


# Silence unused import warning for documentation of feature layout
_ = (EGO_DIM, ENTITY_CONT_DIM)
