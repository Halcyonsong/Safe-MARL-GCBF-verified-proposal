"""Collision proxies for Stage2 L2 (local one/two-step safety)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage2.dynamics import COLLISION_AA, COLLISION_AO, RelState


@dataclass(frozen=True)
class CollisionEval:
    collided: bool
    dist: float
    threshold: float
    kind: str  # "aa" | "ao"

    def as_dict(self) -> dict:
        return {
            "collided": self.collided,
            "dist": self.dist,
            "threshold": self.threshold,
            "kind": self.kind,
        }


def dist_rel(rel: RelState) -> float:
    return float(np.linalg.norm(rel.d_pos))


def eval_aa(rel: RelState, threshold: float = COLLISION_AA) -> CollisionEval:
    d = dist_rel(rel)
    return CollisionEval(d < threshold, d, threshold, "aa")


def eval_ao(rel: RelState, threshold: float = COLLISION_AO) -> CollisionEval:
    """Observed-hit ball: ||p_ego - hit|| < car_radius."""
    d = dist_rel(rel)
    return CollisionEval(d < threshold, d, threshold, "ao")


def eval_aa_ao(
    rel_aa: RelState,
    rel_ao: RelState,
    *,
    thr_aa: float = COLLISION_AA,
    thr_ao: float = COLLISION_AO,
) -> CollisionEval:
    """(1,1) dual threat: collision if either AA or AO proxy is violated.

    ``dist`` reports the tighter normalized clearance in absolute meters of the
    worse threat: min(dist_aa, dist_ao) is not comparable across thresholds, so
    we store the min absolute distance among colliding/near threats and mark
    kind as ``aa_ao``.
    """
    ea = eval_aa(rel_aa, thr_aa)
    eo = eval_ao(rel_ao, thr_ao)
    collided = ea.collided or eo.collided
    # clearance scores in units of radius (higher = safer)
    clear_a = ea.dist / thr_aa
    clear_o = eo.dist / thr_ao
    if clear_a <= clear_o:
        return CollisionEval(collided, ea.dist, thr_aa, "aa_ao")
    return CollisionEval(collided, eo.dist, thr_ao, "aa_ao")
