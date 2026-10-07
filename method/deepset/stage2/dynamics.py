"""DoubleIntegrator Euler used by official env.step, plus L2 next-state maps.

Official ``agent_step_euler`` is:

    accel = a / mass
    p' = p + v * dt
    v' = clip(v + accel * dt, ±v_lim)

so the **first** position update does not depend on the current action.
Collision after one Euler step is therefore independent of π. L2 uses
**two Euler steps with a held action** so acceleration reaches position:

    p'' = p + 2 v dt + accel dt^2
    v'' = clip(clip(v + accel dt, ±v_lim) + accel dt, ±v_lim)

This matches two consecutive env.step calls with the same clipped action
(and frozen neighbor/obstacle motion under the L2 assumptions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from stage2.encode import clip_action

DT = 0.03
MASS = 0.1
V_LIM = 0.5
CAR_RADIUS = 0.05
COLLISION_AA = 2.0 * CAR_RADIUS  # 0.10
COLLISION_AO = CAR_RADIUS  # 0.05 hit-ball proxy
HORIZON_STEPS = 2


def clip_velocity(v: np.ndarray, v_lim: float = V_LIM) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.float64), -v_lim, v_lim)


def accel_from_action(a: np.ndarray, mass: float = MASS) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) / mass


def euler_step(
    p: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    *,
    dt: float = DT,
    mass: float = MASS,
    v_lim: float = V_LIM,
) -> Tuple[np.ndarray, np.ndarray]:
    """One official Euler step: p' = p+v dt, v' = clip(v + a/m dt)."""
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    acc = accel_from_action(a, mass)
    p_n = p + v * dt
    v_n = clip_velocity(v + acc * dt, v_lim)
    return p_n, v_n


def euler_n(
    p: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    n: int = HORIZON_STEPS,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    for _ in range(n):
        p, v = euler_step(p, v, a, **kwargs)
    return p, v


@dataclass(frozen=True)
class RelState:
    """Relative geometry stored in DeepSet entity slots: other relative to ego.

    extract_local_sets uses (s_ego - s_other), so:
        dx, dy   = p_ego - p_other
        dvx, dvy = v_ego - v_other
    """

    dx: float
    dy: float
    dvx: float
    dvy: float

    @property
    def d_pos(self) -> np.ndarray:
        return np.array([self.dx, self.dy], dtype=np.float64)

    @property
    def d_vel(self) -> np.ndarray:
        return np.array([self.dvx, self.dvy], dtype=np.float64)


def rel_after_held_action(
    rel: RelState,
    a_ego: np.ndarray,
    *,
    n: int = HORIZON_STEPS,
    dt: float = DT,
    mass: float = MASS,
    v_lim: float = V_LIM,
    neighbor_accel: np.ndarray | None = None,
    other_static: bool = False,
) -> RelState:
    """Relative pose after n Euler steps.

    AA (other_static=False): neighbor coasts (accel=0) unless neighbor_accel given.
    AO (other_static=True): hit / obstacle does not move; relative vel is just ego vel
    expressed in the stored (v_ego - 0) convention if the hit has zero velocity.
    """
    a_ego = clip_action(a_ego)
    acc_e = accel_from_action(a_ego, mass)
    if other_static:
        acc_o = np.zeros(2, dtype=np.float64)
        # Stored dv = v_ego - v_hit; hits are encoded with zero velocity in extract.
        v_e = rel.d_vel
        v_o = np.zeros(2, dtype=np.float64)
    else:
        acc_o = (
            np.zeros(2, dtype=np.float64)
            if neighbor_accel is None
            else accel_from_action(clip_action(neighbor_accel), mass)
        )
        # We do not have absolute velocities of both; relative kinematics suffice:
        # d_pos' = d_pos + d_vel * dt
        # d_vel' = clip_e(v_e+acc_e dt) - clip_o(v_o+acc_o dt)
        # Without absolute v, velocity clips are not exact. L2 v1 ignores v-clip in
        # the relative map (soundness note: |v|<=v_lim is assumed in the input box).
        v_e = None
        v_o = None

    d_p = rel.d_pos.copy()
    d_v = rel.d_vel.copy()
    for _ in range(n):
        d_p = d_p + d_v * dt
        d_v = d_v + (acc_e - acc_o) * dt
    return RelState(float(d_p[0]), float(d_p[1]), float(d_v[0]), float(d_v[1]))


def next_rel_aa_coast(rel: RelState, a_ego: np.ndarray, n: int = HORIZON_STEPS) -> RelState:
    """L2-AA: neighbor acceleration = 0, held ego action, n Euler steps."""
    return rel_after_held_action(rel, a_ego, n=n, other_static=False, neighbor_accel=None)


def next_rel_ao_static(rel: RelState, a_ego: np.ndarray, n: int = HORIZON_STEPS) -> RelState:
    """L2-AO: lidar hit treated as static point."""
    return rel_after_held_action(rel, a_ego, n=n, other_static=True)
