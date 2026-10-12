"""Planar two-step L2 property (side-branch formal) — NOT radial |a|_∞.

Matches training labels in l2_labels.py / collect_ce.py:

    rel = (dx, dy, dvx, dvy)   # nearest threat, neighbor coast / hit static
    a   = Deploy(x)            # network on 44-D features
    p2  = two-step Euler(rel, a)
    Safe ⇔ ||p2||_2 ≥ R

Signed acceleration matters: braking / lateral a change p2 (unlike radial |a|_∞).

Reachable observation contract (v1, 2026-10-10)
----------------------------------------------
True ``extract_meanfield_features`` satisfies:

  - goal body velocity = 0  ⇒  goal_dv ≡ -v_ego  (not a free input)
  - goal_xy is soft-capped to ‖g‖ ≤ comm_radius (default 0.5)
  - empty quadrant min_d = comm_radius
  - threat quadrant (n, min_d, close, inv_ttc) coupled to ``rel``

The claim feature box MUST over-approx only this reachable set.
Legacy boxes with goal_xy ∈ [-2,2] and free goal_dv produced unreachable CEs.

Soundness
---------
- Neighbor a=0 (coast); obstacle static — same as CE labels.
- Feature box coupled to rel (Q0/Q3 min_d & close from geometry).
- IBP PROVEN uses **min Euclidean distance from origin to p2 AABB** ≥ R
  (NOT the inscribed-square proxy, which is unsound for the circle).
- Axis-aligned goal_xy ∈ [-comm,comm] over-approx the disk ‖g‖≤comm (sound,
  slightly larger than exact extract image). Sampling uses the disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stage2.dynamics import COLLISION_AA, DT, HORIZON_STEPS, MASS, V_LIM
from stage_mf.encode import MFResidualWeights, deploy_action_np, min_d_indices
from stage_mf.ibp_mf import ibp_residual_action, ibp_two_step_next_rel
from stage_mf.l2_labels import _two_step_rel_batch
from stage_mf.property_l2 import close_max_indices
from stage_mf.student import INPUT_DIM

# Must match env params / extract_meanfield_features
COMM_RADIUS_DEFAULT = 0.5
D_EMPTY_DEFAULT = COMM_RADIUS_DEFAULT


def euler_coeffs(n: int = HORIZON_STEPS, dt: float = DT, mass: float = MASS) -> tuple[float, float]:
    cv = float(n * dt)
    ca = float((n * (n - 1) / 2.0) * (dt * dt) / mass)
    return cv, ca


def planar_p2(rel: np.ndarray, a: np.ndarray) -> np.ndarray:
    rel = np.asarray(rel, dtype=np.float64).reshape(4)
    a = np.clip(np.asarray(a, dtype=np.float64).reshape(2), -1.0, 1.0)
    rn = _two_step_rel_batch(rel[None, :], a[None, :], other_static=False)[0]
    return rn[:2].copy()


def planar_safe(
    rel: np.ndarray,
    a: np.ndarray,
    *,
    radius: float = COLLISION_AA,
) -> tuple[bool, dict[str, Any]]:
    p2 = planar_p2(rel, a)
    d = float(np.linalg.norm(p2))
    return d >= radius - 1e-9, {
        "p2": p2.tolist(),
        "d2": d,
        "radius": float(radius),
        "a": np.asarray(a, dtype=np.float64).reshape(2).tolist(),
    }


def is_planar_safe_point(
    w: MFResidualWeights,
    x: np.ndarray,
    rel: np.ndarray,
    *,
    radius: float = COLLISION_AA,
) -> tuple[bool, dict[str, Any]]:
    a = deploy_action_np(w, x)
    ok, info = planar_safe(rel, a, radius=radius)
    info["x_ok"] = True
    return ok, info


def min_norm_aabb(lo: np.ndarray, hi: np.ndarray) -> float:
    """Min ||p||_2 over p in axis-aligned box [lo, hi] (2-D). Sound lower bound."""
    lo = np.asarray(lo, dtype=np.float64).reshape(2)
    hi = np.asarray(hi, dtype=np.float64).reshape(2)
    d2 = 0.0
    for i in range(2):
        if hi[i] < 0.0:
            d2 += float(hi[i] * hi[i])
        elif lo[i] > 0.0:
            d2 += float(lo[i] * lo[i])
    return float(np.sqrt(d2))


def max_norm_aabb(lo: np.ndarray, hi: np.ndarray) -> float:
    """Max ||p||_2 over corners of AABB (upper bound)."""
    lo = np.asarray(lo, dtype=np.float64).reshape(2)
    hi = np.asarray(hi, dtype=np.float64).reshape(2)
    best = 0.0
    for sx in (lo[0], hi[0]):
        for sy in (lo[1], hi[1]):
            best = max(best, float(np.hypot(sx, sy)))
    return best


def _sync_goal_dv_bounds(flo: np.ndarray, fhi: np.ndarray) -> None:
    """goal_dv must track -v_ego: bounds = -flip(v bounds)."""
    # v in [2:4], goal_dv in [6:8]
    flo[6] = -float(fhi[2])
    fhi[6] = -float(flo[2])
    flo[7] = -float(fhi[3])
    fhi[7] = -float(flo[3])
    # ensure lo<=hi
    for i in (6, 7):
        if flo[i] > fhi[i]:
            flo[i], fhi[i] = fhi[i], flo[i]


def _apply_goal_dv_from_v(x: np.ndarray) -> None:
    """In-place: goal_dv := -v_ego (static goal body)."""
    x[6] = -float(x[2])
    x[7] = -float(x[3])


def _sample_goal_xy_disk(rng: np.random.Generator, comm: float) -> np.ndarray:
    """Uniform in disk ‖g‖ ≤ comm (matches extract soft-cap image better than AABB)."""
    # radius^2 uniform for area-uniform disk
    r = float(comm) * np.sqrt(float(rng.random()))
    th = float(rng.random() * 2.0 * np.pi)
    return np.array([r * np.cos(th), r * np.sin(th)], dtype=np.float64)


@dataclass(frozen=True)
class PlanarBox:
    """Joint claim box: reachable features + one threat rel (4-D), coupled."""

    feat_lo: np.ndarray
    feat_hi: np.ndarray
    rel_lo: np.ndarray
    rel_hi: np.ndarray
    radius: float = COLLISION_AA
    name: str = "planar_netdep_reachable"
    comm_radius: float = COMM_RADIUS_DEFAULT

    def __post_init__(self):
        object.__setattr__(self, "feat_lo", np.asarray(self.feat_lo, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "feat_hi", np.asarray(self.feat_hi, dtype=np.float64).reshape(-1))
        object.__setattr__(self, "rel_lo", np.asarray(self.rel_lo, dtype=np.float64).reshape(4))
        object.__setattr__(self, "rel_hi", np.asarray(self.rel_hi, dtype=np.float64).reshape(4))
        if int(self.feat_lo.size) != INPUT_DIM:
            raise ValueError(f"feat dim {self.feat_lo.size} != INPUT_DIM {INPUT_DIM}")

    @property
    def dim(self) -> int:
        return int(self.feat_lo.size + 4)

    def width_feat(self) -> np.ndarray:
        return self.feat_hi - self.feat_lo

    def width_rel(self) -> np.ndarray:
        return self.rel_hi - self.rel_lo

    def longest_axis(self) -> tuple[str, int]:
        """Prefer action-critical dims (rel, v, goal, threat) over ego px/py.

        Never split goal_dv as free dims — slaved to v.
        px/py are split last: they widen MLP IBP a lot but are lower priority
        than v/goal/threat for SafeSet proofs.
        """
        wr = self.width_rel()
        wf = self.width_feat()
        # priority tiers (highest first)
        rel_order = [0, 2, 1, 3]  # dx, dvx, dy, dvy
        # Split order for SafeSet+Marabou decidability:
        #   1) v/goal  2) inv_ttc/threat  3) pose  4) rel
        # inv_ttc root width ~1.1 (coupling over-approx) MUST shrink before Marabou.
        primary = [2, 3, 4, 5]  # vx,vy,goal_xy
        pose = [0, 1]
        threat_md_c = list(min_d_indices()) + list(close_max_indices())
        threat_it = [8 + 5 * q + 4 for q in range(4)] + [28 + 4 * q + 3 for q in range(4)]

        def _best(idxs: list[int]) -> tuple[int, float]:
            bi, bw = idxs[0], -1.0
            for i in idxs:
                if i < wf.size and float(wf[i]) > bw:
                    bw = float(wf[i])
                    bi = int(i)
            return bi, bw

        bp, pw = _best(primary)
        bpose, pose_w = _best(pose)
        bs, sw = _best(threat_md_c)
        bt, tw = _best(threat_it)
        best_rel = max(rel_order, key=lambda j: wr[j])
        rel_w = float(wr[best_rel])
        threat_w = max(sw, tw)

        # Marabou-friendly: vg≤0.2, threat≤0.15, pose≤0.5
        # Prefer shrinking inv_ttc early (dominant timeout cause).
        if threat_w > 0.30:
            return "feat", (bt if tw >= sw else bs)
        if pw > 0.20:
            return "feat", bp
        if threat_w > 0.15:
            return "feat", (bt if tw >= sw else bs)
        if pose_w > 0.50:
            return "feat", bpose
        if pw > 0.05:
            return "feat", bp
        if threat_w > 0.05:
            return "feat", (bt if tw >= sw else bs)
        if pose_w > 0.05:
            return "feat", bpose
        if rel_w > 1e-4:
            return "rel", int(best_rel)
        if pose_w > 1e-6:
            return "feat", bpose
        return "rel", int(best_rel)

    def split(self, which: str | None = None, axis: int | None = None) -> tuple["PlanarBox", "PlanarBox"]:
        if which is None or axis is None:
            which, axis = self.longest_axis()
        if which == "feat":
            # if asked to split goal_dv, redirect to corresponding v axis
            if axis in (6, 7):
                axis = axis - 4  # 6→2, 7→3
            mid = 0.5 * (self.feat_lo[axis] + self.feat_hi[axis])
            flo1, fhi1 = self.feat_lo.copy(), self.feat_hi.copy()
            flo2, fhi2 = self.feat_lo.copy(), self.feat_hi.copy()
            fhi1[axis] = mid
            flo2[axis] = mid
            if axis in (2, 3):
                _sync_goal_dv_bounds(flo1, fhi1)
                _sync_goal_dv_bounds(flo2, fhi2)
            b1 = PlanarBox(flo1, fhi1, self.rel_lo, self.rel_hi, self.radius, self.name, self.comm_radius)
            b2 = PlanarBox(flo2, fhi2, self.rel_lo, self.rel_hi, self.radius, self.name, self.comm_radius)
            return b1, b2
        mid = 0.5 * (self.rel_lo[axis] + self.rel_hi[axis])
        rlo1, rhi1 = self.rel_lo.copy(), self.rel_hi.copy()
        rlo2, rhi2 = self.rel_lo.copy(), self.rel_hi.copy()
        rhi1[axis] = mid
        rlo2[axis] = mid
        b1 = PlanarBox(self.feat_lo, self.feat_hi, rlo1, rhi1, self.radius, self.name, self.comm_radius)
        b2 = PlanarBox(self.feat_lo, self.feat_hi, rlo2, rhi2, self.radius, self.name, self.comm_radius)
        return _recouple_feat(b1), _recouple_feat(b2)

    def sample(self, rng: np.random.Generator, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Sample coupled reachable (x, rel)."""
        xs = np.zeros((n, self.feat_lo.size), dtype=np.float64)
        rels = np.zeros((n, 4), dtype=np.float64)
        for i in range(n):
            rel = self.rel_lo + rng.random(4) * (self.rel_hi - self.rel_lo)
            rels[i] = rel
            x = _feat_from_rel(rel, self.feat_lo, self.feat_hi, rng, comm_radius=self.comm_radius)
            x = np.minimum(np.maximum(x, self.feat_lo), self.feat_hi)
            _apply_goal_dv_from_v(x)
            xs[i] = x
        return xs, rels


def _feat_from_rel(
    rel: np.ndarray,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    comm_radius: float = COMM_RADIUS_DEFAULT,
) -> np.ndarray:
    """Build reachable 44-D feature consistent with nearest threat ``rel``.

    Contract:
      - threat quadrant from rel geometry
      - empty quads: n=0, min_d=comm
      - ego v free in box; goal_dv := -v
      - goal_xy sampled in disk ‖g‖≤comm (or mid if no rng)
    """
    rel = np.asarray(rel, dtype=np.float64).reshape(4)
    dx, dy, dvx, dvy = rel
    d = float(np.hypot(dx, dy)) + 1e-12
    close = float(-(dx * dvx + dy * dvy) / d)
    inv_ttc = float(np.clip(max(0.0, close) / max(d, 1e-3), 0.0, 20.0))
    comm = float(comm_radius)
    d_empty = comm

    if dx >= 0 and dy >= 0:
        q = 0
    elif dx < 0 and dy >= 0:
        q = 1
    elif dx < 0 and dy < 0:
        q = 2
    else:
        q = 3

    x = 0.5 * (feat_lo + feat_hi).copy()

    # ego pose: mild jitter in box (or mid)
    if rng is not None:
        x[0] = float(feat_lo[0] + rng.random() * (feat_hi[0] - feat_lo[0]))
        x[1] = float(feat_lo[1] + rng.random() * (feat_hi[1] - feat_lo[1]))
        x[2] = float(feat_lo[2] + rng.random() * (feat_hi[2] - feat_lo[2]))
        x[3] = float(feat_lo[3] + rng.random() * (feat_hi[3] - feat_lo[3]))
        g = _sample_goal_xy_disk(rng, min(comm, max(abs(feat_lo[4]), abs(feat_hi[4]), abs(feat_lo[5]), abs(feat_hi[5]))))
        # clip to feat box (AABB over-approx of disk)
        x[4] = float(np.clip(g[0], feat_lo[4], feat_hi[4]))
        x[5] = float(np.clip(g[1], feat_lo[5], feat_hi[5]))
    else:
        x[0:2] = 0.5 * (feat_lo[0:2] + feat_hi[0:2])
        x[2:4] = 0.0
        x[4:6] = 0.0

    _apply_goal_dv_from_v(x)

    # agent quads empty then occupy threat
    for qq in range(4):
        b = 8 + qq * 5
        x[b + 0] = 0.0
        x[b + 1] = d_empty
        x[b + 2] = 0.0
        x[b + 3] = 0.0
        x[b + 4] = 0.0
    b = 8 + q * 5
    x[b + 0] = 1.0
    x[b + 1] = d
    x[b + 2] = close
    x[b + 3] = close
    x[b + 4] = inv_ttc

    # obs empty
    for qq in range(4):
        b = 28 + qq * 4
        x[b + 0] = 0.0
        x[b + 1] = d_empty
        x[b + 2] = 0.0
        x[b + 3] = 0.0

    return x


def _recouple_feat(box: PlanarBox) -> PlanarBox:
    """Tighten threat-quad features to rel; keep reachable ego/goal contract."""
    flo, fhi = box.feat_lo.copy(), box.feat_hi.copy()
    rlo, rhi = box.rel_lo, box.rel_hi
    comm = float(box.comm_radius)

    d_lo = min_norm_aabb(rlo[:2], rhi[:2])
    d_hi = max_norm_aabb(rlo[:2], rhi[:2])
    # Bound close = -(dx*dvx + dy*dvy)/||dp|| by interval arithmetic.
    # Corners of this quotient are not generally the extrema, so do not use
    # corner-only evaluation here.
    def _mul_bounds(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float]:
        vals = (a0 * b0, a0 * b1, a1 * b0, a1 * b1)
        return float(min(vals)), float(max(vals))

    nx0, nx1 = _mul_bounds(rlo[0], rhi[0], rlo[2], rhi[2])
    ny0, ny1 = _mul_bounds(rlo[1], rhi[1], rlo[3], rhi[3])
    num_lo, num_hi = nx0 + ny0, nx1 + ny1
    den_lo = max(d_lo, 1e-12)
    den_hi = max(d_hi, den_lo)
    ratio = (
        num_lo / den_lo,
        num_lo / den_hi,
        num_hi / den_lo,
        num_hi / den_hi,
    )
    c_lo, c_hi = -float(max(ratio)), -float(min(ratio))

    quads = set()
    if rhi[1] >= 0:
        quads.add(0)
    if rlo[1] < 0:
        quads.add(3)
    if not quads:
        quads = {0, 3}

    for q in range(4):
        b = 8 + q * 5
        if q in quads:
            flo[b + 0], fhi[b + 0] = 1.0, 1.0  # single nearest threat
            flo[b + 1] = d_lo - 1e-9
            fhi[b + 1] = d_hi + 1e-9
            if flo[b + 1] > fhi[b + 1]:
                flo[b + 1], fhi[b + 1] = d_lo, d_hi
            flo[b + 2] = c_lo - 1e-9
            fhi[b + 2] = c_hi + 1e-9
            if flo[b + 2] > fhi[b + 2]:
                flo[b + 2], fhi[b + 2] = c_lo, c_hi
            flo[b + 3], fhi[b + 3] = flo[b + 2], fhi[b + 2]
            # inv_ttc band from close/d
            it_lo = max(0.0, c_lo) / max(d_hi, 1e-3)
            it_hi = max(0.0, c_hi) / max(d_lo, 1e-3)
            flo[b + 4] = max(0.0, it_lo - 1e-6)
            fhi[b + 4] = min(20.0, it_hi + 1e-6)
        else:
            flo[b + 0], fhi[b + 0] = 0.0, 0.0
            flo[b + 1], fhi[b + 1] = comm, comm
            flo[b + 2], fhi[b + 2] = 0.0, 0.0
            flo[b + 3], fhi[b + 3] = 0.0, 0.0
            flo[b + 4], fhi[b + 4] = 0.0, 0.0

    for q in range(4):
        b = 28 + q * 4
        flo[b + 0], fhi[b + 0] = 0.0, 0.0
        flo[b + 1], fhi[b + 1] = comm, comm
        flo[b + 2], fhi[b + 2] = 0.0, 0.0
        flo[b + 3], fhi[b + 3] = 0.0, 0.0

    # Reachable ego/goal. Preserve narrower bounds made by previous feature
    # splits; rel splits must not reset the already-partitioned feature box.
    flo[0:2] = np.maximum(flo[0:2], 0.0)
    fhi[0:2] = np.minimum(fhi[0:2], 4.0)
    flo[2:4] = np.maximum(flo[2:4], -float(V_LIM))
    fhi[2:4] = np.minimum(fhi[2:4], float(V_LIM))
    flo[4:6] = np.maximum(flo[4:6], -comm)
    fhi[4:6] = np.minimum(fhi[4:6], comm)
    _sync_goal_dv_bounds(flo, fhi)

    return PlanarBox(flo, fhi, rlo, rhi, box.radius, box.name, comm)


def p2_aabb_for_action_box(
    rel_lo: np.ndarray,
    rel_hi: np.ndarray,
    a_lo: np.ndarray,
    a_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Sound p2 AABB and Euclidean min/max norms for a held-action box."""
    nlo, nhi = ibp_two_step_next_rel(rel_lo, rel_hi, a_lo, a_hi)
    return nlo, nhi, min_norm_aabb(nlo, nhi), max_norm_aabb(nlo, nhi)


def safeset_ax_min(
    rel_lo: np.ndarray,
    rel_hi: np.ndarray,
    *,
    radius: float,
    ay_abs: float = 1.0,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Sufficient SafeSet on this rel box: ax ≥ a_min, |ay| ≤ ay_abs.

    Sound: for every rel in the box and every a with ax≥a_min and |ay|≤ay_abs,
    the p2 AABB min-norm is ≥ R (hence planar safe).

    a_min is the *smallest* threshold that still works (binary search on IBP).
    If even ax=1 fails, a_min is set >1 (empty SafeSet under this template).
    """
    rel_lo = np.asarray(rel_lo, dtype=np.float64).reshape(4)
    rel_hi = np.asarray(rel_hi, dtype=np.float64).reshape(4)
    R = float(radius)
    ay = float(abs(ay_abs))

    def ok_at(ax: float) -> tuple[bool, float]:
        a_lo = np.array([ax, -ay], dtype=np.float64)
        a_hi = np.array([ax, ay], dtype=np.float64)
        # ax fixed: still use [ax,ax]×[-ay,ay]
        a_lo[0] = ax
        a_hi[0] = ax
        _, _, d_lo, _ = p2_aabb_for_action_box(rel_lo, rel_hi, a_lo, a_hi)
        return d_lo >= R - 1e-12, d_lo

    # If full a-box already safe under ay bound with ax in [-1,1] worst: need min ax
    ok1, d1 = ok_at(1.0)
    if not ok1:
        return {
            "kind": "ax_ge",
            "a_min": 1.0 + 1.0,  # empty
            "ay_abs": ay,
            "feasible": False,
            "d_lo_at_ax1": d1,
            "radius": R,
            "note": "even ax=1 with |ay|≤ay_abs cannot prove planar safe on this rel box",
        }
    ok_m1, _ = ok_at(-1.0)
    if ok_m1:
        return {
            "kind": "ax_ge",
            "a_min": -1.0,
            "ay_abs": ay,
            "feasible": True,
            "d_lo_at_ax1": d1,
            "radius": R,
            "note": "any ax∈[-1,1] with |ay|≤ay_abs is safe (envelope on this sub-box)",
        }
    lo, hi = -1.0, 1.0
    d_mid = 0.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        ok, d_mid = ok_at(mid)
        if ok:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return {
        "kind": "ax_ge",
        "a_min": float(hi),
        "ay_abs": ay,
        "feasible": True,
        "d_lo_at_amin": float(d_mid if ok_at(hi)[0] else d1),
        "d_lo_at_ax1": d1,
        "radius": R,
        "note": "sufficient: ax≥a_min and |ay|≤ay_abs ⇒ planar safe on rel box",
    }


def action_in_safeset(a: np.ndarray, ss: dict[str, Any], *, eps: float = 1e-9) -> bool:
    a = np.asarray(a, dtype=np.float64).reshape(2)
    if not ss.get("feasible", False):
        return False
    if ss.get("kind") == "ax_ge":
        return float(a[0]) >= float(ss["a_min"]) - eps and abs(float(a[1])) <= float(ss["ay_abs"]) + eps
    return False


def ibp_deploy_in_safeset(
    w: MFResidualWeights,
    feat_lo: np.ndarray,
    feat_hi: np.ndarray,
    ss: dict[str, Any],
) -> dict[str, Any]:
    """IBP: if Deploy action box ⊆ SafeSet ⇒ proven; if disjoint ⇒ conflict."""
    a_lo, a_hi, info = ibp_residual_action(w, feat_lo, feat_hi)
    out = {
        "a_lo": a_lo.tolist(),
        "a_hi": a_hi.tolist(),
        "safeset": {k: v for k, v in ss.items() if k != "note"},
        **info,
    }
    if not ss.get("feasible", False):
        out["status"] = "unknown"
        out["method"] = "ibp_safeset_ax_ge"
        out["detail"] = "safeset_empty"
        return out
    if ss.get("kind") != "ax_ge":
        out["status"] = "unknown"
        out["method"] = "ibp_safeset_ax_ge"
        return out
    a_min = float(ss["a_min"])
    ay_abs = float(ss["ay_abs"])
    # Deploy box ⊆ SafeSet?
    if float(a_lo[0]) >= a_min - 1e-12 and float(a_lo[1]) >= -ay_abs - 1e-12 and float(a_hi[1]) <= ay_abs + 1e-12:
        out["status"] = "proven"
        out["method"] = "ibp_safeset_ax_ge"
        return out
    # Deploy box entirely outside SafeSet? (all ax < a_min)
    if float(a_hi[0]) < a_min - 1e-12:
        out["status"] = "conflict"
        out["method"] = "ibp_safeset_ax_ge"
        return out
    out["status"] = "unknown"
    out["method"] = "ibp_safeset_ax_ge"
    return out


def ibp_planar_status(
    w: MFResidualWeights,
    box: PlanarBox,
) -> dict[str, Any]:
    """Primary planar check: SafeSet(a) then Deploy ⊆ SafeSet (IBP).

    Fallback diagnostic still reports p2 AABB under the IBP action box.
    """
    ss = safeset_ax_min(box.rel_lo, box.rel_hi, radius=box.radius, ay_abs=1.0)
    ss_st = ibp_deploy_in_safeset(w, box.feat_lo, box.feat_hi, ss)
    a_lo = np.asarray(ss_st["a_lo"], dtype=np.float64)
    a_hi = np.asarray(ss_st["a_hi"], dtype=np.float64)
    nlo, nhi, d_lo, d_hi = p2_aabb_for_action_box(box.rel_lo, box.rel_hi, a_lo, a_hi)
    R = float(box.radius)
    status = ss_st["status"]
    # strengthen: direct p2 AABB can still prove even if SafeSet template is coarse
    if status != "proven" and d_lo >= R - 1e-12:
        status = "proven"
        ss_st = {**ss_st, "status": "proven", "method": "ibp_planar_euclid_aabb"}
    if status != "conflict" and d_hi < R + 1e-12:
        status = "conflict"
        ss_st = {**ss_st, "status": "conflict", "method": "ibp_planar_euclid_aabb"}
    return {
        **ss_st,
        "status": status,
        "safeset_full": ss,
        "p2_lo": nlo.tolist(),
        "p2_hi": nhi.tolist(),
        "d_lo": d_lo,
        "d_hi": d_hi,
        "radius": R,
    }


def freeze_threat_midpoint(box: PlanarBox) -> PlanarBox:
    """Pin threat/quad features (dims 8:) to box midpoint (scoped slice).

    Ego/goal dims 0:8 and X_rel stay as-is. Document as threat-mid slice:
    not full threat-feature interval coverage.
    """
    flo, fhi = box.feat_lo.copy(), box.feat_hi.copy()
    for i in range(8, int(flo.size)):
        m = 0.5 * (float(flo[i]) + float(fhi[i]))
        flo[i] = fhi[i] = m
    return PlanarBox(
        feat_lo=flo,
        feat_hi=fhi,
        rel_lo=box.rel_lo.copy(),
        rel_hi=box.rel_hi.copy(),
        radius=float(box.radius),
        name=f"{box.name}_thr_mid",
        comm_radius=float(box.comm_radius),
    )


def default_planar_netdep_claim(
    *,
    radius: float = COLLISION_AA,
    d_min: float = 0.118,
    d_max: float = 0.122,
    lat_max: float = 0.04,
    close_lo: float = 0.25,
    close_hi: float = 0.30,
    v_lat_max: float = 0.05,
    comm_radius: float = COMM_RADIUS_DEFAULT,
    pose_lo: float | None = None,
    pose_hi: float | None = None,
    freeze_pose: bool = False,
    vg_abs: float | None = None,
    threat_mid: bool = False,
) -> PlanarBox:
    """Planar netdep fringe with **reachable** feature contract.

    Physics (cv=0.06, ca=0.009), head-on-ish:
      dx=0.118, dvx=-0.30 → coast p2x≈0.100; ax=-1 → 0.091 < R; ax=+1 → 0.109 ≥ R

    Feature contract (not legacy domain goal_abs=2):
      goal_xy ∈ [-comm, comm] (or tighter vg_abs), goal_dv ≡ -v, empty d = comm,
      threat coupled to rel.

    pose_lo/pose_hi: optional ego px/py bounds (default [0,4]).
    freeze_pose: pin px/py to midpoint of that range (pose-fixed slice).
    vg_abs: if set, pin vx,vy,goal_xy ∈ [-vg_abs, vg_abs] (then goal_dv=-v).
            Used for scoped enlarge from min certificate (e.g. 0.01 → 0.05).
    threat_mid: pin threat feats to midpoint after coupling (scoped slice).
    """
    comm = float(comm_radius)
    # start from zeros then _recouple fills everything
    flo = np.zeros(INPUT_DIM, dtype=np.float64)
    fhi = np.zeros(INPUT_DIM, dtype=np.float64)
    plo = 0.0 if pose_lo is None else float(pose_lo)
    phi = 4.0 if pose_hi is None else float(pose_hi)
    if freeze_pose:
        pmid = 0.5 * (plo + phi)
        flo[0:2], fhi[0:2] = pmid, pmid
    else:
        flo[0:2], fhi[0:2] = plo, phi
    if vg_abs is not None:
        vg = float(vg_abs)
        if vg < 0.0:
            raise ValueError(f"vg_abs must be >= 0, got {vg_abs}")
        vg = min(vg, float(V_LIM), comm)
        flo[2:4], fhi[2:4] = -vg, vg
        flo[4:6], fhi[4:6] = -vg, vg
    else:
        flo[2:4], fhi[2:4] = -float(V_LIM), float(V_LIM)
        flo[4:6], fhi[4:6] = -comm, comm
    _sync_goal_dv_bounds(flo, fhi)
    # placeholder quads; _recouple overwrites
    for q in range(4):
        b = 8 + q * 5
        flo[b + 1], fhi[b + 1] = comm, comm
        b2 = 28 + q * 4
        flo[b2 + 1], fhi[b2 + 1] = comm, comm

    rel_lo = np.array([d_min, -lat_max, -close_hi, -v_lat_max], dtype=np.float64)
    rel_hi = np.array([d_max, lat_max, -close_lo, v_lat_max], dtype=np.float64)
    name = "planar_netdep_reachable"
    if freeze_pose:
        name = "planar_netdep_reachable_pose_fixed"
    if vg_abs is not None:
        name = f"{name}_vg{float(vg_abs):g}"
    box = PlanarBox(
        feat_lo=flo,
        feat_hi=fhi,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
        radius=float(radius),
        name=name,
        comm_radius=comm,
    )
    box = _recouple_feat(box)
    if threat_mid:
        box = freeze_threat_midpoint(box)
    return box


def claim_physics_report(box: PlanarBox) -> dict[str, Any]:
    cv, ca = euler_coeffs()
    dx_lo = float(box.rel_lo[0])
    dx_hi = float(box.rel_hi[0])
    dvx_lo = float(box.rel_lo[2])
    dvx_hi = float(box.rel_hi[2])
    p2_coast_lo = dx_lo + cv * dvx_lo
    p2_coast_hi = dx_hi + cv * dvx_hi
    p2_bad = p2_coast_lo - ca
    p2_good = p2_coast_lo + ca
    R = float(box.radius)
    comm = float(getattr(box, "comm_radius", COMM_RADIUS_DEFAULT))
    # goal contract diagnostics
    g_lo, g_hi = float(box.feat_lo[4]), float(box.feat_hi[4])
    v_lo, v_hi = float(box.feat_lo[2]), float(box.feat_hi[2])
    gdv_lo, gdv_hi = float(box.feat_lo[6]), float(box.feat_hi[6])
    goal_dv_synced = abs(gdv_lo - (-v_hi)) < 1e-9 and abs(gdv_hi - (-v_lo)) < 1e-9
    return {
        "cv": cv,
        "ca": ca,
        "p2x_coast_lo": p2_coast_lo,
        "p2x_coast_hi": p2_coast_hi,
        "p2x_worst_a": p2_bad,
        "p2x_best_a": p2_good,
        "radius": R,
        "network_dependent": bool(p2_bad < R and (p2_good >= R or p2_coast_lo >= R - 1e-6)),
        "envelope_any_a_safe": bool(p2_bad >= R),
        "note": "If envelope_any_a_safe, box is meaningless for controller cert.",
        "signed_a_matters": True,
        "property": "planar_two_step_L2_not_radial",
        "ibp_proof": "Deploy⊆SafeSet(ax≥a_min) or min_norm_aabb(p2_box)≥R",
        "feat_contract": {
            "version": "reachable_v1",
            "comm_radius": comm,
            "goal_xy_aabb": [g_lo, g_hi],
            "goal_xy_note": "AABB over-approx of extract disk ||g||<=comm; sample uses disk",
            "goal_dv_equals_minus_v": bool(goal_dv_synced),
            "goal_dv_bounds": [gdv_lo, gdv_hi],
            "v_bounds": [v_lo, v_hi],
            "legacy_wide_goal_abs2": False,
        },
    }
