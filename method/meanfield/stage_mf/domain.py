"""MF verification domain = 44-D feature box (one free interval per input dim).

Claim shape (agreed):
    ∀ x ∈ X ⊂ R^{44}:  TwoStepL2Safe( Deploy(x) )

- Domain X: axis-aligned product of per-dimension bounds on the **network input**
  (ego speed, goal, each quadrant min_d / close_wmean / inv_ttc / …).
- Property: two-step L2 local safety (specification simplification — same idea as
  DeepSet L2 horizon, but **without** (k,m) bucket splits).
- Not DeepSet: no primary AA/AO/(1,0)/(0,1) claim table.

Two-step geometry is **derived from features** (min_d, closing, inv_ttc, ego v)
via a sound radial over-approximation — no separate DeepSet-style entity slots
and no aa/ao split of the main claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stage2.dynamics import COLLISION_AA, COLLISION_AO, V_LIM
from stage_mf.student import (
    AGENT_QUAD_FEAT,
    EGO_DIM,
    FEATURE_LAYOUT,
    FEATURE_VERSION,
    INPUT_DIM,
    INV_TTC_MAX,
    N_QUAD,
    OBS_QUAD_FEAT,
)


@dataclass(frozen=True)
class Box:
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        lo = np.asarray(self.lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(self.hi, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        if lo.shape != hi.shape:
            raise ValueError("lo/hi shape mismatch")
        if np.any(lo > hi + 1e-12):
            raise ValueError("lo > hi in box")

    @property
    def dim(self) -> int:
        return int(self.lo.shape[0])

    def width(self) -> np.ndarray:
        return self.hi - self.lo

    def longest_axis(self) -> int:
        return int(np.argmax(self.width()))

    def split(self, axis: int | None = None) -> tuple["Box", "Box"]:
        axis = self.longest_axis() if axis is None else int(axis)
        mid = 0.5 * (self.lo[axis] + self.hi[axis])
        lo1, hi1 = self.lo.copy(), self.hi.copy()
        lo2, hi2 = self.lo.copy(), self.hi.copy()
        hi1[axis] = mid
        lo2[axis] = mid
        return Box(lo1, hi1), Box(lo2, hi2)

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        u = rng.random((n, self.dim))
        return self.lo + u * (self.hi - self.lo)

    def as_dict(self) -> dict:
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist(), "dim": self.dim}


@dataclass(frozen=True)
class MFClaim:
    """Single full-feature claim: X ⊂ R^{44} only."""

    name: str
    feature: Box
    radius: float  # two-step collision radius used in the property
    notes: str = ""
    mode: str = "near"

    def dim_labels(self) -> list[str]:
        return feature_dim_labels()


def feature_dim_labels() -> list[str]:
    """Human-readable name for each free input dimension."""
    labels = [
        "ego_px",
        "ego_py",
        "ego_vx",
        "ego_vy",
        "goal_dx",
        "goal_dy",
        "goal_dvx",
        "goal_dvy",
    ]
    for q in range(N_QUAD):
        for name in ("n", "min_d", "close_max", "close_wmean", "inv_ttc_max"):
            labels.append(f"agent_Q{q}_{name}")
    for q in range(N_QUAD):
        for name in ("n", "min_d", "close_max", "inv_ttc_max"):
            labels.append(f"obs_Q{q}_{name}")
    assert len(labels) == INPUT_DIM
    return labels


def feature_box(
    *,
    area: float = 4.0,
    v_abs: float = V_LIM,
    goal_abs: float = 4.0,
    n_agent_max: float = 8.0,
    n_obs_max: float = 8.0,
    d_min: float = 0.08,
    d_max: float = 0.35,
    close_lo: float = -1.0,
    close_hi: float = 1.0,
    inv_ttc_max: float = INV_TTC_MAX,
    # force at least one occupied threat quad (near-danger claims)
    force_occupied: bool = False,
) -> Box:
    """Axis-aligned envelope: one interval per free feature dimension.

    When force_occupied=True, agent Q0 has n∈[1, n_agent_max] so the box
    always contains a real nearest-agent threat (matches near-danger intent).
    """
    lo = np.zeros(INPUT_DIM, dtype=np.float64)
    hi = np.zeros(INPUT_DIM, dtype=np.float64)

    # ego
    lo[0:2], hi[0:2] = 0.0, area
    lo[2:4], hi[2:4] = -v_abs, v_abs
    lo[4:6], hi[4:6] = -goal_abs, goal_abs
    lo[6:8], hi[6:8] = -2.0 * v_abs, 2.0 * v_abs

    # agent quads
    for q in range(N_QUAD):
        b = EGO_DIM + q * AGENT_QUAD_FEAT
        if force_occupied and q == 0:
            lo[b + 0], hi[b + 0] = 1.0, n_agent_max
        else:
            lo[b + 0], hi[b + 0] = 0.0, n_agent_max
        lo[b + 1], hi[b + 1] = d_min, d_max
        lo[b + 2], hi[b + 2] = close_lo, close_hi
        lo[b + 3], hi[b + 3] = close_lo, close_hi
        lo[b + 4], hi[b + 4] = 0.0, inv_ttc_max

    # obs quads
    obs0 = EGO_DIM + N_QUAD * AGENT_QUAD_FEAT
    for q in range(N_QUAD):
        b = obs0 + q * OBS_QUAD_FEAT
        lo[b + 0], hi[b + 0] = 0.0, n_obs_max
        lo[b + 1], hi[b + 1] = d_min, d_max
        lo[b + 2], hi[b + 2] = close_lo, close_hi
        lo[b + 3], hi[b + 3] = 0.0, inv_ttc_max

    return Box(lo, hi)


def default_mf_claims(
    *,
    d_min: float | None = None,
    d_max: float | None = None,
    mode: str = "near",
    radius: float | None = None,
    close_hi: float | None = None,
) -> list[MFClaim]:
    """Build MF claims: **one primary box per mode** (no aa/ao split).

    radius default = COLLISION_AA (0.10).

    **near** defaults target a preventable-ish band:
      min_d ∈ [0.15, 0.35], close ∈ [-0.25, 0.25], agent Q0 occupied.
    so that a=0 coast radial stays ≥ R for most of the box
    (d0 - α·c0 ≥ 0.15 - 0.06·0.25 = 0.135 > 0.10).

    **near_plus** = bounded acceptance target (slightly wider preventable band):
      min_d ∈ [0.14, 0.40], |close| ≤ 0.30, inv_ttc ≤ 8, Q0 occupied.
    Physics envelope: d0 - α·c0 - β ≥ 0.14 - 0.06·0.30 - 0.009 = 0.113 > 0.10.

    Points already doomed under coast are outside the declared claim.
    """
    R = float(COLLISION_AA if radius is None else radius)
    mode = str(mode).strip().lower()

    if mode == "smoke":
        # One occupied safe threat + empty others (consistent with extractor).
        x0 = np.zeros(INPUT_DIM, dtype=np.float64)
        x0[0:2] = 2.0
        x0[2:4] = 0.0
        x0[4:6] = 0.5
        d_empty = 0.5
        for q in range(N_QUAD):
            b = EGO_DIM + q * AGENT_QUAD_FEAT
            x0[b + 0] = 0.0
            x0[b + 1] = d_empty
            x0[b + 2] = 0.0
            x0[b + 3] = 0.0
            x0[b + 4] = 0.0
        # agent Q0 occupied, comfortable clearance
        x0[EGO_DIM + 0] = 1.0
        x0[EGO_DIM + 1] = 0.25
        obs0 = EGO_DIM + N_QUAD * AGENT_QUAD_FEAT
        for q in range(N_QUAD):
            b = obs0 + q * OBS_QUAD_FEAT
            x0[b + 0] = 0.0
            x0[b + 1] = d_empty
            x0[b + 2] = 0.0
            x0[b + 3] = 0.0
        r = 1e-3
        return [
            MFClaim(
                name="mf_smoke",
                feature=Box(x0 - r, x0 + r),
                radius=R,
                mode="smoke",
                notes=(
                    "Tiny 44-D feature box smoke (Q0 occupied min_d≈0.25). "
                    f"Two-step radial L2 with R={R}."
                ),
            )
        ]

    if mode == "near":
        # Preventable-ish near band (not already coast-doomed).
        dm = 0.15 if d_min is None else float(d_min)
        dM = 0.35 if d_max is None else float(d_max)
        ch = 0.25 if close_hi is None else float(close_hi)
        feat = feature_box(
            d_min=dm,
            d_max=dM,
            area=4.0,
            v_abs=V_LIM,
            goal_abs=2.0,
            n_agent_max=4.0,
            n_obs_max=4.0,
            close_lo=-ch,
            close_hi=ch,
            inv_ttc_max=8.0,
            force_occupied=True,
        )
        name = "mf_near"
        notes = (
            "Preventable-ish near-danger 44-D feature product box. "
            f"min_d∈[{dm},{dM}], |close|≤{ch}, agent Q0 occupied. "
            f"Two-step radial L2 R={R}. Excludes coast-doomed band. "
            "No DeepSet (k,m) / aa-ao claim split."
        )
    elif mode in ("near_plus", "nearplus", "plus"):
        # Bounded acceptance target: slightly wider preventable band than near.
        dm = 0.14 if d_min is None else float(d_min)
        dM = 0.40 if d_max is None else float(d_max)
        ch = 0.30 if close_hi is None else float(close_hi)
        feat = feature_box(
            d_min=dm,
            d_max=dM,
            area=4.0,
            v_abs=V_LIM,
            goal_abs=2.0,
            n_agent_max=4.0,
            n_obs_max=4.0,
            close_lo=-ch,
            close_hi=ch,
            inv_ttc_max=8.0,
            force_occupied=True,
        )
        name = "mf_near_plus"
        notes = (
            "Bounded acceptance preventable band (near_plus). "
            f"min_d∈[{dm},{dM}], |close|≤{ch}, inv_ttc≤8, agent Q0 occupied. "
            f"Two-step radial L2 R={R}. "
            "Physics envelope d0-α·c0-β ≥ 0.14-0.06·0.30-0.009=0.113>0.10. "
            "Excludes coast-doomed (d≲0.12) and extreme close (|c|>0.3). "
            "No DeepSet (k,m) / aa-ao claim split."
        )
        mode = "near_plus"
    elif mode == "full":
        dm = 0.12 if d_min is None else float(d_min)
        dM = 0.50 if d_max is None else float(d_max)
        ch = 0.5 if close_hi is None else float(close_hi)
        feat = feature_box(
            d_min=dm,
            d_max=dM,
            close_lo=-ch,
            close_hi=ch,
            force_occupied=True,
        )
        name = "mf_full"
        notes = (
            "Wider 44-D feature envelope (still force Q0 occupied). "
            f"Two-step L2 radial R={R}. May be PARTIAL under resource limits."
        )
    else:
        raise ValueError(
            f"unknown MF claim mode={mode!r}; "
            "expected smoke|near|near_plus|full"
        )

    return [
        MFClaim(
            name=name,
            feature=feat,
            radius=R,
            mode=mode,
            notes=notes,
        )
    ]


def claim_manifest(claim: MFClaim) -> dict[str, Any]:
    """Serialize claim definition for reports."""
    labels = feature_dim_labels()
    dims = []
    for i, lab in enumerate(labels):
        dims.append(
            {
                "i": i,
                "name": lab,
                "lo": float(claim.feature.lo[i]),
                "hi": float(claim.feature.hi[i]),
            }
        )
    return {
        "name": claim.name,
        "mode": claim.mode,
        "feature_version": FEATURE_VERSION,
        "input_dim": INPUT_DIM,
        "feature_layout": FEATURE_LAYOUT,
        "radius": claim.radius,
        "collision_aa_ref": COLLISION_AA,
        "collision_ao_ref": COLLISION_AO,
        "notes": claim.notes,
        "dimensions": dims,
        "query_domain": "R44_feature_box_only",
        "property": "two_step_L2_radial_overapprox",
        "not_deepset_buckets": True,
    }
