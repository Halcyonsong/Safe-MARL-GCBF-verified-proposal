"""Fast two-step L2 collision + min-intervention labels (vectorized numpy).

Same physics as stage2/dynamics.py: 2 Euler steps, neighbor coasts, hits static.
"""

from __future__ import annotations

import numpy as np

from stage2.dynamics import COLLISION_AA, COLLISION_AO, DT, MASS, HORIZON_STEPS
from stage2.encode import clip_action

# Precomputed action grid for min-intervention (shared)
_GRID_N = 11
_AX = np.linspace(-1.0, 1.0, _GRID_N, dtype=np.float64)
_AY = np.linspace(-1.0, 1.0, _GRID_N, dtype=np.float64)
_GX, _GY = np.meshgrid(_AX, _AY, indexing="xy")
_GRID_A = np.stack([_GX.ravel(), _GY.ravel()], axis=1)  # [G, 2]
_G = _GRID_A.shape[0]


def _two_step_rel_batch(
    rel: np.ndarray,
    a: np.ndarray,
    *,
    other_static: bool,
) -> np.ndarray:
    """rel [...,4]=(dx,dy,dvx,dvy), a [...,2] -> rel after 2 steps [...,4]."""
    rel = np.asarray(rel, dtype=np.float64)
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    acc_e = a / MASS
    acc_o = 0.0 if other_static else 0.0  # coast
    d_p = rel[..., :2].copy()
    d_v = rel[..., 2:4].copy()
    for _ in range(HORIZON_STEPS):
        d_p = d_p + d_v * DT
        d_v = d_v + (acc_e - acc_o) * DT
    return np.concatenate([d_p, d_v], axis=-1)


def clearance_batch(
    rel_aa: np.ndarray | None,
    rel_ao: np.ndarray | None,
    a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns margin [N], collided [N] bool.

    rel_* : [N,4] or None; a: [N,2]
    margin = min(d_aa - R_aa, d_ao - R_ao); missing threat ignored.
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    margin = np.full(n, 1e6, dtype=np.float64)
    coll = np.zeros(n, dtype=bool)
    if rel_aa is not None:
        rn = _two_step_rel_batch(rel_aa, a, other_static=False)
        d = np.linalg.norm(rn[..., :2], axis=-1)
        m = d - COLLISION_AA
        margin = np.minimum(margin, m)
        coll |= d < COLLISION_AA - 1e-9
    if rel_ao is not None:
        rn = _two_step_rel_batch(rel_ao, a, other_static=True)
        d = np.linalg.norm(rn[..., :2], axis=-1)
        m = d - COLLISION_AO
        margin = np.minimum(margin, m)
        coll |= d < COLLISION_AO - 1e-9
    return margin, coll


def min_intervention_batch(
    rel_aa: np.ndarray | None,
    rel_ao: np.ndarray | None,
    a0: np.ndarray,
    *,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized grid min-intervention for N samples.

    Returns a_safe[N,2], clearance[N], is_safe[N], delta[N].
    """
    a0 = np.clip(np.asarray(a0, dtype=np.float64).reshape(-1, 2), -1.0, 1.0)
    n = a0.shape[0]
    G = _G
    # Expand: for each sample, evaluate all grid actions
    # a_grid [N,G,2], rel [N,G,4]
    a_grid = np.broadcast_to(_GRID_A[None, :, :], (n, G, 2))
    margin_g = np.full((n, G), 1e6, dtype=np.float64)
    coll_g = np.zeros((n, G), dtype=bool)

    if rel_aa is not None:
        rel = np.asarray(rel_aa, dtype=np.float64).reshape(n, 4)
        rel_e = np.broadcast_to(rel[:, None, :], (n, G, 4)).reshape(n * G, 4)
        a_e = a_grid.reshape(n * G, 2)
        rn = _two_step_rel_batch(rel_e, a_e, other_static=False).reshape(n, G, 4)
        d = np.linalg.norm(rn[..., :2], axis=-1)
        margin_g = np.minimum(margin_g, d - COLLISION_AA)
        coll_g |= d < COLLISION_AA - 1e-9
    if rel_ao is not None:
        rel = np.asarray(rel_ao, dtype=np.float64).reshape(n, 4)
        rel_e = np.broadcast_to(rel[:, None, :], (n, G, 4)).reshape(n * G, 4)
        a_e = a_grid.reshape(n * G, 2)
        rn = _two_step_rel_batch(rel_e, a_e, other_static=True).reshape(n, G, 4)
        d = np.linalg.norm(rn[..., :2], axis=-1)
        margin_g = np.minimum(margin_g, d - COLLISION_AO)
        coll_g |= d < COLLISION_AO - 1e-9

    safe = (~coll_g) & (margin_g >= margin)
    # distance to a0
    da = np.linalg.norm(a_grid - a0[:, None, :], axis=-1)  # [N,G]
    # among safe: min da; tie-break max margin
    big = 1e9
    da_safe = np.where(safe, da, big)
    # primary: min da
    best_da = da_safe.min(axis=1)
    # candidates with da ~= best
    cand = safe & (da <= best_da[:, None] + 1e-9)
    # among cand pick max margin
    marg_cand = np.where(cand, margin_g, -1e9)
    best_j = np.argmax(marg_cand, axis=1)
    has_safe = safe.any(axis=1)

    a_safe = _GRID_A[best_j].copy()
    clr = margin_g[np.arange(n), best_j]
    # fallback: max clearance action if no safe
    j_any = np.argmax(margin_g, axis=1)
    a_any = _GRID_A[j_any]
    clr_any = margin_g[np.arange(n), j_any]
    a_safe = np.where(has_safe[:, None], a_safe, a_any)
    clr = np.where(has_safe, clr, clr_any)
    delta = np.linalg.norm(a_safe - a0, axis=-1)
    return a_safe, clr, has_safe, delta


def nearest_rels_all_agents(
    agents: np.ndarray,
    hits: np.ndarray,
    *,
    comm_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For all agents, nearest AA and AO relative states.

    agents [A,4+], hits [A,R,4+]
    Returns rel_aa[A,4], has_aa[A], rel_ao[A,4], has_ao[A]
    """
    a = np.asarray(agents, dtype=np.float64)
    h = np.asarray(hits, dtype=np.float64)
    n = a.shape[0]
    pos = a[:, :2]
    vel = a[:, 2:4]
    # pairwise agent dist
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, 1e6)
    j_aa = np.argmin(d, axis=1)
    d_aa = d[np.arange(n), j_aa]
    has_aa = d_aa < comm_radius
    rel_aa = np.zeros((n, 4), dtype=np.float64)
    rel_aa[:, :2] = pos - pos[j_aa]
    rel_aa[:, 2:4] = vel - vel[j_aa]
    rel_aa = np.where(has_aa[:, None], rel_aa, 0.0)

    # hits: [A,R,2]
    hp = h[..., :2]
    d_h = np.linalg.norm(pos[:, None, :] - hp, axis=-1)  # [A,R]
    j_ao = np.argmin(d_h, axis=1)
    d_ao = d_h[np.arange(n), j_ao]
    has_ao = d_ao < (comm_radius - 0.1)
    rel_ao = np.zeros((n, 4), dtype=np.float64)
    # gather hit positions
    hit_p = hp[np.arange(n), j_ao]
    rel_ao[:, :2] = pos - hit_p
    rel_ao[:, 2:4] = vel  # hit vel 0
    rel_ao = np.where(has_ao[:, None], rel_ao, 0.0)
    return rel_aa, has_aa, rel_ao, has_ao


# --- thin wrappers for single-sample API (tests / debug) ---

def two_step_clearance(rel_aa, rel_ao, a):
    from stage2.dynamics import RelState

    def _to(r):
        if r is None:
            return None
        if isinstance(r, RelState):
            return np.array([[r.dx, r.dy, r.dvx, r.dvy]])
        return np.asarray(r, dtype=np.float64).reshape(1, 4)

    m, c = clearance_batch(_to(rel_aa), _to(rel_ao), np.asarray(a).reshape(1, 2))
    kind = "aa" if c[0] else "none"
    return float(m[0]), kind, bool(c[0])


def min_intervention_action(rel_aa, rel_ao, a0, *, grid_n: int = 11, margin: float = 0.0):
    from stage2.dynamics import RelState

    def _to(r):
        if r is None:
            return None
        if isinstance(r, RelState):
            return np.array([[r.dx, r.dy, r.dvx, r.dvy]])
        return np.asarray(r, dtype=np.float64).reshape(1, 4)

    a_s, c, ok, da = min_intervention_batch(
        _to(rel_aa), _to(rel_ao), np.asarray(a0).reshape(1, 2), margin=margin
    )
    return a_s[0], float(c[0]), bool(ok[0]), float(da[0])
