"""Mean-field MLP student — feature v5: 4-quadrant max+weighted-mean + inv_ttc.

Input layout (dim = 44), axis-aligned quadrants in ego-relative (dx, dy):

    Q0: dx >= 0, dy >= 0   (right-up)
    Q1: dx <  0, dy >= 0   (left-up)
    Q2: dx <  0, dy <  0   (left-down)
    Q3: dx >= 0, dy <  0   (right-down)

    [0:8]    ego: px,py,vx,vy, goal_dx,goal_dy,goal_dvx,goal_dvy
    [8:28]   agent Q0..Q3 × (n, min_d, close_max, close_wmean, inv_ttc_max)  # 20
    [28:44]  obs   Q0..Q3 × (n, min_d, close_max, inv_ttc_max)                # 16

Per agent quadrant:
  n            density (unweighted count)
  min_d        nearest distance (= max spatial threat)
  close_max    closing of that nearest agent (>0 approaching)
  close_wmean  distance-weighted mean closing, w=exp(-d/τ)
  inv_ttc_max  max(close_max,0)/max(min_d,ε) clipped to [0, INV_TTC_MAX]
               (= time-urgency of the worst threat in the quadrant)

Per obs quadrant:
  n, min_d, close_max, inv_ttc_max
  (closing / inv_ttc use v_rel=-v_ego; static hits)

Empty: n=0, min_d=d_empty, close_*=0, inv_ttc=0.
τ_dist = DIST_TAU_SCALE * comm_radius (default 0.5 → half radius).

Network: pure ReLU MLP 44 → H → H → H → 2, deploy clip ±1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


EGO_DIM = 8
N_QUAD = 4
AGENT_QUAD_FEAT = 5  # n, min_d, close_max, close_wmean, inv_ttc_max
OBS_QUAD_FEAT = 4  # n, min_d, close_max, inv_ttc_max
INPUT_DIM = EGO_DIM + N_QUAD * AGENT_QUAD_FEAT + N_QUAD * OBS_QUAD_FEAT  # 44

FEATURE_VERSION = 5

# Distance weight: w = exp(-d / τ), τ = DIST_TAU_SCALE * comm_radius
DIST_TAU_SCALE = 0.5
# inv_ttc = closing_plus / d, clipped
INV_TTC_MAX = 20.0
EPS = 1e-6

SL_EGO = slice(0, 8)
SL_AGENT_QUADS = slice(8, 28)
SL_OBS_QUADS = slice(28, 44)

FEATURE_LAYOUT = {
    "version": FEATURE_VERSION,
    "ego": [0, 8],
    "agent_quad_n_mind_cmax_cwmean_invttc": [8, 28],
    "obs_quad_n_mind_cmax_invttc": [28, 44],
    "quadrants": "Q0:dx>=0,dy>=0; Q1:dx<0,dy>=0; Q2:dx<0,dy<0; Q3:dx>=0,dy<0",
    "d_empty": "comm_radius",
    "dist_weight": "exp(-d/(DIST_TAU_SCALE*comm_radius))",
    "dist_tau_scale": DIST_TAU_SCALE,
    "inv_ttc_max_clip": INV_TTC_MAX,
    "time_threat": "inv_ttc_max of nearest-in-quad (not mean)",
    "global_nearest": False,
}


class MeanFieldStudent(nn.Module):
    """Verification-friendly pure-ReLU MLP on quadrant mean-field features."""

    hidden: int = 512
    action_dim: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.hidden, name="fc0")(x))
        h = nn.relu(nn.Dense(self.hidden, name="fc1")(h))
        h = nn.relu(nn.Dense(self.hidden, name="fc2")(h))
        return nn.Dense(self.action_dim, name="action")(h)


def _quadrant_id(dx: jnp.ndarray, dy: jnp.ndarray) -> jnp.ndarray:
    right = dx >= 0
    up = dy >= 0
    return jnp.where(
        right & up,
        0,
        jnp.where(~right & up, 1, jnp.where(~right & ~up, 2, 3)),
    )


def _closing_speed(dx, dy, dvx, dvy, eps: float = EPS):
    dist = jnp.sqrt(dx * dx + dy * dy)
    dist_s = jnp.maximum(dist, eps)
    return -(dvx * dx + dvy * dy) / dist_s


def _inv_ttc(closing: jnp.ndarray, dist: jnp.ndarray) -> jnp.ndarray:
    """Time urgency: max(closing,0)/dist, clipped. Far/receding → 0."""
    return jnp.clip(
        jnp.maximum(closing, 0.0) / jnp.maximum(dist, EPS),
        0.0,
        INV_TTC_MAX,
    )


def _agent_quadrant_feats(
    rel: jnp.ndarray,
    presence: jnp.ndarray,
    d_empty: float,
    tau: float,
) -> jnp.ndarray:
    """Flat [20] = 4 quads × (n, min_d, close_max, close_wmean, inv_ttc_max)."""
    n_ent = rel.shape[0]
    dx, dy = rel[:, 0], rel[:, 1]
    dvx, dvy = rel[:, 2], rel[:, 3]
    dist = jnp.linalg.norm(rel[:, :2], axis=-1)
    qid = _quadrant_id(dx, dy)
    closing = _closing_speed(dx, dy, dvx, dvy)
    mask = presence > 0.5
    big = jnp.full((n_ent,), 1e6, dtype=rel.dtype)
    d_empty_t = jnp.asarray(d_empty, dtype=rel.dtype)
    tau_t = jnp.maximum(jnp.asarray(tau, dtype=rel.dtype), EPS)

    outs = []
    for j in range(N_QUAD):
        in_j = mask & (qid == j)
        n_j = jnp.sum(in_j.astype(rel.dtype))
        d_j = jnp.where(in_j, dist, big)
        min_d = jnp.min(d_j)
        j_star = jnp.argmin(d_j)
        c_max = closing[j_star]
        # distance-weighted mean closing (near weighs more)
        w = jnp.where(in_j, jnp.exp(-dist / tau_t), 0.0)
        w_sum = jnp.sum(w)
        c_wmean = jnp.where(
            w_sum > EPS,
            jnp.sum(w * closing) / w_sum,
            0.0,
        )
        ittc = _inv_ttc(c_max, min_d)
        # empty fill
        min_d = jnp.where(n_j > 0.5, min_d, d_empty_t)
        c_max = jnp.where(n_j > 0.5, c_max, 0.0)
        c_wmean = jnp.where(n_j > 0.5, c_wmean, 0.0)
        ittc = jnp.where(n_j > 0.5, ittc, 0.0)
        outs.append(jnp.stack([n_j, min_d, c_max, c_wmean, ittc]))
    return jnp.concatenate(outs, axis=0)  # [20]


def _obs_quadrant_feats(
    rel_xy: jnp.ndarray,
    presence: jnp.ndarray,
    v_ego: jnp.ndarray,
    d_empty: float,
) -> jnp.ndarray:
    """Flat [16] = 4 quads × (n, min_d, close_max, inv_ttc_max)."""
    n_ent = rel_xy.shape[0]
    dx, dy = rel_xy[:, 0], rel_xy[:, 1]
    dist = jnp.linalg.norm(rel_xy[:, :2], axis=-1)
    qid = _quadrant_id(dx, dy)
    vx, vy = v_ego[0], v_ego[1]
    closing = _closing_speed(dx, dy, -vx, -vy)
    mask = presence > 0.5
    big = jnp.full((n_ent,), 1e6, dtype=rel_xy.dtype)
    d_empty_t = jnp.asarray(d_empty, dtype=rel_xy.dtype)

    outs = []
    for j in range(N_QUAD):
        in_j = mask & (qid == j)
        n_j = jnp.sum(in_j.astype(rel_xy.dtype))
        d_j = jnp.where(in_j, dist, big)
        min_d = jnp.min(d_j)
        j_star = jnp.argmin(d_j)
        c_max = closing[j_star]
        ittc = _inv_ttc(c_max, min_d)
        min_d = jnp.where(n_j > 0.5, min_d, d_empty_t)
        c_max = jnp.where(n_j > 0.5, c_max, 0.0)
        ittc = jnp.where(n_j > 0.5, ittc, 0.0)
        outs.append(jnp.stack([n_j, min_d, c_max, ittc]))
    return jnp.concatenate(outs, axis=0)  # [16]


def extract_meanfield_features(
    graph,
    n_agents: int,
    n_rays: int,
    comm_radius: float,
) -> jnp.ndarray:
    """Build [n_agents, 44] quadrant features (weighted mean + inv_ttc_max)."""
    agents = graph.type_states(type_idx=0, n_type=n_agents)
    goals = graph.type_states(type_idx=1, n_type=n_agents)
    hits = graph.type_states(type_idx=2, n_type=n_rays * n_agents)
    hits = hits.reshape(n_agents, n_rays, -1)

    goal_rel = goals - agents
    goal_xy = goal_rel[:, :2]
    goal_norm = jnp.linalg.norm(goal_xy, axis=-1, keepdims=True)
    safe_norm = jnp.maximum(goal_norm, comm_radius)
    coef = jnp.where(goal_norm > comm_radius, comm_radius / safe_norm, 1.0)
    goal_rel = goal_rel.at[:, :2].set(goal_xy * coef)
    ego = jnp.concatenate([agents, goal_rel], axis=-1)  # [A, 8]

    rel_agents = agents[:, None, :] - agents[None, :, :]
    dist_a = jnp.linalg.norm(rel_agents[..., :2], axis=-1)
    dist_a = dist_a + jnp.eye(n_agents, dtype=agents.dtype) * (comm_radius + 1.0)
    presence_a = jnp.less(dist_a, comm_radius)
    tau = float(DIST_TAU_SCALE) * float(comm_radius)

    def _one_agent(i):
        a_quad = _agent_quadrant_feats(
            rel_agents[i], presence_a[i], float(comm_radius), tau
        )
        rel_h = agents[i, None, :] - hits[i]
        d_h = jnp.linalg.norm(rel_h[:, :2], axis=-1)
        p_h = jnp.less(d_h, comm_radius - 1e-1)
        o_quad = _obs_quadrant_feats(
            rel_h[:, :2], p_h, agents[i, 2:4], float(comm_radius)
        )
        return jnp.concatenate([ego[i], a_quad, o_quad], axis=0)

    return jax.vmap(_one_agent)(jnp.arange(n_agents))


def extract_rollout_meanfield(graphs, n_agents: int, n_rays: int, comm_radius: float):
    return jax.vmap(
        lambda graph: extract_meanfield_features(graph, n_agents, n_rays, comm_radius)
    )(graphs)


def danger_weights_mf(
    x: jnp.ndarray,
    danger_radius: float,
    danger_coef: float,
) -> jnp.ndarray:
    """Higher weight when any quadrant min_d is close or inv_ttc is large."""
    # agent min_d at 8+5*q+1; obs min_d at 28+4*q+1
    # agent inv_ttc at 8+5*q+4; obs inv_ttc at 28+4*q+3
    mins = []
    ittc_max = x[..., 8 + 4]  # start with Q0 agent inv_ttc
    for q in range(4):
        mins.append(x[..., 8 + 5 * q + 1])
        mins.append(x[..., 28 + 4 * q + 1])
        ittc_max = jnp.maximum(ittc_max, x[..., 8 + 5 * q + 4])
        ittc_max = jnp.maximum(ittc_max, x[..., 28 + 4 * q + 3])
    qmin = mins[0]
    for m in mins[1:]:
        qmin = jnp.minimum(qmin, m)
    closeness = jnp.clip(danger_radius / jnp.maximum(qmin, 1e-3), 0.0, 10.0)
    urgency = jnp.clip(ittc_max / 5.0, 0.0, 10.0)
    return 1.0 + danger_coef * (0.5 * closeness + 0.5 * urgency)
