"""Shared DeepSet student and local-set feature extraction for Stage1."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


EGO_DIM = 8          # [px, py, vx, vy, goal_dx, goal_dy, goal_dvx, goal_dvy]
ENTITY_DIM = 7       # [dx, dy, dvx, dvy, type_agent, type_obs, presence]


class DeepSetStudent(nn.Module):
    """Masked DeepSet policy with fixed maximum entity slots."""

    hidden: int = 256
    action_dim: int = 2

    @nn.compact
    def __call__(self, ego: jnp.ndarray, entities: jnp.ndarray) -> jnp.ndarray:
        presence = entities[..., -1:]
        x = entities
        h = nn.relu(nn.Dense(self.hidden, name="phi0")(x))
        h = nn.relu(nn.Dense(self.hidden, name="phi1")(h))
        h = nn.relu(nn.Dense(self.hidden, name="phi2")(h))
        pooled = jnp.sum(h * presence, axis=-2)

        y = jnp.concatenate([ego, pooled], axis=-1)
        y = nn.relu(nn.Dense(self.hidden, name="rho0")(y))
        y = nn.relu(nn.Dense(self.hidden, name="rho1")(y))
        y = nn.relu(nn.Dense(self.hidden, name="rho2")(y))
        return nn.Dense(self.action_dim, name="action")(y)


def extract_local_sets(
    graph,
    n_agents: int,
    n_rays: int,
    comm_radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract padded local sets for every agent from one official graph."""
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
    ego = jnp.concatenate([agents, goal_rel], axis=-1)

    rel_agents = agents[:, None, :] - agents[None, :, :]
    dist = jnp.linalg.norm(rel_agents[..., :2], axis=-1)
    dist = dist + jnp.eye(n_agents) * (comm_radius + 1.0)
    presence_n = jnp.less(dist, comm_radius)[..., None]
    neigh = jnp.concatenate(
        [
            rel_agents,
            jnp.ones((n_agents, n_agents, 1), dtype=agents.dtype),
            jnp.zeros((n_agents, n_agents, 1), dtype=agents.dtype),
            presence_n.astype(agents.dtype),
        ],
        axis=-1,
    )
    neigh = neigh * presence_n

    rel_hits = agents[:, None, :] - hits
    dist_hits = jnp.linalg.norm(rel_hits[..., :2], axis=-1)
    presence_o = jnp.less(dist_hits, comm_radius - 1e-1)[..., None]
    obst = jnp.concatenate(
        [
            rel_hits,
            jnp.zeros((n_agents, n_rays, 1), dtype=agents.dtype),
            jnp.ones((n_agents, n_rays, 1), dtype=agents.dtype),
            presence_o.astype(agents.dtype),
        ],
        axis=-1,
    )
    obst = obst * presence_o

    entities = jnp.concatenate([neigh, obst], axis=1)
    return ego, entities


def extract_rollout_sets(graphs, n_agents: int, n_rays: int, comm_radius: float):
    return jax.vmap(
        lambda graph: extract_local_sets(graph, n_agents, n_rays, comm_radius)
    )(graphs)


def danger_weights(entities: jnp.ndarray, danger_radius: float, danger_coef: float) -> jnp.ndarray:
    """Higher weight when any present neighbor/obstacle is nearby."""
    rel_xy = entities[..., :2]
    presence = entities[..., -1]
    dist = jnp.linalg.norm(rel_xy, axis=-1)
    dist = jnp.where(presence > 0.5, dist, 1e6)
    min_dist = jnp.min(dist, axis=-1)
    closeness = jnp.clip(danger_radius / jnp.maximum(min_dist, 1e-3), 0.0, 10.0)
    return 1.0 + danger_coef * closeness
