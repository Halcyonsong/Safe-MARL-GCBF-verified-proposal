"""GNN Gaussian actor for InforMARL (no tensorflow_probability)."""

from __future__ import annotations

import functools as ft
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from gcbfplus.nn.gnn import GNN
from gcbfplus.nn.mlp import MLP
from gcbfplus.nn.utils import default_nn_init
from gcbfplus.utils.graph import GraphsTuple


class GNNGaussianActor(nn.Module):
    """Attention-GNN backbone → diagonal Gaussian over residual actions."""

    n_agents: int
    action_dim: int = 2
    gnn_layers: int = 2
    gnn_out_dim: int = 64
    log_std_init: float = -0.5

    @nn.compact
    def __call__(self, graph: GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        gnn = GNN(
            msg_dim=self.gnn_out_dim,
            hid_size_msg=(128, 128),
            hid_size_aggr=(128, 128),
            hid_size_update=(128, 128),
            out_dim=self.gnn_out_dim,
            n_layers=self.gnn_layers,
        )
        x = gnn(graph, node_type=0, n_type=self.n_agents)  # [A, D]
        h = MLP(hid_sizes=(256, 256), act=nn.relu, act_final=False, name="ActorHead")(x)
        mean = nn.Dense(self.action_dim, kernel_init=default_nn_init(), name="mean")(h)
        log_std = self.param(
            "log_std",
            lambda *_: jnp.full((self.action_dim,), self.log_std_init, dtype=jnp.float32),
        )
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mean, log_std


class GNNCritic(nn.Module):
    """Centralized value: GNN on agents → attention pool → scalar."""

    n_agents: int
    gnn_layers: int = 2
    gnn_out_dim: int = 64

    @nn.compact
    def __call__(self, graph: GraphsTuple) -> jnp.ndarray:
        gnn = GNN(
            msg_dim=self.gnn_out_dim,
            hid_size_msg=(128, 128),
            hid_size_aggr=(128, 128),
            hid_size_update=(128, 128),
            out_dim=self.gnn_out_dim,
            n_layers=self.gnn_layers,
        )
        x = gnn(graph, node_type=0, n_type=self.n_agents)  # [A, D]
        gate = MLP(hid_sizes=(128, 128), act=nn.relu, act_final=False, name="CriticAttn")(x)
        gate = nn.Dense(1, kernel_init=default_nn_init(), name="CriticGate")(gate).squeeze(-1)
        attn = jax.nn.softmax(gate, axis=-1)
        pooled = jnp.sum(attn[:, None] * x, axis=0)
        h = MLP(hid_sizes=(128, 128), act=nn.relu, act_final=False, name="CriticHead")(pooled)
        v = nn.Dense(1, kernel_init=default_nn_init(), name="v")(h)
        return v.squeeze()


def sample_action(
    mean: jnp.ndarray, log_std: jnp.ndarray, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape)
    action = mean + std * noise
    log_prob = -0.5 * (
        ((action - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    ).sum(axis=-1)
    return action, log_prob


def log_prob_action(
    mean: jnp.ndarray, log_std: jnp.ndarray, action: jnp.ndarray
) -> jnp.ndarray:
    std = jnp.exp(log_std)
    return -0.5 * (
        ((action - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    ).sum(axis=-1)


def entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    return (0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + log_std).sum(axis=-1)
