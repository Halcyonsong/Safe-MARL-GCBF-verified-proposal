"""Shared-parameter actor-critic for MAPPO-Lag (JAX/Flax)."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy: mean MLP + state-independent log_std."""

    hidden: int = 256
    action_dim: int = 2
    log_std_init: float = -0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = nn.relu(nn.Dense(self.hidden, name="fc0")(x))
        h = nn.relu(nn.Dense(self.hidden, name="fc1")(h))
        h = nn.relu(nn.Dense(self.hidden, name="fc2")(h))
        mean = nn.Dense(self.action_dim, name="mean")(h)
        log_std = self.param(
            "log_std",
            lambda *_: jnp.full((self.action_dim,), self.log_std_init, dtype=jnp.float32),
        )
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mean, log_std


class ValueNet(nn.Module):
    """Scalar value / cost-value head on features."""

    hidden: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.hidden, name="fc0")(x))
        h = nn.relu(nn.Dense(self.hidden, name="fc1")(h))
        h = nn.relu(nn.Dense(self.hidden, name="fc2")(h))
        v = nn.Dense(1, name="v")(h)
        return v.squeeze(-1)


def sample_action(
    mean: jnp.ndarray, log_std: jnp.ndarray, key: jax.Array
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a ~ N(mean, diag(exp(log_std)^2)); return (a, log_prob)."""
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape)
    action = mean + std * noise
    # sum over action dims
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
    """Per-state entropy of diagonal Gaussian (sum over action dims)."""
    return (0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + log_std).sum(axis=-1)
