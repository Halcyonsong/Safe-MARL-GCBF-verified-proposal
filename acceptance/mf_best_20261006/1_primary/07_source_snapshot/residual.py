"""Frozen MF base + physics-gated residual correction.

Deployment:
    a = clip( base(x) + g_phys(x) * delta(x) )

- base: frozen meanfield_mlp (feature v5 step9000)
- g_phys: ReLU-style gate from min_d / inv_ttc in the 44-D feature
- delta: small trainable MLP on the same x

Far from danger g≈0 ⇒ policy equals frozen base (protects finish).
Verifier-friendly: no learned sigmoid gate by default.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from stage_mf.student import (
    FEATURE_VERSION,
    INPUT_DIM,
    MeanFieldStudent,
    extract_meanfield_features,
)
from stage2.dynamics import COLLISION_AA, COLLISION_AO


class CorrectionMLP(nn.Module):
    """Small residual head δ(x) → R^{action_dim} (unclipped)."""

    hidden: int = 128
    action_dim: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.hidden, name="d0")(x))
        h = nn.relu(nn.Dense(self.hidden, name="d1")(h))
        return nn.Dense(self.action_dim, name="d_out")(h)


def min_dist_from_features(x: jnp.ndarray) -> jnp.ndarray:
    """Global min_d over agent/obs quadrants (feature v5 layout)."""
    # agent min_d at 8 + 5*q + 1; obs at 28 + 4*q + 1
    md = x[..., 8 + 1]
    for q in range(4):
        md = jnp.minimum(md, x[..., 8 + 5 * q + 1])
        md = jnp.minimum(md, x[..., 28 + 4 * q + 1])
    return md


def max_inv_ttc_from_features(x: jnp.ndarray) -> jnp.ndarray:
    it = x[..., 8 + 4]
    for q in range(4):
        it = jnp.maximum(it, x[..., 8 + 5 * q + 4])
        it = jnp.maximum(it, x[..., 28 + 4 * q + 3])
    return it


def physics_gate_mf(
    x: jnp.ndarray,
    *,
    open_mult: float = 3.0,
    close_mult: float = 1.0,
    r_ref: float = COLLISION_AA,
    inv_ttc_ref: float = 5.0,
    mix: float = 0.5,
) -> jnp.ndarray:
    """Gate in [0,1]: near / urgent → 1; far and calm → 0.

    Distance part: 1 at close_mult*r_ref, 0 beyond open_mult*r_ref.
    Time part: inv_ttc / inv_ttc_ref clipped.
    mix blends the two (default 0.5).
    """
    d = min_dist_from_features(x)
    n = d / max(r_ref, 1e-6)
    g_d = (open_mult - n) / max(open_mult - close_mult, 1e-6)
    g_d = jnp.clip(g_d, 0.0, 1.0)
    it = max_inv_ttc_from_features(x)
    g_t = jnp.clip(it / max(inv_ttc_ref, 1e-6), 0.0, 1.0)
    return jnp.clip((1.0 - mix) * g_d + mix * g_t, 0.0, 1.0)


def apply_residual_mf(
    base: MeanFieldStudent,
    base_params: Any,
    delta_mod: CorrectionMLP,
    delta_params: Any,
    x: jnp.ndarray,
    *,
    open_mult: float = 3.0,
    close_mult: float = 1.0,
    inv_ttc_ref: float = 5.0,
    gate_mix: float = 0.5,
    action_lim: Tuple[float, float] = (-1.0, 1.0),
    delta_clip: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (a_clip, g, delta)."""
    a0 = base.apply(base_params, x)
    g = physics_gate_mf(
        x,
        open_mult=open_mult,
        close_mult=close_mult,
        inv_ttc_ref=inv_ttc_ref,
        mix=gate_mix,
    )
    delta = delta_mod.apply(delta_params, x)
    delta = jnp.clip(delta, -delta_clip, delta_clip)
    a = jnp.clip(a0 + g[..., None] * delta, action_lim[0], action_lim[1])
    return a, g, delta


def load_mf_base(checkpoint_path: str | pathlib.Path) -> tuple[dict, MeanFieldStudent, dict]:
    path = pathlib.Path(checkpoint_path)
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("architecture") != "meanfield_mlp":
        raise ValueError(f"expected meanfield_mlp, got {ckpt.get('architecture')}")
    ver = int(ckpt.get("feature_version", -1))
    if ver != FEATURE_VERSION:
        raise ValueError(f"feature_version ckpt={ver} code={FEATURE_VERSION}")
    dim = int(ckpt.get("input_dim", INPUT_DIM))
    if dim != INPUT_DIM:
        raise ValueError(f"input_dim ckpt={dim} code={INPUT_DIM}")
    hidden = int(ckpt["hidden"])
    base = MeanFieldStudent(hidden=hidden, action_dim=2)
    return ckpt, base, ckpt["student_params"]


def save_mf_residual_checkpoint(
    path: str | pathlib.Path,
    *,
    base_checkpoint: str,
    base_params: Any,
    delta_params: Any,
    hidden: int,
    correction_hidden: int,
    open_mult: float,
    close_mult: float,
    inv_ttc_ref: float,
    gate_mix: float,
    delta_clip: float,
    n_agents: int,
    n_rays: int,
    metrics: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": "meanfield_residual_phys",
        "track": "meanfield",
        "feature_version": FEATURE_VERSION,
        "input_dim": INPUT_DIM,
        "base_checkpoint": str(base_checkpoint),
        "base_params": base_params,
        "delta_params": delta_params,
        "hidden": int(hidden),
        "correction_hidden": int(correction_hidden),
        "gate_mode": "physics",
        "open_mult": float(open_mult),
        "close_mult": float(close_mult),
        "inv_ttc_ref": float(inv_ttc_ref),
        "gate_mix": float(gate_mix),
        "delta_clip": float(delta_clip),
        "n_max_agents": int(n_agents),
        "n_max_obs": int(n_rays),
        "metrics": metrics or {},
        "note": (
            "MF residual: a=clip(base + g_phys*delta). "
            "base frozen meanfield_mlp; only delta trained. "
            "Empirical repair only — not a formal certificate."
        ),
    }
    if extra:
        payload.update(extra)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_mf_residual(path: str | pathlib.Path) -> dict:
    with pathlib.Path(path).open("rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("architecture") != "meanfield_residual_phys":
        raise ValueError(f"expected meanfield_residual_phys, got {ckpt.get('architecture')}")
    return ckpt
