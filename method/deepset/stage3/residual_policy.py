"""Frozen DeepSet base + danger-gated residual correction.

Deployment:
    a = clip( base(x) + g(x) * delta(x) )

- ``base`` is the Stage1 distilled DeepSet (frozen).
- ``delta`` is a small DeepSet correction head (trainable).
- ``g`` is a scalar gate in [0, 1] (trainable / physics / hybrid).

Far from danger the gate is near zero, so the policy reduces to the frozen
base and does not overwrite already-safe behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from stage1.student import DeepSetStudent, EGO_DIM, ENTITY_DIM
from stage2.dynamics import COLLISION_AA, COLLISION_AO


# Feature layout for the gate MLP (fixed, documented for checkpoints).
GATE_FEAT_DIM = 10


class GateMLP(nn.Module):
    """Maps handcrafted danger features -> scalar gate in (0, 1).

    Final bias is initialized positive so the soft gate starts partially open
    (~0.7–0.9); hybrid mode can still zero it via the physics gate far away.
    """

    hidden: int = 32
    init_logit_bias: float = 1.5  # sigmoid(1.5) ≈ 0.82

    @nn.compact
    def __call__(self, feat: jnp.ndarray) -> jnp.ndarray:
        h = nn.relu(nn.Dense(self.hidden, name="g0")(feat))
        h = nn.relu(nn.Dense(self.hidden, name="g1")(h))
        logit = nn.Dense(
            1,
            name="g_out",
            bias_init=nn.initializers.constant(self.init_logit_bias),
        )(h)
        return nn.sigmoid(logit).squeeze(-1)


class CorrectionDeepSet(nn.Module):
    """Small DeepSet that outputs an action residual (not clipped)."""

    hidden: int = 64
    action_dim: int = 2

    @nn.compact
    def __call__(self, ego: jnp.ndarray, entities: jnp.ndarray) -> jnp.ndarray:
        presence = entities[..., -1:]
        h = nn.relu(nn.Dense(self.hidden, name="c_phi0")(entities))
        h = nn.relu(nn.Dense(self.hidden, name="c_phi1")(h))
        pooled = jnp.sum(h * presence, axis=-2)
        y = jnp.concatenate([ego, pooled], axis=-1)
        y = nn.relu(nn.Dense(self.hidden, name="c_rho0")(y))
        y = nn.relu(nn.Dense(self.hidden, name="c_rho1")(y))
        return nn.Dense(self.action_dim, name="c_action")(y)


def min_type_dist(entities: jnp.ndarray, type_idx: int) -> jnp.ndarray:
    """Min Euclidean distance over present entities of a given one-hot type.

    type_idx: 4 = agent neighbor, 5 = obstacle hit.
    Returns large distance when no entity of that type is present.
    """
    rel_xy = entities[..., :2]
    presence = entities[..., -1]
    type_flag = entities[..., type_idx]
    dist = jnp.linalg.norm(rel_xy, axis=-1)
    mask = (presence > 0.5) & (type_flag > 0.5)
    dist = jnp.where(mask, dist, 1e6)
    return jnp.min(dist, axis=-1)


def danger_features(
    ego: jnp.ndarray,
    entities: jnp.ndarray,
    *,
    r_aa: float = COLLISION_AA,
    r_ao: float = COLLISION_AO,
    dist_cap: float = 1.0,
) -> jnp.ndarray:
    """Handcrafted gate features from ego + local entity set.

    All continuous channels are bounded so a small MLP is not saturated by
    the 1e6 sentinel used for missing entity types.

    Layout (GATE_FEAT_DIM=10):
      0 d_aa_cap, 1 d_ao_cap,          # min dist clipped to [0, dist_cap]
      2 close_aa, 3 close_ao,          # r / max(d, r) in (0, 1]
      4 ||v_ego||, 5 ||goal_xy||/cap,
      6 n_aa_near, 7 n_ao_near,
      8 has_aa, 9 has_ao
    """
    d_aa_raw = min_type_dist(entities, 4)
    d_ao_raw = min_type_dist(entities, 5)
    has_aa = (d_aa_raw < 1e5).astype(ego.dtype)
    has_ao = (d_ao_raw < 1e5).astype(ego.dtype)
    d_aa = jnp.minimum(d_aa_raw, dist_cap)
    d_ao = jnp.minimum(d_ao_raw, dist_cap)
    # Missing type -> treat as far (close score 0).
    close_aa = jnp.where(has_aa > 0.5, r_aa / jnp.maximum(d_aa_raw, r_aa), 0.0)
    close_ao = jnp.where(has_ao > 0.5, r_ao / jnp.maximum(d_ao_raw, r_ao), 0.0)
    v_norm = jnp.linalg.norm(ego[..., 2:4], axis=-1)
    g_norm = jnp.minimum(jnp.linalg.norm(ego[..., 4:6], axis=-1), dist_cap)

    presence = entities[..., -1]
    is_aa = entities[..., 4]
    is_ao = entities[..., 5]
    near_aa = (
        (presence > 0.5) & (is_aa > 0.5) & (jnp.linalg.norm(entities[..., :2], axis=-1) < 0.3)
    ).sum(axis=-1).astype(ego.dtype)
    near_ao = (
        (presence > 0.5) & (is_ao > 0.5) & (jnp.linalg.norm(entities[..., :2], axis=-1) < 0.3)
    ).sum(axis=-1).astype(ego.dtype)

    return jnp.stack(
        [d_aa, d_ao, close_aa, close_ao, v_norm, g_norm, near_aa, near_ao, has_aa, has_ao],
        axis=-1,
    )


def physics_gate(
    feat: jnp.ndarray,
    *,
    open_mult: float = 2.5,
    close_mult: float = 1.0,
) -> jnp.ndarray:
    """Hard-ish proximity gate from normalized min distances.

    Uses the tighter of the two present threats. Gate is 1 at close_mult·r and
    0 beyond open_mult·r. Missing threats are ignored (not treated as distance 0).
    """
    d_aa = feat[..., 0]
    d_ao = feat[..., 1]
    has_aa = feat[..., 8]
    has_ao = feat[..., 9]
    # If a type is absent, push its normalized distance to +inf so min ignores it.
    n_aa = jnp.where(has_aa > 0.5, d_aa / COLLISION_AA, 1e6)
    n_ao = jnp.where(has_ao > 0.5, d_ao / COLLISION_AO, 1e6)
    n_min = jnp.minimum(n_aa, n_ao)
    g = (open_mult - n_min) / max(open_mult - close_mult, 1e-6)
    return jnp.clip(g, 0.0, 1.0)


@dataclass
class ResidualModules:
    base: DeepSetStudent
    gate: GateMLP
    correction: CorrectionDeepSet


def make_modules(
    *,
    base_hidden: int = 512,
    gate_hidden: int = 32,
    correction_hidden: int = 64,
    action_dim: int = 2,
) -> ResidualModules:
    return ResidualModules(
        base=DeepSetStudent(hidden=base_hidden, action_dim=action_dim),
        gate=GateMLP(hidden=gate_hidden),
        correction=CorrectionDeepSet(hidden=correction_hidden, action_dim=action_dim),
    )


def apply_residual(
    modules: ResidualModules,
    base_params: Any,
    train_params: Dict[str, Any],
    ego: jnp.ndarray,
    entities: jnp.ndarray,
    *,
    gate_mode: str = "hybrid",
    open_mult: float = 2.5,
    close_mult: float = 1.0,
    action_lim: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (a_clip, gate, delta)."""
    a0 = modules.base.apply(base_params, ego, entities)
    feat = danger_features(ego, entities)
    g_soft = modules.gate.apply({"params": train_params["gate"]}, feat)
    g_hard = physics_gate(feat, open_mult=open_mult, close_mult=close_mult)
    if gate_mode == "soft":
        g = g_soft
    elif gate_mode == "hard":
        g = g_hard
    elif gate_mode == "hybrid":
        # Physics envelope × learned intensity. Far away g_hard=0 ⇒ equals base.
        g = g_soft * g_hard
    else:
        raise ValueError(f"unknown gate_mode={gate_mode}")
    delta = modules.correction.apply({"params": train_params["correction"]}, ego, entities)
    a = jnp.clip(a0 + g[..., None] * delta, action_lim[0], action_lim[1])
    return a, g, delta


def init_train_params(
    modules: ResidualModules,
    key,
    ego_dim: int = EGO_DIM,
    entity_dim: int = ENTITY_DIM,
    entity_slots: int = 40,
) -> Dict[str, Any]:
    k1, k2 = jax.random.split(key)
    ego = jnp.zeros((1, ego_dim))
    ent = jnp.zeros((1, entity_slots, entity_dim))
    feat = danger_features(ego, ent)
    gate_vars = modules.gate.init(k1, feat)
    corr_vars = modules.correction.init(k2, ego, ent)
    return {
        "gate": gate_vars["params"],
        "correction": corr_vars["params"],
    }


def save_residual_checkpoint(
    path,
    *,
    base_params,
    train_params,
    base_checkpoint: str,
    args,
    ego_dim: int,
    entity_dim: int,
    entity_slots: int,
    n_agents: int,
    n_rays: int,
    metrics: Optional[dict] = None,
) -> None:
    import pathlib
    import pickle

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": 3,
        "architecture": "deepset_residual_gate",
        "base_checkpoint": str(base_checkpoint),
        "base_params": base_params,
        "student_params": train_params,  # gate + correction only
        "hidden": int(args.hidden),
        "gate_hidden": int(args.gate_hidden),
        "correction_hidden": int(args.correction_hidden),
        "gate_mode": str(args.gate_mode),
        "open_mult": float(args.open_mult),
        "close_mult": float(args.close_mult),
        "ego_dim": int(ego_dim),
        "entity_dim": int(entity_dim),
        "entity_slots": int(entity_slots),
        "n_max_agents": int(n_agents),
        "n_max_obs": int(n_rays),
        "environment": vars(args) if hasattr(args, "__dict__") else dict(args),
        "metrics": metrics or {},
        "note": (
            "Residual-gate repair: a=clip(base + g*delta). "
            "base is frozen Stage1 DeepSet; only gate/correction are trained. "
            "Empirical repair only — not a formal safety certificate."
        ),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_residual_checkpoint(path) -> Tuple[dict, ResidualModules, Any, Dict[str, Any]]:
    import pathlib
    import pickle

    with pathlib.Path(path).open("rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("architecture") != "deepset_residual_gate":
        raise ValueError(f"expected deepset_residual_gate, got {ckpt.get('architecture')}")
    modules = make_modules(
        base_hidden=int(ckpt["hidden"]),
        gate_hidden=int(ckpt.get("gate_hidden", 32)),
        correction_hidden=int(ckpt.get("correction_hidden", 64)),
    )
    return ckpt, modules, ckpt["base_params"], ckpt["student_params"]
