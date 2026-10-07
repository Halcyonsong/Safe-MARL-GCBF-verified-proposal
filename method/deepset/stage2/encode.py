"""Load frozen Stage1 DeepSet weights and evaluate fixed-(k,m) buckets.

For verification we do **not** expand all 40 padded slots. Each query fixes a
presence mode (k agent neighbors, m obstacle hits) and only encodes those
active entities. Empty slots are omitted (equivalent to presence=0).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


EGO_DIM = 8
ENTITY_CONT_DIM = 4  # dx, dy, dvx, dvy
ENTITY_DIM = 7       # + type_agent, type_obs, presence


@dataclass(frozen=True)
class DeepSetWeights:
    phi0_w: np.ndarray
    phi0_b: np.ndarray
    phi1_w: np.ndarray
    phi1_b: np.ndarray
    phi2_w: np.ndarray
    phi2_b: np.ndarray
    rho0_w: np.ndarray
    rho0_b: np.ndarray
    rho1_w: np.ndarray
    rho1_b: np.ndarray
    rho2_w: np.ndarray
    rho2_b: np.ndarray
    act_w: np.ndarray
    act_b: np.ndarray
    hidden: int

    @property
    def n_relu(self) -> int:
        # Approximate ReLU count when encoding k agent + m obs slots.
        # Exact count depends on (k+m); this is per-layer width reference.
        return self.hidden


def load_weights(checkpoint: str | Path) -> Tuple[DeepSetWeights, dict]:
    path = Path(checkpoint)
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    p = ckpt["student_params"]["params"]
    w = DeepSetWeights(
        phi0_w=np.asarray(p["phi0"]["kernel"], dtype=np.float64),
        phi0_b=np.asarray(p["phi0"]["bias"], dtype=np.float64),
        phi1_w=np.asarray(p["phi1"]["kernel"], dtype=np.float64),
        phi1_b=np.asarray(p["phi1"]["bias"], dtype=np.float64),
        phi2_w=np.asarray(p["phi2"]["kernel"], dtype=np.float64),
        phi2_b=np.asarray(p["phi2"]["bias"], dtype=np.float64),
        rho0_w=np.asarray(p["rho0"]["kernel"], dtype=np.float64),
        rho0_b=np.asarray(p["rho0"]["bias"], dtype=np.float64),
        rho1_w=np.asarray(p["rho1"]["kernel"], dtype=np.float64),
        rho1_b=np.asarray(p["rho1"]["bias"], dtype=np.float64),
        rho2_w=np.asarray(p["rho2"]["kernel"], dtype=np.float64),
        rho2_b=np.asarray(p["rho2"]["bias"], dtype=np.float64),
        act_w=np.asarray(p["action"]["kernel"], dtype=np.float64),
        act_b=np.asarray(p["action"]["bias"], dtype=np.float64),
        hidden=int(ckpt["hidden"]),
    )
    meta = {
        "checkpoint": str(path),
        "architecture": ckpt.get("architecture"),
        "hidden": int(ckpt["hidden"]),
        "ego_dim": int(ckpt["ego_dim"]),
        "entity_dim": int(ckpt["entity_dim"]),
        "entity_slots": int(ckpt["entity_slots"]),
        "n_max_agents": int(ckpt["n_max_agents"]),
        "n_max_obs": int(ckpt["n_max_obs"]),
        "note": ckpt.get("note", ""),
    }
    return w, meta


def build_entity_row(
    cont: Sequence[float],
    *,
    is_agent: bool,
) -> np.ndarray:
    """Build one 7-D entity feature with fixed type/presence."""
    dx, dy, dvx, dvy = map(float, cont)
    row = np.array(
        [
            dx,
            dy,
            dvx,
            dvy,
            1.0 if is_agent else 0.0,
            0.0 if is_agent else 1.0,
            1.0,  # presence
        ],
        dtype=np.float64,
    )
    return row


def dense(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def clip_action(a: np.ndarray, limit: float = 1.0) -> np.ndarray:
    """Same saturation as env.clip_action / [-1,1]^2."""
    return np.clip(np.asarray(a, dtype=np.float64), -limit, limit)


def forward_bucket(
    weights: DeepSetWeights,
    ego: np.ndarray,
    agent_conts: Sequence[Sequence[float]],
    obs_conts: Sequence[Sequence[float]],
) -> np.ndarray:
    """Exact numpy forward for a fixed-(k,m) bucket."""
    ego = np.asarray(ego, dtype=np.float64).reshape(-1)
    assert ego.shape == (EGO_DIM,)

    pooled = np.zeros((weights.hidden,), dtype=np.float64)
    for cont in agent_conts:
        e = build_entity_row(cont, is_agent=True)
        h = relu(dense(e, weights.phi0_w, weights.phi0_b))
        h = relu(dense(h, weights.phi1_w, weights.phi1_b))
        h = relu(dense(h, weights.phi2_w, weights.phi2_b))
        pooled = pooled + h
    for cont in obs_conts:
        e = build_entity_row(cont, is_agent=False)
        h = relu(dense(e, weights.phi0_w, weights.phi0_b))
        h = relu(dense(h, weights.phi1_w, weights.phi1_b))
        h = relu(dense(h, weights.phi2_w, weights.phi2_b))
        pooled = pooled + h

    y = np.concatenate([ego, pooled], axis=0)
    y = relu(dense(y, weights.rho0_w, weights.rho0_b))
    y = relu(dense(y, weights.rho1_w, weights.rho1_b))
    y = relu(dense(y, weights.rho2_w, weights.rho2_b))
    return dense(y, weights.act_w, weights.act_b)


def input_dim_for_bucket(k: int, m: int) -> int:
    """Continuous free inputs: ego(8) + k*4 + m*4."""
    return EGO_DIM + ENTITY_CONT_DIM * (k + m)


def pack_inputs(
    ego: Sequence[float],
    agent_conts: Sequence[Sequence[float]],
    obs_conts: Sequence[Sequence[float]],
) -> np.ndarray:
    parts = [np.asarray(ego, dtype=np.float64).reshape(-1)]
    for c in agent_conts:
        parts.append(np.asarray(c, dtype=np.float64).reshape(-1))
    for c in obs_conts:
        parts.append(np.asarray(c, dtype=np.float64).reshape(-1))
    return np.concatenate(parts, axis=0)


def unpack_inputs(
    x: np.ndarray, k: int, m: int
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    assert x.shape[0] == input_dim_for_bucket(k, m)
    ego = x[:EGO_DIM]
    off = EGO_DIM
    agents = []
    for _ in range(k):
        agents.append(x[off : off + ENTITY_CONT_DIM])
        off += ENTITY_CONT_DIM
    obs = []
    for _ in range(m):
        obs.append(x[off : off + ENTITY_CONT_DIM])
        off += ENTITY_CONT_DIM
    return ego, agents, obs


def relu_count_for_bucket(hidden: int, k: int, m: int) -> int:
    """ReLU neurons if φ is unrolled per active entity (verification encoding)."""
    phi = 3 * hidden * (k + m)
    rho = 3 * hidden
    return phi + rho
