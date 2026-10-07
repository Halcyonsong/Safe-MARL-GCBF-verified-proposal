"""Extract pure-ReLU MLP weights from MF base / residual checkpoints."""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from stage_mf.student import FEATURE_VERSION, INPUT_DIM


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@dataclass
class MLPWeights:
    """Dense layers: kernels (in, out), biases (out,). ReLU after all but last."""

    kernels: list[np.ndarray]
    biases: list[np.ndarray]
    name: str = "mlp"

    @property
    def n_layers(self) -> int:
        return len(self.kernels)

    @property
    def input_dim(self) -> int:
        return int(self.kernels[0].shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.kernels[-1].shape[1])

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        for i, (w, b) in enumerate(zip(self.kernels, self.biases)):
            h = h @ w + b
            if i < len(self.kernels) - 1:
                h = np.maximum(h, 0.0)
        return h


def _flax_dense(params: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    p = params[name]
    return _np(p["kernel"]), _np(p["bias"])


def mlp_from_flax_base(params_tree: dict) -> MLPWeights:
    p = params_tree.get("params", params_tree)
    layers = []
    for name in ("fc0", "fc1", "fc2", "action"):
        layers.append(_flax_dense(p, name))
    return MLPWeights(
        kernels=[w for w, _ in layers],
        biases=[b for _, b in layers],
        name="base",
    )


def mlp_from_flax_delta(params_tree: dict) -> MLPWeights:
    p = params_tree.get("params", params_tree)
    layers = []
    for name in ("d0", "d1", "d_out"):
        layers.append(_flax_dense(p, name))
    return MLPWeights(
        kernels=[w for w, _ in layers],
        biases=[b for _, b in layers],
        name="delta",
    )


@dataclass
class MFResidualWeights:
    base: MLPWeights
    delta: MLPWeights
    open_mult: float
    close_mult: float
    inv_ttc_ref: float
    gate_mix: float
    delta_clip: float
    r_ref: float
    feature_version: int
    input_dim: int
    base_checkpoint: str
    residual_path: str

    def base_action(self, x: np.ndarray) -> np.ndarray:
        return self.base.forward(x)

    def delta_raw(self, x: np.ndarray) -> np.ndarray:
        return self.delta.forward(x)


def load_mf_residual_weights(path: str | pathlib.Path) -> MFResidualWeights:
    path = pathlib.Path(path)
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("architecture") != "meanfield_residual_phys":
        raise ValueError(f"expected meanfield_residual_phys, got {ckpt.get('architecture')}")
    ver = int(ckpt.get("feature_version", -1))
    if ver != FEATURE_VERSION:
        raise ValueError(f"feature_version ckpt={ver} code={FEATURE_VERSION}")
    dim = int(ckpt.get("input_dim", INPUT_DIM))
    if dim != INPUT_DIM:
        raise ValueError(f"input_dim ckpt={dim} code={INPUT_DIM}")
    from stage2.dynamics import COLLISION_AA

    return MFResidualWeights(
        base=mlp_from_flax_base(ckpt["base_params"]),
        delta=mlp_from_flax_delta(ckpt["delta_params"]),
        open_mult=float(ckpt["open_mult"]),
        close_mult=float(ckpt["close_mult"]),
        inv_ttc_ref=float(ckpt.get("inv_ttc_ref", 5.0)),
        gate_mix=float(ckpt.get("gate_mix", 0.5)),
        delta_clip=float(ckpt.get("delta_clip", 1.0)),
        r_ref=float(COLLISION_AA),
        feature_version=ver,
        input_dim=dim,
        base_checkpoint=str(ckpt.get("base_checkpoint", "")),
        residual_path=str(path),
    )


def load_mf_base_weights(path: str | pathlib.Path) -> MLPWeights:
    path = pathlib.Path(path)
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("architecture") != "meanfield_mlp":
        raise ValueError(f"expected meanfield_mlp, got {ckpt.get('architecture')}")
    return mlp_from_flax_base(ckpt["student_params"])


# ---------------------------------------------------------------------------
# Physics gate on features (exact, piecewise-linear / clip)
# ---------------------------------------------------------------------------

# Feature indices for min_d and inv_ttc (v5 layout)
def agent_min_d_indices() -> list[int]:
    return [8 + 5 * q + 1 for q in range(4)]


def obs_min_d_indices() -> list[int]:
    return [28 + 4 * q + 1 for q in range(4)]


def agent_inv_ttc_indices() -> list[int]:
    return [8 + 5 * q + 4 for q in range(4)]


def obs_inv_ttc_indices() -> list[int]:
    return [28 + 4 * q + 3 for q in range(4)]


def min_d_indices() -> list[int]:
    return agent_min_d_indices() + obs_min_d_indices()


def inv_ttc_indices() -> list[int]:
    return agent_inv_ttc_indices() + obs_inv_ttc_indices()


def physics_gate_np(
    x: np.ndarray,
    *,
    open_mult: float,
    close_mult: float,
    r_ref: float,
    inv_ttc_ref: float,
    mix: float,
) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d = min(float(x[i]) for i in min_d_indices())
    n = d / max(r_ref, 1e-6)
    g_d = (open_mult - n) / max(open_mult - close_mult, 1e-6)
    g_d = float(np.clip(g_d, 0.0, 1.0))
    it = max(float(x[i]) for i in inv_ttc_indices())
    g_t = float(np.clip(it / max(inv_ttc_ref, 1e-6), 0.0, 1.0))
    return float(np.clip((1.0 - mix) * g_d + mix * g_t, 0.0, 1.0))


def deploy_action_np(w: MFResidualWeights, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    a0 = w.base.forward(x)
    g = physics_gate_np(
        x,
        open_mult=w.open_mult,
        close_mult=w.close_mult,
        r_ref=w.r_ref,
        inv_ttc_ref=w.inv_ttc_ref,
        mix=w.gate_mix,
    )
    d = np.clip(w.delta.forward(x), -w.delta_clip, w.delta_clip)
    return np.clip(a0 + g * d, -1.0, 1.0)
