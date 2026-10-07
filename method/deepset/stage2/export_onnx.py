"""Export fixed-(k,m) DeepSet buckets as Marabou-friendly ONNX.

Uses only MatMul / Gemm / Relu / Add. Avoids Slice / Gather / Concat by:
  - selecting entity features with MatMul against constant selectors
  - splitting rho0 into ego and pooled branches: y = ego@We + pooled@Wp + b
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.encode import EGO_DIM, ENTITY_CONT_DIM, DeepSetWeights, input_dim_for_bucket


def clip_pm1_relu(a: torch.Tensor, neg_eye: torch.Tensor, one: torch.Tensor, neg_one: torch.Tensor) -> torch.Tensor:
    """clip(a,-1,1) using Add / Relu / MatMul only (no Sub)."""
    t1 = F.relu(a + one)
    t2 = F.relu(a + neg_one)
    return t1 + t2.matmul(neg_eye) + neg_one


class BucketDeepSetONNX(nn.Module):
    def __init__(self, weights: DeepSetWeights, k: int, m: int, with_clip: bool = False):
        super().__init__()
        self.k = int(k)
        self.m = int(m)
        self.with_clip = bool(with_clip)
        self.hidden = int(weights.hidden)
        self.input_dim = input_dim_for_bucket(self.k, self.m)

        # φ1/φ2 as Gemm
        self.phi1 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.phi2 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.rho1 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.rho2 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.action = nn.Linear(self.hidden, 2, bias=True)

        with torch.no_grad():
            self.phi1.weight.copy_(torch.tensor(weights.phi1_w.T, dtype=torch.float32))
            self.phi1.bias.copy_(torch.tensor(weights.phi1_b, dtype=torch.float32))
            self.phi2.weight.copy_(torch.tensor(weights.phi2_w.T, dtype=torch.float32))
            self.phi2.bias.copy_(torch.tensor(weights.phi2_b, dtype=torch.float32))
            self.rho1.weight.copy_(torch.tensor(weights.rho1_w.T, dtype=torch.float32))
            self.rho1.bias.copy_(torch.tensor(weights.rho1_b, dtype=torch.float32))
            self.rho2.weight.copy_(torch.tensor(weights.rho2_w.T, dtype=torch.float32))
            self.rho2.bias.copy_(torch.tensor(weights.rho2_b, dtype=torch.float32))
            self.action.weight.copy_(torch.tensor(weights.act_w.T, dtype=torch.float32))
            self.action.bias.copy_(torch.tensor(weights.act_b, dtype=torch.float32))

        # φ0 continuous weights + folded type/presence bias
        w0 = weights.phi0_w
        b0 = weights.phi0_b
        self.register_buffer("phi0_w_cont", torch.tensor(w0[:4, :], dtype=torch.float32))
        agent_const = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        obs_const = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        self.register_buffer(
            "phi0_bias_agent",
            torch.tensor(agent_const @ w0[4:, :] + b0, dtype=torch.float32),
        )
        self.register_buffer(
            "phi0_bias_obs",
            torch.tensor(obs_const @ w0[4:, :] + b0, dtype=torch.float32),
        )

        # rho0 split: avoids Concat([ego, pooled])
        self.register_buffer("rho0_we", torch.tensor(weights.rho0_w[:EGO_DIM, :], dtype=torch.float32))
        self.register_buffer("rho0_wp", torch.tensor(weights.rho0_w[EGO_DIM:, :], dtype=torch.float32))
        self.register_buffer("rho0_b", torch.tensor(weights.rho0_b, dtype=torch.float32))
        self.register_buffer("clip_neg_eye", torch.tensor([[-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32))
        self.register_buffer("clip_one", torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.register_buffer("clip_neg_one", torch.tensor([-1.0, -1.0], dtype=torch.float32))

        # ego selector (dim, 8)
        Se = np.zeros((self.input_dim, EGO_DIM), dtype=np.float32)
        Se[:EGO_DIM, :EGO_DIM] = np.eye(EGO_DIM, dtype=np.float32)
        self.register_buffer("select_ego", torch.tensor(Se, dtype=torch.float32))

        selectors = []
        for i in range(self.k + self.m):
            S = np.zeros((self.input_dim, ENTITY_CONT_DIM), dtype=np.float32)
            off = EGO_DIM + i * ENTITY_CONT_DIM
            for j in range(ENTITY_CONT_DIM):
                S[off + j, j] = 1.0
            selectors.append(torch.tensor(S, dtype=torch.float32))
        if selectors:
            self.register_buffer("select", torch.stack(selectors, dim=0))
        else:
            self.register_buffer("select", torch.zeros(0, self.input_dim, 4))

    def _phi_cont(self, cont: torch.Tensor, bias0: torch.Tensor) -> torch.Tensor:
        h = F.relu(cont @ self.phi0_w_cont + bias0)
        h = F.relu(self.phi1(h))
        h = F.relu(self.phi2(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        ego = x.matmul(self.select_ego)
        pooled = x.new_zeros((x.shape[0], self.hidden))
        for i in range(self.k):
            cont = x.matmul(self.select[i])
            pooled = pooled + self._phi_cont(cont, self.phi0_bias_agent)
        for j in range(self.m):
            cont = x.matmul(self.select[self.k + j])
            pooled = pooled + self._phi_cont(cont, self.phi0_bias_obs)

        # rho0 without Concat
        y = ego.matmul(self.rho0_we) + pooled.matmul(self.rho0_wp) + self.rho0_b
        y = F.relu(y)
        y = F.relu(self.rho1(y))
        y = F.relu(self.rho2(y))
        a = self.action(y)
        if self.with_clip:
            a = clip_pm1_relu(a, self.clip_neg_eye, self.clip_one, self.clip_neg_one)
        return a.squeeze(0) if squeeze else a


def export_bucket_onnx(
    weights: DeepSetWeights,
    k: int,
    m: int,
    out_path: str | Path,
    *,
    with_clip: bool = False,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = BucketDeepSetONNX(weights, k, m, with_clip=with_clip).eval()
    dim = input_dim_for_bucket(k, m)
    dummy = torch.zeros(1, dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["inputs"],
        output_names=["actions"],
        opset_version=12,
        dynamo=False,
        do_constant_folding=True,
    )
    return out_path


def check_torch_matches_numpy(
    weights: DeepSetWeights,
    k: int,
    m: int,
    x: np.ndarray,
    atol: float = 1e-4,
    *,
    with_clip: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    from stage2.encode import clip_action, forward_bucket, unpack_inputs

    ego, agents, obs = unpack_inputs(x, k, m)
    np_out = forward_bucket(weights, ego, agents, obs)
    if with_clip:
        np_out = clip_action(np_out)
    model = BucketDeepSetONNX(weights, k, m, with_clip=with_clip).eval()
    with torch.no_grad():
        torch_out = model(torch.tensor(x, dtype=torch.float32)).numpy()
    if not np.allclose(np_out, torch_out, atol=atol, rtol=1e-4):
        raise AssertionError(f"torch/numpy mismatch: np={np_out} torch={torch_out}")
    return np_out, torch_out


def list_onnx_ops(path: str | Path) -> set[str]:
    import onnx

    model = onnx.load(str(path))
    return {n.op_type for n in model.graph.node}
