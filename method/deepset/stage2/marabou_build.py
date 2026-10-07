"""Build a Marabou network for a fixed-(k,m) DeepSet bucket (no ONNX)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from maraboupy import Marabou
from maraboupy.MarabouUtils import Equation

from stage2.encode import EGO_DIM, ENTITY_CONT_DIM, DeepSetWeights


def _add_affine(
    net: Marabou.MarabouNetwork,
    x_vars: Sequence[int],
    w: np.ndarray,
    b: np.ndarray,
) -> List[int]:
    """Encode y = x @ W + b with W shape (in_dim, out_dim)."""
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert w.shape == (len(x_vars), b.shape[0])
    y_vars = [net.getNewVariable() for _ in range(b.shape[0])]
    for j in range(b.shape[0]):
        e = Equation()
        for i, xv in enumerate(x_vars):
            c = float(w[i, j])
            if c != 0.0:
                e.addAddend(c, xv)
        e.addAddend(-1.0, y_vars[j])
        e.setScalar(float(-b[j]))  # sum_i W_ij x_i - y_j = -b_j
        net.addEquation(e)
    return y_vars


def _add_relu_layer(net: Marabou.MarabouNetwork, pre_vars: Sequence[int]) -> List[int]:
    post = []
    for v in pre_vars:
        o = net.getNewVariable()
        net.addRelu(v, o)
        post.append(o)
    return post


def _phi_entity(
    net: Marabou.MarabouNetwork,
    cont_vars: Sequence[int],
    weights: DeepSetWeights,
    *,
    is_agent: bool,
) -> List[int]:
    """Encode φ on one entity with folded type/presence constants."""
    assert len(cont_vars) == ENTITY_CONT_DIM
    w0 = weights.phi0_w  # (7, H)
    b0 = weights.phi0_b
    const = np.array([1.0, 0.0, 1.0] if is_agent else [0.0, 1.0, 1.0], dtype=np.float64)
    # pre0 = cont @ W[:4] + (const @ W[4:] + b0)
    bias = const @ w0[4:, :] + b0
    pre0 = _add_affine(net, cont_vars, w0[:4, :], bias)
    h0 = _add_relu_layer(net, pre0)
    pre1 = _add_affine(net, h0, weights.phi1_w, weights.phi1_b)
    h1 = _add_relu_layer(net, pre1)
    pre2 = _add_affine(net, h1, weights.phi2_w, weights.phi2_b)
    h2 = _add_relu_layer(net, pre2)
    return h2


def build_bucket_network(
    weights: DeepSetWeights,
    k: int,
    m: int,
) -> Tuple[Marabou.MarabouNetwork, List[int], List[int]]:
    """Return (network, input_vars, output_vars)."""
    net = Marabou.MarabouNetwork()
    input_vars: List[int] = []

    ego_vars = [net.getNewVariable() for _ in range(EGO_DIM)]
    input_vars.extend(ego_vars)

    agent_conts: List[List[int]] = []
    for _ in range(k):
        vs = [net.getNewVariable() for _ in range(ENTITY_CONT_DIM)]
        agent_conts.append(vs)
        input_vars.extend(vs)

    obs_conts: List[List[int]] = []
    for _ in range(m):
        vs = [net.getNewVariable() for _ in range(ENTITY_CONT_DIM)]
        obs_conts.append(vs)
        input_vars.extend(vs)

    # pooled = sum φ(entity)
    hidden = weights.hidden
    pooled = [net.getNewVariable() for _ in range(hidden)]
    # Start pooled = 0 via equations collected from entity embeddings
    entity_embeds: List[List[int]] = []
    for vs in agent_conts:
        entity_embeds.append(_phi_entity(net, vs, weights, is_agent=True))
    for vs in obs_conts:
        entity_embeds.append(_phi_entity(net, vs, weights, is_agent=False))

    if not entity_embeds:
        for p in pooled:
            e = Equation()
            e.addAddend(1.0, p)
            e.setScalar(0.0)  # p = 0
            net.addEquation(e)
    else:
        for j in range(hidden):
            e = Equation()
            for emb in entity_embeds:
                e.addAddend(1.0, emb[j])
            e.addAddend(-1.0, pooled[j])
            e.setScalar(0.0)  # sum emb - pooled = 0
            net.addEquation(e)

    # ρ([ego; pooled])
    y_in = ego_vars + pooled
    pre0 = _add_affine(net, y_in, weights.rho0_w, weights.rho0_b)
    h0 = _add_relu_layer(net, pre0)
    pre1 = _add_affine(net, h0, weights.rho1_w, weights.rho1_b)
    h1 = _add_relu_layer(net, pre1)
    pre2 = _add_affine(net, h1, weights.rho2_w, weights.rho2_b)
    h2 = _add_relu_layer(net, pre2)
    out_vars = _add_affine(net, h2, weights.act_w, weights.act_b)

    net.inputVars = [np.array(input_vars, dtype=np.int32)]
    net.outputVars = [np.array(out_vars, dtype=np.int32)]
    return net, input_vars, out_vars


def set_input_box(net: Marabou.MarabouNetwork, input_vars: Sequence[int], lo: np.ndarray, hi: np.ndarray) -> None:
    assert len(input_vars) == len(lo) == len(hi)
    for v, a, b in zip(input_vars, lo, hi):
        net.setLowerBound(int(v), float(a))
        net.setUpperBound(int(v), float(b))
