#!/usr/bin/env python3
"""Small Marabou feasibility check for the deployed residual policy.

This is deliberately conservative.  The trained gate contains sigmoid,
norm/minimum operations, and a physics gate, so it is not encoded directly.
Instead, the verifier allows any ``g in [0, 1]`` and encodes ``g * delta`` by
the exact McCormick linear envelope using interval bounds for ``delta``.

Consequences:
  * UNSAT proves the real residual policy safe for the queried square proxy,
    because the real policy is contained in the relaxed policy family.
  * SAT is only a witness for the relaxation, not necessarily for the real
    learned gate.
  * TIMEOUT/UNKNOWN is not a proof.

The query includes the exact frozen base DeepSet, correction DeepSet, action
clip, and two-step held-action relative dynamics.  It is intended as a small
feasibility PoC, not a replacement for the full Stage2 runner.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from maraboupy import Marabou, MarabouCore
from maraboupy.MarabouUtils import Equation

# Allow ``python stage2/verify_residual_marabou.py`` from the repository root.
import pathlib as _pathlib
import sys as _sys
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "core" / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from core.paths import project_root, ensure_import_path
ROOT = project_root()
ensure_import_path()

from stage2.bounds import tiny_l2_box
from stage2.dynamics import COLLISION_AA, COLLISION_AO, DT, HORIZON_STEPS, MASS
from stage2.marabou_query import _solve


EGO_DIM = 8
ENTITY_DIM = 7
ENTITY_CONT_DIM = 4


def _p(tree: dict[str, Any], name: str) -> dict[str, np.ndarray]:
    p = tree.get("params", tree)
    return {
        key: np.asarray(value, dtype=np.float64)
        for key, value in p[name].items()
    }


def _linear(
    net: Marabou.MarabouNetwork,
    xs: Sequence[int],
    w: np.ndarray,
    b: np.ndarray,
) -> list[int]:
    """Encode y = xs @ w + b, with Flax's (input, output) kernels."""
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if w.shape != (len(xs), len(b)):
        raise ValueError(f"linear shape {w.shape} vs {(len(xs), len(b))}")
    ys = [net.getNewVariable() for _ in range(len(b))]
    for j, y in enumerate(ys):
        e = Equation()
        for i, x in enumerate(xs):
            if w[i, j] != 0.0:
                e.addAddend(float(w[i, j]), int(x))
        e.addAddend(-1.0, y)
        e.setScalar(float(-b[j]))
        net.addEquation(e)
    return ys


def _relu(net: Marabou.MarabouNetwork, xs: Sequence[int]) -> list[int]:
    ys = [net.getNewVariable() for _ in xs]
    for x, y in zip(xs, ys):
        net.addRelu(int(x), int(y))
    return ys


def _eq(net: Marabou.MarabouNetwork, terms: Iterable[tuple[float, int]], scalar: float) -> int:
    """Add sum(coeff*var) = scalar and return a fresh lhs variable if needed."""
    e = Equation()
    for coeff, var in terms:
        e.addAddend(float(coeff), int(var))
    e.setScalar(float(scalar))
    net.addEquation(e)
    return 0


def _le(net: Marabou.MarabouNetwork, terms: Iterable[tuple[float, int]], scalar: float) -> None:
    """Add sum(coeff*var) <= scalar."""
    terms = list(terms)
    net.addInequality(
        [int(var) for _coeff, var in terms],
        [float(coeff) for coeff, _var in terms],
        float(scalar),
    )


def _entity_first_layer(
    net: Marabou.MarabouNetwork,
    cont: Sequence[int],
    params: dict[str, np.ndarray],
    prefix: str,
    is_agent: bool,
) -> list[int]:
    del prefix  # names are not needed by the Marabou API
    w = params["kernel"]
    b = params["bias"]
    # Entity input is [dx, dy, dvx, dvy, type_agent, type_obs, presence].
    const = np.array([1.0, 0.0, 1.0] if is_agent else [0.0, 1.0, 1.0])
    folded_b = const @ w[ENTITY_CONT_DIM:, :] + b
    return _relu(net, _linear(net, cont, w[:ENTITY_CONT_DIM, :], folded_b))


def _build_deepset(
    net: Marabou.MarabouNetwork,
    ego: Sequence[int],
    agent_conts: Sequence[Sequence[int]],
    obs_conts: Sequence[Sequence[int]],
    params_tree: dict[str, Any],
    *,
    prefix: str,
) -> list[int]:
    """Encode either the base or correction DeepSet exactly."""
    p = params_tree.get("params", params_tree)
    phi = [_p(p, f"{prefix}phi{i}") for i in range(3) if f"{prefix}phi{i}" in p]
    rho = [_p(p, f"{prefix}rho{i}") for i in range(3) if f"{prefix}rho{i}" in p]
    if not phi or not rho:
        raise ValueError(f"missing DeepSet layers for prefix {prefix!r}")
    action = _p(p, f"{prefix}action")

    embeds: list[list[int]] = []
    for i, cont in enumerate(agent_conts):
        h = _entity_first_layer(net, cont, phi[0], f"{prefix}a{i}", True)
        for layer in phi[1:]:
            h = _relu(net, _linear(net, h, layer["kernel"], layer["bias"]))
        embeds.append(h)
    for i, cont in enumerate(obs_conts):
        h = _entity_first_layer(net, cont, phi[0], f"{prefix}o{i}", False)
        for layer in phi[1:]:
            h = _relu(net, _linear(net, h, layer["kernel"], layer["bias"]))
        embeds.append(h)

    hidden = int(phi[0]["bias"].shape[0])
    pooled = [net.getNewVariable() for _ in range(hidden)]
    for j, out in enumerate(pooled):
        if embeds:
            terms = [(1.0, h[j]) for h in embeds] + [(-1.0, out)]
            _eq(net, terms, 0.0)
        else:
            _eq(net, [(1.0, out)], 0.0)

    h = _relu(net, _linear(net, list(ego) + pooled, rho[0]["kernel"], rho[0]["bias"]))
    for layer in rho[1:]:
        h = _relu(net, _linear(net, h, layer["kernel"], layer["bias"]))
    return _linear(net, h, action["kernel"], action["bias"])


def _interval_affine(lo: np.ndarray, hi: np.ndarray, w: np.ndarray, b: np.ndarray):
    pos = np.maximum(w, 0.0)
    neg = np.minimum(w, 0.0)
    return lo @ pos + hi @ neg + b, hi @ pos + lo @ neg + b


def _interval_deepset(
    lo: np.ndarray,
    hi: np.ndarray,
    params_tree: dict[str, Any],
    *,
    prefix: str,
    k: int,
    m: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sound interval bounds for a DeepSet output over a flat input box."""
    p = params_tree.get("params", params_tree)
    ego_lo, ego_hi = lo[:EGO_DIM], hi[:EGO_DIM]
    off = EGO_DIM
    phi = [_p(p, f"{prefix}phi{i}") for i in range(3) if f"{prefix}phi{i}" in p]
    emb_lo: list[np.ndarray] = []
    emb_hi: list[np.ndarray] = []
    for idx in range(k + m):
        c_lo = lo[off : off + ENTITY_CONT_DIM]
        c_hi = hi[off : off + ENTITY_CONT_DIM]
        off += ENTITY_CONT_DIM
        is_agent = idx < k
        const = np.array([1.0, 0.0, 1.0] if is_agent else [0.0, 1.0, 1.0])
        p0 = phi[0]
        b0 = const @ p0["kernel"][ENTITY_CONT_DIM:, :] + p0["bias"]
        a_lo, a_hi = _interval_affine(c_lo, c_hi, p0["kernel"][:ENTITY_CONT_DIM, :], b0)
        a_lo, a_hi = np.maximum(a_lo, 0.0), np.maximum(a_hi, 0.0)
        for pj in phi[1:]:
            a_lo, a_hi = _interval_affine(a_lo, a_hi, pj["kernel"], pj["bias"])
            a_lo, a_hi = np.maximum(a_lo, 0.0), np.maximum(a_hi, 0.0)
        emb_lo.append(a_lo)
        emb_hi.append(a_hi)
    if emb_lo:
        pool_lo, pool_hi = np.sum(emb_lo, axis=0), np.sum(emb_hi, axis=0)
    else:
        pool_lo = pool_hi = np.zeros(phi[0]["bias"].shape[0])
    rlo, rhi = ego_lo, ego_hi
    ylo, yhi = _interval_affine(
        np.concatenate([rlo, pool_lo]),
        np.concatenate([rhi, pool_hi]),
        _p(p, f"{prefix}rho0")["kernel"],
        _p(p, f"{prefix}rho0")["bias"],
    )
    ylo, yhi = np.maximum(ylo, 0.0), np.maximum(yhi, 0.0)
    for i in range(1, 3):
        if f"{prefix}rho{i}" not in p:
            break
        pi = _p(p, f"{prefix}rho{i}")
        ylo, yhi = _interval_affine(ylo, yhi, pi["kernel"], pi["bias"])
        ylo, yhi = np.maximum(ylo, 0.0), np.maximum(yhi, 0.0)
    pa = _p(p, f"{prefix}action")
    return _interval_affine(ylo, yhi, pa["kernel"], pa["bias"])


def _add_mccormick(net, g: int, d: int, z: int, dlo: float, dhi: float) -> None:
    # g in [0,1], d in [dlo,dhi], z = g*d relaxed by its convex hull.
    _le(net, [(dlo, g), (-1.0, z)], 0.0)                 # z >= dlo*g
    _le(net, [(1.0, d), (dhi, g), (-1.0, z)], dhi)      # z >= d+dhi*g-dhi
    _le(net, [(1.0, z), (-1.0, d), (-dlo, g)], -dlo)    # z <= d+dlo*g-dlo
    _le(net, [(-1.0, z), (dhi, g)], 0.0)                # z <= dhi*g


def _add_clip(net, raw: Sequence[int]) -> list[int]:
    out: list[int] = []
    for r in raw:
        plus = _linear(net, [r], np.array([[1.0]]), np.array([1.0]))[0]
        minus = _linear(net, [r], np.array([[1.0]]), np.array([-1.0]))[0]
        hp, hm = _relu(net, [plus])[0], _relu(net, [minus])[0]
        a = net.getNewVariable()
        _eq(net, [(1.0, hp), (-1.0, hm), (-1.0, a)], 1.0)
        net.setLowerBound(a, -1.0)
        net.setUpperBound(a, 1.0)
        out.append(a)
    return out


def _add_two_step_square(net, xvars: Sequence[int], action: Sequence[int], *, k: int, threshold: float):
    # First active entity is the threat. Flat input starts with ego(8).
    off = EGO_DIM
    dx, dy, dvx, dvy = xvars[off : off + 4]
    n = HORIZON_STEPS
    cv = n * DT
    ca = (n * (n - 1) / 2.0) * (DT * DT) / MASS
    next_xy = []
    for d, dv, a in ((dx, dvx, action[0]), (dy, dvy, action[1])):
        q = net.getNewVariable()
        _eq(net, [(1.0, d), (cv, dv), (ca, a), (-1.0, q)], 0.0)
        next_xy.append(q)
    for q in next_xy:
        net.setLowerBound(q, -threshold)
        net.setUpperBound(q, threshold)


def run_one(checkpoint: Path, bucket: str, radius: float, dist: float, timeout: int) -> dict:
    with checkpoint.open("rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("architecture") != "deepset_residual_gate":
        raise ValueError(f"unexpected checkpoint architecture: {ckpt.get('architecture')}")
    if bucket == "aa":
        k, m, threshold = 1, 0, COLLISION_AA / np.sqrt(2.0)
    elif bucket == "ao":
        k, m, threshold = 0, 1, COLLISION_AO / np.sqrt(2.0)
    else:
        raise ValueError(bucket)
    box = tiny_l2_box(k, m, radius=radius, dist=dist)
    net = Marabou.MarabouNetwork()
    xvars = [net.getNewVariable() for _ in range(box.dim)]
    for v, lo, hi in zip(xvars, box.lo, box.hi):
        net.setLowerBound(v, float(lo))
        net.setUpperBound(v, float(hi))
    ego = xvars[:EGO_DIM]
    cont = xvars[EGO_DIM : EGO_DIM + ENTITY_CONT_DIM]
    agents = [cont] if k else []
    obs = [cont] if m else []
    base_raw = _build_deepset(net, ego, agents, obs, ckpt["base_params"], prefix="")
    corr_raw = _build_deepset(
        net,
        ego,
        agents,
        obs,
        ckpt["student_params"]["correction"],
        prefix="c_",
    )
    dlo, dhi = _interval_deepset(
        box.lo,
        box.hi,
        ckpt["student_params"]["correction"],
        prefix="c_",
        k=k,
        m=m,
    )
    g = net.getNewVariable()
    net.setLowerBound(g, 0.0)
    net.setUpperBound(g, 1.0)
    zvars = [net.getNewVariable() for _ in range(2)]
    for z, d, lo, hi in zip(zvars, corr_raw, dlo, dhi):
        net.setLowerBound(z, min(0.0, float(lo), float(hi)))
        net.setUpperBound(z, max(0.0, float(lo), float(hi)))
        _add_mccormick(net, g, d, z, float(lo), float(hi))
    raw = []
    for b, z in zip(base_raw, zvars):
        a = net.getNewVariable()
        _eq(net, [(1.0, b), (1.0, z), (-1.0, a)], 0.0)
        raw.append(a)
    action = _add_clip(net, raw)
    _add_two_step_square(net, xvars, action, k=k, threshold=threshold)
    t0 = time.time()
    code, vals, stats = _solve(net, timeout)
    elapsed = time.time() - t0
    # Marabou stores both equalities and inequalities in equList.
    n_ineq = sum(e.EquationType != MarabouCore.Equation.EQ for e in net.equList)
    result = {
        "bucket": bucket,
        "status": code.lower(),
        "elapsed_s": elapsed,
        "timeout_s": timeout,
        "input_box": {"lo": box.lo.tolist(), "hi": box.hi.tolist()},
        "square_half": threshold,
        "relaxation": "g in [0,1] + McCormick envelope for g*delta",
        "delta_interval": {"lo": dlo.tolist(), "hi": dhi.tolist()},
        "network_vars": int(net.numVars),
        "network_equations": int(len(net.equList)),
        "network_inequalities": int(n_ineq),
        "meaning": (
            "UNSAT is a conservative proof for the real residual on this square proxy; "
            "SAT is only a witness for the relaxed gate/product model; timeout is unknown."
        ),
    }
    if code.lower() == "sat" and vals:
        result["witness_input"] = [float(vals[v]) for v in xvars]
        result["witness_action"] = [float(vals[v]) for v in action]
    del stats
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--buckets", default="aa,ao")
    ap.add_argument("--radius", type=float, default=1e-3)
    ap.add_argument("--dist", type=float, default=0.15)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for bucket in [x.strip() for x in args.buckets.split(",") if x.strip()]:
        print(f"[residual-marabou] bucket={bucket} dist={args.dist} radius={args.radius}", flush=True)
        row = run_one(Path(args.checkpoint), bucket, args.radius, args.dist, args.timeout)
        rows.append(row)
        print(f"  status={row['status']} vars={row['network_vars']} time={row['elapsed_s']:.2f}s", flush=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "kind": "residual_marabou_feasibility_poc",
        "rows": rows,
        "note": (
            "The learned sigmoid/physics gate is over-approximated by arbitrary g in [0,1]. "
            "Only UNSAT is a sound certificate for the deployed residual on the queried box."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[residual-marabou] wrote {out}")


if __name__ == "__main__":
    main()
