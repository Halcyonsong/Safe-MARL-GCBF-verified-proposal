#!/usr/bin/env python3
"""Benchmark control-computation latency (not full env step).

Methods (default):
  - mf_r2:     feature extract + residual forward
  - gcbfplus:  GNN actor act (deploy; no QP)
  - handcbf:   DecShareCBF QP solve

Protocol env: DoubleIntegrator · 8 · area4 · obs4 · rays32 · seed=1234
Reports median / mean / p95 ms per control call (all 8 agents in one call).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import pathlib as _pathlib
import sys as _sys
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "core" / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from core.paths import project_root, ensure_import_path
_ROOT = project_root()
ensure_import_path()



from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from stage_mf.residual import CorrectionMLP, apply_residual_mf, load_mf_base, load_mf_residual
from stage_mf.student import extract_meanfield_features


def _percentile(xs: np.ndarray, q: float) -> float:
    return float(np.percentile(xs, q))


def _bench(fn: Callable[[], None], *, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    # block_until_ready if jax arrays returned — callers should include that
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n": int(repeats),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": _percentile(arr, 95),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std()),
    }


def _sample_graphs(env, n: int, seed: int):
    keys = jr.split(jr.PRNGKey(seed), n)
    graphs = []
    for k in keys:
        g = env.reset(k)
        graphs.append(g)
    return graphs


def build_mf_fn(env, ckpt_path: pathlib.Path):
    ck = load_mf_residual(ckpt_path)
    bp = pathlib.Path(ck["base_checkpoint"])
    if not bp.exists():
        bp = _ROOT / "method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl"
    _, base, base_params = load_mf_base(bp)
    if ck.get("base_params") is not None:
        base_params = ck["base_params"]
    delta = CorrectionMLP(hidden=int(ck.get("correction_hidden", 128)), action_dim=2)
    delta_params = ck["delta_params"]
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()
    open_mult = float(ck.get("open_mult", 3.0))
    close_mult = float(ck.get("close_mult", 1.0))
    inv_ttc_ref = float(ck.get("inv_ttc_ref", 5.0))
    gate_mix = float(ck.get("gate_mix", 0.5))
    delta_clip = float(ck.get("delta_clip", 1.0))

    @jax.jit
    def control(graph):
        x = extract_meanfield_features(graph, n_agents, n_rays, comm)
        a, _, _ = apply_residual_mf(
            base,
            base_params,
            delta,
            delta_params,
            x,
            open_mult=open_mult,
            close_mult=close_mult,
            inv_ttc_ref=inv_ttc_ref,
            gate_mix=gate_mix,
            action_lim=(lo, hi),
            delta_clip=delta_clip,
        )
        return a

    return control, str(ckpt_path)


def build_gcbfplus_fn(env, run_dir: pathlib.Path, step: int):
    import yaml

    with open(run_dir / "config.yaml") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=config.gnn_layers,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        horizon=config.horizon,
        lr_actor=config.lr_actor,
        lr_cbf=config.lr_cbf,
        alpha=config.alpha,
        eps=0.02,
        inner_epoch=8,
        loss_action_coef=config.loss_action_coef,
        loss_unsafe_coef=config.loss_unsafe_coef,
        loss_safe_coef=config.loss_safe_coef,
        loss_h_dot_coef=config.loss_h_dot_coef,
        max_grad_norm=2.0,
        seed=config.seed,
    )
    algo.load(str(run_dir / "models"), step)
    act = jax.jit(algo.act)
    return act, f"{run_dir}/models/{step}"


def build_handcbf_fn(env, alpha: float):
    algo = make_algo(
        algo="dec_share_cbf",
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        alpha=alpha,
    )
    act = jax.jit(algo.act)
    return act, f"dec_share_cbf(alpha={alpha})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n-graphs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--mf-ckpt",
        type=str,
        default="acceptance/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl",
    )
    ap.add_argument(
        "--gcbfplus-run",
        type=str,
        default="baselines/gcbfplus/log/gcbf+/seed0_20261001232611",
    )
    ap.add_argument("--gcbfplus-step", type=int, default=1000)
    ap.add_argument(
        "--out-json",
        type=str,
        default="baselines/compare/output/latency/control_latency.json",
    )
    ap.add_argument("--methods", type=str, default="mf_r2,gcbfplus,handcbf")
    args = ap.parse_args()

    env = make_env(
        "DoubleIntegrator",
        num_agents=8,
        num_obs=4,
        area_size=4.0,
        n_rays=32,
    )
    graphs = _sample_graphs(env, args.n_graphs, args.seed)
    # cycle graphs to avoid measuring only one state
    idx = {"i": 0}

    def next_graph():
        g = graphs[idx["i"] % len(graphs)]
        idx["i"] += 1
        return g

    builders = {
        "mf_r2": lambda: build_mf_fn(env, _ROOT / args.mf_ckpt),
        "gcbfplus": lambda: build_gcbfplus_fn(
            env, _ROOT / args.gcbfplus_run, args.gcbfplus_step
        ),
        "handcbf": lambda: build_handcbf_fn(env, args.alpha),
    }

    results = {
        "protocol": {
            "env": "DoubleIntegrator",
            "num_agents": 8,
            "area_size": 4.0,
            "obs": 4,
            "n_rays": 32,
            "seed": args.seed,
            "n_graphs": args.n_graphs,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "metric": "ms per control call (all agents); excludes env.step",
            "device": str(jax.default_backend()),
        },
        "methods": {},
    }

    for name in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if name not in builders:
            raise SystemExit(f"unknown method {name}; choose from {list(builders)}")
        print(f"[bench] building {name} ...", flush=True)
        control, ckpt = builders[name]()
        # warmup compile on first graph
        g0 = graphs[0]
        a0 = control(g0)
        jax.block_until_ready(a0)

        def once():
            g = next_graph()
            a = control(g)
            jax.block_until_ready(a)

        print(f"[bench] timing {name} ...", flush=True)
        stats = _bench(once, warmup=args.warmup, repeats=args.repeats)
        entry = {"checkpoint": ckpt, "compute": name, **stats}
        results["methods"][name] = entry
        print(
            f"  {name}: median={stats['median_ms']:.3f}ms "
            f"mean={stats['mean_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms",
            flush=True,
        )

    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
