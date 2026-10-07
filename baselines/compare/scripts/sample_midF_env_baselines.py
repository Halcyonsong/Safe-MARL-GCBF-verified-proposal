#!/usr/bin/env python3
"""Collect real env graphs in mid-F *semantic* band, then score baselines on the SAME set.

mid F (formal probe box): min_d∈[0.125,0.40], |close|≤0.30, R=0.10
  → physics d2_lo≈0.098 < R  (network-dependent; r2 formal VIOLATED with CEs)

This script is **empirical**, not formal:
  1) Roll out env, keep timesteps where some agent has (d0,c0) in the mid-F band.
  2) Persist those GraphsTuple (+ per-agent features / band mask).
  3) Query each baseline on the *same* graphs; score two-step radial L2 per in-band agent.

Graph policies (DS / GCBF+ / HandCBF / InforMARL) and feature policies (MF / MAPPO)
all consume the same env state.

Usage:
  python scripts/sample_midF_env_baselines.py --target 3000 --episodes 200
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

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
from gcbfplus.utils.utils import jax_jit_np

from baseline.informarl.nets import GNNGaussianActor
from baseline.mappo_lag.nets import GaussianActor
from stage1.student import extract_local_sets
from stage2.dynamics import DT, HORIZON_STEPS, MASS
from stage3.residual_policy import apply_residual, load_residual_checkpoint
from stage_mf.property_l2 import D_EMPTY_DEFAULT, N_OCCUPIED, occupied_threat_pairs
from stage_mf.residual import (
    CorrectionMLP,
    apply_residual_mf,
    load_mf_base,
    load_mf_residual,
)
from stage_mf.student import FEATURE_VERSION, INPUT_DIM, extract_meanfield_features

R = 0.10
# Default = mid-F *hard fringe* where formal CEs concentrate
# (full mid-F d∈[0.125,0.40],|c|≤0.30 is mostly easy under real rollouts).
D0_LO, D0_HI = 0.125, 0.135
C0_LO, C0_HI = 0.20, 0.30


def d0_c0_from_feats(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xs [A,44] → d0,c0 per agent."""
    xs = np.asarray(xs, dtype=np.float64)
    a = xs.shape[0]
    d0 = np.full(a, D_EMPTY_DEFAULT, dtype=np.float64)
    c0 = np.zeros(a, dtype=np.float64)
    any_occ = np.zeros(a, dtype=bool)
    for n_i, d_i, c_i in occupied_threat_pairs():
        occ = xs[:, n_i] >= N_OCCUPIED
        any_occ |= occ
        d0 = np.where(occ, np.minimum(d0, xs[:, d_i]), d0)
        c0 = np.where(occ, np.maximum(c0, xs[:, c_i]), c0)
    d0 = np.where(any_occ, d0, D_EMPTY_DEFAULT)
    c0 = np.maximum(c0, 0.0)
    return d0, c0


def d2_from_dc_a(d0: np.ndarray, c0: np.ndarray, acts: np.ndarray) -> np.ndarray:
    a_mag = np.max(np.abs(acts), axis=-1)
    d, c = d0.astype(np.float64).copy(), c0.astype(np.float64).copy()
    acc = a_mag / max(MASS, 1e-9)
    for _ in range(HORIZON_STEPS):
        d = d - c * DT
        c = np.maximum(c + acc * DT, 0.0)
    return d


def in_mid_f(d0: np.ndarray, c0: np.ndarray) -> np.ndarray:
    return (d0 >= D0_LO) & (d0 <= D0_HI) & (c0 >= C0_LO) & (c0 <= C0_HI)


def summarize(name: str, d0, c0, acts) -> dict:
    d2 = d2_from_dc_a(d0, c0, acts)
    margin = d2 - R
    a_inf = np.max(np.abs(acts), axis=-1)
    return {
        "policy": name,
        "n": int(len(d0)),
        "l2_safe_rate": float(np.mean(d2 >= R - 1e-9)),
        "n_fail": int(np.sum(d2 < R - 1e-9)),
        "fail_rate": float(np.mean(d2 < R - 1e-9)),
        "margin_mean": float(margin.mean()),
        "margin_p10": float(np.percentile(margin, 10)),
        "margin_min": float(margin.min()),
        "mean_abs_a_inf": float(a_inf.mean()),
        "p90_abs_a_inf": float(np.percentile(a_inf, 90)),
        "mean_d0": float(d0.mean()),
        "mean_c0": float(c0.mean()),
        "mean_d2": float(d2.mean()),
    }


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


def collect_band(
    env,
    *,
    episodes: int,
    seed: int,
    target: int,
    stride: int,
    max_graphs: int,
    explorer: str = "u_ref",
    explorer_act: Callable | None = None,
) -> dict:
    """Roll out with explorer policy; keep graphs with ≥1 agent in band."""
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])

    if explorer_act is not None:
        act_fn = jax.jit(explorer_act)
        explorer_name = explorer
    else:
        act_fn = jax.jit(env.u_ref)
        explorer_name = "env.u_ref"
    rollout_fn = jax_jit_np(env.rollout_fn(act_fn, env.max_episode_steps))

    kept_graphs = []
    kept_feats = []  # [A,44] np
    kept_mask = []  # [A] bool in-band
    kept_meta = []
    n_in_band = 0
    n_agent_steps = 0
    t0 = time.time()
    keys = jr.split(jr.PRNGKey(seed), episodes)

    print(
        f"[collect] midF band d0∈[{D0_LO},{D0_HI}] c0∈[{C0_LO},{C0_HI}] "
        f"target_agents={target} episodes≤{episodes}",
        flush=True,
    )

    checkpoint_every = 25
    ckpt_path = pathlib.Path("baselines/compare/output/midF_env_sample/_partial_pack.pkl")

    for ep_i, key in enumerate(keys):
        if n_in_band >= target or len(kept_graphs) >= max_graphs:
            break
        rollout = rollout_fn(key)
        graphs = rollout.Tp1_graph
        T = int(np.asarray(rollout.T_action).shape[0])
        for t in range(0, T, max(stride, 1)):
            if n_in_band >= target or len(kept_graphs) >= max_graphs:
                break
            g = jax.tree_util.tree_map(lambda x, _t=t: x[_t], graphs)
            feats = np.asarray(extract_meanfield_features(g, n_agents, n_rays, comm))
            d0, c0 = d0_c0_from_feats(feats)
            mask = in_mid_f(d0, c0)
            n_agent_steps += n_agents
            if not mask.any():
                continue
            # store numpy-friendly graph
            g_np = jax.tree_util.tree_map(lambda x: np.asarray(x), g)
            kept_graphs.append(g_np)
            kept_feats.append(feats.astype(np.float64))
            kept_mask.append(mask)
            kept_meta.append(
                {
                    "ep": int(ep_i),
                    "t": int(t),
                    "n_in_band": int(mask.sum()),
                    "d0_band": d0[mask].tolist(),
                    "c0_band": c0[mask].tolist(),
                }
            )
            n_in_band += int(mask.sum())
        if (ep_i + 1) % 10 == 0 or ep_i == 0:
            print(
                f"  epi={ep_i+1} graphs={len(kept_graphs)} "
                f"agents_in_band={n_in_band}/{target} "
                f"wall={time.time()-t0:.1f}s",
                flush=True,
            )
        if (ep_i + 1) % checkpoint_every == 0 and kept_graphs:
            partial = {
                "graphs": kept_graphs,
                "feats": kept_feats,
                "masks": kept_mask,
                "meta": kept_meta,
                "n_in_band": int(n_in_band),
                "n_graphs": len(kept_graphs),
            }
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "wb") as f:
                pickle.dump(partial, f)
            print(f"  [ckpt] partial pack n_in_band={n_in_band} → {ckpt_path}", flush=True)

    return {
        "graphs": kept_graphs,
        "feats": kept_feats,
        "masks": kept_mask,
        "meta": kept_meta,
        "n_in_band": int(n_in_band),
        "n_graphs": len(kept_graphs),
        "n_agent_steps_scanned": int(n_agent_steps),
        "episodes_used": int(ep_i + 1),
        "wall_s": round(time.time() - t0, 2),
        "band": {"d0": [D0_LO, D0_HI], "c0": [C0_LO, C0_HI], "R": R},
        "explorer": explorer_name,
        "note": (
            "Semantic (d0,c0) band on real GraphsTuple — NOT the formal 44-D product box. "
            "Other feature dims follow the env, not box bounds."
        ),
    }


# ---------------------------------------------------------------------------
# Policy builders: graph → actions [A,2]
# ---------------------------------------------------------------------------


def build_coast(env) -> Callable:
    n = env.num_agents

    def act(graph):
        return np.zeros((n, 2), dtype=np.float64)

    return act


def build_mf(env, ckpt_path: pathlib.Path, *, residual: bool) -> Callable:
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()
    if residual:
        ck = load_mf_residual(ckpt_path)
        bp = pathlib.Path(ck["base_checkpoint"])
        if not bp.exists():
            bp = _ROOT / pathlib.Path(
                "method/meanfield/output/distill/mf_distill_20261005_v5/seed0/"
                "meanfield_mlp512_student_step9000.pkl"
            )
        _, base, base_params = load_mf_base(bp)
        if ck.get("base_params") is not None:
            base_params = ck["base_params"]
        delta = CorrectionMLP(hidden=int(ck.get("correction_hidden", 128)), action_dim=2)
        delta_params = ck["delta_params"]
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

    else:
        _, base, base_params = load_mf_base(ckpt_path)

        @jax.jit
        def control(graph):
            x = extract_meanfield_features(graph, n_agents, n_rays, comm)
            return jnp.clip(base.apply(base_params, x), lo, hi)

    def act(graph):
        g = _to_jax_graph(graph)
        return np.asarray(control(g))

    return act


def build_mappo(env, ckpt_path: pathlib.Path) -> Callable:
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    actor = GaussianActor(hidden=int(ck["hidden"]), action_dim=2)
    params = ck["actor_params"]
    mode = ck.get("action_mode", "uref_plus_tanh_residual")
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()

    @jax.jit
    def control(graph):
        x = extract_meanfield_features(graph, n_agents, n_rays, comm)
        mean, _ = actor.apply(params, x)
        if mode == "uref_plus_tanh_residual":
            a = env.u_ref(graph) + jnp.tanh(mean)
        else:
            a = jnp.tanh(mean)
        return jnp.clip(a, lo, hi)

    def act(graph):
        return np.asarray(control(_to_jax_graph(graph)))

    return act


def build_ds(env, ckpt_path: pathlib.Path) -> Callable:
    ckpt, modules, base_params, train_params = load_residual_checkpoint(ckpt_path)
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()
    gate_mode = str(ckpt.get("gate_mode", "hybrid"))
    open_mult = float(ckpt.get("open_mult", 2.0))
    close_mult = float(ckpt.get("close_mult", 1.0))

    @jax.jit
    def control(graph):
        ego, entities = extract_local_sets(graph, n_agents, n_rays, comm)
        a, _, _ = apply_residual(
            modules,
            base_params,
            train_params,
            ego,
            entities,
            gate_mode=gate_mode,
            open_mult=open_mult,
            close_mult=close_mult,
            action_lim=(lo, hi),
        )
        return a

    def act(graph):
        return np.asarray(control(_to_jax_graph(graph)))

    return act


def build_gcbfplus(env, run_dir: pathlib.Path, step: int) -> Callable:
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
    act_j = jax.jit(algo.act)

    def act(graph):
        return np.asarray(act_j(_to_jax_graph(graph)))

    return act


def build_informarl(env, ckpt_path: pathlib.Path) -> Callable:
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    actor = GNNGaussianActor(
        n_agents=env.num_agents,
        action_dim=env.action_dim,
        gnn_layers=int(ck.get("gnn_layers", 2)),
    )
    params = ck["actor_params"]
    mode = ck.get("action_mode", "uref_plus_tanh_residual")
    lo, hi = env.action_lim()

    @jax.jit
    def control(graph):
        mean, _ = actor.apply(params, graph)
        if mode == "uref_plus_tanh_residual":
            a = env.u_ref(graph) + jnp.tanh(mean)
        else:
            a = mean
        return jnp.clip(a, lo, hi)

    def act(graph):
        return np.asarray(control(_to_jax_graph(graph)))

    return act


def build_handcbf(env, alpha: float = 1.0) -> Callable:
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
    act_j = jax.jit(algo.act)

    def act(graph):
        return np.asarray(act_j(_to_jax_graph(graph)))

    return act


def _to_jax_graph(graph):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), graph)


def score_policy(
    name: str,
    act_fn: Callable,
    pack: dict,
    *,
    max_graphs: int | None = None,
) -> dict:
    graphs = pack["graphs"]
    feats = pack["feats"]
    masks = pack["masks"]
    if max_graphs is not None:
        graphs = graphs[:max_graphs]
        feats = feats[:max_graphs]
        masks = masks[:max_graphs]

    d0_all, c0_all, a_all = [], [], []
    t0 = time.time()
    for i, (g, feat, mask) in enumerate(zip(graphs, feats, masks)):
        acts = np.asarray(act_fn(g), dtype=np.float64)
        if acts.ndim != 2 or acts.shape[0] != feat.shape[0]:
            raise RuntimeError(f"{name}: bad action shape {acts.shape} vs feat {feat.shape}")
        d0, c0 = d0_c0_from_feats(feat)
        d0_all.append(d0[mask])
        c0_all.append(c0[mask])
        a_all.append(acts[mask])
        if (i + 1) % 50 == 0:
            print(f"  [{name}] scored {i+1}/{len(graphs)} graphs", flush=True)
    d0 = np.concatenate(d0_all)
    c0 = np.concatenate(c0_all)
    acts = np.concatenate(a_all, axis=0)
    st = summarize(name, d0, c0, acts)
    st["fwd_s"] = round(time.time() - t0, 2)
    print(
        f"[score] {name}: n={st['n']} L2-safe={100*st['l2_safe_rate']:.2f}% "
        f"fail={st['n_fail']} margin_mean={st['margin_mean']:.4f} "
        f"‖a‖∞={st['mean_abs_a_inf']:.3f} wall={st['fwd_s']}s",
        flush=True,
    )
    return st


def main() -> None:
    global D0_LO, D0_HI, C0_LO, C0_HI
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000, help="in-band agent-states to collect")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--d0-lo", type=float, default=0.125)
    ap.add_argument("--d0-hi", type=float, default=0.135)
    ap.add_argument("--c0-lo", type=float, default=0.20)
    ap.add_argument("--c0-hi", type=float, default=0.30)
    ap.add_argument("--out-dir", type=str, default="baselines/compare/output/midF_env_sample")
    ap.add_argument(
        "--methods",
        type=str,
        default="coast_a0,mf_base,mf_r2,mappo_lag,ds_r2,gcbfplus,informarl",
        help="comma list; add handcbf if needed (slow QP)",
    )
    ap.add_argument("--handcbf-max-graphs", type=int, default=200)
    ap.add_argument(
        "--explorer",
        type=str,
        default="mf_base",
        choices=["u_ref", "mf_base", "mf_r2"],
        help="rollout policy used only to *visit* near-band states",
    )
    ap.add_argument("--skip-collect", action="store_true", help="reuse saved pack.pkl")
    args = ap.parse_args()

    D0_LO, D0_HI = float(args.d0_lo), float(args.d0_hi)
    C0_LO, C0_HI = float(args.c0_lo), float(args.c0_hi)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pack_path = out / "env_pack.pkl"

    env = make_env(
        "DoubleIntegrator",
        num_agents=8,
        num_obs=4,
        area_size=4.0,
        n_rays=32,
    )

    if args.skip_collect and pack_path.exists():
        print(f"[load] {pack_path}", flush=True)
        with open(pack_path, "rb") as f:
            pack = pickle.load(f)
    else:
        explorer_act = None
        n_agents = env.num_agents
        n_rays = int(env.params["n_rays"])
        comm = float(env.params["comm_radius"])
        lo, hi = env.action_lim()
        if args.explorer == "mf_base":
            _, base, base_params = load_mf_base(
                pathlib.Path("method/meanfield/output/ACTIVE_MF_BASE.txt").read_text().strip()
            )

            @jax.jit
            def explorer_act(graph):
                x = extract_meanfield_features(graph, n_agents, n_rays, comm)
                return jnp.clip(base.apply(base_params, x), lo, hi)

        elif args.explorer == "mf_r2":
            ck = load_mf_residual(
                pathlib.Path("method/meanfield/output/ACTIVE_MF_RESIDUAL.txt").read_text().strip()
            )
            _, base, base_params = load_mf_base(
                pathlib.Path("method/meanfield/output/ACTIVE_MF_BASE.txt").read_text().strip()
            )
            if ck.get("base_params") is not None:
                base_params = ck["base_params"]
            delta = CorrectionMLP(hidden=int(ck.get("correction_hidden", 128)), action_dim=2)
            delta_params = ck["delta_params"]

            @jax.jit
            def explorer_act(graph):
                x = extract_meanfield_features(graph, n_agents, n_rays, comm)
                a, _, _ = apply_residual_mf(
                    base,
                    base_params,
                    delta,
                    delta_params,
                    x,
                    open_mult=float(ck.get("open_mult", 3.0)),
                    close_mult=float(ck.get("close_mult", 1.0)),
                    inv_ttc_ref=float(ck.get("inv_ttc_ref", 5.0)),
                    gate_mix=float(ck.get("gate_mix", 0.5)),
                    action_lim=(lo, hi),
                    delta_clip=float(ck.get("delta_clip", 1.0)),
                )
                return a

        pack = collect_band(
            env,
            episodes=args.episodes,
            seed=args.seed,
            target=args.target,
            stride=args.stride,
            max_graphs=args.max_graphs,
            explorer=args.explorer,
            explorer_act=explorer_act,
        )
        # persist: full pack + feats-only (avoids GraphsTuple pickle fragility on reload)
        slim = {
            k: pack[k]
            for k in (
                "feats",
                "masks",
                "meta",
                "n_in_band",
                "n_graphs",
                "n_agent_steps_scanned",
                "episodes_used",
                "wall_s",
                "band",
                "explorer",
                "note",
            )
        }
        with open(pack_path, "wb") as f:
            pickle.dump(pack, f)
        with open(out / "env_feats_only.pkl", "wb") as f:
            pickle.dump({"feats": pack["feats"], "masks": pack["masks"], "meta": pack["meta"], "band": pack.get("band")}, f)
        (out / "collect_meta.json").write_text(json.dumps(slim, indent=2, default=str) + "\n")
        print(
            f"[collect] saved {pack_path} graphs={pack['n_graphs']} "
            f"in_band={pack['n_in_band']} wall={pack['wall_s']}s",
            flush=True,
        )

    if pack["n_in_band"] == 0:
        raise SystemExit("no in-band agents collected — widen band or more episodes")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    paths = {
        "mf_base": pathlib.Path("method/meanfield/output/ACTIVE_MF_BASE.txt").read_text().strip(),
        "mf_r2": pathlib.Path("method/meanfield/output/ACTIVE_MF_RESIDUAL.txt").read_text().strip(),
        "mappo_lag": pathlib.Path("baselines/mappo_lag/output/mappo_lag/ACTIVE.txt").read_text().strip(),
        "ds_r2": pathlib.Path("method/deepset/output/phase2/ACTIVE_RESIDUAL.txt").read_text().strip(),
        "informarl": pathlib.Path("baselines/informarl/output/informarl/ACTIVE.txt").read_text().strip(),
        "gcbfplus_run": "baselines/gcbfplus/log/gcbf+/seed0_20261001232611",
        "gcbfplus_step": 1000,
    }

    builders: dict[str, Callable[[], Callable]] = {
        "coast_a0": lambda: build_coast(env),
        "mf_base": lambda: build_mf(env, pathlib.Path(paths["mf_base"]), residual=False),
        "mf_r2": lambda: build_mf(env, pathlib.Path(paths["mf_r2"]), residual=True),
        "mappo_lag": lambda: build_mappo(env, pathlib.Path(paths["mappo_lag"])),
        "ds_r2": lambda: build_ds(env, pathlib.Path(paths["ds_r2"])),
        "gcbfplus": lambda: build_gcbfplus(
            env, pathlib.Path(paths["gcbfplus_run"]), int(paths["gcbfplus_step"])
        ),
        "informarl": lambda: build_informarl(env, pathlib.Path(paths["informarl"])),
        "handcbf": lambda: build_handcbf(env, alpha=1.0),
    }

    rows = []
    t_all = time.time()
    for name in methods:
        if name not in builders:
            print(f"[skip] unknown method {name}", flush=True)
            continue
        print(f"[build] {name} …", flush=True)
        act_fn = builders[name]()
        max_g = args.handcbf_max_graphs if name == "handcbf" else None
        rows.append(score_policy(name, act_fn, pack, max_graphs=max_g))

    report = {
        "claim": (
            "Empirical two-step radial L2 on real env graphs whose (d0,c0) lie in "
            "the mid-F semantic band. NOT formal verification. "
            "Formal mid-F product-box probe on MF-r2 was VIOLATED (204 CEs)."
        ),
        "band": pack.get("band", {"d0": [D0_LO, D0_HI], "c0": [C0_LO, C0_HI], "R": R}),
        "collect": {
            "n_in_band": pack["n_in_band"],
            "n_graphs": pack["n_graphs"],
            "episodes_used": pack.get("episodes_used"),
            "explorer": pack.get("explorer"),
            "note": pack.get("note"),
        },
        "paths": paths,
        "rows": rows,
        "wall_s": round(time.time() - t_all, 2),
        "feature_version": FEATURE_VERSION,
        "input_dim": INPUT_DIM,
    }
    (out / "midF_env_l2.json").write_text(json.dumps(report, indent=2) + "\n")

    md = [
        "# mid-F semantic band · shared env graphs (empirical)",
        "",
        "> Same GraphsTuple batch for all methods. Filter: "
        f"`d0∈[{D0_LO},{D0_HI}]`, `c0∈[{C0_LO},{C0_HI}]` (mid-F semantics).",
        "> **Not** the formal 44-D product box. Formal mid-F on MF-r2 = VIOLATED (204 CE).",
        "",
        f"- in-band agents={pack['n_in_band']} · graphs={pack['n_graphs']} · "
        f"seed={args.seed} · R={R}",
        "",
        "| policy | n | L2-safe% | n_fail | fail% | margin mean | p10 | min | ‖a‖∞ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['policy']} | {r['n']} | {100*r['l2_safe_rate']:.2f} | "
            f"{r['n_fail']} | {100*r['fail_rate']:.2f} | {r['margin_mean']:.4f} | "
            f"{r['margin_p10']:.4f} | {r['margin_min']:.4f} | {r['mean_abs_a_inf']:.3f} |"
        )
    md += [
        "",
        "## How to read",
        "",
        "1. Failures here mean two-step **radial** L2 unsafe under that policy's action.",
        "2. Expect **coast** often safer (small ‖a‖); aggressive policies fail more in this band.",
        "3. MF-r2 may still fail empirically here (matches formal mid-F VIOLATED) — "
        "repair target, not a claim of zero fail yet.",
        "4. After CE repair + PROVEN on formal mid-F, re-score this pack to show separation.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(md))
    print(json.dumps(rows, indent=2))
    print(f"wrote {out}/REPORT.md")


if __name__ == "__main__":
    main()
