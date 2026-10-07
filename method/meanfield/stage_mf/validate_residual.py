#!/usr/bin/env python3
"""Empirical rollout for meanfield residual (MF-r2). Supports --num-agents for transfer."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

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



from gcbfplus.env import make_env
from gcbfplus.utils.utils import jax_jit_np, jax_vmap
from stage_mf.residual import (
    CorrectionMLP,
    apply_residual_mf,
    load_mf_base,
    load_mf_residual,
)
from stage_mf.student import FEATURE_VERSION, INPUT_DIM, MeanFieldStudent, extract_meanfield_features


def evaluate_mf_residual(
    checkpoint_path: str | pathlib.Path,
    *,
    episodes: int = 32,
    seed: int = 1234,
    num_agents: int | None = None,
    obs: int = 4,
    n_rays: int | None = None,
    area_size: float = 4.0,
    max_step: int | None = None,
) -> dict:
    ckpt = load_mf_residual(checkpoint_path)
    n_agents = int(num_agents or ckpt.get("n_max_agents", 8))
    n_rays_v = int(n_rays or ckpt.get("n_max_obs", 32))
    hidden = int(ckpt.get("hidden", 256))
    corr_h = int(ckpt.get("correction_hidden", 128))
    open_mult = float(ckpt.get("open_mult", 2.0))
    close_mult = float(ckpt.get("close_mult", 1.0))
    inv_ttc_ref = float(ckpt.get("inv_ttc_ref", 8.0))
    gate_mix = float(ckpt.get("gate_mix", 1.0))
    delta_clip = float(ckpt.get("delta_clip", 0.5))

    base_path = ckpt.get("base_checkpoint")
    if base_path and pathlib.Path(base_path).exists():
        base_ck, base, base_params = load_mf_base(base_path)
        hidden = int(base_ck.get("hidden", hidden))
    else:
        base_params = ckpt["base_params"]
        base = MeanFieldStudent(hidden=hidden, action_dim=2)

    delta = CorrectionMLP(hidden=corr_h, action_dim=2)
    delta_params = ckpt["delta_params"]

    env = make_env(
        "DoubleIntegrator",
        num_agents=n_agents,
        area_size=area_size,
        num_obs=obs,
        n_rays=n_rays_v,
        max_step=max_step,
    )
    lo, hi = env.action_lim()
    comm = float(env.params["comm_radius"])

    def policy(graph):
        x = extract_meanfield_features(graph, n_agents, n_rays_v, comm)
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

    rollout_fn = jax_jit_np(env.rollout_fn(policy, env.max_episode_steps))
    is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
    is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    keys = jr.split(jr.PRNGKey(seed), episodes)

    rewards, costs, unsafe, finish = [], [], [], []
    t0 = time.time()
    for key in keys:
        rollout = rollout_fn(key)
        is_unsafe = is_unsafe_fn(rollout.Tp1_graph)
        is_finish = is_finish_fn(rollout.Tp1_graph)
        rewards.append(float(rollout.T_reward.sum()))
        costs.append(float(rollout.T_cost.sum()))
        unsafe.append(np.asarray(is_unsafe).max(axis=0))
        finish.append(np.asarray(is_finish).max(axis=0))

    unsafe_a = np.stack(unsafe)
    finish_a = np.stack(finish)
    agent_safe = 1.0 - unsafe_a
    agent_success = agent_safe * finish_a

    return {
        "policy_id": "mf_r2",
        "architecture": "meanfield_residual_phys",
        "checkpoint": str(pathlib.Path(checkpoint_path).resolve()),
        "feature_version": FEATURE_VERSION,
        "input_dim": INPUT_DIM,
        "episodes": int(episodes),
        "eval_seed": int(seed),
        "num_agents": n_agents,
        "n_rays": n_rays_v,
        "area_size": float(area_size),
        "obs": int(obs),
        "max_step": int(env.max_episode_steps),
        "reward": float(np.mean(rewards)),
        "cost": float(np.mean(costs)),
        "safe_rate": round(float(agent_safe.mean()), 6),
        "finish_rate": round(float(finish_a.mean()), 6),
        "success_rate": round(float(agent_success.mean()), 6),
        "n_agent_trials": int(agent_success.size),
        "wall_time_s": round(time.time() - t0, 3),
        "claim": "Empirical MF residual transfer. Not formal verification.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-agents", type=int, default=None)
    ap.add_argument("--obs", type=int, default=4)
    ap.add_argument("--n-rays", type=int, default=None)
    ap.add_argument("--area-size", type=float, default=4.0)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not ckpt:
        active = pathlib.Path("method/meanfield/output/ACTIVE_MF_RESIDUAL.txt")
        if active.exists():
            ckpt = active.read_text().strip()
        else:
            raise SystemExit("pass --checkpoint or ACTIVE_MF_RESIDUAL.txt")

    m = evaluate_mf_residual(
        ckpt,
        episodes=args.episodes,
        seed=args.seed,
        num_agents=args.num_agents,
        obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
    )
    print(
        f"MF-r2 N={m['num_agents']} suc={m['success_rate']*100:.1f}% "
        f"safe={m['safe_rate']*100:.1f}% fin={m['finish_rate']*100:.1f}% cost={m['cost']:.3f}"
    )
    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(m, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
