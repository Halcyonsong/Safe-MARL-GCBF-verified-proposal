#!/usr/bin/env python3
"""Validate InforMARL checkpoint under project eval protocol.

  DoubleIntegrator · 8 · area4 · obs4 · rays32 · seed=1234

Usage:
  python baselines/informarl/code/validate.py --checkpoint PATH --epi 32 [--out-json PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time

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



from gcbfplus.env import make_env
from gcbfplus.utils.utils import jax_jit_np, jax_vmap

from baseline.informarl.nets import GNNGaussianActor


def evaluate_informarl(
    checkpoint_path: str | pathlib.Path,
    *,
    episodes: int = 32,
    seed: int = 1234,
    num_agents: int = 8,
    obs: int = 4,
    n_rays: int = 32,
    area_size: float = 4.0,
) -> dict:
    checkpoint_path = pathlib.Path(checkpoint_path)
    with checkpoint_path.open("rb") as f:
        ck = pickle.load(f)
    if ck.get("architecture") not in ("informarl", "InforMARL"):
        raise ValueError(f"unexpected architecture {ck.get('architecture')!r}")
    gnn_layers = int(ck.get("gnn_layers", 2))
    params = ck["actor_params"]
    action_mode = ck.get("action_mode", "uref_plus_tanh_residual")

    env = make_env(
        "DoubleIntegrator",
        num_agents=num_agents,
        area_size=area_size,
        num_obs=obs,
        n_rays=n_rays,
    )
    actor = GNNGaussianActor(
        n_agents=env.num_agents,
        action_dim=env.action_dim,
        gnn_layers=gnn_layers,
    )
    lo, hi = env.action_lim()

    def policy(graph):
        mean, _ = actor.apply(params, graph)
        if action_mode == "uref_plus_tanh_residual":
            a = env.u_ref(graph) + jnp.tanh(mean)
        else:
            a = mean
        return jnp.clip(a, lo, hi)

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
        "policy_id": "informarl",
        "policy_kind": "informarl",
        "label": "InforMARL",
        "architecture": "informarl",
        "checkpoint": str(checkpoint_path.resolve()),
        "gnn_layers": gnn_layers,
        "action_mode": action_mode,
        "cost_weight": float(ck.get("cost_weight", 0.0)),
        "train_step": int(ck.get("step", -1)),
        "episodes": int(episodes),
        "eval_seed": int(seed),
        "num_agents": env.num_agents,
        "n_rays": int(env.params["n_rays"]),
        "area_size": float(area_size),
        "max_step": int(env.max_episode_steps),
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "safe_rate": round(float(agent_safe.mean()), 6),
        "finish_rate": round(float(finish_a.mean()), 6),
        "success_rate": round(float(agent_success.mean()), 6),
        "n_agent_trials": int(agent_success.size),
        "wall_time_s": round(time.time() - t0, 3),
        "claim": "Empirical InforMARL (GNN+PPO, soft cost). Not formal verification.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--epi", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-agents", type=int, default=8)
    ap.add_argument("--obs", type=int, default=4)
    ap.add_argument("--n-rays", type=int, default=32)
    ap.add_argument("--area-size", type=float, default=4.0)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not ckpt:
        active = pathlib.Path("baselines/informarl/output/informarl/ACTIVE.txt")
        if active.exists():
            ckpt = active.read_text().strip()
        else:
            raise SystemExit("pass --checkpoint or create baselines/informarl/output/informarl/ACTIVE.txt")

    metrics = evaluate_informarl(
        ckpt,
        episodes=args.epi,
        seed=args.seed,
        num_agents=args.num_agents,
        obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
    )
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "success_rate",
                    "safe_rate",
                    "finish_rate",
                    "cost",
                    "reward",
                    "wall_time_s",
                    "train_step",
                    "cost_weight",
                )
            },
            indent=2,
        )
    )
    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
