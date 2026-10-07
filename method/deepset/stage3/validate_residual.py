"""Empirical rollout validation for residual-gate checkpoints.

Not a formal safety certificate. Supports architecture=deepset_residual_gate.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

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

import jax.numpy as jnp
import jax.random as jr
import numpy as np



from gcbfplus.env import make_env
from gcbfplus.utils.utils import jax_jit_np, jax_vmap
from stage1.student import extract_local_sets
from stage3.residual_policy import apply_residual, load_residual_checkpoint


def evaluate_residual_checkpoint(
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
    ckpt, modules, base_params, train_params = load_residual_checkpoint(checkpoint_path)
    n_agents = int(num_agents or ckpt.get("n_max_agents", 8))
    n_rays_v = int(n_rays or ckpt.get("n_max_obs", 32))
    gate_mode = str(ckpt.get("gate_mode", "hybrid"))
    open_mult = float(ckpt.get("open_mult", 2.0))
    close_mult = float(ckpt.get("close_mult", 1.0))

    env = make_env(
        "DoubleIntegrator",
        num_agents=n_agents,
        area_size=area_size,
        num_obs=obs,
        n_rays=n_rays_v,
        max_step=max_step,
    )
    lo, hi = env.action_lim()
    comm_radius = float(env.params["comm_radius"])

    def policy(graph):
        ego, entities = extract_local_sets(graph, n_agents, n_rays_v, comm_radius)
        a, _g, _d = apply_residual(
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
    safe_rate = float(agent_safe.mean())
    finish_rate = float(finish_a.mean())
    success_rate = float(agent_success.mean())
    n_trials = int(agent_success.size)

    return {
        "policy_kind": "deepset_residual_gate",
        "checkpoint": str(checkpoint_path),
        "base_checkpoint": ckpt.get("base_checkpoint"),
        "architecture": "deepset_residual_gate",
        "gate_mode": gate_mode,
        "episodes": int(episodes),
        "eval_seed": int(seed),
        "num_agents": n_agents,
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "safe_rate": safe_rate,
        "finish_rate": finish_rate,
        "success_rate": success_rate,
        "n_agent_trials": n_trials,
        "success_se_agent": math.sqrt(max(success_rate * (1.0 - success_rate), 1e-12) / n_trials),
        "safe_se_agent": math.sqrt(max(safe_rate * (1.0 - safe_rate), 1e-12) / n_trials),
        "wall_time_s": round(time.time() - t0, 3),
        "claim": "Empirical rollout only. Not formal verification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out residual-gate Stage3 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=4)
    parser.add_argument("--n-rays", type=int, default=None)
    parser.add_argument("--area-size", type=float, default=4.0)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    metrics = evaluate_residual_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        num_agents=args.num_agents,
        obs=args.obs,
        n_rays=args.n_rays,
        area_size=args.area_size,
        max_step=args.max_step,
    )
    print(
        f"checkpoint={metrics['checkpoint']} architecture=deepset_residual_gate "
        f"base={metrics.get('base_checkpoint')} gate_mode={metrics.get('gate_mode')} "
        f"episodes={metrics['episodes']} "
        f"reward={metrics['reward']:.3f} cost={metrics['cost']:.3f} "
        f"safe_rate={metrics['safe_rate'] * 100:.3f}% finish_rate={metrics['finish_rate'] * 100:.3f}% "
        f"success_rate={metrics['success_rate'] * 100:.3f}%"
    )
    print("NOTE: empirical rollout metrics are not formal safety verification.")
    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
