"""Validate the exact DeepSet Student checkpoint used for deployment.

This is an empirical rollout check only. It does not constitute formal
verification or a safety certificate. The local-set construction and network
definition are shared with ``train.py``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
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
from stage1.student import DeepSetStudent, extract_local_sets


def evaluate_deepset_checkpoint(
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
    """Run official-style rollout eval; returns a JSON-serializable dict."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)
    params = checkpoint["student_params"]
    hidden = int(checkpoint["hidden"])
    n_agents = int(num_agents or checkpoint.get("n_max_agents", 8))
    n_rays_v = int(n_rays or checkpoint.get("n_max_obs", 32))
    if checkpoint.get("architecture", "deepset") != "deepset":
        raise ValueError(f"Unsupported student architecture: {checkpoint.get('architecture')}")

    env = make_env(
        "DoubleIntegrator",
        num_agents=n_agents,
        area_size=area_size,
        num_obs=obs,
        n_rays=n_rays_v,
        max_step=max_step,
    )
    student = DeepSetStudent(hidden=hidden, action_dim=env.action_dim)
    comm_radius = float(env.params["comm_radius"])

    def policy(graph):
        ego, entities = extract_local_sets(graph, n_agents, n_rays_v, comm_radius)
        if ego.shape[-1] != int(checkpoint["ego_dim"]):
            raise ValueError(
                f"checkpoint ego_dim={checkpoint['ego_dim']}, current={ego.shape[-1]}"
            )
        if entities.shape[-1] != int(checkpoint["entity_dim"]):
            raise ValueError(
                f"checkpoint entity_dim={checkpoint['entity_dim']}, current={entities.shape[-1]}"
            )
        if entities.shape[-2] != int(checkpoint["entity_slots"]):
            raise ValueError(
                f"checkpoint entity_slots={checkpoint['entity_slots']}, current={entities.shape[-2]}"
            )
        return jnp.clip(student.apply(params, ego, entities), *env.action_lim())

    rollout_fn = jax_jit_np(env.rollout_fn(policy, env.max_episode_steps))
    is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
    is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    keys = jr.split(jr.PRNGKey(seed), episodes)

    rewards = []
    costs = []
    unsafe = []
    finish = []
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
    # per-agent episode outcomes then mean (official style)
    agent_safe = 1.0 - unsafe_a
    agent_success = agent_safe * finish_a
    safe_rate = float(agent_safe.mean())
    finish_rate = float(finish_a.mean())
    success_rate = float(agent_success.mean())
    # episode-level (any-agent) extras for SE analysis
    epi_safe = agent_safe.min(axis=1)  # all agents safe
    epi_finish = finish_a.mean(axis=1)
    epi_success = agent_success.mean(axis=1)

    return {
        "policy_kind": "deepset",
        "checkpoint": str(checkpoint_path),
        "architecture": "deepset",
        "episodes": int(episodes),
        "eval_seed": int(seed),
        "num_agents": n_agents,
        "n_rays": n_rays_v,
        "area_size": float(area_size),
        "max_step": int(env.max_episode_steps),
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "safe_rate": safe_rate,
        "finish_rate": finish_rate,
        "success_rate": success_rate,
        # binomial-ish SE on agent-level Bernoulli mean (approx; agents correlated)
        "n_agent_trials": int(agent_success.size),
        "success_se_agent": float(
            np.sqrt(max(success_rate * (1.0 - success_rate), 1e-12) / agent_success.size)
        ),
        "safe_se_agent": float(
            np.sqrt(max(safe_rate * (1.0 - safe_rate), 1e-12) / agent_safe.size)
        ),
        "epi_success_mean": float(epi_success.mean()),
        "epi_success_std": float(epi_success.std()),
        "wall_time_s": round(time.time() - t0, 3),
        "claim": "Empirical rollout only. Not formal verification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out an exact Stage1 DeepSet Student")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=4)
    parser.add_argument("--n-rays", type=int, default=None)
    parser.add_argument("--area-size", type=float, default=4.0)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument(
        "--out-json",
        default=None,
        help="optional path to write full metrics JSON (for compare tables)",
    )
    args = parser.parse_args()

    metrics = evaluate_deepset_checkpoint(
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
        f"checkpoint={metrics['checkpoint']} architecture=deepset episodes={metrics['episodes']} "
        f"reward={metrics['reward']:.3f} cost={metrics['cost']:.3f} "
        f"safe_rate={metrics['safe_rate'] * 100:.3f}% finish_rate={metrics['finish_rate'] * 100:.3f}% "
        f"success_rate={metrics['success_rate'] * 100:.3f}%"
    )
    print(
        f"  approx agent-level SE(success)={metrics['success_se_agent'] * 100:.2f}pt "
        f"(n_agent_trials={metrics['n_agent_trials']}); epi_success_std={metrics['epi_success_std'] * 100:.2f}pt"
    )
    print("NOTE: empirical rollout metrics are not formal safety verification.")

    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
