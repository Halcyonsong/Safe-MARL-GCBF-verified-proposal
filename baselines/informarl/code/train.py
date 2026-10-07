#!/usr/bin/env python3
"""Train InforMARL on DoubleIntegrator (same protocol as baselines).

Local agent–entity GNN (env GraphsTuple) + PPO-clip.
No CBF / no Lagrangian dual — safety via soft cost in train reward:
  r_train = r_env - cost_weight * cost_env

Action: a = clip(u_ref + tanh(μ(GNN(graph)))) so unconstrained opt ≈ LQR ref.

Usage:
  python baselines/informarl/code/train.py [--steps 400] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax.training.train_state import TrainState

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
from gcbfplus.utils.graph import GraphsTuple

from baseline.informarl.nets import (
    GNNCritic,
    GNNGaussianActor,
    entropy,
    log_prob_action,
    sample_action,
)


def _gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """rewards/values/dones: [T]; last_value: scalar. Returns adv, returns [T]."""

    def scan_fn(last_gae, inputs):
        r, v, d, v_next = inputs
        delta = r + gamma * v_next * (1.0 - d) - v
        gae = delta + gamma * gae_lambda * (1.0 - d) * last_gae
        return gae, gae

    v_next = jnp.concatenate([values[1:], last_value[None]], axis=0)
    _, adv_rev = jax.lax.scan(
        scan_fn,
        jnp.zeros((), dtype=rewards.dtype),
        (rewards[::-1], values[::-1], dones[::-1], v_next[::-1]),
    )
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns


def make_train_state(module, key, lr, graph0):
    params = module.init(key, graph0)
    tx = optax.adam(lr, eps=1e-5)
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)


def _strip_env_states(graph: GraphsTuple) -> GraphsTuple:
    return graph._replace(env_states=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400, help="PPO update iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cost-weight", type=float, default=1.0,
                    help="Soft cost in train reward: r - cw * cost")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-epi", type=int, default=16)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(
        args.out_dir or f"baselines/informarl/output/informarl/seed{args.seed}_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    env = make_env(
        "DoubleIntegrator",
        num_agents=8,
        area_size=4.0,
        num_obs=4,
        n_rays=32,
    )
    n_agents = env.num_agents
    lo, hi = env.action_lim()
    T = int(env.max_episode_steps)
    n_envs = args.n_envs

    actor = GNNGaussianActor(
        n_agents=n_agents,
        action_dim=env.action_dim,
        gnn_layers=args.gnn_layers,
    )
    critic = GNNCritic(n_agents=n_agents, gnn_layers=args.gnn_layers)

    key = jr.PRNGKey(args.seed)
    key, k0, k1, rk = jr.split(key, 4)
    graph0 = _strip_env_states(env.reset(rk))
    actor_state = make_train_state(actor, k0, args.lr, graph0)
    critic_state = make_train_state(critic, k1, args.lr, graph0)

    def act_det(params, graph):
        mean, _ = actor.apply(params, graph)
        uref = env.u_ref(graph)
        return jnp.clip(uref + jnp.tanh(mean), lo, hi)

    def collect_batch(actor_params, critic_params, keys):
        def one_env(key):
            key, rk = jr.split(key)
            graph0 = env.reset(rk)

            def step_fn(carry, _):
                graph, key = carry
                key, sk = jr.split(key)
                mean, log_std = actor.apply(actor_params, graph)
                uref = env.u_ref(graph)
                raw, logp_raw = sample_action(mean, log_std, sk)
                delta = jnp.tanh(raw)
                action = jnp.clip(uref + delta, lo, hi)
                v = critic.apply(critic_params, graph)
                graph2, reward, cost, done, _ = env.step(graph, action)
                # InforMARL soft cost (dgppo-style): fold cost into train reward
                r_train = reward - args.cost_weight * cost
                return (graph2, key), {
                    "graph": _strip_env_states(graph),
                    "action_raw": raw,
                    "logp": logp_raw,
                    "reward": r_train,
                    "env_reward": reward,
                    "cost": cost,
                    "value": v,
                    "done": done.astype(jnp.float32),
                }

            (graph_f, key_f), traj = jax.lax.scan(step_fn, (graph0, key), None, length=T)
            v_f = critic.apply(critic_params, graph_f)
            traj["bootstrap_v"] = v_f
            return traj

        return jax.vmap(one_env)(keys)

    collect_batch = jax.jit(collect_batch)

    def ppo_update(actor_state, critic_state, batch, key):
        # batch fields: [E, T, ...]
        # Flatten env×time for PPO; keep per-agent logp [E,T,A] → [E*T*A]
        E, Tt = batch["reward"].shape
        graphs = batch["graph"]  # pytree [E,T,...]
        # Flatten graphs over E*T
        flat_graphs = jax.tree.map(
            lambda x: x.reshape((E * Tt,) + x.shape[2:]), graphs
        )
        action_raw = batch["action_raw"].reshape(E * Tt, n_agents, env.action_dim)
        old_logp = batch["logp"].reshape(E * Tt, n_agents)
        adv = batch["adv"].reshape(E * Tt)  # team adv per (e,t)
        ret = batch["ret"].reshape(E * Tt)
        old_v = batch["value"].reshape(E * Tt)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # broadcast team adv to agents
        adv_a = adv[:, None].repeat(n_agents, axis=1)  # [E*T, A]

        n = E * Tt
        mb = max(n // args.num_minibatches, 1)

        def one_epoch(carry, _):
            actor_state, critic_state, key = carry
            key, pk = jr.split(key)
            perm = jr.permutation(pk, n)

            def mb_step(carry, start):
                actor_state, critic_state = carry
                idx = jax.lax.dynamic_slice(perm, (start,), (mb,))
                g = jax.tree.map(lambda x: x[idx], flat_graphs)
                ar = action_raw[idx]
                olp = old_logp[idx]
                ad = adv_a[idx]
                rr = ret[idx]

                def actor_loss_fn(params):
                    # vmap over minibatch graphs
                    def one(graph, a_raw):
                        mean, log_std = actor.apply(params, graph)
                        logp = log_prob_action(mean, log_std, a_raw)
                        ent = entropy(log_std)
                        return logp, ent, log_std

                    logp, ent, _ = jax.vmap(one)(g, ar)
                    ratio = jnp.exp(logp - olp)
                    surr1 = ratio * ad
                    surr2 = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * ad
                    pg = -jnp.minimum(surr1, surr2).mean()
                    return pg - args.ent_coef * ent.mean(), (pg, ent.mean())

                (aloss, (pg, ent)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                    actor_state.params
                )
                g_norm = optax.global_norm(grads)
                grads = jax.tree.map(
                    lambda g: g * jnp.minimum(1.0, args.max_grad_norm / (g_norm + 1e-8)),
                    grads,
                )
                actor_state = actor_state.apply_gradients(grads=grads)

                def critic_loss_fn(params):
                    def one(graph):
                        return critic.apply(params, graph)

                    v = jax.vmap(one)(g)
                    return 0.5 * ((v - rr) ** 2).mean()

                closs, cgrads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
                c_norm = optax.global_norm(cgrads)
                cgrads = jax.tree.map(
                    lambda g: g * jnp.minimum(1.0, args.max_grad_norm / (c_norm + 1e-8)),
                    cgrads,
                )
                critic_state = critic_state.apply_gradients(grads=cgrads)

                return (actor_state, critic_state), {
                    "actor_loss": aloss,
                    "pg": pg,
                    "ent": ent,
                    "critic_loss": closs,
                }

            starts = jnp.arange(0, args.num_minibatches * mb, mb)
            (actor_state, critic_state), info = jax.lax.scan(
                mb_step, (actor_state, critic_state), starts
            )
            return (actor_state, critic_state, key), info

        (actor_state, critic_state, key), infos = jax.lax.scan(
            one_epoch,
            (actor_state, critic_state, key),
            None,
            length=args.ppo_epochs,
        )
        metrics = {k: v.mean() for k, v in infos.items()}
        return actor_state, critic_state, key, metrics

    ppo_update = jax.jit(ppo_update)

    def eval_policy(actor_params, seed: int, episodes: int) -> Dict[str, float]:
        keys = jr.split(jr.PRNGKey(seed), episodes)
        is_unsafe_fn = jax.jit(jax.vmap(env.collision_mask))
        is_finish_fn = jax.jit(jax.vmap(env.finish_mask))

        def policy(graph):
            return act_det(actor_params, graph)

        rollout_fn = jax.jit(env.rollout_fn(policy, env.max_episode_steps))
        rewards, costs, unsafe, finish = [], [], [], []
        for k in keys:
            rollout = rollout_fn(k)
            is_u = np.asarray(is_unsafe_fn(rollout.Tp1_graph))
            is_f = np.asarray(is_finish_fn(rollout.Tp1_graph))
            rewards.append(float(np.asarray(rollout.T_reward).sum()))
            costs.append(float(np.asarray(rollout.T_cost).sum()))
            unsafe.append(is_u.max(axis=0))
            finish.append(is_f.max(axis=0))
        unsafe_a = np.stack(unsafe)
        finish_a = np.stack(finish)
        agent_safe = 1.0 - unsafe_a
        agent_success = agent_safe * finish_a
        return {
            "reward": float(np.mean(rewards)),
            "cost": float(np.mean(costs)),
            "safe_rate": float(agent_safe.mean()),
            "finish_rate": float(finish_a.mean()),
            "success_rate": float(agent_success.mean()),
        }

    history = []
    best = {"success_rate": -1.0, "step": -1}
    t0 = time.time()
    print(
        f"[informarl] out={out_dir} steps={args.steps} cost_weight={args.cost_weight}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        key, bk = jr.split(key)
        env_keys = jr.split(bk, n_envs)
        traj = collect_batch(actor_state.params, critic_state.params, env_keys)

        # GAE over time for each env (team reward/value)
        def gae_one(rew, val, done, boot):
            return _gae(rew, val, done, boot, gamma=args.gamma, gae_lambda=args.gae_lambda)

        adv, ret = jax.vmap(gae_one)(
            traj["reward"], traj["value"], traj["done"], traj["bootstrap_v"]
        )
        batch = {**traj, "adv": adv, "ret": ret}

        epi_team_cost = traj["cost"].sum(axis=1).mean()
        epi_env_reward = traj["env_reward"].sum(axis=1).mean()

        key, uk = jr.split(key)
        actor_state, critic_state, key, metrics = ppo_update(
            actor_state, critic_state, batch, uk
        )

        if step % 10 == 0 or step == 1:
            print(
                f"  step {step:4d} | team_cost={float(epi_team_cost):.3f} "
                f"env_R={float(epi_env_reward):.2f} "
                f"pg={float(metrics['pg']):.4f} ent={float(metrics['ent']):.3f}",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            ev = eval_policy(actor_state.params, seed=1234, episodes=args.eval_epi)
            ev["step"] = step
            ev["train_team_cost"] = float(epi_team_cost)
            history.append(ev)
            print(
                f"  [eval@{step}] suc={ev['success_rate']*100:.1f}% "
                f"safe={ev['safe_rate']*100:.1f}% fin={ev['finish_rate']*100:.1f}% "
                f"cost={ev['cost']:.3f}",
                flush=True,
            )
            ckpt = {
                "architecture": "informarl",
                "policy_kind": "informarl",
                "gnn_layers": args.gnn_layers,
                "n_agents": n_agents,
                "action_dim": env.action_dim,
                "actor_params": actor_state.params,
                "critic_params": critic_state.params,
                "cost_weight": args.cost_weight,
                "step": step,
                "seed": args.seed,
                "action_mode": "uref_plus_tanh_residual",
                "eval": ev,
                "claim": "Empirical InforMARL (GNN+PPO, soft cost). Not formal verification.",
            }
            with open(out_dir / "latest.pkl", "wb") as f:
                pickle.dump(ckpt, f)
            if ev["success_rate"] > best["success_rate"]:
                best = dict(ev)
                best["step"] = step
                with open(out_dir / "best.pkl", "wb") as f:
                    pickle.dump(ckpt, f)
                print(f"  [best] suc={best['success_rate']*100:.1f}% @ step {step}", flush=True)

    wall = time.time() - t0
    summary = {
        "out_dir": str(out_dir),
        "steps": args.steps,
        "wall_time_s": round(wall, 1),
        "best": best,
        "history": history,
        "cost_weight": args.cost_weight,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    mirror = pathlib.Path("runs/baseline/informarl")
    mirror.mkdir(parents=True, exist_ok=True)
    with open(mirror / "ACTIVE.txt", "w") as f:
        f.write(str((out_dir / "best.pkl").resolve()) + "\n")
    print(f"[informarl] done wall={wall:.0f}s best={best}", flush=True)


if __name__ == "__main__":
    main()
