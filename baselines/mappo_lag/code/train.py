#!/usr/bin/env python3
"""Train MAPPO-Lagrangian on DoubleIntegrator (same protocol as baselines).

Homogeneous parameter sharing · 44-D MF features · PPO-clip + dual λ on env cost.
Action: a = clip(u_ref + tanh(μ(x))) so unconstrained optimum recovers LQR ref.

Usage:
  python baselines/mappo_lag/code/train.py [--steps 300] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import time
from typing import Any, Dict, Tuple

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
from stage_mf.student import INPUT_DIM, extract_meanfield_features

from baseline.mappo_lag.nets import (
    GaussianActor,
    ValueNet,
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
    """rewards/values/dones: [T, N]; last_value: [N]. Returns adv, returns."""

    def scan_fn(last_gae, inputs):
        r, v, d, v_next = inputs
        delta = r + gamma * v_next * (1.0 - d) - v
        gae = delta + gamma * gae_lambda * (1.0 - d) * last_gae
        return gae, gae

    v_next = jnp.concatenate([values[1:], last_value[None, :]], axis=0)
    _, adv_rev = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(last_value),
        (rewards[::-1], values[::-1], dones[::-1], v_next[::-1]),
    )
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns


def make_train_state(module, key: jax.Array, lr: float, x0: jnp.ndarray):
    params = module.init(key, x0)
    tx = optax.adam(lr, eps=1e-5)
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400, help="PPO update iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-lambda", type=float, default=5e-3)
    ap.add_argument("--cost-limit", type=float, default=0.5, help="Threshold on mean epi team-cost")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=8)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-epi", type=int, default=16)
    ap.add_argument("--lambda-init", type=float, default=1.0)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(
        args.out_dir
        or f"baselines/mappo_lag/output/mappo_lag/seed{args.seed}_{stamp}"
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
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()
    T = int(env.max_episode_steps)  # 256
    n_envs = args.n_envs

    actor = GaussianActor(hidden=args.hidden, action_dim=env.action_dim)
    critic = ValueNet(hidden=args.hidden)
    cost_critic = ValueNet(hidden=args.hidden)

    key = jr.PRNGKey(args.seed)
    key, k0, k1, k2 = jr.split(key, 4)
    x0 = jnp.zeros((n_agents, INPUT_DIM), dtype=jnp.float32)
    # init with single-agent feature row for Dense
    actor_state = make_train_state(actor, k0, args.lr, x0[0])
    critic_state = make_train_state(critic, k1, args.lr, x0[0])
    cost_critic_state = make_train_state(cost_critic, k2, args.lr, x0[0])

    # dual variable (numpy scalar, updated outside JIT)
    lag_lambda = float(args.lambda_init)

    def feats(graph):
        return extract_meanfield_features(graph, n_agents, n_rays, comm)

    def act_det(params, graph):
        x = feats(graph)
        mean, _ = actor.apply(params, x)
        uref = env.u_ref(graph)
        a = jnp.clip(uref + jnp.tanh(mean), lo, hi)
        return a

    # ---------- vectorized rollout of one full episode per env ----------
    def collect_batch(actor_params, critic_params, cost_params, keys):
        """keys: [n_envs]. Returns trajectory dict."""

        def one_env(key):
            key, rk = jr.split(key)
            graph0 = env.reset(rk)

            def step_fn(carry, _):
                graph, key = carry
                key, sk = jr.split(key)
                x = feats(graph)
                mean, log_std = actor.apply(actor_params, x)
                uref = env.u_ref(graph)
                # sample residual in pre-tanh space then map
                raw, logp_raw = sample_action(mean, log_std, sk)
                # reparameterize through tanh residual: a = clip(uref + tanh(raw))
                # log_prob under raw Gaussian (ignore tanh Jacobian — common approx for clipped residual)
                delta = jnp.tanh(raw)
                action = jnp.clip(uref + delta, lo, hi)
                v = critic.apply(critic_params, x)
                vc = cost_critic.apply(cost_params, x)
                graph2, reward, cost, done, _ = env.step(graph, action)
                # per-agent cost from collision mask (aligns with safe metric)
                c_agent = env.collision_mask(graph).astype(jnp.float32)
                # broadcast team reward
                r_agent = jnp.full((n_agents,), reward, dtype=jnp.float32)
                done_f = jnp.full((n_agents,), done.astype(jnp.float32))
                return (graph2, key), {
                    "obs": x,
                    "action_raw": raw,
                    "action": action,
                    "logp": logp_raw,
                    "reward": r_agent,
                    "cost": c_agent,
                    "team_cost": jnp.full((n_agents,), cost, dtype=jnp.float32),
                    "value": v,
                    "cost_value": vc,
                    "done": done_f,
                    "uref": uref,
                }

            (graph_f, key_f), traj = jax.lax.scan(step_fn, (graph0, key), None, length=T)
            # bootstrap values
            x_f = feats(graph_f)
            v_f = critic.apply(critic_params, x_f)
            vc_f = cost_critic.apply(cost_params, x_f)
            traj["bootstrap_v"] = v_f
            traj["bootstrap_vc"] = vc_f
            return traj

        return jax.vmap(one_env)(keys)

    collect_batch = jax.jit(collect_batch)

    def ppo_update(
        actor_state,
        critic_state,
        cost_critic_state,
        batch,
        lag_lambda: float,
        key,
    ):
        """One PPO epoch over flattened [n_envs*T*n_agents] transitions — multi minibatch."""
        # obs: [E,T,A,D] → [E*T*A, D]
        obs = batch["obs"].reshape(-1, INPUT_DIM)
        action_raw = batch["action_raw"].reshape(-1, env.action_dim)
        old_logp = batch["logp"].reshape(-1)
        adv_r = batch["adv_r"].reshape(-1)
        adv_c = batch["adv_c"].reshape(-1)
        ret_r = batch["ret_r"].reshape(-1)
        ret_c = batch["ret_c"].reshape(-1)

        # normalize reward advantage
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        # combined advantage (Lagrangian)
        adv = adv_r - lag_lambda * adv_c

        n = obs.shape[0]
        mb = max(n // args.num_minibatches, 1)

        def one_epoch(carry, _):
            actor_state, critic_state, cost_critic_state, key = carry
            key, pk = jr.split(key)
            perm = jr.permutation(pk, n)

            def mb_step(carry, start):
                actor_state, critic_state, cost_critic_state = carry
                idx = jax.lax.dynamic_slice(perm, (start,), (mb,))
                o = obs[idx]
                ar = action_raw[idx]
                olp = old_logp[idx]
                ad = adv[idx]
                rr = ret_r[idx]
                rc = ret_c[idx]

                def actor_loss_fn(params):
                    mean, log_std = actor.apply(params, o)
                    logp = log_prob_action(mean, log_std, ar)
                    ratio = jnp.exp(logp - olp)
                    surr1 = ratio * ad
                    surr2 = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * ad
                    pg = -jnp.minimum(surr1, surr2).mean()
                    ent = entropy(log_std).mean()
                    return pg - args.ent_coef * ent, (pg, ent)

                (aloss, (pg, ent)), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                    actor_state.params
                )
                grads = jax.tree.map(
                    lambda g: jnp.clip(g, -args.max_grad_norm, args.max_grad_norm)
                    if False
                    else g,
                    grads,
                )
                # global norm clip
                g_norm = optax.global_norm(grads)
                grads = jax.tree.map(
                    lambda g: g * jnp.minimum(1.0, args.max_grad_norm / (g_norm + 1e-8)),
                    grads,
                )
                actor_state = actor_state.apply_gradients(grads=grads)

                def critic_loss_fn(params):
                    v = critic.apply(params, o)
                    return 0.5 * ((v - rr) ** 2).mean()

                closs, cgrads = jax.value_and_grad(critic_loss_fn)(critic_state.params)
                c_norm = optax.global_norm(cgrads)
                cgrads = jax.tree.map(
                    lambda g: g * jnp.minimum(1.0, args.max_grad_norm / (c_norm + 1e-8)),
                    cgrads,
                )
                critic_state = critic_state.apply_gradients(grads=cgrads)

                def cost_critic_loss_fn(params):
                    v = cost_critic.apply(params, o)
                    return 0.5 * ((v - rc) ** 2).mean()

                ccloss, ccgrads = jax.value_and_grad(cost_critic_loss_fn)(
                    cost_critic_state.params
                )
                cc_norm = optax.global_norm(ccgrads)
                ccgrads = jax.tree.map(
                    lambda g: g * jnp.minimum(1.0, args.max_grad_norm / (cc_norm + 1e-8)),
                    ccgrads,
                )
                cost_critic_state = cost_critic_state.apply_gradients(grads=ccgrads)

                return (actor_state, critic_state, cost_critic_state), {
                    "actor_loss": aloss,
                    "pg": pg,
                    "ent": ent,
                    "critic_loss": closs,
                    "cost_critic_loss": ccloss,
                }

            starts = jnp.arange(0, args.num_minibatches * mb, mb)
            (actor_state, critic_state, cost_critic_state), info = jax.lax.scan(
                mb_step, (actor_state, critic_state, cost_critic_state), starts
            )
            return (actor_state, critic_state, cost_critic_state, key), info

        (actor_state, critic_state, cost_critic_state, key), infos = jax.lax.scan(
            one_epoch,
            (actor_state, critic_state, cost_critic_state, key),
            None,
            length=args.ppo_epochs,
        )
        metrics = {k: v.mean() for k, v in infos.items()}
        return actor_state, critic_state, cost_critic_state, key, metrics

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
            # convert
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
    print(f"[mappo_lag] out={out_dir} steps={args.steps} cost_limit={args.cost_limit}", flush=True)

    for step in range(1, args.steps + 1):
        key, bk = jr.split(key)
        env_keys = jr.split(bk, n_envs)
        traj = collect_batch(
            actor_state.params, critic_state.params, cost_critic_state.params, env_keys
        )

        # GAE per env, per agent — reshape [E,T,A]
        def gae_one(rew, val, done, boot):
            # rew/val/done: [T,A]
            adv, ret = _gae(rew, val, done, boot, gamma=args.gamma, gae_lambda=args.gae_lambda)
            return adv, ret

        adv_r, ret_r = jax.vmap(gae_one)(
            traj["reward"], traj["value"], traj["done"], traj["bootstrap_v"]
        )
        adv_c, ret_c = jax.vmap(gae_one)(
            traj["cost"], traj["cost_value"], traj["done"], traj["bootstrap_vc"]
        )
        batch = {
            **traj,
            "adv_r": adv_r,
            "adv_c": adv_c,
            "ret_r": ret_r,
            "ret_c": ret_c,
        }

        # mean epi team cost for dual (sum over T of team cost; team_cost is broadcast)
        # team_cost[e,t,a] identical across a → take [:,:,0], sum T, mean E
        epi_team_cost = traj["team_cost"][:, :, 0].sum(axis=1).mean()
        epi_agent_cost = traj["cost"].sum(axis=1).mean()  # mean over E,A of sum_T

        key, uk = jr.split(key)
        actor_state, critic_state, cost_critic_state, key, metrics = ppo_update(
            actor_state,
            critic_state,
            cost_critic_state,
            batch,
            float(lag_lambda),
            uk,
        )

        # dual ascent on constraint: J_C - c ≤ 0  (use team epi cost to match eval)
        cost_violation = float(epi_team_cost) - args.cost_limit
        lag_lambda = max(0.0, lag_lambda + args.lr_lambda * cost_violation)

        if step % 10 == 0 or step == 1:
            print(
                f"  step {step:4d} | team_cost={float(epi_team_cost):.3f} "
                f"agent_cost={float(epi_agent_cost):.3f} λ={lag_lambda:.3f} "
                f"pg={float(metrics['pg']):.4f} ent={float(metrics['ent']):.3f}",
                flush=True,
            )

        if step % args.eval_every == 0 or step == args.steps:
            ev = eval_policy(actor_state.params, seed=1234, episodes=args.eval_epi)
            ev["step"] = step
            ev["lambda"] = lag_lambda
            ev["train_team_cost"] = float(epi_team_cost)
            history.append(ev)
            print(
                f"  [eval@{step}] suc={ev['success_rate']*100:.1f}% "
                f"safe={ev['safe_rate']*100:.1f}% fin={ev['finish_rate']*100:.1f}% "
                f"cost={ev['cost']:.3f} λ={lag_lambda:.3f}",
                flush=True,
            )
            # save latest
            ckpt = {
                "architecture": "mappo_lag",
                "policy_kind": "mappo_lagrangian",
                "feature_version": 5,
                "input_dim": INPUT_DIM,
                "hidden": args.hidden,
                "actor_params": actor_state.params,
                "critic_params": critic_state.params,
                "cost_critic_params": cost_critic_state.params,
                "lambda": lag_lambda,
                "cost_limit": args.cost_limit,
                "step": step,
                "seed": args.seed,
                "action_mode": "uref_plus_tanh_residual",
                "eval": ev,
                "claim": "Empirical CMDP MAPPO-Lag. Not formal verification.",
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
        "final_lambda": lag_lambda,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # mirror best into baselines/mappo_lag/output/mappo_lag/
    mirror = pathlib.Path("runs/baseline/mappo_lag")
    mirror.mkdir(parents=True, exist_ok=True)
    with open(mirror / "ACTIVE.txt", "w") as f:
        f.write(str((out_dir / "best.pkl").resolve()) + "\n")
    print(f"[mappo_lag] done wall={wall:.0f}s best={best}", flush=True)


if __name__ == "__main__":
    main()
