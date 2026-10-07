#!/usr/bin/env python3
"""Mean-field MLP distillation from frozen GCBF+ Actor (parallel to stage1/train.py).

Writes only under method/meanfield/output/ — never touches DeepSet paths or ACTIVE_* pointers.
Checkpoint architecture tag: ``meanfield_mlp``.
"""

from __future__ import annotations

import argparse
import csv
import functools as ft
import json
import pathlib
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax

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
from gcbfplus.utils.utils import jax_jit_np, jax_vmap
from stage_mf.student import (
    FEATURE_LAYOUT,
    FEATURE_VERSION,
    INPUT_DIM,
    MeanFieldStudent,
    danger_weights_mf,
    extract_meanfield_features,
    extract_rollout_meanfield,
)

CSV_FIELDS = (
    "step",
    "wall_time_s",
    "loss",
    "mae",
    "buffer_size",
    "reward",
    "cost",
    "safe_rate",
    "finish_rate",
    "success_rate",
    "is_best",
)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})


def make_teacher(args, env):
    algo = make_algo(
        algo="gcbf+",
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=args.gnn_layers,
        batch_size=256,
        buffer_size=args.buffer_size,
        horizon=args.horizon,
        lr_actor=args.lr_actor,
        lr_cbf=args.lr_cbf,
        alpha=args.alpha,
        eps=0.02,
        inner_epoch=8,
        loss_action_coef=1e-4,
        loss_unsafe_coef=1.0,
        loss_safe_coef=1.0,
        loss_h_dot_coef=0.01,
        max_grad_norm=2.0,
        seed=args.seed,
    )
    algo.load(str(pathlib.Path(args.teacher_run) / "models"), args.teacher_step)
    return algo


def collect_teacher_batch(args, env, teacher, keys):
    teacher_params = teacher.actor_params
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm_radius = float(env.params["comm_radius"])

    def policy(graph):
        return teacher.act(graph, teacher_params)

    rollout_fn = jax.jit(env.rollout_fn(policy, args.rollout_length))
    rollout_fn = jax.jit(jax.vmap(rollout_fn))
    rollouts = rollout_fn(keys)
    graphs = jtu.tree_map(lambda x: x[:, :-1], rollouts.Tp1_graph)

    feats = jax.vmap(
        lambda gs: extract_rollout_meanfield(gs, n_agents, n_rays, comm_radius)
    )(graphs)
    # feats: [B, T, A, 22]
    feats = feats.reshape(-1, feats.shape[-1])
    targets = jnp.clip(rollouts.T_action, *env.action_lim())
    targets = targets.reshape(-1, targets.shape[-1])
    return feats, targets


@ft.partial(jax.jit, static_argnums=(0, 1))
def update_student(
    student,
    tx,
    opt_state,
    params,
    feats,
    targets,
    danger_radius: float,
    danger_coef: float,
):
    def loss_fn(p):
        pred = student.apply(p, feats)
        pred = jnp.clip(pred, -1.0, 1.0)
        per = jnp.mean(jnp.square(pred - targets), axis=-1)
        weights = danger_weights_mf(feats, danger_radius, danger_coef)
        return jnp.mean(weights * per)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def save_checkpoint(
    path: pathlib.Path,
    params,
    args,
    input_dim: int,
    n_agents: int,
    n_rays: int,
    metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "mf1",
        "architecture": "meanfield_mlp",
        "track": "meanfield",
        "feature_version": int(FEATURE_VERSION),
        "student_params": params,
        "hidden": int(args.hidden),
        "input_dim": int(input_dim),
        "feature_layout": dict(FEATURE_LAYOUT),
        "n_max_agents": int(n_agents),
        "n_max_obs": int(n_rays),
        "teacher_run": str(args.teacher_run),
        "teacher_step": int(args.teacher_step),
        "environment": vars(args),
        "metrics": metrics or {},
        "note": (
            "Mean-field v5: 4-quadrant max + distance-weighted mean closing "
            "+ inv_ttc_max (time urgency of nearest-in-quad). "
            "No global nearest. Keep 4 quadrants (not 8). "
            "Formal verification must use this exact MLP and layout. "
            "Not a DeepSet checkpoint. Outputs under method/meanfield/output/."
        ),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def evaluate_student(env, student, params, n_agents, n_rays, episodes, seed) -> dict:
    comm_radius = float(env.params["comm_radius"])

    def policy(graph):
        x = extract_meanfield_features(graph, n_agents, n_rays, comm_radius)
        return jnp.clip(student.apply(params, x), *env.action_lim())

    rollout_fn = jax_jit_np(env.rollout_fn(policy, env.max_episode_steps))
    is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
    is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    keys = jr.split(jr.PRNGKey(seed), episodes)

    rewards, costs, unsafe, finish = [], [], [], []
    for key in keys:
        rollout = rollout_fn(key)
        is_unsafe = is_unsafe_fn(rollout.Tp1_graph)
        is_finish = is_finish_fn(rollout.Tp1_graph)
        rewards.append(float(rollout.T_reward.sum()))
        costs.append(float(rollout.T_cost.sum()))
        unsafe.append(np.asarray(is_unsafe).max(axis=0))
        finish.append(np.asarray(is_finish).max(axis=0))

    unsafe = np.stack(unsafe)
    finish = np.stack(finish)
    safe_rate = float((1 - unsafe).mean())
    finish_rate = float(finish.mean())
    success_rate = float(((1 - unsafe) * finish).mean())
    return {
        "reward": float(np.mean(rewards)),
        "cost": float(np.mean(costs)),
        "safe_rate": safe_rate,
        "finish_rate": finish_rate,
        "success_rate": success_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GCBF+ Actor → Mean-field MLP distillation (parallel track)"
    )
    parser.add_argument("--teacher-run", required=True)
    parser.add_argument("--teacher-step", type=int, default=None)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--buffer-capacity", type=int, default=131072)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--refresh-every", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--obs", type=int, default=4)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--area-size", type=float, default=4.0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        default="method/meanfield/output/distill/mf_default/seed0/meanfield_mlp256_student.pkl",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--danger-radius", type=float, default=0.3)
    parser.add_argument("--danger-coef", type=float, default=4.0)
    parser.add_argument("--init-from", type=str, default="")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-train-every", type=int, default=50)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-cbf", type=float, default=3e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    model_dir = pathlib.Path(args.teacher_run) / "models"
    if args.teacher_step is None:
        steps = [int(p.name) for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if not steps:
            raise FileNotFoundError(f"No official checkpoints in {model_dir}")
        args.teacher_step = max(steps)

    env = make_env(
        "DoubleIntegrator",
        num_agents=args.num_agents,
        area_size=args.area_size,
        num_obs=args.obs,
        n_rays=args.n_rays,
    )
    teacher = make_teacher(args, env)

    key = jr.PRNGKey(args.seed)
    key, init_key, sample_key = jr.split(key, 3)
    sample_keys = jr.split(sample_key, args.batch_size)
    feats_new, targets_new = collect_teacher_batch(args, env, teacher, sample_keys)
    if int(feats_new.shape[-1]) != INPUT_DIM:
        raise RuntimeError(f"feature dim {feats_new.shape[-1]} != INPUT_DIM {INPUT_DIM}")

    student = MeanFieldStudent(hidden=args.hidden, action_dim=env.action_dim)
    params = student.init(init_key, feats_new[:1])
    if args.init_from:
        with pathlib.Path(args.init_from).open("rb") as f:
            warm = pickle.load(f)
        if warm.get("architecture") != "meanfield_mlp":
            raise ValueError(f"--init-from must be meanfield_mlp, got {warm.get('architecture')}")
        if int(warm.get("hidden", -1)) != args.hidden:
            raise ValueError("hidden mismatch for warm-start")
        params = warm["student_params"]
        print(f"[mf1] warm-start from {args.init_from}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=max(1, args.steps // 20),
        decay_steps=args.steps,
        end_value=args.lr * 0.1,
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    optimizer = tx.init(params)

    buf_x = feats_new
    buf_y = targets_new
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    out = pathlib.Path(args.checkpoint)
    best_path = out.with_name(out.stem + "_best.pkl")
    best_success = -1.0
    best_mae = float("inf")
    best_step = -1
    best_tuple = (-1.0, float("inf"))
    history: list[dict] = []
    train_log: list[dict] = []
    t0 = time.time()
    last_mae = float("nan")
    log_every = max(1, args.steps // 20)

    print(
        f"[mf1] arch=meanfield_mlp feature_v={FEATURE_VERSION} "
        f"(4quad wmean+invttc) track=meanfield teacher={args.teacher_run} "
        f"step={args.teacher_step} input_dim={INPUT_DIM} hidden={args.hidden} "
        f"samples={feats_new.shape[0]} n_agents={n_agents} n_rays={n_rays} "
        f"steps={args.steps} batch={args.batch_size} minibatch={args.minibatch_size} "
        f"danger_r={args.danger_radius} danger_c={args.danger_coef} "
        f"eval_every={args.eval_every} eval_seed={args.eval_seed} lr={args.lr} seed={args.seed}"
    )

    for step in range(1, args.steps + 1):
        if step == 1 or step % args.refresh_every == 0:
            key, batch_key = jr.split(key)
            sample_keys = jr.split(batch_key, args.batch_size)
            feats_new, targets_new = collect_teacher_batch(args, env, teacher, sample_keys)
            buf_x = jnp.concatenate([buf_x, feats_new], axis=0)
            buf_y = jnp.concatenate([buf_y, targets_new], axis=0)
            if buf_x.shape[0] > args.buffer_capacity:
                buf_x = buf_x[-args.buffer_capacity :]
                buf_y = buf_y[-args.buffer_capacity :]

        key, mb_key = jr.split(key)
        n = buf_x.shape[0]
        mb = min(args.minibatch_size, n)
        idx = jr.choice(mb_key, n, shape=(mb,), replace=False)
        x = buf_x[idx]
        y = buf_y[idx]

        params, optimizer, loss = update_student(
            student,
            tx,
            optimizer,
            params,
            x,
            y,
            args.danger_radius,
            args.danger_coef,
        )
        loss_f = float(loss)
        wall_t = time.time() - t0

        do_train_log = (
            step == 1
            or step % log_every == 0
            or step == args.steps
            or (args.log_train_every > 0 and step % args.log_train_every == 0)
        )
        if do_train_log:
            pred = jnp.clip(student.apply(params, x), -1.0, 1.0)
            last_mae = float(jnp.mean(jnp.abs(pred - y)))
            print(
                f"step={step:04d} distill_loss={loss_f:.6f} mae={last_mae:.6f} "
                f"buffer={buf_x.shape[0]} elapsed={wall_t:.1f}s"
            )
            if last_mae < best_mae:
                best_mae = last_mae
            train_log.append(
                {
                    "step": step,
                    "wall_time_s": round(wall_t, 3),
                    "loss": loss_f,
                    "mae": last_mae,
                    "buffer_size": int(buf_x.shape[0]),
                    "reward": None,
                    "cost": None,
                    "safe_rate": None,
                    "finish_rate": None,
                    "success_rate": None,
                    "is_best": False,
                }
            )

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            metrics = evaluate_student(
                env,
                student,
                params,
                n_agents,
                n_rays,
                episodes=args.eval_episodes,
                seed=int(args.eval_seed),
            )
            print(
                f"[eval] step={step:04d} "
                f"safe={metrics['safe_rate'] * 100:.2f}% "
                f"finish={metrics['finish_rate'] * 100:.2f}% "
                f"success={metrics['success_rate'] * 100:.2f}% "
                f"reward={metrics['reward']:.3f} cost={metrics['cost']:.3f}"
            )
            cand = (float(metrics["success_rate"]), loss_f)
            is_best = cand[0] > best_tuple[0] or (
                cand[0] == best_tuple[0] and cand[1] < best_tuple[1]
            )
            row = {
                "step": step,
                "wall_time_s": round(wall_t, 3),
                "loss": loss_f,
                "mae": last_mae,
                "buffer_size": int(buf_x.shape[0]),
                "is_best": False,
                **metrics,
            }
            if is_best:
                best_tuple = cand
                best_success = metrics["success_rate"]
                best_step = step
                row["is_best"] = True
                for h in history:
                    h["is_best"] = False
                save_checkpoint(
                    best_path,
                    params,
                    args,
                    INPUT_DIM,
                    n_agents,
                    n_rays,
                    metrics={"step": step, "loss": loss_f, "mae": last_mae, **metrics},
                )
                print(
                    f"[mf1] best checkpoint={best_path} "
                    f"success={best_success * 100:.2f}% loss={loss_f:.6f}"
                )
            history.append(row)

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt = out.with_name(f"{out.stem}_step{step}.pkl")
            save_checkpoint(
                ckpt,
                params,
                args,
                INPUT_DIM,
                n_agents,
                n_rays,
                metrics={"step": step, "loss": loss_f, "mae": last_mae},
            )

    save_checkpoint(
        out,
        params,
        args,
        INPUT_DIM,
        n_agents,
        n_rays,
        metrics={
            "step": args.steps,
            "best_success": best_success,
            "best_mae": best_mae,
            "best_step": best_step,
        },
    )

    csv_by_step: dict[int, dict] = {}
    for r in train_log:
        csv_by_step[int(r["step"])] = dict(r)
    for r in history:
        prev = csv_by_step.get(int(r["step"]), {})
        merged = dict(prev)
        merged.update(r)
        csv_by_step[int(r["step"])] = merged
    csv_rows = [csv_by_step[s] for s in sorted(csv_by_step)]
    for r in csv_rows:
        r["is_best"] = bool(r.get("is_best")) and int(r.get("step", -1)) == best_step

    csv_path = out.with_suffix(".csv")
    _write_csv(csv_path, csv_rows)

    summary = {
        "mode": "mf1_actor_distill",
        "architecture": "meanfield_mlp",
        "track": "meanfield",
        "feature_version": FEATURE_VERSION,
        "feature_spec": "4quad_agent(n,min_d,cmax,cwmean,invttc)+obs(n,min_d,cmax,invttc)",
        "teacher_run": str(args.teacher_run),
        "teacher_step": args.teacher_step,
        "hidden": args.hidden,
        "input_dim": INPUT_DIM,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "minibatch_size": args.minibatch_size,
        "buffer_capacity": args.buffer_capacity,
        "refresh_every": args.refresh_every,
        "danger_radius": args.danger_radius,
        "danger_coef": args.danger_coef,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "eval_history": history,
        "best_success_short_eval": best_success,
        "best_mae_seen": best_mae,
        "best_step": best_step,
        "wall_time_total_s": round(time.time() - t0, 3),
        "checkpoint": str(out),
        "best_checkpoint": str(best_path) if best_success >= 0.0 else None,
        "eval_csv": str(csv_path),
        "run_meta": {
            "run_id": args.run_id,
            "out_root": "method/meanfield/output/distill",
            "deepset_untouched": True,
        },
        "note": "Parallel mean-field track. Does not update method/deepset/output/phase2/ACTIVE_*.",
    }
    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[mf1] done wall={summary['wall_time_total_s']}s "
        f"best_success_short={best_success * 100:.2f}% @ step {best_step} "
        f"ckpt={out} best={best_path} csv={csv_path}"
    )


if __name__ == "__main__":
    main()
