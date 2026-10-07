"""Stage3 residual-gate repair: freeze distilled DeepSet, train danger correction.

    a = clip( base(x) + g(x) * delta(x) )

- base: frozen Stage1 distilled DeepSet (default) or any DeepSet checkpoint
- g: danger gate (soft / hard / hybrid)
- delta: small trainable DeepSet correction

Far from danger, g≈0 so the policy equals the frozen base. Only gate and
correction parameters are optimized. This is empirical repair, not a proof.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
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

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax



from gcbfplus.env import make_env
from gcbfplus.utils.utils import jax_jit_np, jax_vmap
from stage1.student import extract_local_sets, extract_rollout_sets, EGO_DIM, ENTITY_DIM
from stage2.bounds import default_l2_buckets
from stage2.encode import ENTITY_CONT_DIM
from stage3.retrain_l2_safe import (
    bucket_x_to_student_inputs,
    resample_hit_rate_on_D,
    _selection_score,
)
from stage3.metrics_io import attach_run_meta, write_eval_csv, write_summary_json
from stage3.residual_policy import (
    apply_residual,
    init_train_params,
    load_residual_checkpoint,
    make_modules,
    save_residual_checkpoint,
)

N_MAX_AGENTS = 8
N_MAX_OBS = 32
ENTITY_SLOTS = N_MAX_AGENTS + N_MAX_OBS


def load_ce_dataset_weighted(path: pathlib.Path, *, require_safe_label: bool = True):
    """Load CEs and per-sample threat tags for AA/AO reweighting.

    Returns ego, entities, targets, threat_code in {0=aa, 1=ao, 2=dual/other}.
    """
    raw = json.loads(path.read_text())
    egos, ents, acts, threats = [], [], [], []
    n_skip = 0
    deltas = []
    modes = set()
    counts = {"aa": 0, "ao": 0, "dual": 0, "other": 0}
    for bucket in raw["buckets"]:
        k, m = int(bucket["k"]), int(bucket["m"])
        prop = str(bucket.get("property", ""))
        for ce in bucket["counterexamples"]:
            if require_safe_label and not ce.get("a_l2_label_ok", False):
                n_skip += 1
                continue
            label = ce.get("a_l2_safe") or ce.get("a")
            ego, ent = bucket_x_to_student_inputs(np.asarray(ce["x"]), k, m)
            kind = str(ce.get("kind", "")).lower()
            if prop == "l2_aa" or kind == "aa" or (k >= 1 and m == 0):
                code, tag = 0, "aa"
            elif prop == "l2_ao" or kind == "ao" or (k == 0 and m >= 1):
                code, tag = 1, "ao"
            elif prop == "l2_aa_ao" or "aa_ao" in kind or (k >= 1 and m >= 1):
                code, tag = 2, "dual"
            else:
                code, tag = 3, "other"
            counts[tag if tag in counts else "other"] = counts.get(tag, 0) + 1
            egos.append(ego)
            ents.append(ent)
            acts.append(np.asarray(label, dtype=np.float32))
            threats.append(code)
            if "a_l2_delta" in ce:
                deltas.append(float(ce["a_l2_delta"]))
            if "a_l2_label_mode" in ce:
                modes.add(str(ce["a_l2_label_mode"]))
    if not egos:
        raise ValueError(f"no usable CEs in {path} (skipped_unsafe_labels={n_skip})")
    mean_da = float(np.mean(deltas)) if deltas else None
    print(
        f"[residual] loaded {len(egos)} CEs (skipped {n_skip}); "
        f"modes={sorted(modes) or ['legacy']}; mean_Δa={mean_da}; counts={counts}"
    )
    return (
        np.stack(egos),
        np.stack(ents),
        np.stack(acts),
        np.asarray(threats, dtype=np.int32),
    )


def collect_base_batch(args, env, modules, base_params, keys):
    """On-policy states + base DeepSet actions (behavior anchor when g=0)."""
    import jax.tree_util as jtu

    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm_radius = float(env.params["comm_radius"])

    def base_policy(graph):
        ego, entities = extract_local_sets(graph, n_agents, n_rays, comm_radius)
        a0 = modules.base.apply(base_params, ego, entities)
        return jnp.clip(a0, *env.action_lim())

    rollout_fn = jax.jit(env.rollout_fn(base_policy, args.rollout_length))
    rollout_fn = jax.jit(jax.vmap(rollout_fn))
    rollouts = rollout_fn(keys)
    graphs = jtu.tree_map(lambda x: x[:, :-1], rollouts.Tp1_graph)
    ego, entities = jax.vmap(
        lambda gs: extract_rollout_sets(gs, n_agents, n_rays, comm_radius)
    )(graphs)
    ego = ego.reshape(-1, ego.shape[-1])
    entities = entities.reshape(-1, entities.shape[-2], entities.shape[-1])
    targets = jnp.clip(rollouts.T_action, *env.action_lim()).reshape(-1, env.action_dim)
    return ego, entities, targets


def evaluate_residual(
    env,
    modules,
    base_params,
    train_params,
    *,
    n_agents: int,
    n_rays: int,
    episodes: int,
    seed: int,
    gate_mode: str,
    open_mult: float,
    close_mult: float,
) -> dict:
    """Official-style short empirical rollout for residual policy."""
    comm_radius = float(env.params["comm_radius"])
    lo, hi = env.action_lim()

    def policy(graph):
        ego, entities = extract_local_sets(graph, n_agents, n_rays, comm_radius)
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
    return {
        "reward": float(np.mean(rewards)),
        "cost": float(np.mean(costs)),
        "safe_rate": float((1 - unsafe).mean()),
        "finish_rate": float(finish.mean()),
        "success_rate": float(((1 - unsafe) * finish).mean()),
    }


def residual_action_at_x(modules, base_params, train_params, x, spec, *, gate_mode, open_mult, close_mult):
    """Numpy-friendly action for D-hit resampling (matches deployment clip)."""
    ego_np, ent_np = bucket_x_to_student_inputs(np.asarray(x), spec.k, spec.m)
    ego = jnp.asarray(ego_np[None, ...])
    ent = jnp.asarray(ent_np[None, ...])
    a, g, _ = apply_residual(
        modules,
        base_params,
        train_params,
        ego,
        ent,
        gate_mode=gate_mode,
        open_mult=open_mult,
        close_mult=close_mult,
    )
    return np.asarray(a[0], dtype=np.float64), float(np.asarray(g[0]))


def resample_hit_rate_residual(
    modules,
    base_params,
    train_params,
    specs,
    *,
    n_samples: int,
    seed: int,
    gate_mode: str,
    open_mult: float,
    close_mult: float,
) -> dict:
    from stage2.l2_domain import in_preventable_domain_x
    from stage2.l2_query import eval_collision_dual, next_rels_for_spec, rels_from_x

    rng = np.random.default_rng(seed)
    out = {}
    for spec in specs:
        n_eval = n_hit = n_draw = 0
        min_d = 1e9
        gate_sum = 0.0
        budget = max(n_samples * 40, 20000)
        while n_eval < n_samples and n_draw < budget:
            x = spec.box.sample(rng)
            n_draw += 1
            if not in_preventable_domain_x(x, spec, grid_n=7):
                continue
            n_eval += 1
            a, g = residual_action_at_x(
                modules,
                base_params,
                train_params,
                x,
                spec,
                gate_mode=gate_mode,
                open_mult=open_mult,
                close_mult=close_mult,
            )
            gate_sum += g
            ra0, ro0 = rels_from_x(x, spec)
            ran, ron = next_rels_for_spec(ra0, ro0, a, spec)
            ev = eval_collision_dual(ran, ron, spec)
            min_d = min(min_d, ev.dist)
            n_hit += int(ev.collided)
        out[spec.name] = {
            "n_eval_on_D": n_eval,
            "n_draw": n_draw,
            "n_hit": n_hit,
            "hit_rate_on_D": n_hit / max(n_eval, 1),
            "min_distn": float(min_d) if n_eval else None,
            "mean_gate_on_D": gate_sum / max(n_eval, 1),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual-gate L2 CE repair (frozen base)")
    parser.add_argument(
        "--base-checkpoint",
        default="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl",
        help="Frozen distilled DeepSet base (default: Stage1 baseline)",
    )
    parser.add_argument("--ce-json", default="method/deepset/output/stage3/ce/l2_counterexamples.json")
    parser.add_argument("--teacher-run", default="baselines/gcbfplus/log/gcbf+/seed0_20261001232611")
    parser.add_argument("--teacher-step", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--ce-batch", type=int, default=256)
    parser.add_argument("--normal-batch", type=int, default=768)
    parser.add_argument("--ce-weight", type=float, default=3.0)
    parser.add_argument(
        "--ao-ce-weight",
        type=float,
        default=1.0,
        help="extra multiplier on AO CE samples (on top of --ce-weight)",
    )
    parser.add_argument(
        "--dual-ce-weight",
        type=float,
        default=1.0,
        help="extra multiplier on dual (1,1) CE samples",
    )
    parser.add_argument("--idle-weight", type=float, default=1.0, help="weight on far-from-danger idle loss")
    parser.add_argument("--delta-l2", type=float, default=1e-3, help="L2 penalty on correction magnitude")
    parser.add_argument("--gate-sparse", type=float, default=1e-3, help="encourage small gate on normals")
    parser.add_argument(
        "--gate-ce-open",
        type=float,
        default=0.0,
        help="encourage gate open on CE: weight * mean((1-g)^2 on CE)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=512, help="base DeepSet hidden (must match ckpt)")
    parser.add_argument("--gate-hidden", type=int, default=32)
    parser.add_argument("--correction-hidden", type=int, default=64)
    parser.add_argument(
        "--gate-mode",
        default="hybrid",
        choices=["soft", "hard", "hybrid"],
        help="soft=learned only; hard=physics proximity; hybrid=product",
    )
    parser.add_argument("--open-mult", type=float, default=2.5, help="physics gate fully closed beyond this · radius")
    parser.add_argument("--close-mult", type=float, default=1.0, help="physics gate fully open at/below this · radius")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--resample", type=int, default=2000)
    parser.add_argument("--eval-buckets", default="l2_aa_near,l2_ao_near,l2_11_near")
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    parser.add_argument("--min-success", type=float, default=0.70)
    parser.add_argument("--min-safe", type=float, default=0.90)
    parser.add_argument("--max-aa-hit", type=float, default=0.20)
    parser.add_argument("--max-ao-hit", type=float, default=0.35)
    parser.add_argument(
        "--init-residual-from",
        default=None,
        help="optional warm-start of gate/correction from a prior residual checkpoint",
    )
    parser.add_argument("--checkpoint", default="method/deepset/output/stage3/repair/deepset_residual_gate.pkl")
    parser.add_argument(
        "--method-tag",
        default=None,
        help="optional run_meta.method (e.g. V2, V2d) written into summary JSON",
    )
    parser.add_argument(
        "--phase2-run-id",
        default=None,
        help="optional run_meta.phase2_run_id for multi-seed aggregation",
    )
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--obs", type=int, default=4)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--area-size", type=float, default=4.0)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    # unused teacher kwargs kept for env/arg parity with other stage3 scripts
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-cbf", type=float, default=3e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    ce_ego, ce_ent, ce_tgt, ce_threat = load_ce_dataset_weighted(
        pathlib.Path(args.ce_json), require_safe_label=True
    )

    env = make_env(
        "DoubleIntegrator",
        num_agents=args.num_agents,
        area_size=args.area_size,
        num_obs=args.obs,
        n_rays=args.n_rays,
    )
    modules = make_modules(
        base_hidden=args.hidden,
        gate_hidden=args.gate_hidden,
        correction_hidden=args.correction_hidden,
        action_dim=env.action_dim,
    )

    with pathlib.Path(args.base_checkpoint).open("rb") as f:
        base_ckpt = pickle.load(f)
    if base_ckpt.get("architecture", "deepset") not in ("deepset",):
        raise ValueError(f"base must be plain deepset, got {base_ckpt.get('architecture')}")
    if int(base_ckpt.get("hidden", args.hidden)) != args.hidden:
        raise ValueError("base hidden mismatch")
    base_params = base_ckpt["student_params"]
    # Freeze: never updated; only referenced.
    base_params = jax.tree_util.tree_map(lambda x: jax.device_put(x), base_params)

    key = jr.PRNGKey(args.seed)
    key, ik = jr.split(key)
    if args.init_residual_from:
        prev, _prev_modules, _prev_base, prev_train = load_residual_checkpoint(args.init_residual_from)
        if int(prev.get("correction_hidden", -1)) != args.correction_hidden:
            raise ValueError(
                f"warm-start correction_hidden mismatch: "
                f"ckpt={prev.get('correction_hidden')} vs args={args.correction_hidden}"
            )
        if int(prev.get("gate_hidden", -1)) != args.gate_hidden:
            raise ValueError(
                f"warm-start gate_hidden mismatch: "
                f"ckpt={prev.get('gate_hidden')} vs args={args.gate_hidden}"
            )
        train_params = prev_train
        print(f"[residual] warm-start gate/correction from {args.init_residual_from}")
    else:
        train_params = init_train_params(
            modules, ik, ego_dim=EGO_DIM, entity_dim=ENTITY_DIM, entity_slots=ENTITY_SLOTS
        )

    key, nk = jr.split(key)
    sample_keys = jr.split(nk, args.batch_size)
    n_ego, n_ent, n_tgt = collect_base_batch(args, env, modules, base_params, sample_keys)
    print(
        f"[residual] base={args.base_checkpoint} gate_mode={args.gate_mode} "
        f"corr_h={args.correction_hidden} gate_h={args.gate_hidden} "
        f"normals={n_ego.shape[0]} ce={ce_ego.shape[0]} "
        f"ce_weight={args.ce_weight} ao×={args.ao_ce_weight} dual×={args.dual_ce_weight}"
    )

    lo, hi = env.action_lim()
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = tx.init(train_params)

    def loss_fn(tp, ego, ent, tgt, w, is_ce):
        a, g, delta = apply_residual(
            modules,
            base_params,
            tp,
            ego,
            ent,
            gate_mode=args.gate_mode,
            open_mult=args.open_mult,
            close_mult=args.close_mult,
            action_lim=(lo, hi),
        )
        per = jnp.mean(jnp.square(a - tgt), axis=-1)
        # CE: match safe labels (w already includes ao/dual multipliers).
        # Normal: match base (idle).
        w_eff = jnp.where(is_ce > 0.5, w * args.ce_weight, w * args.idle_weight)
        mse = jnp.mean(w_eff * per)
        # Prefer controlled corrections; sparse gates on non-CE; open gates on CE.
        d_pen = args.delta_l2 * jnp.mean(jnp.square(delta))
        g_pen = args.gate_sparse * jnp.mean((1.0 - is_ce) * jnp.square(g))
        g_open = args.gate_ce_open * jnp.mean(is_ce * jnp.square(1.0 - g))
        return mse + d_pen + g_pen + g_open, {
            "mse": mse,
            "d_pen": d_pen,
            "g_pen": g_pen,
            "g_open": g_open,
            "mean_gate": jnp.mean(g),
            "mean_gate_ce": jnp.sum(g * is_ce) / jnp.maximum(jnp.sum(is_ce), 1.0),
            "mean_gate_n": jnp.sum(g * (1.0 - is_ce)) / jnp.maximum(jnp.sum(1.0 - is_ce), 1.0),
            "mean_delta": jnp.mean(jnp.linalg.norm(delta, axis=-1)),
        }

    @jax.jit
    def update(tp, os, ego, ent, tgt, w, is_ce):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(tp, ego, ent, tgt, w, is_ce)
        updates, os = tx.update(grads, os, tp)
        return optax.apply_updates(tp, updates), os, loss, aux

    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    out = pathlib.Path(args.checkpoint)
    best_path = out.with_name(out.stem + "_best.pkl")
    best_success = -1.0
    best_hit = 1e9
    best_tuple = (1e9, 1.0)
    history = []
    best_steps: set[int] = set()
    wall_times: dict[int, float] = {}
    t0 = time.time()

    bucket_names = {b.strip() for b in args.eval_buckets.split(",") if b.strip()}
    specs = [s for s in default_l2_buckets() if s.name in bucket_names]
    if not specs:
        raise ValueError(f"no specs for {bucket_names}")

    before = resample_hit_rate_residual(
        modules,
        base_params,
        train_params,
        specs,
        n_samples=min(400, args.resample),
        seed=123,
        gate_mode=args.gate_mode,
        open_mult=args.open_mult,
        close_mult=args.close_mult,
    )
    print("[residual] hit-rate on D BEFORE (random init corr):", json.dumps(before))

    # Also report pure base D-hit once (reference).
    from stage2.encode import load_weights

    base_w, _ = load_weights(args.base_checkpoint)
    before_base = resample_hit_rate_on_D(base_w, specs, n_samples=args.resample, seed=123)
    print("[residual] hit-rate on D BASE only:", json.dumps(before_base))

    ce_ego_j, ce_ent_j, ce_tgt_j = map(jnp.asarray, (ce_ego, ce_ent, ce_tgt))
    ce_threat_j = jnp.asarray(ce_threat)
    # Per-CE multipliers: AA=1, AO=ao_ce_weight, dual=dual_ce_weight, other=1
    ce_mult = jnp.where(
        ce_threat_j == 1,
        args.ao_ce_weight,
        jnp.where(ce_threat_j == 2, args.dual_ce_weight, 1.0),
    )
    n_ce = ce_ego_j.shape[0]
    n_norm = n_ego.shape[0]

    for step in range(1, args.steps + 1):
        key, k1, k2 = jr.split(key, 3)
        ci = jr.choice(k1, n_ce, shape=(min(args.ce_batch, n_ce),), replace=False)
        ni = jr.choice(k2, n_norm, shape=(min(args.normal_batch, n_norm),), replace=False)
        ego = jnp.concatenate([ce_ego_j[ci], n_ego[ni]], axis=0)
        ent = jnp.concatenate([ce_ent_j[ci], n_ent[ni]], axis=0)
        tgt = jnp.concatenate([ce_tgt_j[ci], n_tgt[ni]], axis=0)
        w = jnp.concatenate([ce_mult[ci], jnp.ones((ni.shape[0],))], axis=0)
        is_ce = jnp.concatenate(
            [jnp.ones((ci.shape[0],)), jnp.zeros((ni.shape[0],))],
            axis=0,
        )
        train_params, opt_state, loss, aux = update(
            train_params, opt_state, ego, ent, tgt, w, is_ce
        )

        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(
                f"step={step:04d} loss={float(loss):.6f} mse={float(aux['mse']):.6f} "
                f"g_ce={float(aux['mean_gate_ce']):.3f} g_n={float(aux['mean_gate_n']):.3f} "
                f"|δ|={float(aux['mean_delta']):.4f} elapsed={time.time()-t0:.1f}s"
            )

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            metrics = evaluate_residual(
                env,
                modules,
                base_params,
                train_params,
                n_agents=n_agents,
                n_rays=n_rays,
                episodes=args.eval_episodes,
                seed=args.eval_seed if args.eval_seed is not None else args.seed + 50_000 + step,
                gate_mode=args.gate_mode,
                open_mult=args.open_mult,
                close_mult=args.close_mult,
            )
            hit = resample_hit_rate_residual(
                modules,
                base_params,
                train_params,
                specs,
                n_samples=min(800, args.resample),
                seed=99,
                gate_mode=args.gate_mode,
                open_mult=args.open_mult,
                close_mult=args.close_mult,
            )
            mean_hit = float(np.mean([hit[s.name]["hit_rate_on_D"] for s in specs]))
            hit_str = " ".join(f"{k}={v['hit_rate_on_D']*100:.2f}%" for k, v in hit.items())
            print(
                f"[eval] step={step:04d} safe={metrics['safe_rate']*100:.2f}% "
                f"finish={metrics['finish_rate']*100:.2f}% success={metrics['success_rate']*100:.2f}% "
                f"mean_D_hit={mean_hit*100:.2f}% D_hits[{hit_str}]"
            )
            score = _selection_score(hit, metrics, specs)
            eligible = (
                metrics["success_rate"] >= args.min_success
                and metrics["safe_rate"] >= args.min_safe
                and hit.get("l2_aa_near", {}).get("hit_rate_on_D", 0.0) <= args.max_aa_hit
                and hit.get("l2_ao_near", {}).get("hit_rate_on_D", 0.0) <= args.max_ao_hit
            )
            wall_t = time.time() - t0
            eval_metrics = {
                "step": step,
                "loss": float(loss),
                "mean_hit_D": mean_hit,
                "hit_D": {k: v["hit_rate_on_D"] for k, v in hit.items()},
                "mean_gate_on_D": {k: v.get("mean_gate_on_D") for k, v in hit.items()},
                "eligible": eligible,
                "wall_time_s": round(wall_t, 3),
                **metrics,
            }
            history.append(eval_metrics)
            wall_times[step] = wall_t
            if args.save_eval_checkpoints:
                save_residual_checkpoint(
                    out.with_name(out.stem + f"_step{step:04d}.pkl"),
                    base_params=base_params,
                    train_params=train_params,
                    base_checkpoint=args.base_checkpoint,
                    args=args,
                    ego_dim=EGO_DIM,
                    entity_dim=ENTITY_DIM,
                    entity_slots=ENTITY_SLOTS,
                    n_agents=n_agents,
                    n_rays=n_rays,
                    metrics=eval_metrics,
                )
            if not eligible:
                print("[residual] rejected by success/safe/AA/AO selection floors")
            if eligible and score < best_tuple:
                best_tuple = score
                best_hit = mean_hit
                best_success = metrics["success_rate"]
                best_steps = {step}
                save_residual_checkpoint(
                    best_path,
                    base_params=base_params,
                    train_params=train_params,
                    base_checkpoint=args.base_checkpoint,
                    args=args,
                    ego_dim=EGO_DIM,
                    entity_dim=ENTITY_DIM,
                    entity_slots=ENTITY_SLOTS,
                    n_agents=n_agents,
                    n_rays=n_rays,
                    metrics=eval_metrics,
                )
                print(
                    f"[residual] best={best_path} mean_D_hit={mean_hit*100:.2f}% "
                    f"success={best_success*100:.2f}%"
                )

        if step % 100 == 0:
            key, rk = jr.split(key)
            n_keys = jr.split(rk, args.batch_size)
            n_ego, n_ent, n_tgt = collect_base_batch(args, env, modules, base_params, n_keys)
            n_norm = n_ego.shape[0]

    save_residual_checkpoint(
        out,
        base_params=base_params,
        train_params=train_params,
        base_checkpoint=args.base_checkpoint,
        args=args,
        ego_dim=EGO_DIM,
        entity_dim=ENTITY_DIM,
        entity_slots=ENTITY_SLOTS,
        n_agents=n_agents,
        n_rays=n_rays,
        metrics={"step": args.steps, "best_success": best_success, "best_mean_hit_D": best_hit},
    )
    after = resample_hit_rate_residual(
        modules,
        base_params,
        train_params,
        specs,
        n_samples=args.resample,
        seed=123,
        gate_mode=args.gate_mode,
        open_mult=args.open_mult,
        close_mult=args.close_mult,
    )
    print("[residual] hit-rate on D AFTER (last):", json.dumps(after))
    if best_success >= 0.0 and best_path.exists():
        _c, _m, bp, tp = load_residual_checkpoint(best_path)
        after_best = resample_hit_rate_residual(
            modules,
            bp,
            tp,
            specs,
            n_samples=args.resample,
            seed=123,
            gate_mode=args.gate_mode,
            open_mult=args.open_mult,
            close_mult=args.close_mult,
        )
        print("[residual] hit-rate on D AFTER (best):", json.dumps(after_best))
    else:
        after_best = after

    ce_meta = json.loads(pathlib.Path(args.ce_json).read_text())
    summary = {
        "mode": "residual_gate_frozen_base",
        "architecture": "deepset_residual_gate",
        "base_checkpoint": args.base_checkpoint,
        "label_mode": ce_meta.get("label_mode"),
        "ce_json": args.ce_json,
        "gate_mode": args.gate_mode,
        "open_mult": args.open_mult,
        "close_mult": args.close_mult,
        "gate_hidden": args.gate_hidden,
        "correction_hidden": args.correction_hidden,
        "ce_weight": args.ce_weight,
        "ao_ce_weight": args.ao_ce_weight,
        "dual_ce_weight": args.dual_ce_weight,
        "idle_weight": args.idle_weight,
        "delta_l2": args.delta_l2,
        "gate_sparse": args.gate_sparse,
        "gate_ce_open": args.gate_ce_open,
        "lr": args.lr,
        "seed": args.seed,
        "init_residual_from": args.init_residual_from,
        "n_ce": int(n_ce),
        "steps": args.steps,
        "selection_limits": {
            k: getattr(args, k) for k in ("min_success", "min_safe", "max_aa_hit", "max_ao_hit")
        },
        "eval_seed": args.eval_seed,
        "eval_history": history,
        "has_eligible_checkpoint": best_success >= 0.0,
        "before_D_init_residual": before,
        "before_D_base": before_base,
        "after_last_D": after,
        "after_best_D": after_best,
        "best_success_short_eval": best_success,
        "best_mean_hit_D": best_hit,
        "wall_time_total_s": round(time.time() - t0, 3),
        "checkpoint": str(out),
        "best_checkpoint": str(best_path) if best_success >= 0.0 else None,
        "eval_csv": str(out.with_suffix(".csv")),
        "claim": (
            "Empirical residual repair only. Base remains the frozen distilled DeepSet. "
            "Not a formal safety proof."
        ),
    }
    summary = attach_run_meta(
        summary,
        method=args.method_tag,
        train_seed=args.seed,
        phase2_run_id=args.phase2_run_id,
    )
    sum_path = write_summary_json(out.with_suffix(".json"), summary)
    csv_path = write_eval_csv(
        out.with_suffix(".csv"),
        history,
        best_steps=best_steps,
        wall_times=wall_times,
    )
    print(f"[residual] summary {sum_path}")
    print(f"[residual] eval_csv {csv_path}")


if __name__ == "__main__":
    main()
