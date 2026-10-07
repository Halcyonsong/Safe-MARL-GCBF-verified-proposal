"""Stage3: repair DeepSet with L2-dynamics safe labels on preventable CEs.

Labels for CE inputs are ``a_l2_safe`` from ``collect_counterexamples.py``
(min-intervention by default, or max-clearance). Student inputs use the same
sparse (k,m) packing as Stage2 L2 queries. Normal samples default to frozen
DeepSet on-policy actions (behavior anchor).

Does not overwrite the frozen Stage1 baseline.
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
from stage1.student import DeepSetStudent, extract_rollout_sets
from stage1.train import collect_teacher_batch, evaluate_student, make_teacher, save_checkpoint
from stage2.bounds import default_l2_buckets
from stage2.encode import ENTITY_CONT_DIM, ENTITY_DIM, EGO_DIM, load_weights
from stage2.l2_query import action_at
from stage3.metrics_io import attach_run_meta, write_eval_csv, write_summary_json

N_MAX_AGENTS = 8
N_MAX_OBS = 32
ENTITY_SLOTS = N_MAX_AGENTS + N_MAX_OBS


def bucket_x_to_student_inputs(x: np.ndarray, k: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    ego = x[:EGO_DIM].astype(np.float32)
    entities = np.zeros((ENTITY_SLOTS, ENTITY_DIM), dtype=np.float32)
    off = EGO_DIM
    for i in range(k):
        cont = x[off : off + ENTITY_CONT_DIM]
        off += ENTITY_CONT_DIM
        entities[i, :4] = cont
        entities[i, 4:] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    for j in range(m):
        cont = x[off : off + ENTITY_CONT_DIM]
        off += ENTITY_CONT_DIM
        slot = N_MAX_AGENTS + j
        entities[slot, :4] = cont
        entities[slot, 4:] = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    return ego, entities


def load_ce_dataset(path: pathlib.Path, *, require_safe_label: bool = True):
    raw = json.loads(path.read_text())
    egos, ents, acts = [], [], []
    n_skip = 0
    deltas = []
    modes = set()
    for bucket in raw["buckets"]:
        k, m = int(bucket["k"]), int(bucket["m"])
        for ce in bucket["counterexamples"]:
            if require_safe_label and not ce.get("a_l2_label_ok", False):
                n_skip += 1
                continue
            label = ce.get("a_l2_safe") or ce.get("a")
            ego, ent = bucket_x_to_student_inputs(np.asarray(ce["x"]), k, m)
            egos.append(ego)
            ents.append(ent)
            acts.append(np.asarray(label, dtype=np.float32))
            if "a_l2_delta" in ce:
                deltas.append(float(ce["a_l2_delta"]))
            if "a_l2_label_mode" in ce:
                modes.add(str(ce["a_l2_label_mode"]))
    if not egos:
        raise ValueError(f"no usable CEs in {path} (skipped_unsafe_labels={n_skip})")
    mean_da = float(np.mean(deltas)) if deltas else None
    print(
        f"[repair-l2] loaded {len(egos)} CEs with L2-safe labels (skipped {n_skip}); "
        f"modes={sorted(modes) or ['legacy']}; mean_Δa={mean_da}"
    )
    return np.stack(egos), np.stack(ents), np.stack(acts)


def collect_frozen_batch(args, env, student, frozen_params, keys):
    """On-policy states + frozen DeepSet actions (behavior anchor)."""
    import jax.tree_util as jtu

    from stage1.student import extract_local_sets

    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm_radius = float(env.params["comm_radius"])

    def frozen_policy(graph):
        ego, entities = extract_local_sets(graph, n_agents, n_rays, comm_radius)
        return jnp.clip(student.apply(frozen_params, ego, entities), *env.action_lim())

    rollout_fn = jax.jit(env.rollout_fn(frozen_policy, args.rollout_length))
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


def _selection_score(hit: dict, metrics: dict, specs) -> tuple:
    """Lower is better: mean D-hit, then -success (prefer higher success)."""
    hits = [hit.get(s.name, {}).get("hit_rate_on_D", 1.0) for s in specs]
    mean_hit = float(np.mean(hits)) if hits else 1.0
    return (mean_hit, -float(metrics["success_rate"]))


def resample_hit_rate_on_D(weights, specs, *, n_samples: int, seed: int) -> dict:
    """Hit-rate of π on preventable domain D only (supports dual (1,1))."""
    from stage2.l2_domain import in_preventable_domain_x
    from stage2.l2_query import eval_collision_dual, next_rels_for_spec, rels_from_x

    rng = np.random.default_rng(seed)
    out = {}
    for spec in specs:
        n_eval = n_hit = n_draw = 0
        min_d = 1e9
        budget = max(n_samples * 40, 20000)
        while n_eval < n_samples and n_draw < budget:
            x = spec.box.sample(rng)
            n_draw += 1
            if not in_preventable_domain_x(x, spec, grid_n=7):
                continue
            n_eval += 1
            a = action_at(weights, x, spec)
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
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 preventable-CE repair with L2-safe labels")
    parser.add_argument("--init-from", default="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl")
    parser.add_argument("--ce-json", default="method/deepset/output/stage3/ce/l2_counterexamples.json")
    parser.add_argument("--teacher-run", default="baselines/gcbfplus/log/gcbf+/seed0_20261001232611")
    parser.add_argument("--teacher-step", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--ce-batch", type=int, default=256)
    parser.add_argument("--normal-batch", type=int, default=768)
    parser.add_argument("--ce-weight", type=float, default=2.0)
    parser.add_argument(
        "--normal-source",
        default="frozen",
        choices=["frozen", "teacher", "mixed"],
        help="Behavior anchor: frozen DeepSet on-policy, GCBF+ Actor, or both",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--resample", type=int, default=2000)
    parser.add_argument("--eval-buckets", default="", help="independent evaluation buckets, including untouched buckets")
    parser.add_argument("--eval-seed", type=int, default=None, help="fixed rollout seed across evaluation checkpoints")
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    parser.add_argument("--min-success", type=float, default=0.0)
    parser.add_argument("--min-safe", type=float, default=0.0)
    parser.add_argument("--max-aa-hit", type=float, default=1.0)
    parser.add_argument("--max-ao-hit", type=float, default=1.0)
    parser.add_argument("--replay-ce-json", default=None, help="anchor initial-policy actions on older CE inputs")
    parser.add_argument("--replay-batch", type=int, default=256)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint", default="method/deepset/output/stage3/repair/deepset_l2_safe_repair.pkl")
    parser.add_argument(
        "--method-tag",
        default=None,
        help="optional run_meta.method (e.g. R1) written into summary JSON",
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
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-cbf", type=float, default=3e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    ce_ego, ce_ent, ce_tgt = load_ce_dataset(pathlib.Path(args.ce_json), require_safe_label=True)

    env = make_env(
        "DoubleIntegrator",
        num_agents=args.num_agents,
        area_size=args.area_size,
        num_obs=args.obs,
        n_rays=args.n_rays,
    )
    student = DeepSetStudent(hidden=args.hidden, action_dim=env.action_dim)

    with pathlib.Path(args.init_from).open("rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["student_params"]
    frozen_params = params  # immutable reference for on-policy anchor
    if int(ckpt.get("hidden", args.hidden)) != args.hidden:
        raise ValueError("hidden mismatch")

    replay_arrays = None
    if args.replay_ce_json:
        replay_ego, replay_ent, _old_labels = load_ce_dataset(pathlib.Path(args.replay_ce_json))
        replay_ego, replay_ent = map(jnp.asarray, (replay_ego, replay_ent))
        replay_tgt = jnp.clip(student.apply(frozen_params, replay_ego, replay_ent), -1.0, 1.0)
        replay_arrays = (replay_ego, replay_ent, replay_tgt)
        print(f"[repair-l2] replay anchor={args.init_from} on {replay_ego.shape[0]} older CE inputs")

    teacher = None
    if args.normal_source in ("teacher", "mixed"):
        teacher = make_teacher(args, env)

    key = jr.PRNGKey(args.seed)
    key, nk = jr.split(key)
    sample_keys = jr.split(nk, args.batch_size)

    def refresh_normals(k):
        parts_e, parts_n, parts_t = [], [], []
        if args.normal_source in ("frozen", "mixed"):
            e, n, t = collect_frozen_batch(args, env, student, frozen_params, k)
            parts_e.append(e)
            parts_n.append(n)
            parts_t.append(t)
        if args.normal_source in ("teacher", "mixed"):
            assert teacher is not None
            e, n, t = collect_teacher_batch(args, env, teacher, k)
            parts_e.append(e)
            parts_n.append(n)
            parts_t.append(t)
        return jnp.concatenate(parts_e), jnp.concatenate(parts_n), jnp.concatenate(parts_t)

    n_ego, n_ent, n_tgt = refresh_normals(sample_keys)
    print(
        f"[repair-l2] normal samples ({args.normal_source}): {n_ego.shape[0]} "
        f"ce_weight={args.ce_weight}"
    )

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = tx.init(params)

    def loss_fn(p, ego, ent, tgt, w):
        pred = jnp.clip(student.apply(p, ego, ent), -1.0, 1.0)
        per = jnp.mean(jnp.square(pred - tgt), axis=-1)
        return jnp.mean(w * per)

    @jax.jit
    def update(p, os, ego, ent, tgt, w):
        loss, grads = jax.value_and_grad(loss_fn)(p, ego, ent, tgt, w)
        updates, os = tx.update(grads, os, p)
        return optax.apply_updates(p, updates), os, loss

    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    out = pathlib.Path(args.checkpoint)
    best_path = out.with_name(out.stem + "_best.pkl")
    best_success = -1.0
    best_hit = 1e9
    best_tuple = (1e9, 1.0)  # (mean_hit, -success)
    history = []
    best_steps: set[int] = set()
    wall_times: dict[int, float] = {}
    t0 = time.time()

    # Evaluate the buckets present in the CE json (supports multi-bucket joint).
    ce_bucket_names = {
        b["bucket"] for b in json.loads(pathlib.Path(args.ce_json).read_text())["buckets"]
    }
    if args.eval_buckets:
        ce_bucket_names = {b.strip() for b in args.eval_buckets.split(",") if b.strip()}
        missing = ce_bucket_names - {s.name for s in default_l2_buckets()}
        if missing:
            raise ValueError(f"unknown evaluation buckets: {sorted(missing)}")
    specs = [s for s in default_l2_buckets() if s.name in ce_bucket_names]
    if not specs:
        specs = [s for s in default_l2_buckets() if s.name in ("l2_aa_near", "l2_ao_near")]
    numpy_w, _ = load_weights(args.init_from)
    before = resample_hit_rate_on_D(numpy_w, specs, n_samples=args.resample, seed=123)
    print("[repair-l2] hit-rate on D BEFORE:", json.dumps(before))

    ce_ego_j, ce_ent_j, ce_tgt_j = map(jnp.asarray, (ce_ego, ce_ent, ce_tgt))
    n_ce = ce_ego_j.shape[0]
    n_norm = n_ego.shape[0]

    for step in range(1, args.steps + 1):
        key, k1, k2 = jr.split(key, 3)
        ci = jr.choice(k1, n_ce, shape=(min(args.ce_batch, n_ce),), replace=False)
        ni = jr.choice(k2, n_norm, shape=(min(args.normal_batch, n_norm),), replace=False)
        ego = jnp.concatenate([ce_ego_j[ci], n_ego[ni]], axis=0)
        ent = jnp.concatenate([ce_ent_j[ci], n_ent[ni]], axis=0)
        tgt = jnp.concatenate([ce_tgt_j[ci], n_tgt[ni]], axis=0)
        w = jnp.concatenate(
            [jnp.full((ci.shape[0],), args.ce_weight), jnp.ones((ni.shape[0],))]
        )
        if replay_arrays is not None:
            key, kr = jr.split(key)
            re, rn, rt = replay_arrays
            ri = jr.choice(kr, re.shape[0], shape=(min(args.replay_batch, re.shape[0]),), replace=False)
            ego = jnp.concatenate([ego, re[ri]])
            ent = jnp.concatenate([ent, rn[ri]])
            tgt = jnp.concatenate([tgt, rt[ri]])
            w = jnp.concatenate([w, jnp.full((ri.shape[0],), args.replay_weight)])
        params, opt_state, loss = update(params, opt_state, ego, ent, tgt, w)

        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f"step={step:04d} loss={float(loss):.6f} elapsed={time.time()-t0:.1f}s")

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            metrics = evaluate_student(
                env, student, params, n_agents, n_rays, args.eval_episodes,
                args.eval_seed if args.eval_seed is not None else args.seed + 50_000 + step,
            )
            tmp = out.with_name(out.stem + "_tmp_eval.pkl")
            save_checkpoint(
                tmp, params, args, EGO_DIM, ENTITY_DIM, ENTITY_SLOTS, n_agents, n_rays, metrics=metrics
            )
            w_tmp, _ = load_weights(tmp)
            hit = resample_hit_rate_on_D(w_tmp, specs, n_samples=min(800, args.resample), seed=99)
            mean_hit = float(np.mean([hit[s.name]["hit_rate_on_D"] for s in specs]))
            hit_str = " ".join(
                f"{k}={v['hit_rate_on_D']*100:.2f}%" for k, v in hit.items()
            )
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
                "eligible": eligible,
                "wall_time_s": round(wall_t, 3),
                **metrics,
            }
            history.append(eval_metrics)
            wall_times[step] = wall_t
            if args.save_eval_checkpoints:
                save_checkpoint(
                    out.with_name(out.stem + f"_step{step:04d}.pkl"),
                    params, args, EGO_DIM, ENTITY_DIM, ENTITY_SLOTS,
                    n_agents, n_rays, metrics=eval_metrics,
                )
            if not eligible:
                print("[repair-l2] rejected by success/safe/AA/AO selection floors")
            if eligible and score < best_tuple:
                best_tuple = score
                best_hit = mean_hit
                best_success = metrics["success_rate"]
                best_steps = {step}
                save_checkpoint(
                    best_path,
                    params,
                    args,
                    EGO_DIM,
                    ENTITY_DIM,
                    ENTITY_SLOTS,
                    n_agents,
                    n_rays,
                    metrics=eval_metrics,
                )
                print(
                    f"[repair-l2] best={best_path} mean_D_hit={mean_hit*100:.2f}% "
                    f"success={best_success*100:.2f}%"
                )
            tmp.unlink(missing_ok=True)

        if step % 100 == 0:
            key, rk = jr.split(key)
            n_keys = jr.split(rk, args.batch_size)
            n_ego, n_ent, n_tgt = refresh_normals(n_keys)
            n_norm = n_ego.shape[0]

    save_checkpoint(
        out,
        params,
        args,
        EGO_DIM,
        ENTITY_DIM,
        ENTITY_SLOTS,
        n_agents,
        n_rays,
        metrics={"step": args.steps, "best_success": best_success, "best_mean_hit_D": best_hit},
    )
    numpy_w, _ = load_weights(out)
    after = resample_hit_rate_on_D(numpy_w, specs, n_samples=args.resample, seed=123)
    print("[repair-l2] hit-rate on D AFTER (last):", json.dumps(after))
    if best_success >= 0.0:
        numpy_b, _ = load_weights(best_path)
        after_best = resample_hit_rate_on_D(numpy_b, specs, n_samples=args.resample, seed=123)
        print("[repair-l2] hit-rate on D AFTER (best):", json.dumps(after_best))
    else:
        after_best = after

    # Infer label mode from CE json meta when present.
    ce_meta = json.loads(pathlib.Path(args.ce_json).read_text())
    label_mode = ce_meta.get("label_mode", "unknown")
    summary = {
        "mode": f"ce_{label_mode}_on_D_with_init_checkpoint_anchor",
        "anchor_type": "init_from_checkpoint",
        "label_mode": label_mode,
        "hybrid_lam": ce_meta.get("hybrid_lam"),
        "hybrid_mu": ce_meta.get("hybrid_mu"),
        "hybrid_near": ce_meta.get("hybrid_near"),
        "init_from": args.init_from,
        "ce_json": args.ce_json,
        "normal_source": args.normal_source,
        "anchor_checkpoint": args.init_from if args.normal_source in ("frozen", "mixed") else None,
        "replay_ce_json": args.replay_ce_json,
        "selection_limits": {
            k: getattr(args, k) for k in ("min_success", "min_safe", "max_aa_hit", "max_ao_hit")
        },
        "eval_seed": args.eval_seed,
        "seed": args.seed,
        "lr": args.lr,
        "eval_history": history,
        "has_eligible_checkpoint": best_success >= 0.0,
        "n_ce": int(n_ce),
        "steps": args.steps,
        "ce_weight": args.ce_weight,
        "before_D": before,
        "after_last_D": after,
        "after_best_D": after_best,
        "best_success_short_eval": best_success,
        "best_mean_hit_D": best_hit,
        "wall_time_total_s": round(time.time() - t0, 3),
        "checkpoint": str(out),
        "best_checkpoint": str(best_path) if best_success >= 0.0 else None,
        "eval_csv": str(out.with_suffix(".csv")),
        "claim": (
            "Empirical full-net L2 CE repair from frozen init. "
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
    print(f"[repair-l2] summary {sum_path}")
    print(f"[repair-l2] eval_csv {csv_path}")


if __name__ == "__main__":
    main()
