"""Train physics-gated residual on MF base using rollout CEs + base anchoring.

    a = clip( base(x) + g_phys(x) * delta(x) )

Only delta is trained. base frozen. Outputs under method/meanfield/output/repair/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
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



from gcbfplus.env import make_env
from gcbfplus.utils.utils import jax_jit_np, jax_vmap
from stage_mf.collect_ce import collect_ces
from stage_mf.residual import (
    CorrectionMLP,
    apply_residual_mf,
    load_mf_base,
    load_mf_residual,
    physics_gate_mf,
    save_mf_residual_checkpoint,
)
from stage_mf.student import INPUT_DIM, MeanFieldStudent, extract_meanfield_features, extract_rollout_meanfield


def load_ce_arrays(
    paths: list[pathlib.Path] | pathlib.Path,
    *,
    require_safe_label: bool = True,
):
    if isinstance(paths, (str, pathlib.Path)):
        paths = [pathlib.Path(paths)]
    xs, targets, kinds = [], [], []
    n_skip = 0
    n_raw = 0
    for path in paths:
        path = pathlib.Path(path)
        raw = json.loads(path.read_text())
        for ce in raw.get("counterexamples", []):
            n_raw += 1
            if require_safe_label and not ce.get("a_l2_label_ok", False):
                n_skip += 1
                continue
            label = ce.get("a_l2_safe") or ce.get("a")
            xs.append(np.asarray(ce["x"], dtype=np.float32))
            targets.append(np.asarray(label, dtype=np.float32))
            k = str(ce.get("kind", "other"))
            kinds.append({"aa": 0, "ao": 1, "both": 2}.get(k, 3))
    if not xs:
        raise ValueError(f"no usable CEs in {paths} (raw={n_raw} skipped={n_skip})")
    print(
        f"[mf-res] loaded {len(xs)} CEs from {len(paths)} file(s) "
        f"(raw={n_raw} skipped={n_skip})"
    )
    return np.stack(xs), np.stack(targets), np.asarray(kinds, dtype=np.int32)


def collect_base_anchor(env, base, base_params, keys, rollout_length: int):
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()

    def pol(graph):
        x = extract_meanfield_features(graph, n_agents, n_rays, comm)
        return jnp.clip(base.apply(base_params, x), lo, hi)

    rollout_fn = jax.jit(jax.vmap(env.rollout_fn(pol, rollout_length)))
    rollouts = rollout_fn(keys)
    graphs = jax.tree_util.tree_map(lambda x: x[:, :-1], rollouts.Tp1_graph)
    feats = jax.vmap(
        lambda gs: extract_rollout_meanfield(gs, n_agents, n_rays, comm)
    )(graphs)
    feats = feats.reshape(-1, feats.shape[-1])
    acts = jnp.clip(rollouts.T_action, lo, hi).reshape(-1, env.action_dim)
    return feats, acts


def evaluate_policy(
    env,
    base,
    base_params,
    delta,
    delta_params,
    *,
    episodes: int,
    seed: int,
    open_mult: float,
    close_mult: float,
    inv_ttc_ref: float,
    gate_mix: float,
    delta_clip: float,
) -> dict:
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()

    def policy(graph):
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

    rollout_fn = jax_jit_np(env.rollout_fn(policy, env.max_episode_steps))
    is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
    is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    keys = jr.split(jr.PRNGKey(seed), episodes)
    rewards, costs, unsafe, finish = [], [], [], []
    for key in keys:
        rollout = rollout_fn(key)
        unsafe.append(np.asarray(is_unsafe_fn(rollout.Tp1_graph)).max(axis=0))
        finish.append(np.asarray(is_finish_fn(rollout.Tp1_graph)).max(axis=0))
        rewards.append(float(rollout.T_reward.sum()))
        costs.append(float(rollout.T_cost.sum()))
    unsafe = np.stack(unsafe)
    finish = np.stack(finish)
    return {
        "reward": float(np.mean(rewards)),
        "cost": float(np.mean(costs)),
        "safe_rate": float((1 - unsafe).mean()),
        "finish_rate": float(finish.mean()),
        "success_rate": float(((1 - unsafe) * finish).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument(
        "--ce-json",
        default="",
        help="CE json path(s), comma-separated; if empty, collect first",
    )
    ap.add_argument(
        "--init-residual",
        default="",
        help="warm-start delta from prior meanfield_residual_phys ckpt",
    )
    ap.add_argument(
        "--allow-unsafe-labels",
        action="store_true",
        help="keep CEs even when min-intervention found no fully-safe action",
    )
    ap.add_argument("--collect-episodes", type=int, default=48)
    ap.add_argument("--collect-seed", type=int, default=123)
    ap.add_argument("--danger-radius", type=float, default=0.35)
    ap.add_argument("--max-ce", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--correction-hidden", type=int, default=128)
    ap.add_argument("--ce-batch", type=int, default=256)
    ap.add_argument("--normal-batch", type=int, default=512)
    ap.add_argument("--ce-weight", type=float, default=3.0)
    ap.add_argument("--ao-ce-weight", type=float, default=1.5)
    ap.add_argument("--both-ce-weight", type=float, default=2.0)
    ap.add_argument("--idle-weight", type=float, default=1.5)
    ap.add_argument("--delta-l2", type=float, default=1e-3)
    ap.add_argument("--open-mult", type=float, default=3.0)
    ap.add_argument("--close-mult", type=float, default=1.0)
    ap.add_argument("--inv-ttc-ref", type=float, default=5.0)
    ap.add_argument("--gate-mix", type=float, default=0.5)
    ap.add_argument("--delta-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--resample-episodes", type=int, default=32)
    ap.add_argument("--resample-seed", type=int, default=123)
    ap.add_argument("--collect-stride", type=int, default=2)
    ap.add_argument("--prior-summary", default="", help="optional r1 ce_post_summary.json for SELECTION")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--checkpoint",
        default="method/meanfield/output/repair/mf_residual_v5/seed0/repair.pkl",
    )
    ap.add_argument("--rollout-length", type=int, default=64)
    ap.add_argument(
        "--no-update-active",
        action="store_true",
        help="do not rewrite method/meanfield/output/ACTIVE_MF_RESIDUAL.txt (side-branch / ablation)",
    )
    ap.add_argument(
        "--score-mode",
        choices=("success_safe", "success_finish"),
        default="success_safe",
        help="best-ckpt selection: success_safe=2*suc+safe (default); success_finish=2*suc+fin (finish-oriented)",
    )
    args = ap.parse_args()

    out = pathlib.Path(args.checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.parent / "train.log"

    # CE
    ce_paths: list[pathlib.Path] = []
    if args.ce_json.strip():
        for part in args.ce_json.split(","):
            part = part.strip()
            if not part:
                continue
            p = pathlib.Path(part)
            if not p.exists():
                raise FileNotFoundError(f"CE json not found: {p}")
            ce_paths.append(p)
    else:
        ce_path = out.parent / "ce_baseline.json"
        print(f"[mf-res] collecting CEs → {ce_path}")
        report = collect_ces(
            args.base_checkpoint,
            residual=False,
            episodes=args.collect_episodes,
            seed=args.collect_seed,
            danger_radius=args.danger_radius,
            max_keep=args.max_ce,
        )
        ce_path.write_text(json.dumps(report) + "\n")
        summary = {k: v for k, v in report.items() if k != "counterexamples"}
        (out.parent / "ce_baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"[mf-res] baseline hit_on_danger={report['hit_rate_on_danger']*100:.2f}% "
            f"kept={report['n_kept']}"
        )
        ce_paths = [ce_path]

    x_ce, y_ce, kind_ce = load_ce_arrays(
        ce_paths, require_safe_label=not args.allow_unsafe_labels
    )
    ckpt, base, base_params = load_mf_base(args.base_checkpoint)
    hidden = int(ckpt["hidden"])
    delta = CorrectionMLP(hidden=args.correction_hidden, action_dim=2)

    env = make_env("DoubleIntegrator", num_agents=8, area_size=4.0, num_obs=4, n_rays=32)
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    key = jr.PRNGKey(args.seed)
    key, k_init = jr.split(key)
    dummy = jnp.zeros((1, INPUT_DIM))
    if args.init_residual:
        r0 = load_mf_residual(args.init_residual)
        if int(r0.get("correction_hidden", -1)) != int(args.correction_hidden):
            raise ValueError(
                f"init-residual correction_hidden={r0.get('correction_hidden')} "
                f"!= {args.correction_hidden}"
            )
        delta_params = r0["delta_params"]
        # keep gate hyperparams consistent with prior residual if not overridden
        print(f"[mf-res] warm-start delta from {args.init_residual}")
    else:
        delta_params = delta.init(k_init, dummy)

    opt = optax.adam(args.lr)
    opt_state = opt.init(delta_params)

    # threat weights
    w_kind = jnp.array(
        [args.ce_weight, args.ce_weight * args.ao_ce_weight, args.ce_weight * args.both_ce_weight, args.ce_weight],
        dtype=jnp.float32,
    )

    def loss_fn(dparams, x_b, y_b, k_b, x_n, y_n):
        a_ce, g_ce, d_ce = apply_residual_mf(
            base,
            base_params,
            delta,
            dparams,
            x_b,
            open_mult=args.open_mult,
            close_mult=args.close_mult,
            inv_ttc_ref=args.inv_ttc_ref,
            gate_mix=args.gate_mix,
            delta_clip=args.delta_clip,
        )
        w = w_kind[k_b]
        ce_l = jnp.mean(w * jnp.sum((a_ce - y_b) ** 2, axis=-1))
        a_n, g_n, d_n = apply_residual_mf(
            base,
            base_params,
            delta,
            dparams,
            x_n,
            open_mult=args.open_mult,
            close_mult=args.close_mult,
            inv_ttc_ref=args.inv_ttc_ref,
            gate_mix=args.gate_mix,
            delta_clip=args.delta_clip,
        )
        # anchor to base action; stronger when gate small (idle)
        idle = 1.0 - g_n
        n_l = jnp.mean((args.idle_weight * idle + 1.0) * jnp.sum((a_n - y_n) ** 2, axis=-1))
        reg = args.delta_l2 * jnp.mean(d_ce**2)
        return ce_l + n_l + reg, (ce_l, n_l, reg, jnp.mean(g_ce), jnp.mean(g_n))

    @jax.jit
    def train_step(dparams, opt_state, x_b, y_b, k_b, x_n, y_n):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            dparams, x_b, y_b, k_b, x_n, y_n
        )
        updates, opt_state = opt.update(grads, opt_state, dparams)
        dparams = optax.apply_updates(dparams, updates)
        return dparams, opt_state, loss, aux

    history = []
    best_score = -1e9
    best_params = delta_params
    best_metrics = {}
    t0 = time.time()
    n_ce = x_ce.shape[0]

    def log(msg: str):
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    log(
        f"[mf-res] base={args.base_checkpoint} ce={[str(p) for p in ce_paths]} n_ce={n_ce} "
        f"init={args.init_residual or 'random'} steps={args.steps} "
        f"corr_h={args.correction_hidden} gate=physics"
    )
    ce_path_note = ",".join(str(p) for p in ce_paths)

    # Prefetch base-anchor buffer once (refresh every refresh_anchor steps)
    key, k_anchor = jr.split(key)
    anchor_keys = jr.split(k_anchor, 4)
    x_anchor, y_anchor = collect_base_anchor(
        env, base, base_params, anchor_keys, args.rollout_length
    )
    x_anchor = np.asarray(x_anchor)
    y_anchor = np.asarray(y_anchor)
    refresh_anchor = 50

    for step in range(1, args.steps + 1):
        key, k1, k2, k3 = jr.split(key, 4)
        idx = jr.randint(k1, (args.ce_batch,), 0, n_ce)
        x_b = jnp.asarray(x_ce[np.asarray(idx)])
        y_b = jnp.asarray(y_ce[np.asarray(idx)])
        k_b = jnp.asarray(kind_ce[np.asarray(idx)])
        if step % refresh_anchor == 1 and step > 1:
            key, k_anchor = jr.split(key)
            anchor_keys = jr.split(k_anchor, 4)
            x_anchor, y_anchor = collect_base_anchor(
                env, base, base_params, anchor_keys, args.rollout_length
            )
            x_anchor = np.asarray(x_anchor)
            y_anchor = np.asarray(y_anchor)
        j = np.asarray(jr.randint(k3, (min(args.normal_batch, x_anchor.shape[0]),), 0, x_anchor.shape[0]))
        x_n = jnp.asarray(x_anchor[j])
        y_n = jnp.asarray(y_anchor[j])

        delta_params, opt_state, loss, aux = train_step(
            delta_params, opt_state, x_b, y_b, k_b, x_n, y_n
        )
        ce_l, n_l, reg, g_ce, g_n = aux

        if step % 50 == 0 or step == 1:
            log(
                f"step={step:04d} loss={float(loss):.5f} ce={float(ce_l):.5f} "
                f"anchor={float(n_l):.5f} reg={float(reg):.5f} "
                f"g_ce={float(g_ce):.3f} g_n={float(g_n):.3f}"
            )

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate_policy(
                env,
                base,
                base_params,
                delta,
                delta_params,
                episodes=args.eval_episodes,
                seed=args.eval_seed,
                open_mult=args.open_mult,
                close_mult=args.close_mult,
                inv_ttc_ref=args.inv_ttc_ref,
                gate_mix=args.gate_mix,
                delta_clip=args.delta_clip,
            )
            # score: default prefer success then safe; finish-oriented side-branch available
            if args.score_mode == "success_finish":
                score = 2.0 * m["success_rate"] + m["finish_rate"]
            else:
                score = 2.0 * m["success_rate"] + m["safe_rate"]
            is_best = score > best_score
            if is_best:
                best_score = score
                best_params = delta_params
                best_metrics = dict(m)
            rec = {
                "step": step,
                "loss": float(loss),
                "success_rate": m["success_rate"],
                "safe_rate": m["safe_rate"],
                "finish_rate": m["finish_rate"],
                "reward": m["reward"],
                "cost": m["cost"],
                "is_best": is_best,
                "wall_time_s": round(time.time() - t0, 2),
            }
            history.append(rec)
            log(
                f"[eval] step={step} safe={m['safe_rate']*100:.2f}% "
                f"finish={m['finish_rate']*100:.2f}% success={m['success_rate']*100:.2f}% "
                f"{'BEST' if is_best else ''}"
            )
            save_mf_residual_checkpoint(
                out if not is_best else out.with_name(out.stem + "_best.pkl"),
                base_checkpoint=args.base_checkpoint,
                base_params=base_params,
                delta_params=delta_params if not is_best else best_params,
                hidden=hidden,
                correction_hidden=args.correction_hidden,
                open_mult=args.open_mult,
                close_mult=args.close_mult,
                inv_ttc_ref=args.inv_ttc_ref,
                gate_mix=args.gate_mix,
                delta_clip=args.delta_clip,
                n_agents=n_agents,
                n_rays=n_rays,
                metrics=m,
                extra={
                    "step": step,
                    "ce_json": ce_path_note,
                    "init_residual": args.init_residual or None,
                },
            )

    # final save best + last
    save_mf_residual_checkpoint(
        out,
        base_checkpoint=args.base_checkpoint,
        base_params=base_params,
        delta_params=delta_params,
        hidden=hidden,
        correction_hidden=args.correction_hidden,
        open_mult=args.open_mult,
        close_mult=args.close_mult,
        inv_ttc_ref=args.inv_ttc_ref,
        gate_mix=args.gate_mix,
        delta_clip=args.delta_clip,
        n_agents=n_agents,
        n_rays=n_rays,
        metrics=history[-1] if history else {},
        extra={
            "step": args.steps,
            "ce_json": ce_path_note,
            "init_residual": args.init_residual or None,
            "tag": "last",
        },
    )
    best_path = out.with_name(out.stem + "_best.pkl")
    save_mf_residual_checkpoint(
        best_path,
        base_checkpoint=args.base_checkpoint,
        base_params=base_params,
        delta_params=best_params,
        hidden=hidden,
        correction_hidden=args.correction_hidden,
        open_mult=args.open_mult,
        close_mult=args.close_mult,
        inv_ttc_ref=args.inv_ttc_ref,
        gate_mix=args.gate_mix,
        delta_clip=args.delta_clip,
        n_agents=n_agents,
        n_rays=n_rays,
        metrics=best_metrics,
        extra={
            "ce_json": ce_path_note,
            "init_residual": args.init_residual or None,
            "tag": "best",
        },
    )

    # post resample CE hit rate
    log("[mf-res] resampling CEs on residual best…")
    post = collect_ces(
        best_path,
        residual=True,
        episodes=args.resample_episodes,
        seed=args.resample_seed,
        danger_radius=args.danger_radius,
        max_keep=args.max_ce,
        stride=args.collect_stride,
    )
    post_path = out.parent / "ce_post_residual.json"
    post_path.write_text(json.dumps(post) + "\n")
    post_sum = {k: v for k, v in post.items() if k != "counterexamples"}
    (out.parent / "ce_post_summary.json").write_text(json.dumps(post_sum, indent=2) + "\n")

    # 32-epi validate residual
    m32 = evaluate_policy(
        env,
        base,
        base_params,
        delta,
        best_params,
        episodes=32,
        seed=1234,
        open_mult=args.open_mult,
        close_mult=args.close_mult,
        inv_ttc_ref=args.inv_ttc_ref,
        gate_mix=args.gate_mix,
        delta_clip=args.delta_clip,
    )
    (out.parent / "validate_32epi.json").write_text(json.dumps(m32, indent=2) + "\n")

    # prior / baseline rows for SELECTION
    rows = []
    if args.prior_summary and pathlib.Path(args.prior_summary).exists():
        prior = json.loads(pathlib.Path(args.prior_summary).read_text())
        rows.append(
            (
                "prior_residual",
                prior.get("hit_rate_on_danger"),
                prior.get("n_hit_two_step"),
                prior.get("n_danger"),
                prior.get("n_kept"),
            )
        )
    bs = out.parent / "ce_train_summary.json"
    if not bs.exists():
        bs = out.parent / "ce_baseline_summary.json"
    if bs.exists():
        train_ce = json.loads(bs.read_text())
        rows.append(
            (
                "train_ce_pool",
                train_ce.get("hit_rate_on_danger"),
                train_ce.get("n_hit_two_step"),
                train_ce.get("n_danger"),
                train_ce.get("n_kept"),
            )
        )
    rows.append(
        (
            "this_residual",
            post["hit_rate_on_danger"],
            post["n_hit_two_step"],
            post["n_danger"],
            post["n_kept"],
        )
    )

    def _pct(x):
        try:
            return f"{100.0 * float(x):.2f}%"
        except Exception:
            return "?"

    sel = [
        "# MF residual repair",
        "",
        f"- base: `{args.base_checkpoint}`",
        f"- init residual: `{args.init_residual or '(random delta)'}`",
        f"- residual best: `{best_path}`",
        f"- gate: physics (open={args.open_mult}, close={args.close_mult}, mix={args.gate_mix})",
        f"- ce sources: `{ce_path_note}`",
        f"- lr={args.lr} idle_w={args.idle_weight} delta_l2={args.delta_l2} steps={args.steps}",
        "",
        "## CE hit_rate on danger (two-step L2)",
        "",
        "| phase | hit_on_danger | n_hit | n_danger | n_kept |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, h, nh, nd, nk in rows:
        sel.append(f"| {name} | {_pct(h)} | {nh} | {nd} | {nk} |")
    sel += [
        "",
        "## Task 32-epi (seed=1234)",
        "",
        "| policy | success | safe | finish |",
        "|---|---:|---:|---:|",
        f"| residual best | {100*m32['success_rate']:.2f}% | {100*m32['safe_rate']:.2f}% | {100*m32['finish_rate']:.2f}% |",
        "",
        "Empirical repair only. Formal verification deferred until CE rate is low.",
        "",
    ]
    (out.parent / "SELECTION.md").write_text("\n".join(sel))

    # csv
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "success_rate",
                "safe_rate",
                "finish_rate",
                "reward",
                "cost",
                "is_best",
                "wall_time_s",
            ],
        )
        w.writeheader()
        for r in history:
            w.writerow(r)

    # ACTIVE residual pointer (meanfield only) — skip for side-branch ablations
    ptr = pathlib.Path("method/meanfield/output/ACTIVE_MF_RESIDUAL.txt")
    if args.no_update_active:
        log(f"[mf-res] --no-update-active: left {ptr} untouched")
    else:
        ptr.write_text(str(best_path) + "\n")

    prior_hit = None
    if args.prior_summary and pathlib.Path(args.prior_summary).exists():
        prior_hit = json.loads(pathlib.Path(args.prior_summary).read_text()).get(
            "hit_rate_on_danger"
        )
    prior_s = f"{100*prior_hit:.2f}%" if prior_hit is not None else "n/a"
    log(
        f"[mf-res] done. prior_hit={prior_s} "
        f"post_hit={100*post['hit_rate_on_danger']:.2f}% "
        f"task32 success={100*m32['success_rate']:.2f}% safe={100*m32['safe_rate']:.2f}% "
        f"finish={100*m32['finish_rate']:.2f}% best={best_path}"
    )


if __name__ == "__main__":
    main()
