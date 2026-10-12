#!/usr/bin/env python3
"""Box→env from reachable planar claim: shared GraphsTuple for all baselines.

Construction (reachable_v1):
  rel ~ planar_netdep box (controllable)
  ego v free; goal_xy in disk ≤comm; goal body v=0 ⇒ feature goal_dv=-v
  threat agent1 = ego - rel; others/obstacles far
  graph = env.get_graph(state)

Score: planar ||p2(rel, a_ego)||_2 ≥ R on agent0.
Does not touch ACTIVE_MF_* / acceptance.
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "core" / "paths.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
from core.paths import ensure_import_path, project_root

_ROOT = project_root()
ensure_import_path()

from baseline.informarl.nets import GNNGaussianActor
from baseline.mappo_lag.nets import GaussianActor
from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from stage_mf.l2_labels import clearance_batch, nearest_rels_all_agents
from stage_mf.property_planar_l2 import claim_physics_report, default_planar_netdep_claim
from stage_mf.residual import CorrectionMLP, apply_residual_mf, load_mf_base, load_mf_residual
from stage_mf.student import extract_meanfield_features


def main() -> None:
    out = pathlib.Path("baselines/compare/output/planar_reachable_env")
    out.mkdir(parents=True, exist_ok=True)
    SEED, N, R = 20261010, 256, 0.10
    rng = np.random.default_rng(SEED)

    box = default_planar_netdep_claim()
    phys = claim_physics_report(box)
    assert phys["feat_contract"]["goal_dv_equals_minus_v"]
    print("physics", {k: phys[k] for k in ("network_dependent", "envelope_any_a_safe")}, flush=True)
    print("feat_contract", phys["feat_contract"], flush=True)

    env = make_env("DoubleIntegrator", num_agents=8, num_obs=4, area_size=4.0, n_rays=32)
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()
    EnvState = type(env.reset(jr.PRNGKey(0)).env_states)

    agrid = np.array([[ax, ay] for ax in np.linspace(-1, 1, 11) for ay in np.linspace(-1, 1, 11)])

    def controllable(rel):
        rb = np.repeat(rel[None], len(agrid), 0)
        m, c = clearance_batch(rb, None, agrid)
        return bool(np.any(c | (m < 0))) and bool(np.any(~c & (m >= 0)))

    # sample coupled (x,rel) then build state from x,rel
    rels, xs = [], []
    guard = 0
    while len(rels) < N and guard < N * 50:
        guard += 1
        x, rel = box.sample(rng, 1)
        x, rel = x[0], rel[0]
        if controllable(rel):
            rels.append(rel)
            xs.append(x)
    rels = np.asarray(rels)
    xs = np.asarray(xs)
    print(f"sampled N={len(rels)}", flush=True)

    obs = env.create_obstacles(
        jnp.asarray([[12.0, 12.0], [12.0, -8.0], [-8.0, 12.0], [-8.0, -8.0]], dtype=jnp.float32),
        jnp.ones(4) * 0.2,
        jnp.ones(4) * 0.2,
        jnp.zeros(4),
    )

    def build_state(x_star, rel_star, rr):
        ego_p = np.clip(x_star[0:2].copy(), 0.8, 3.2)
        ego_v = np.clip(x_star[2:4].copy(), -float(hi[0]) if hasattr(hi, "__len__") else -1.0, 1.0)
        # prefer claim v bounds
        ego_v = np.clip(ego_v, -0.5, 0.5)
        g_xy = x_star[4:6].copy()
        gn = float(np.linalg.norm(g_xy))
        if gn > comm:
            g_xy = g_xy * (comm / gn)
        other_p = ego_p - rel_star[:2]
        other_v = ego_v - rel_star[2:4]
        agents = np.zeros((n_agents, 4), dtype=np.float32)
        goals = np.zeros((n_agents, 4), dtype=np.float32)
        agents[0, :2] = ego_p
        agents[0, 2:4] = ego_v
        agents[1, :2] = other_p
        agents[1, 2:4] = other_v
        goals[0, :2] = np.clip(ego_p + g_xy, 0.15, 3.85)
        # goal body vel = 0
        far = np.array(
            [[0.15, 0.15], [0.15, 3.85], [3.85, 0.15], [3.85, 3.85], [0.15, 2.0], [3.85, 2.0]],
            dtype=np.float64,
        )
        for i in range(2, n_agents):
            fp = far[i - 2].copy()
            if np.linalg.norm(fp - ego_p) < comm + 0.25:
                fp = np.clip(ego_p + np.array([comm + 0.4, comm + 0.4]), 0.1, 3.9)
            agents[i, :2] = fp
            goals[i, :2] = fp
        goals[1, :2] = np.clip(other_p + rr.uniform(-0.15, 0.15, 2), 0.2, 3.8)
        return EnvState(
            agent=jnp.asarray(agents, dtype=jnp.float32),
            goal=jnp.asarray(goals, dtype=jnp.float32),
            obstacle=obs,
        )

    print("building graphs…", flush=True)
    graphs, meta = [], []
    agents_all, goals_all = [], []
    for i in range(N):
        st = build_state(xs[i], rels[i], np.random.default_rng(SEED + 3000 + i))
        g = env.get_graph(st)
        graphs.append(g)
        agents_all.append(np.asarray(st.agent))
        goals_all.append(np.asarray(st.goal))
        hits = np.asarray(g.type_states(2, n_rays * n_agents)).reshape(n_agents, n_rays, -1)
        raa, haa, rao, hao = nearest_rels_all_agents(np.asarray(st.agent), hits, comm_radius=comm)
        rel_got = raa[0] if haa[0] else (rao[0] if hao[0] else rels[i])
        # feature match
        x_got = np.asarray(extract_meanfield_features(g, n_agents, n_rays, comm))[0]
        meta.append(
            {
                "rel_err": float(np.linalg.norm(rel_got - rels[i])),
                "feat_l2": float(np.linalg.norm(x_got - xs[i])),
                "feat_goal_err": float(np.linalg.norm(x_got[4:6] - xs[i][4:6])),
                "goal_dv_err": float(np.linalg.norm(x_got[6:8] + x_got[2:4])),
                "has_aa": bool(haa[0]),
            }
        )
        if (i + 1) % 64 == 0:
            print(
                f"  {i+1}/{N} rel_err={np.mean([m['rel_err'] for m in meta]):.2e} "
                f"feat_l2={np.mean([m['feat_l2'] for m in meta]):.3f}",
                flush=True,
            )

    print(
        f"rel_err mean={np.mean([m['rel_err'] for m in meta]):.2e} "
        f"feat_l2 mean={np.mean([m['feat_l2'] for m in meta]):.3f} "
        f"goal_dv_err max={max(m['goal_dv_err'] for m in meta):.2e}",
        flush=True,
    )
    np.savez_compressed(
        out / "constructed_states.npz",
        agents=np.asarray(agents_all),
        goals=np.asarray(goals_all),
        rels=rels,
        xs=xs,
        seed=np.array([SEED]),
    )

    def score(name, act_fn, max_g=None):
        n_g = N if max_g is None else min(max_g, N)
        a_all = []
        t0 = time.time()
        for i in range(n_g):
            acts = np.asarray(act_fn(graphs[i]), dtype=np.float64)
            a_all.append(acts[0])
            if (i + 1) % 64 == 0:
                print(f"  [{name}] {i+1}/{n_g}", flush=True)
        a_c = np.asarray(a_all)
        margin, coll = clearance_batch(rels[:n_g], None, a_c)
        safe = ~coll & (margin >= -1e-12)
        n_fail = int((~safe).sum())
        n_rep = 0
        fail_idx = np.where(~safe)[0]
        for j in fail_idx[: min(200, len(fail_idx))]:
            rb = np.repeat(rels[j][None], len(agrid), 0)
            mm, cc = clearance_batch(rb, None, agrid)
            if np.any(~cc & (mm >= 0)):
                n_rep += 1
        n_chk = min(200, len(fail_idx))
        row = dict(
            policy=name,
            n=n_g,
            n_fail=n_fail,
            fail_rate=n_fail / n_g,
            safe_rate=float(safe.mean()),
            margin_mean=float(margin.mean()),
            ax_mean=float(a_c[:, 0].mean()),
            repairable_frac=(n_rep / n_chk) if n_chk else None,
            fwd_s=round(time.time() - t0, 2),
        )
        print(f"[score] {name}: fail={100*row['fail_rate']:.2f}% ax={row['ax_mean']:.3f}", flush=True)
        return row

    def build_mf(path, residual=True):
        if residual:
            ck = load_mf_residual(path)
            bp = pathlib.Path(ck["base_checkpoint"])
            if not bp.exists():
                bp = _ROOT / "method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl"
            _, b, bpms = load_mf_base(bp)
            if ck.get("base_params") is not None:
                bpms = ck["base_params"]
            delta = CorrectionMLP(hidden=int(ck.get("correction_hidden", 128)), action_dim=2)
            dp = ck["delta_params"]

            @jax.jit
            def control(graph):
                x = extract_meanfield_features(graph, n_agents, n_rays, comm)
                a, _, _ = apply_residual_mf(
                    b,
                    bpms,
                    delta,
                    dp,
                    x,
                    open_mult=float(ck.get("open_mult", 3)),
                    close_mult=float(ck.get("close_mult", 1)),
                    inv_ttc_ref=float(ck.get("inv_ttc_ref", 5)),
                    gate_mix=float(ck.get("gate_mix", 0.5)),
                    action_lim=(lo, hi),
                    delta_clip=float(ck.get("delta_clip", 1)),
                )
                return a
        else:
            _, b, bpms = load_mf_base(path)

            @jax.jit
            def control(graph):
                x = extract_meanfield_features(graph, n_agents, n_rays, comm)
                return jnp.clip(b.apply(bpms, x), lo, hi)

        return lambda g: np.asarray(control(g))

    def build_mappo(path):
        with open(path, "rb") as f:
            ck = pickle.load(f)
        actor = GaussianActor(hidden=int(ck["hidden"]), action_dim=2)
        params = ck["actor_params"]
        mode = ck.get("action_mode", "uref_plus_tanh_residual")

        @jax.jit
        def control(graph):
            x = extract_meanfield_features(graph, n_agents, n_rays, comm)
            mean, _ = actor.apply(params, x)
            a = env.u_ref(graph) + jnp.tanh(mean) if mode == "uref_plus_tanh_residual" else jnp.tanh(mean)
            return jnp.clip(a, lo, hi)

        return lambda g: np.asarray(control(g))

    def build_gcbf(run_dir, step):
        with open(pathlib.Path(run_dir) / "config.yaml") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
        algo = make_algo(
            algo=config.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            gnn_layers=config.gnn_layers,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            horizon=config.horizon,
            lr_actor=config.lr_actor,
            lr_cbf=config.lr_cbf,
            alpha=config.alpha,
            eps=0.02,
            inner_epoch=8,
            loss_action_coef=config.loss_action_coef,
            loss_unsafe_coef=config.loss_unsafe_coef,
            loss_safe_coef=config.loss_safe_coef,
            loss_h_dot_coef=config.loss_h_dot_coef,
            max_grad_norm=2.0,
            seed=config.seed,
        )
        algo.load(str(pathlib.Path(run_dir) / "models"), step)
        act_j = jax.jit(algo.act)
        return lambda g: np.asarray(act_j(g))

    def build_informarl(path):
        with open(path, "rb") as f:
            ck = pickle.load(f)
        actor = GNNGaussianActor(
            n_agents=env.num_agents,
            action_dim=env.action_dim,
            gnn_layers=int(ck.get("gnn_layers", 2)),
        )
        params = ck["actor_params"]
        mode = ck.get("action_mode", "uref_plus_tanh_residual")

        @jax.jit
        def control(graph):
            mean, _ = actor.apply(params, graph)
            a = env.u_ref(graph) + jnp.tanh(mean) if mode == "uref_plus_tanh_residual" else mean
            return jnp.clip(a, lo, hi)

        return lambda g: np.asarray(control(g))

    def build_handcbf():
        algo = make_algo(
            algo="dec_share_cbf",
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            alpha=1.0,
        )
        act_j = jax.jit(algo.act)
        return lambda g: np.asarray(act_j(g))

    paths = dict(
        mf_base=pathlib.Path("method/meanfield/output/ACTIVE_MF_BASE.txt").read_text().strip(),
        mf_r2=pathlib.Path("method/meanfield/output/ACTIVE_MF_RESIDUAL.txt").read_text().strip(),
        netdep_r3="method/meanfield/output/repair/mf_residual_v5_netdep_r3_20261010/seed0/repair_manual_pick.pkl",
        mappo=pathlib.Path("baselines/mappo_lag/output/mappo_lag/ACTIVE.txt").read_text().strip(),
        info=pathlib.Path("baselines/informarl/output/informarl/ACTIVE.txt").read_text().strip(),
        gcbf="baselines/gcbfplus/log/gcbf+/seed0_20261001232611",
    )

    rows = []
    for name, b, mg in [
        ("coast_a0", lambda: (lambda g: np.zeros((n_agents, 2))), None),
        ("const_a=-1", lambda: (lambda g: np.tile([-1.0, 0.0], (n_agents, 1))), None),
        ("const_a=+1", lambda: (lambda g: np.tile([1.0, 0.0], (n_agents, 1))), None),
        ("mf_base", lambda: build_mf(paths["mf_base"], False), None),
        ("mf_r2", lambda: build_mf(paths["mf_r2"], True), None),
        ("netdep_r3", lambda: build_mf(paths["netdep_r3"], True), None),
        ("mappo_lag", lambda: build_mappo(paths["mappo"]), None),
        ("gcbfplus", lambda: build_gcbf(paths["gcbf"], 1000), None),
        ("informarl", lambda: build_informarl(paths["info"]), None),
        ("handcbf", build_handcbf, 128),
    ]:
        print("[build]", name, flush=True)
        rows.append(score(name, b(), max_g=mg))

    oa = np.zeros((N, 2))
    for i in range(N):
        rb = np.repeat(rels[i][None], len(agrid), 0)
        m, c = clearance_batch(rb, None, agrid)
        oa[i] = agrid[int(np.argmax(np.where(c, -1e9, m)))]
    m, c = clearance_batch(rels, None, oa)
    safe = ~c & (m >= -1e-12)
    rows.append(
        dict(
            policy="oracle_11x11",
            n=N,
            n_fail=int((~safe).sum()),
            fail_rate=float((~safe).mean()),
            safe_rate=float(safe.mean()),
            margin_mean=float(m.mean()),
            ax_mean=float(oa[:, 0].mean()),
            repairable_frac=None,
        )
    )

    rep = dict(
        title="planar_reachable_v1 box→env shared baselines",
        N=N,
        seed=SEED,
        physics=phys,
        construction="reachable claim sample → EnvState (goal_v=0, goal_xy from x, threat=ego-rel)",
        meta_summary={
            "rel_err_mean": float(np.mean([m["rel_err"] for m in meta])),
            "feat_l2_mean": float(np.mean([m["feat_l2"] for m in meta])),
            "goal_dv_err_max": float(max(m["goal_dv_err"] for m in meta)),
        },
        rows=rows,
    )
    (out / "REPORT.json").write_text(json.dumps(rep, indent=2) + "\n")
    md = [
        "# Planar reachable_v1 · box→env · all baselines",
        "",
        f"- N={N}, seed={SEED}",
        f"- feat_contract: goal_xy≤comm, goal_dv=-v",
        f"- rel_err mean={rep['meta_summary']['rel_err_mean']:.2e}, feat_l2 mean={rep['meta_summary']['feat_l2_mean']:.3f}",
        "",
        "| policy | n | planar fail% | ax_mean | repairable |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        repf = f"{100*r['repairable_frac']:.0f}%" if r.get("repairable_frac") is not None else "—"
        md.append(
            f"| `{r['policy']}` | {r['n']} | {100*r['fail_rate']:.2f}% | {r['ax_mean']:.3f} | {repf} |"
        )
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n" + (out / "REPORT.md").read_text())
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
