"""Fast MF two-step L2 CE collection from short rollouts (vectorized).

Design:
  - JIT env rollout (GPU) for a batch of episodes
  - Per timestep: numpy vectorized nearest-rel + two-step check for all agents
  - Grid min-intervention only on hits, batched
  - Progress logs every episode batch

Typical: 32 epi baseline in ~1–3 min (not 40+).
"""

from __future__ import annotations

import argparse
import json
import pathlib
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
from gcbfplus.utils.utils import jax_jit_np
from stage_mf.l2_labels import (
    clearance_batch,
    min_intervention_batch,
    nearest_rels_all_agents,
)
from stage_mf.residual import (
    CorrectionMLP,
    apply_residual_mf,
    load_mf_base,
    load_mf_residual,
)
from stage_mf.student import (
    FEATURE_VERSION,
    INPUT_DIM,
    MeanFieldStudent,
    extract_meanfield_features,
)
from stage2.dynamics import COLLISION_AA, COLLISION_AO


def _policy_from_ckpt(ckpt_path: pathlib.Path, env, *, residual: bool):
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    lo, hi = env.action_lim()

    if residual:
        rck = load_mf_residual(ckpt_path)
        base = MeanFieldStudent(hidden=int(rck["hidden"]), action_dim=2)
        delta = CorrectionMLP(hidden=int(rck["correction_hidden"]), action_dim=2)
        bparams, dparams = rck["base_params"], rck["delta_params"]

        def policy(graph):
            x = extract_meanfield_features(graph, n_agents, n_rays, comm)
            a, _, _ = apply_residual_mf(
                base,
                bparams,
                delta,
                dparams,
                x,
                open_mult=float(rck["open_mult"]),
                close_mult=float(rck["close_mult"]),
                inv_ttc_ref=float(rck.get("inv_ttc_ref", 5.0)),
                gate_mix=float(rck.get("gate_mix", 0.5)),
                action_lim=(lo, hi),
                delta_clip=float(rck.get("delta_clip", 1.0)),
            )
            return a

        meta = {"kind": "residual", "path": str(ckpt_path)}
    else:
        ckpt, base, params = load_mf_base(ckpt_path)

        def policy(graph):
            x = extract_meanfield_features(graph, n_agents, n_rays, comm)
            return jnp.clip(base.apply(params, x), lo, hi)

        meta = {"kind": "base", "feature_version": int(ckpt.get("feature_version", -1)), "path": str(ckpt_path)}
    return policy, meta


def _process_timestep(
    agents: np.ndarray,
    hits: np.ndarray,
    x_all: np.ndarray,
    actions: np.ndarray,
    *,
    comm: float,
    danger_radius: float,
    ep: int,
    t: int,
) -> list[dict]:
    """Vectorized CE extraction for one graph timestep. agents[A,4], hits[A,R,*], x[A,D], a[A,2]."""
    n = agents.shape[0]
    rel_aa, has_aa, rel_ao, has_ao = nearest_rels_all_agents(agents, hits, comm_radius=comm)
    d_aa = np.linalg.norm(rel_aa[:, :2], axis=-1)
    d_ao = np.linalg.norm(rel_ao[:, :2], axis=-1)
    d_aa = np.where(has_aa, d_aa, 1e6)
    d_ao = np.where(has_ao, d_ao, 1e6)
    d_min = np.minimum(d_aa, d_ao)
    in_danger = (d_min <= danger_radius) & (has_aa | has_ao)
    if not np.any(in_danger):
        return [], 0, 0

    idx = np.where(in_danger)[0]
    n_danger = int(len(idx))
    raa = rel_aa[idx]
    rao = rel_ao[idx]
    use_aa = has_aa[idx]
    use_ao = has_ao[idx]
    a = np.asarray(actions[idx], dtype=np.float64)
    far = np.array([10.0, 10.0, 0.0, 0.0], dtype=np.float64)
    # missing threat → far (never collides under AA/AO radii)
    raa_u = np.where(use_aa[:, None], raa, far[None, :])
    rao_u = np.where(use_ao[:, None], rao, far[None, :])

    _, coll = clearance_batch(raa_u, rao_u, a)
    _, coll_aa = clearance_batch(raa_u, None, a)
    _, coll_ao = clearance_batch(None, rao_u, a)
    coll_aa &= use_aa
    coll_ao &= use_ao
    coll = coll_aa | coll_ao

    hit_local = np.where(coll)[0]
    n_hit = int(len(hit_local))
    if n_hit == 0:
        return [], n_danger, 0

    hi = hit_local
    a_hit = a[hi]
    a_safe, clr, ok, da = min_intervention_batch(raa_u[hi], rao_u[hi], a_hit)

    ces = []
    for j, s in enumerate(hi):
        ii = int(idx[s])
        if coll_aa[s] and coll_ao[s]:
            tag = "both"
        elif coll_aa[s]:
            tag = "aa"
        elif coll_ao[s]:
            tag = "ao"
        else:
            tag = "other"
        ces.append(
            {
                "x": x_all[ii].tolist(),
                "a": a_hit[j].tolist(),
                "a_l2_safe": a_safe[j].tolist(),
                "a_l2_safe_dist": float(clr[j]),
                "a_l2_label_ok": bool(ok[j]),
                "a_l2_label_mode": "min_intervention",
                "a_l2_delta": float(da[j]),
                "kind": tag,
                "d_aa0": float(d_aa[ii]) if has_aa[ii] else None,
                "d_ao0": float(d_ao[ii]) if has_ao[ii] else None,
                "episode": int(ep),
                "t": int(t),
                "agent": ii,
            }
        )
    return ces, n_danger, n_hit


def collect_ces(
    checkpoint: str | pathlib.Path,
    *,
    residual: bool = False,
    episodes: int = 32,
    seed: int = 123,
    danger_radius: float = 0.35,
    max_keep: int = 4000,
    obs: int = 4,
    area_size: float = 4.0,
    num_agents: int = 8,
    stride: int = 1,
) -> dict:
    checkpoint = pathlib.Path(checkpoint)
    env = make_env(
        "DoubleIntegrator",
        num_agents=num_agents,
        area_size=area_size,
        num_obs=obs,
        n_rays=32,
    )
    n_agents = env.num_agents
    n_rays = int(env.params["n_rays"])
    comm = float(env.params["comm_radius"])
    policy, meta = _policy_from_ckpt(checkpoint, env, residual=residual)
    rollout_fn = jax_jit_np(env.rollout_fn(policy, env.max_episode_steps))

    ces: list[dict] = []
    n_steps = 0
    n_danger = 0
    n_hit = 0
    kind_counts = {"aa": 0, "ao": 0, "both": 0, "other": 0}
    t0 = time.time()
    keys = jr.split(jr.PRNGKey(seed), episodes)

    print(
        f"[mf-ce] start episodes={episodes} danger_r={danger_radius} stride={stride} "
        f"ckpt={checkpoint.name}",
        flush=True,
    )

    for ep_i, key in enumerate(keys):
        if len(ces) >= max_keep:
            break
        rollout = rollout_fn(key)
        actions = np.asarray(rollout.T_action)  # [T,A,2]
        T = actions.shape[0]
        graphs = rollout.Tp1_graph
        t_list = range(0, T, max(stride, 1))
        for t in t_list:
            if len(ces) >= max_keep:
                break
            g = jax.tree_util.tree_map(lambda x, _t=t: np.asarray(x[_t]), graphs)
            # type_states may need jax graph — use original jax slice
            g_jax = jax.tree_util.tree_map(lambda x, _t=t: x[_t], graphs)
            agents = np.asarray(g_jax.type_states(type_idx=0, n_type=n_agents))
            hits = np.asarray(
                g_jax.type_states(type_idx=2, n_type=n_rays * n_agents)
            ).reshape(n_agents, n_rays, -1)
            x_all = np.asarray(extract_meanfield_features(g_jax, n_agents, n_rays, comm))
            n_steps += n_agents
            batch_ces, nd, nh = _process_timestep(
                agents,
                hits,
                x_all,
                actions[t],
                comm=comm,
                danger_radius=danger_radius,
                ep=ep_i,
                t=t,
            )
            n_danger += nd
            n_hit += nh
            for e in batch_ces:
                kind_counts[e["kind"]] = kind_counts.get(e["kind"], 0) + 1
                ces.append(e)
                if len(ces) >= max_keep:
                    break
        if (ep_i + 1) % 4 == 0 or ep_i == 0:
            print(
                f"[mf-ce] ep {ep_i+1}/{episodes} kept={len(ces)} "
                f"hit={n_hit} danger={n_danger} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    report = {
        "track": "meanfield",
        "domain": "rollout_danger_two_step_L2",
        "feature_version": FEATURE_VERSION,
        "input_dim": INPUT_DIM,
        "checkpoint": str(checkpoint),
        "policy_meta": meta,
        "episodes": int(episodes),
        "seed": int(seed),
        "danger_radius": float(danger_radius),
        "stride": int(stride),
        "collision_aa": COLLISION_AA,
        "collision_ao": COLLISION_AO,
        "n_agent_steps": int(n_steps),
        "n_danger": int(n_danger),
        "n_hit_two_step": int(n_hit),
        "hit_rate_on_danger": float(n_hit / max(n_danger, 1)),
        "n_kept": len(ces),
        "n_labels_safe": int(sum(1 for c in ces if c["a_l2_label_ok"])),
        "kind_counts": kind_counts,
        "wall_time_s": round(time.time() - t0, 3),
        "counterexamples": ces,
        "claim": (
            "Empirical two-step L2 CEs (vectorized). Neighbor coasts; hits static. "
            "Not formal verification."
        ),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--danger-radius", type=float, default=0.35)
    p.add_argument("--max-keep", type=int, default=4000)
    p.add_argument("--stride", type=int, default=2, help="subsample timesteps (2=faster)")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    report = collect_ces(
        args.checkpoint,
        residual=args.residual,
        episodes=args.episodes,
        seed=args.seed,
        danger_radius=args.danger_radius,
        max_keep=args.max_keep,
        stride=args.stride,
    )
    out = pathlib.Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in report.items() if k != "counterexamples"}
    print(json.dumps(summary, indent=2), flush=True)
    out.write_text(json.dumps(report) + "\n")
    print(
        f"[mf-ce] DONE kept={report['n_kept']} "
        f"hit_on_danger={report['hit_rate_on_danger']*100:.2f}% "
        f"wall={report['wall_time_s']}s → {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
