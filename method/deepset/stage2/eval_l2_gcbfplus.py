"""Same-standard L2 preventable-D sampling for official GCBF+ GNN teacher.

Compares GCBF+ Actor (and optionally DeepSet / residual) on the *same* claim:

    D = initially safe AND exists crashing a AND exists avoiding a
    hit = two-step held-action collision under policy π

Protocol matches residual/DeepSet reports:
  - buckets: l2_aa_near, l2_ao_near, l2_11_near
  - domain filter: in_preventable_domain_x
  - dynamics: two Euler steps, neighbor coasts / hit static
  - collision radii: AA=0.10, AO=0.05

Embedding note (important claim boundary):
  L2 boxes are *local* (ego + k neigh + m hit). GCBF+ needs a full 8-agent
  graph with lidar obstacles. We embed each local x into a full EnvState by:
    - ego = agent 0 from box ego state
    - AA neighbor = agent 1 from relative cont (ego - other)
    - remaining agents placed far outside comm_radius of ego
    - AO hit approximated by a tiny rectangle at the hit point so lidar
      returns a nearby hit (AO proxy matches DeepSet hit-ball as closely
      as geometry allows)
  This is the fairest practical apples-to-apples sampling on D; it is still
  *not* a formal proof of the GNN, and AO embedding is an approximation.

Empirical only — not Marabou/IBP proven.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
import numpy as np



from gcbfplus.env import make_env
from gcbfplus.env.obstacle import Rectangle
from stage1.train import make_teacher
from stage2.bounds import default_l2_buckets
from stage2.encode import clip_action, load_weights, unpack_inputs
from stage2.l2_domain import in_preventable_domain_x
from stage2.l2_query import (
    action_at,
    eval_collision_dual,
    next_rels_for_spec,
    rels_from_x,
)


def _simple_ns(**kwargs):
    return argparse.Namespace(**kwargs)


def load_gcbf_teacher(
    *,
    teacher_run: str,
    teacher_step: int,
    num_agents: int = 8,
    area_size: float = 4.0,
    obs: int = 4,
    n_rays: int = 32,
):
    env = make_env(
        "DoubleIntegrator",
        num_agents=num_agents,
        area_size=area_size,
        num_obs=obs,
        n_rays=n_rays,
    )
    args = _simple_ns(
        teacher_run=teacher_run,
        teacher_step=teacher_step,
        gnn_layers=1,
        buffer_size=256,
        horizon=32,
        lr_actor=3e-5,
        lr_cbf=3e-5,
        alpha=1.0,
        seed=0,
    )
    teacher = make_teacher(args, env)
    return env, teacher


def _far_agent_pose(i: int, ego_xy: np.ndarray, area: float = 4.0) -> np.ndarray:
    """Place filler agents outside ego comm range, still in-ish the map."""
    # Corners / grid far from typical ego center; push away if too close.
    candidates = np.array(
        [
            [0.2, 0.2],
            [0.2, 3.8],
            [3.8, 0.2],
            [3.8, 3.8],
            [0.2, 2.0],
            [3.8, 2.0],
            [2.0, 0.2],
            [2.0, 3.8],
        ],
        dtype=np.float64,
    )
    p = candidates[i % len(candidates)].copy()
    if np.linalg.norm(p - ego_xy) < 1.0:
        # flip to opposite corner
        p = np.array([area - p[0], area - p[1]], dtype=np.float64)
    return np.array([p[0], p[1], 0.0, 0.0], dtype=np.float64)


def bucket_x_to_env_state(env, x: np.ndarray, spec) -> object:
    """Embed L2 local vector x into a full DoubleIntegrator EnvState.

    Relative convention (DeepSet / extract_local_sets):
        cont = s_ego - s_other  =>  s_other = s_ego - cont
    """
    ego, agents_c, obs_c = unpack_inputs(x, spec.k, spec.m)
    px, py, vx, vy = map(float, ego[:4])
    gdx, gdy, gdvx, gdvy = map(float, ego[4:8])
    ego_state = np.array([px, py, vx, vy], dtype=np.float64)
    goal0 = np.array([px + gdx, py + gdy, vx + gdvx, vy + gdvy], dtype=np.float64)

    n = env.num_agents
    agent = np.zeros((n, 4), dtype=np.float64)
    goal = np.zeros((n, 4), dtype=np.float64)
    agent[0] = ego_state
    goal[0] = goal0

    next_idx = 1
    if spec.k >= 1 and agents_c:
        cont = np.asarray(agents_c[0], dtype=np.float64)
        agent[1] = ego_state - cont
        # neighbor goal: park nearby so u_ref not wild, still far from collision logic
        goal[1] = agent[1] + np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float64)
        next_idx = 2

    for i in range(next_idx, n):
        agent[i] = _far_agent_pose(i, ego_state[:2], env.area_size)
        goal[i] = agent[i] + np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float64)

    # Obstacles: always provide env.params['n_obs'] rectangles.
    n_obs = int(env.params["n_obs"])
    centers = []
    widths = []
    heights = []
    thetas = []
    if spec.m >= 1 and obs_c:
        # Hit point in world: hit = ego_pos - cont_xy  (cont = ego - hit)
        cont = np.asarray(obs_c[0], dtype=np.float64)
        hit_xy = ego_state[:2] - cont[:2]
        # Tiny square so lidar can return a point near the hit.
        centers.append(hit_xy)
        widths.append(0.04)
        heights.append(0.04)
        thetas.append(0.0)
        start = 1
    else:
        start = 0
    # Filler obstacles far from ego (and outside typical lidar range).
    far = [
        np.array([0.15, 0.15]),
        np.array([0.15, 3.85]),
        np.array([3.85, 0.15]),
        np.array([3.85, 3.85]),
        np.array([0.15, 2.0]),
        np.array([3.85, 2.0]),
        np.array([2.0, 0.15]),
        np.array([2.0, 3.85]),
    ]
    fi = 0
    while len(centers) < n_obs:
        c = far[fi % len(far)].copy()
        if np.linalg.norm(c - ego_state[:2]) < 1.2:
            c = np.array([env.area_size - c[0], env.area_size - c[1]])
        centers.append(c)
        widths.append(0.15)
        heights.append(0.15)
        thetas.append(0.0)
        fi += 1

    centers_a = jnp.asarray(np.stack(centers[:n_obs]), dtype=jnp.float32)
    widths_a = jnp.asarray(np.asarray(widths[:n_obs], dtype=np.float32))
    heights_a = jnp.asarray(np.asarray(heights[:n_obs], dtype=np.float32))
    thetas_a = jnp.asarray(np.asarray(thetas[:n_obs], dtype=np.float32))
    obstacles = env.create_obstacles(centers_a, widths_a, heights_a, thetas_a)

    return env.EnvState(
        agent=jnp.asarray(agent, dtype=jnp.float32),
        goal=jnp.asarray(goal, dtype=jnp.float32),
        obstacle=obstacles,
    )


def gcbf_action_ego(env, teacher, x: np.ndarray, spec) -> np.ndarray:
    """Return clipped ego (agent-0) action from GCBF+ on embedded graph."""
    state = bucket_x_to_env_state(env, x, spec)
    graph = env.get_graph(state)
    acts = teacher.act(graph, teacher.actor_params)
    a0 = np.asarray(acts[0], dtype=np.float64)
    return clip_action(a0)


def sample_hit_rate(
    action_fn,
    specs,
    *,
    n_samples: int,
    seed: int,
    max_draw_factor: int = 40,
) -> dict:
    """action_fn(x, spec) -> a in R^2 clipped."""
    out = {}
    for bi, spec in enumerate(specs):
        rng = np.random.default_rng(seed + bi * 97)
        n_eval = n_hit = n_draw = n_skip = 0
        min_dist = 1e9
        gate_like = []  # unused; placeholder for symmetry
        budget = max(n_samples * max_draw_factor, 20000)
        t0 = time.time()
        while n_eval < n_samples and n_draw < budget:
            x = spec.box.sample(rng)
            n_draw += 1
            if not in_preventable_domain_x(x, spec, grid_n=7):
                n_skip += 1
                continue
            n_eval += 1
            a = action_fn(x, spec)
            rel_aa0, rel_ao0 = rels_from_x(x, spec)
            rel_aan, rel_aon = next_rels_for_spec(rel_aa0, rel_ao0, a, spec)
            ev = eval_collision_dual(rel_aan, rel_aon, spec)
            if ev.dist is not None:
                min_dist = min(min_dist, float(ev.dist))
            if ev.collided:
                n_hit += 1
        out[spec.name] = {
            "n_eval_on_D": n_eval,
            "n_draw": n_draw,
            "n_skip_out_of_D": n_skip,
            "n_hit": n_hit,
            "hit_rate_on_D": n_hit / max(n_eval, 1),
            "min_distn": None if min_dist >= 1e8 else float(min_dist),
            "seconds": round(time.time() - t0, 2),
        }
        print(
            f"  {spec.name}: in_D={n_eval}/{n_draw} hits={n_hit} "
            f"rate={out[spec.name]['hit_rate_on_D']*100:.2f}% "
            f"min_dist={out[spec.name]['min_distn']} ({out[spec.name]['seconds']}s)"
        )
    return out


def make_deepset_fn(checkpoint: str):
    weights, meta = load_weights(checkpoint)

    def fn(x, spec):
        return action_at(weights, x, spec)

    return fn, meta


def make_residual_fn(checkpoint: str):
    from stage3.retrain_l2_safe import bucket_x_to_student_inputs
    from stage3.residual_policy import apply_residual, load_residual_checkpoint

    ckpt, modules, base_params, train_params = load_residual_checkpoint(checkpoint)
    gate_mode = str(ckpt.get("gate_mode", "hybrid"))
    open_mult = float(ckpt.get("open_mult", 2.5))
    close_mult = float(ckpt.get("close_mult", 1.0))

    def fn(x, spec):
        ego_np, ent_np = bucket_x_to_student_inputs(np.asarray(x), spec.k, spec.m)
        a, _g, _d = apply_residual(
            modules,
            base_params,
            train_params,
            jnp.asarray(ego_np[None, ...]),
            jnp.asarray(ent_np[None, ...]),
            gate_mode=gate_mode,
            open_mult=open_mult,
            close_mult=close_mult,
        )
        return np.asarray(a[0], dtype=np.float64)

    return fn, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="L2-D hit-rate: GCBF+ vs DeepSet/residual")
    parser.add_argument(
        "--teacher-run",
        default="baselines/gcbfplus/log/gcbf+/seed0_20261001232611",
    )
    parser.add_argument("--teacher-step", type=int, default=1000)
    parser.add_argument(
        "--deepset",
        default="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl",
    )
    parser.add_argument(
        "--residual",
        default="method/deepset/output/stage3/repair/deepset_residual_gate_v2_best.pkl",
    )
    parser.add_argument("--r1", default="method/deepset/output/stage3/repair/deepset_l2_safe_repair.pkl")
    parser.add_argument("--buckets", default="l2_aa_near,l2_ao_near,l2_11_near")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-gcbf", action="store_true")
    parser.add_argument("--skip-deepset", action="store_true")
    parser.add_argument("--skip-residual", action="store_true")
    parser.add_argument("--skip-r1", action="store_true")
    parser.add_argument(
        "--out",
        default="method/deepset/output/stage2/l2_gcbfplus_vs_student_sampling_report.json",
    )
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--obs", type=int, default=4)
    parser.add_argument("--n-rays", type=int, default=32)
    args = parser.parse_args()

    want = {b.strip() for b in args.buckets.split(",") if b.strip()}
    specs = [s for s in default_l2_buckets() if s.name in want]
    if not specs:
        raise SystemExit(f"no specs in {want}")

    report = {
        "protocol": {
            "domain": "preventable_D",
            "horizon": "two Euler steps held action",
            "buckets": [s.name for s in specs],
            "samples_target_per_bucket": args.samples,
            "seed": args.seed,
            "claim": (
                "Sampling hit-rate on D. NOT formal proof. "
                "GCBF+ uses full-graph embedding of local L2 boxes "
                "(far filler agents; AO via tiny rectangle lidar hit)."
            ),
        },
        "policies": {},
    }

    if not args.skip_gcbf:
        print("=== GCBF+ teacher on L2-D ===")
        env, teacher = load_gcbf_teacher(
            teacher_run=args.teacher_run,
            teacher_step=args.teacher_step,
            num_agents=args.num_agents,
            obs=args.obs,
            n_rays=args.n_rays,
        )
        # JIT graph+act path roughly via closure
        def gcbf_fn(x, spec, _env=env, _teacher=teacher):
            return gcbf_action_ego(_env, _teacher, x, spec)

        hit = sample_hit_rate(gcbf_fn, specs, n_samples=args.samples, seed=args.seed)
        report["policies"]["gcbfplus_teacher"] = {
            "teacher_run": args.teacher_run,
            "teacher_step": args.teacher_step,
            "architecture": "gcbf+_gnn_actor",
            "embedding": "local_L2_box_into_8agent_graph",
            "buckets": hit,
        }

    if not args.skip_deepset and Path(args.deepset).exists():
        print("=== frozen DeepSet on L2-D ===")
        fn, meta = make_deepset_fn(args.deepset)
        hit = sample_hit_rate(fn, specs, n_samples=args.samples, seed=args.seed)
        report["policies"]["deepset_frozen"] = {
            "checkpoint": args.deepset,
            "architecture": meta.get("architecture", "deepset"),
            "buckets": hit,
        }

    if not args.skip_residual and Path(args.residual).exists():
        print("=== residual v2 best on L2-D ===")
        fn, meta = make_residual_fn(args.residual)
        hit = sample_hit_rate(fn, specs, n_samples=args.samples, seed=args.seed)
        report["policies"]["residual_v2_best"] = {
            "checkpoint": args.residual,
            "architecture": meta.get("architecture"),
            "buckets": hit,
        }

    if not args.skip_r1 and Path(args.r1).exists():
        print("=== r1 L2-safe repair on L2-D ===")
        fn, meta = make_deepset_fn(args.r1)
        hit = sample_hit_rate(fn, specs, n_samples=args.samples, seed=args.seed)
        report["policies"]["r1_l2_safe"] = {
            "checkpoint": args.r1,
            "architecture": meta.get("architecture", "deepset"),
            "buckets": hit,
        }

    # compact table
    print("\n=== summary hit_rate_on_D (%) ===")
    header = f"{'policy':<22}" + "".join(f"{s.name[3:]:>14}" for s in specs)
    print(header)
    for name, block in report["policies"].items():
        row = f"{name:<22}"
        for s in specs:
            b = block["buckets"].get(s.name, {})
            row += f"{100.0 * b.get('hit_rate_on_D', float('nan')):>13.2f}%"
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
