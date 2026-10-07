#!/usr/bin/env python3
"""Accumulative Stage1 compare: GCBF+ (GNN) vs DeepSet bases.

Writes JSON rows under a compare dir so later seeds/episode budgets can be
appended and plotted together.

Examples:
  # protocol 32-epi + optional 64/128, task only
  python scripts/eval_stage1_compare.py \\
    --compare-dir method/deepset/output/stage1/compare/base_gcbf_vs_seed0 \\
    --episodes 32,64

  # also same-std D-hit
  python scripts/eval_stage1_compare.py ... --with-d-hit

  # plot later (after more rows exist)
  python scripts/plot_stage1_compare.py --compare-dir method/deepset/output/stage1/compare/base_gcbf_vs_seed0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
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



PY = os.environ.get("PY", sys.executable)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_gcbf_log(text: str) -> dict:
    """Parse baselines/gcbfplus/code/test_official.py final summary line."""
    # Final: reward: x ± y, cost: ..., safe_rate: a%, finish_rate: b%, success_rate: c%
    out: dict = {}
    # prefer last success_rate line that looks like aggregate
    m = re.search(
        r"reward:\s*([-\d.]+)\s*±\s*([-\d.]+).*?"
        r"cost:\s*([-\d.]+)\s*±\s*([-\d.]+).*?"
        r"safe_rate:\s*([-\d.]+)%.*?"
        r"finish_rate:\s*([-\d.]+)%.*?"
        r"success_rate:\s*([-\d.]+)%",
        text,
        flags=re.S,
    )
    if m:
        out = {
            "reward": float(m.group(1)),
            "reward_std": float(m.group(2)),
            "cost": float(m.group(3)),
            "cost_std": float(m.group(4)),
            "safe_rate": float(m.group(5)) / 100.0,
            "finish_rate": float(m.group(6)) / 100.0,
            "success_rate": float(m.group(7)) / 100.0,
        }
        return out
    # fallback: mean of per-epi printed rates
    safes = [float(x) / 100 for x in re.findall(r"safe rate:\s*([-\d.]+)%", text)]
    fins = [float(x) / 100 for x in re.findall(r"finish rate:\s*([-\d.]+)%", text)]
    succs = [float(x) / 100 for x in re.findall(r"success rate:\s*([-\d.]+)%", text)]
    rews = [float(x) for x in re.findall(r"reward:\s*([-\d.]+),", text)]
    costs = [float(x) for x in re.findall(r"cost:\s*([-\d.]+),", text)]
    if succs:
        import numpy as np

        out = {
            "reward": float(np.mean(rews)) if rews else None,
            "cost": float(np.mean(costs)) if costs else None,
            "safe_rate": float(np.mean(safes)) if safes else None,
            "finish_rate": float(np.mean(fins)) if fins else None,
            "success_rate": float(np.mean(succs)),
            "parse": "per_epi_mean_fallback",
        }
    return out


def eval_gcbf(
    *,
    teacher_run: str,
    teacher_step: int,
    episodes: int,
    seed: int,
    area_size: float,
    obs: int,
    log_path: Path,
) -> dict:
    cmd = [
        PY,
        str(_ROOT / "baselines/gcbfplus/code/validate.py"),
        "--path",
        teacher_run,
        "--env",
        "DoubleIntegrator",
        "--area-size",
        str(area_size),
        "--obs",
        str(obs),
        "--n-rays",
        "32",
        "--epi",
        str(episodes),
        "--seed",
        str(seed),
        "--step",
        str(teacher_step),
        "--no-video",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"GCBF+ validate failed rc={proc.returncode}; see {log_path}")
    parsed = _parse_gcbf_log(proc.stdout)
    if not parsed.get("success_rate") and parsed.get("success_rate") != 0.0:
        raise RuntimeError(f"failed to parse GCBF+ metrics from {log_path}")
    n_agents = 8
    succ = float(parsed["success_rate"])
    safe = float(parsed["safe_rate"])
    n_trials = episodes * n_agents
    import math

    return {
        "policy_id": "gcbfplus_teacher",
        "policy_kind": "gcbfplus_gnn",
        "label": "GCBF+ teacher (GNN)",
        "checkpoint": teacher_run,
        "teacher_step": teacher_step,
        "episodes": episodes,
        "eval_seed": seed,
        "reward": parsed.get("reward"),
        "reward_std": parsed.get("reward_std"),
        "cost": parsed.get("cost"),
        "cost_std": parsed.get("cost_std"),
        "safe_rate": safe,
        "finish_rate": float(parsed["finish_rate"]),
        "success_rate": succ,
        "n_agent_trials": n_trials,
        "success_se_agent": math.sqrt(max(succ * (1 - succ), 1e-12) / n_trials),
        "safe_se_agent": math.sqrt(max(safe * (1 - safe), 1e-12) / n_trials),
        "wall_time_s": round(time.time() - t0, 3),
        "log": str(log_path),
        "claim": "Empirical rollout only. Not formal verification.",
    }


def eval_deepset(
    *,
    policy_id: str,
    label: str,
    checkpoint: str,
    episodes: int,
    seed: int,
    out_json: Path,
) -> dict:
    from stage1.validate import evaluate_deepset_checkpoint

    m = evaluate_deepset_checkpoint(checkpoint, episodes=episodes, seed=seed)
    m.update(
        {
            "policy_id": policy_id,
            "label": label,
        }
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(m, indent=2))
    return m


def run_d_hit(
    *,
    teacher_run: str,
    teacher_step: int,
    deepset_ckpt: str | None,
    frozen_ckpt: str | None,
    out_path: Path,
    samples: int,
    seed: int,
) -> dict:
    cmd = [
        PY,
        str(_ROOT / "stage2/eval_l2_gcbfplus.py"),
        "--teacher-run",
        teacher_run,
        "--teacher-step",
        str(teacher_step),
        "--buckets",
        "l2_aa_near,l2_ao_near,l2_11_near",
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--skip-residual",
        "--skip-r1",
        "--out",
        str(out_path),
    ]
    if deepset_ckpt:
        cmd += ["--deepset", deepset_ckpt]
    else:
        cmd += ["--skip-deepset"]
    # frozen as r1 slot reuse? better run twice or only one deepset.
    # Script only has one --deepset; we call separately if needed.
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    (out_path.parent / (out_path.stem + "_run.log")).write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr
    )
    if proc.returncode != 0:
        raise RuntimeError(f"D-hit failed; log {out_path.stem}_run.log")
    return json.loads(out_path.read_text())


def append_row(compare_dir: Path, row: dict) -> Path:
    compare_dir.mkdir(parents=True, exist_ok=True)
    rows_path = compare_dir / "rows.jsonl"
    stamp = _utc()
    row = dict(row)
    row["recorded_at"] = stamp
    with rows_path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    # also dump individual
    pid = row.get("policy_id", "pol")
    epi = row.get("episodes", 0)
    kind = row.get("metric_kind", "task")
    single = compare_dir / "raw" / f"{pid}_epi{epi}_{kind}_{stamp}.json"
    single.parent.mkdir(parents=True, exist_ok=True)
    single.write_text(json.dumps(row, indent=2, default=str))
    return rows_path


def write_snapshot_table(compare_dir: Path) -> None:
    rows_path = compare_dir / "rows.jsonl"
    if not rows_path.exists():
        return
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    task = [r for r in rows if r.get("metric_kind", "task") == "task"]
    lines = [
        f"# Stage1 compare snapshot — `{compare_dir.name}`",
        "",
        f"Updated: {_utc()}",
        "",
        "## Task / empirical safety",
        "",
        "| policy | epi | seed | success% | ±SE | safe% | finish% | reward | cost | when |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in sorted(task, key=lambda x: (x.get("policy_id", ""), x.get("episodes", 0))):
        se = r.get("success_se_agent")
        se_s = f"{se*100:.2f}" if se is not None else "—"
        lines.append(
            f"| {r.get('label') or r.get('policy_id')} | {r.get('episodes')} | {r.get('eval_seed')} | "
            f"{(r.get('success_rate') or 0)*100:.2f} | {se_s} | "
            f"{(r.get('safe_rate') or 0)*100:.2f} | {(r.get('finish_rate') or 0)*100:.2f} | "
            f"{r.get('reward')} | {r.get('cost')} | {r.get('recorded_at')} |"
        )
    lines += [
        "",
        "SE is **approximate agent-level binomial SE** (agents in one epi are correlated → optimistic).",
        "Prefer comparing **same episodes + same eval_seed**. Larger epi mainly shrinks Monte-Carlo noise.",
        "",
        f"Machine-readable: `{rows_path}`",
        "",
    ]
    (compare_dir / "SNAPSHOT.md").write_text("\n".join(lines))
    (compare_dir / "latest_task.json").write_text(json.dumps(task, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare-dir",
        default=None,
        help="method/deepset/output/stage1/compare/<id> (default: timestamped)",
    )
    parser.add_argument(
        "--teacher-run",
        default="baselines/gcbfplus/log/gcbf+/seed0_20261001232611",
    )
    parser.add_argument("--teacher-step", type=int, default=1000)
    parser.add_argument(
        "--seed0",
        default=(
            "method/deepset/output/stage1/redistill/stage1_redistill_20261004_183322/"
            "seed0/deepset_actor512_student_best.pkl"
        ),
    )
    parser.add_argument(
        "--frozen",
        default="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl",
    )
    parser.add_argument("--skip-frozen", action="store_true")
    parser.add_argument("--skip-gcbf", action="store_true")
    parser.add_argument("--skip-seed0", action="store_true")
    parser.add_argument(
        "--episodes",
        default="32",
        help="comma list, e.g. 32 or 32,64,128",
    )
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--with-d-hit", action="store_true")
    parser.add_argument("--d-hit-samples", type=int, default=2000)
    parser.add_argument("--d-hit-seed", type=int, default=123)
    parser.add_argument("--area-size", type=float, default=4.0)
    parser.add_argument("--obs", type=int, default=4)
    args = parser.parse_args()

    compare_dir = Path(
        args.compare_dir
        or f"method/deepset/output/stage1/compare/gcbf_vs_bases_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    compare_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "compare_dir": str(compare_dir),
        "teacher_run": args.teacher_run,
        "teacher_step": args.teacher_step,
        "seed0": args.seed0,
        "frozen": args.frozen,
        "eval_seed": args.eval_seed,
        "episodes_list": [int(x) for x in args.episodes.split(",") if x.strip()],
        "started_at": _utc(),
    }
    (compare_dir / "META.json").write_text(json.dumps(meta, indent=2))

    epi_list = meta["episodes_list"]
    print(f"[compare] dir={compare_dir}")
    print(f"[compare] episodes={epi_list} eval_seed={args.eval_seed}")

    for epi in epi_list:
        print(f"\n=== task eval episodes={epi} ===")
        if not args.skip_gcbf:
            print("[compare] GCBF+ teacher ...")
            row = eval_gcbf(
                teacher_run=args.teacher_run,
                teacher_step=args.teacher_step,
                episodes=epi,
                seed=args.eval_seed,
                area_size=args.area_size,
                obs=args.obs,
                log_path=compare_dir / "raw" / f"gcbf_epi{epi}.log",
            )
            row["metric_kind"] = "task"
            append_row(compare_dir, row)
            print(
                f"  GCBF+ success={row['success_rate']*100:.2f}% "
                f"safe={row['safe_rate']*100:.2f}% (±SE {row['success_se_agent']*100:.2f}pt)"
            )

        if not args.skip_frozen and Path(args.frozen).exists():
            print("[compare] Phase-1 frozen DeepSet ...")
            row = eval_deepset(
                policy_id="deepset_phase1_frozen",
                label="Phase-1 frozen DeepSet",
                checkpoint=args.frozen,
                episodes=epi,
                seed=args.eval_seed,
                out_json=compare_dir / "raw" / f"frozen_epi{epi}.json",
            )
            row["metric_kind"] = "task"
            append_row(compare_dir, row)
            print(
                f"  frozen success={row['success_rate']*100:.2f}% "
                f"safe={row['safe_rate']*100:.2f}% (±SE {row['success_se_agent']*100:.2f}pt)"
            )

        if not args.skip_seed0 and Path(args.seed0).exists():
            print("[compare] redistill seed0 best ...")
            row = eval_deepset(
                policy_id="deepset_redistill_seed0",
                label="redistill seed0 best",
                checkpoint=args.seed0,
                episodes=epi,
                seed=args.eval_seed,
                out_json=compare_dir / "raw" / f"seed0_epi{epi}.json",
            )
            row["metric_kind"] = "task"
            append_row(compare_dir, row)
            print(
                f"  seed0 success={row['success_rate']*100:.2f}% "
                f"safe={row['safe_rate']*100:.2f}% (±SE {row['success_se_agent']*100:.2f}pt)"
            )
        elif not args.skip_seed0:
            print(f"[compare] WARN missing seed0 ckpt: {args.seed0}")

    if args.with_d_hit:
        print("\n=== same-std L2-D hit ===")
        # GCBF + seed0
        if Path(args.seed0).exists() and not args.skip_seed0:
            out_p = compare_dir / "raw" / "d_hit_gcbf_vs_seed0.json"
            rep = run_d_hit(
                teacher_run=args.teacher_run,
                teacher_step=args.teacher_step,
                deepset_ckpt=args.seed0,
                frozen_ckpt=None,
                out_path=out_p,
                samples=args.d_hit_samples,
                seed=args.d_hit_seed,
            )
            for name, pol in (rep.get("policies") or {}).items():
                hits = {
                    k: float(v.get("hit_rate_on_D", 0.0))
                    for k, v in (pol.get("buckets") or {}).items()
                    if isinstance(v, dict)
                }
                row = {
                    "metric_kind": "d_hit",
                    "policy_id": name if name != "deepset_frozen" else "deepset_redistill_seed0",
                    "label": name,
                    "checkpoint": pol.get("checkpoint"),
                    "d_hit": hits,
                    "mean_d_hit": sum(hits.values()) / len(hits) if hits else None,
                    "d_hit_seed": args.d_hit_seed,
                    "d_hit_samples": args.d_hit_samples,
                    "source_report": str(out_p),
                }
                append_row(compare_dir, row)
        if not args.skip_frozen and Path(args.frozen).exists():
            out_p = compare_dir / "raw" / "d_hit_frozen_only.json"
            # skip gcbf second time
            cmd = [
                PY,
                str(_ROOT / "stage2/eval_l2_gcbfplus.py"),
                "--deepset",
                args.frozen,
                "--buckets",
                "l2_aa_near,l2_ao_near,l2_11_near",
                "--samples",
                str(args.d_hit_samples),
                "--seed",
                str(args.d_hit_seed),
                "--skip-gcbf",
                "--skip-residual",
                "--skip-r1",
                "--out",
                str(out_p),
            ]
            proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
            (compare_dir / "raw" / "d_hit_frozen_only_run.log").write_text(
                proc.stdout + "\n" + proc.stderr
            )
            if proc.returncode == 0 and out_p.exists():
                rep = json.loads(out_p.read_text())
                for name, pol in (rep.get("policies") or {}).items():
                    hits = {
                        k: float(v.get("hit_rate_on_D", 0.0))
                        for k, v in (pol.get("buckets") or {}).items()
                        if isinstance(v, dict)
                    }
                    append_row(
                        compare_dir,
                        {
                            "metric_kind": "d_hit",
                            "policy_id": "deepset_phase1_frozen",
                            "label": "Phase-1 frozen DeepSet",
                            "checkpoint": pol.get("checkpoint"),
                            "d_hit": hits,
                            "mean_d_hit": sum(hits.values()) / len(hits) if hits else None,
                            "d_hit_seed": args.d_hit_seed,
                            "d_hit_samples": args.d_hit_samples,
                            "source_report": str(out_p),
                        },
                    )

    write_snapshot_table(compare_dir)
    print(f"\n[compare] wrote {compare_dir / 'rows.jsonl'}")
    print(f"[compare] snapshot {compare_dir / 'SNAPSHOT.md'}")
    print(
        "Plot later:\n"
        f"  python scripts/plot_stage1_compare.py --compare-dir {compare_dir}"
    )


if __name__ == "__main__":
    main()
