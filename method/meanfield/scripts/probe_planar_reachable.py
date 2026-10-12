#!/usr/bin/env python3
"""Dense planar probe on reachable_v1 claim (goal_xy≤comm, goal_dv=-v).

Does NOT write ACTIVE_*. Side-branch diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "core" / "paths.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
from core.paths import ensure_import_path, project_root

_ROOT = project_root()
ensure_import_path()

from stage_mf.encode import load_mf_base_weights, load_mf_residual_weights, deploy_action_np
from stage_mf.l2_labels import clearance_batch
from stage_mf.property_planar_l2 import claim_physics_report, default_planar_netdep_claim


def _controllable(rel: np.ndarray, agrid: np.ndarray) -> bool:
    rb = np.repeat(rel[None], len(agrid), 0)
    m, c = clearance_batch(rb, None, agrid)
    return bool(np.any(c | (m < 0))) and bool(np.any(~c & (m >= 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="method/meanfield/output/verify/planar_reachable_v1/probe")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    box = default_planar_netdep_claim()
    phys = claim_physics_report(box)
    assert phys["feat_contract"]["goal_dv_equals_minus_v"]
    assert not phys["feat_contract"]["legacy_wide_goal_abs2"]
    assert phys["network_dependent"] and not phys["envelope_any_a_safe"]

    rng = np.random.default_rng(args.seed)
    agrid = np.array([[ax, ay] for ax in np.linspace(-1, 1, 11) for ay in np.linspace(-1, 1, 11)])

    xs, rels = [], []
    guard = 0
    while len(xs) < args.n and guard < args.n * 40:
        guard += 1
        x, rel = box.sample(rng, 1)
        x, rel = x[0], rel[0]
        if not _controllable(rel, agrid):
            continue
        xs.append(x)
        rels.append(rel)
    xs = np.asarray(xs)
    rels = np.asarray(rels)
    assert len(xs) == args.n

    # contract checks
    gnorm = np.linalg.norm(xs[:, 4:6], axis=1)
    gdv_err = np.max(np.abs(xs[:, 6:8] + xs[:, 2:4]))
    np.savez_compressed(out / "probe_reachable_v1.npz", xs=xs, rels=rels, seed=np.array([args.seed]))

    def eval_acts(name: str, acts: np.ndarray) -> dict:
        acts = np.asarray(acts, dtype=np.float64).reshape(-1, 2)
        m, c = clearance_batch(rels, None, acts)
        safe = ~c & (m >= -1e-12)
        n_fail = int((~safe).sum())
        fail_idx = np.where(~safe)[0]
        n_rep = 0
        for j in fail_idx[: min(300, len(fail_idx))]:
            rb = np.repeat(rels[j][None], len(agrid), 0)
            mm, cc = clearance_batch(rb, None, agrid)
            if np.any(~cc & (mm >= 0)):
                n_rep += 1
        n_chk = min(300, len(fail_idx))
        return {
            "name": name,
            "N": args.n,
            "n_fail": n_fail,
            "fail_rate": n_fail / args.n,
            "safe_rate": float(safe.mean()),
            "margin_mean": float(m.mean()),
            "margin_min": float(m.min()),
            "ax_mean": float(acts[:, 0].mean()),
            "fail_ax_mean": float(acts[~safe, 0].mean()) if n_fail else None,
            "repairable_frac": (n_rep / n_chk) if n_chk else None,
        }

    rows = []
    rows.append(eval_acts("const_a0_coast", np.zeros((args.n, 2))))
    rows.append(eval_acts("const_a=-1", np.tile([-1.0, 0.0], (args.n, 1))))
    rows.append(eval_acts("const_a=+1", np.tile([1.0, 0.0], (args.n, 1))))

    ckpts = [
        ("MF_base_v5", "acceptance/mf_best_20261006/1_primary/01_models/base_v5/meanfield_mlp512_student_step9000.pkl", "base"),
        ("MF_main_r2", "acceptance/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl", "res"),
        ("netdep_r3_pick", "method/meanfield/output/repair/mf_residual_v5_netdep_r3_20261010/seed0/repair_manual_pick.pkl", "res"),
    ]
    ces = []
    for name, path, kind in ckpts:
        p = pathlib.Path(path)
        if not p.is_file():
            rows.append({"name": name, "error": f"missing {path}"})
            continue
        if kind == "base":
            bw = load_mf_base_weights(path)
            acts = np.clip(np.array([bw.forward(xs[i]) for i in range(args.n)]), -1, 1)
        else:
            w = load_mf_residual_weights(path)
            acts = np.array([deploy_action_np(w, xs[i]) for i in range(args.n)])
        row = eval_acts(name, acts)
        rows.append(row)
        # store CEs for env construction
        m, c = clearance_batch(rels, None, acts)
        bad = np.where(c | (m < 0))[0]
        for j in bad[:200]:
            ces.append(
                {
                    "policy": name,
                    "x": xs[j].tolist(),
                    "rel": rels[j].tolist(),
                    "a": acts[j].tolist(),
                }
            )
        print(f"  {name}: fail={100*row['fail_rate']:.2f}% ax={row['ax_mean']:.3f}", flush=True)

    # oracle
    oa = np.zeros((args.n, 2))
    for i in range(args.n):
        rb = np.repeat(rels[i][None], len(agrid), 0)
        m, c = clearance_batch(rb, None, agrid)
        oa[i] = agrid[int(np.argmax(np.where(c, -1e9, m)))]
    rows.append(eval_acts("oracle_11x11", oa))

    rep = {
        "title": "planar_reachable_v1 dense probe",
        "N": args.n,
        "seed": args.seed,
        "physics": phys,
        "contract_checks": {
            "goal_norm_max": float(gnorm.max()),
            "goal_norm_mean": float(gnorm.mean()),
            "frac_goal_norm_gt_comm": float(np.mean(gnorm > box.comm_radius + 1e-9)),
            "max_abs_goal_dv_plus_v": float(gdv_err),
        },
        "rows": rows,
        "n_ce_stored": len(ces),
    }
    (out / "PROBE.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out / "counterexamples.json").write_text(json.dumps({"n": len(ces), "ces": ces}, indent=2) + "\n")

    md = [
        "# Planar reachable_v1 probe",
        "",
        f"- N={args.n}, seed={args.seed}",
        f"- goal_dv=-v check max|err|={gdv_err:.2e}",
        f"- |goal| max={gnorm.max():.3f} (comm={box.comm_radius})",
        f"- physics netdep={phys['network_dependent']} envelope={phys['envelope_any_a_safe']}",
        "",
        "| policy | N | planar fail% | ax_mean | repairable |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "fail_rate" not in r:
            continue
        repf = f"{100*r['repairable_frac']:.0f}%" if r.get("repairable_frac") is not None else "—"
        md.append(
            f"| `{r['name']}` | {r['N']} | {100*r['fail_rate']:.2f}% | {r['ax_mean']:.3f} | {repf} |"
        )
    (out / "PROBE_REPORT.md").write_text("\n".join(md) + "\n")
    print((out / "PROBE_REPORT.md").read_text())
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
