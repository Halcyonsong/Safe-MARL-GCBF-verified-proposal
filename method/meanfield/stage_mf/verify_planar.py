#!/usr/bin/env python3
"""Planar two-step L2 verify via SafeSet(a) (side-branch).

Correct pipeline (not loose p2-IBP alone):
  1) Geometry on rel box → sufficient SafeSet, default {ax ≥ a_min, |ay|≤1}
  2) Network Deploy on feat box → prove Deploy(x) ∈ SafeSet
       IBP first; if unknown and box small enough → Marabou
  3) True CE: sample / Marabou candidate rechecked with real Deploy + planar p2

Does NOT write ACTIVE_MF_* / acceptance pack.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "core" / "paths.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
from core.paths import project_root, ensure_import_path

_ROOT = project_root()
ensure_import_path()

from stage_mf.encode import load_mf_residual_weights, deploy_action_np
from stage_mf.marabou_mf import marabou_safeset_subbox
from stage_mf.property_planar_l2 import (
    PlanarBox,
    action_in_safeset,
    claim_physics_report,
    default_planar_netdep_claim,
    ibp_planar_status,
    is_planar_safe_point,
    planar_p2,
    safeset_ax_min,
)


def _box_dict(box: PlanarBox) -> dict[str, Any]:
    return {
        "name": box.name,
        "feat_lo": box.feat_lo.tolist(),
        "feat_hi": box.feat_hi.tolist(),
        "rel_lo": box.rel_lo.tolist(),
        "rel_hi": box.rel_hi.tolist(),
        "radius": float(box.radius),
        "comm_radius": float(box.comm_radius),
        "width_rel": box.width_rel().tolist(),
        "width_feat_max": float(np.max(box.width_feat())),
        "width_v": float(max(box.feat_hi[2] - box.feat_lo[2], box.feat_hi[3] - box.feat_lo[3])),
        "width_goal": float(max(box.feat_hi[4] - box.feat_lo[4], box.feat_hi[5] - box.feat_lo[5])),
    }


def _pose_width(box: PlanarBox) -> float:
    wf = box.width_feat()
    return max(float(wf[0]), float(wf[1]))


def _vg_width(box: PlanarBox) -> float:
    wf = box.width_feat()
    return max(float(wf[2]), float(wf[3]), float(wf[4]), float(wf[5]))


def _threat_width(box: PlanarBox) -> float:
    """Max width over threat min_d / close / inv_ttc (agent+obs quads)."""
    wf = box.width_feat()
    idxs = []
    for q in range(4):
        b = 8 + 5 * q
        idxs.extend([b + 1, b + 2, b + 3, b + 4])  # md, cmax, cmean?, inv_ttc
        b2 = 28 + 4 * q
        idxs.extend([b2 + 1, b2 + 2, b2 + 3])
    return max((float(wf[i]) for i in idxs if i < wf.size), default=0.0)


def _critical_width(box: PlanarBox) -> float:
    """Priority score: pose + v/goal + threat + rel.

    Marabou needs inv_ttc narrowed (root ~1.1 from coupling over-approx);
    free inv_ttc alone makes SafeSet queries time out.
    """
    wr = box.width_rel()
    return max(_pose_width(box), _vg_width(box), _threat_width(box), float(np.max(wr)))


def _marabou_ready(
    box: PlanarBox,
    *,
    pose_max: float,
    vg_max: float,
    threat_max: float = 0.30,
) -> bool:
    """True when box is small enough for Marabou SafeSet to typically finish."""
    return (
        _pose_width(box) <= pose_max + 1e-12
        and _vg_width(box) <= vg_max + 1e-12
        and _threat_width(box) <= threat_max + 1e-12
    )


@dataclass
class Stats:
    proven: int = 0
    violated: int = 0
    unknown: int = 0
    splits: int = 0
    ibp_hits: int = 0
    marabou_calls: int = 0
    marabou_proven: int = 0
    marabou_timeouts: int = 0
    marabou_false_sat: int = 0
    leaves: int = 0
    ces: list = field(default_factory=list)
    ce_boxes: list = field(default_factory=list)
    unknown_boxes: list = field(default_factory=list)
    proven_sample: list = field(default_factory=list)
    incomplete_reasons: dict = field(default_factory=dict)
    max_ce: int = 500


def _inc_incomplete(stats: Stats, reason: str) -> None:
    stats.incomplete_reasons[reason] = stats.incomplete_reasons.get(reason, 0) + 1


def _record_ce(
    stats: Stats,
    *,
    box: PlanarBox,
    x: np.ndarray,
    rel: np.ndarray,
    info: dict[str, Any],
    source: str,
    depth: int,
    ss: dict[str, Any] | None = None,
) -> None:
    if len(stats.ces) >= stats.max_ce:
        stats.violated += 1
        return
    stats.violated += 1
    a = np.asarray(info.get("a", [0, 0]), dtype=np.float64)
    stats.ces.append(
        {
            "source": source,
            "depth": int(depth),
            "x": np.asarray(x, dtype=np.float64).tolist(),
            "rel": np.asarray(rel, dtype=np.float64).tolist(),
            "a": a.tolist(),
            "p2": info.get("p2"),
            "d2": info.get("d2"),
            "radius": float(box.radius),
            "in_safeset": bool(action_in_safeset(a, ss)) if ss else None,
            "box_rel_lo": box.rel_lo.tolist(),
            "box_rel_hi": box.rel_hi.tolist(),
            "box_feat_ego_lo": box.feat_lo[:8].tolist(),
            "box_feat_ego_hi": box.feat_hi[:8].tolist(),
        }
    )
    if len(stats.ce_boxes) < 200:
        stats.ce_boxes.append({**_box_dict(box), "depth": depth, "source": source})


def _sample_true_ces(
    w,
    box: PlanarBox,
    *,
    stats: Stats,
    rng: np.random.Generator,
    n: int,
    source: str,
    depth: int,
    ss: dict[str, Any],
) -> int:
    n_found = 0
    xs, rels = box.sample(rng, n)
    for x, rel in zip(xs, rels):
        ok, info = is_planar_safe_point(w, x, rel, radius=box.radius)
        if not ok:
            _record_ce(stats, box=box, x=x, rel=rel, info=info, source=source, depth=depth, ss=ss)
            n_found += 1
            if len(stats.ces) >= stats.max_ce:
                break
    return n_found


def verify_planar(
    w,
    root: PlanarBox,
    *,
    max_splits: int = 2000,
    max_depth: int = 24,
    wall_limit_s: float = 900.0,
    sample_per_unknown: int = 32,
    seed: int = 0,
    use_marabou: bool = True,
    marabou_timeout: int = 20,
    marabou_max_width: float = 0.15,  # legacy: applied to vg if pose/vg not set
    marabou_pose_max: float = 0.50,
    marabou_vg_max: float = 0.20,
    marabou_threat_max: float = 0.15,
    min_width: float = 1e-3,
    progress_every: int = 25,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    stats = Stats()
    t0 = time.time()
    # DFS stack: deepen one branch to Marabou-ready, prove, then backtrack.
    # (Best-first on width stays shallow forever on free pose [0,4].)
    q: list[tuple[PlanarBox, int]] = [(root, 0)]
    leaves_done = 0

    def progress(tag: str) -> None:
        nonlocal leaves_done
        leaves_done += 1
        if leaves_done % progress_every == 0 or tag.startswith("marabou"):
            top_w = _critical_width(q[-1][0]) if q else 0.0
            print(
                f"[planar-safeset] leaves={leaves_done} proven={stats.proven} "
                f"CE={len(stats.ces)} unk={stats.unknown} splits={stats.splits} "
                f"mara={stats.marabou_calls}/{stats.marabou_proven} "
                f"to={stats.marabou_timeouts} q={len(q)} topW={top_w:.3f} "
                f"t={time.time()-t0:.1f}s last={tag}",
                flush=True,
            )

    def mark_unknown(box: PlanarBox, depth: int, reason: str, ss: dict[str, Any]) -> None:
        stats.unknown += 1
        stats.leaves += 1
        _inc_incomplete(stats, reason)
        if len(stats.unknown_boxes) < 40:
            stats.unknown_boxes.append(
                {**_box_dict(box), "depth": depth, "reason": reason, "a_min": ss.get("a_min")}
            )
        _sample_true_ces(
            w, box, stats=stats, rng=rng, n=sample_per_unknown, source=f"unk_{reason}", depth=depth, ss=ss
        )
        progress(f"unk_{reason}")

    while q:
        if time.time() - t0 > wall_limit_s:
            while q:
                b, d = q.pop()
                ss = safeset_ax_min(b.rel_lo, b.rel_hi, radius=b.radius)
                mark_unknown(b, d, "wall", ss)
            break
        if stats.splits >= max_splits:
            while q:
                b, d = q.pop()
                ss = safeset_ax_min(b.rel_lo, b.rel_hi, radius=b.radius)
                mark_unknown(b, d, "split_cap", ss)
            break

        box, depth = q.pop()
        ss = safeset_ax_min(box.rel_lo, box.rel_hi, radius=box.radius, ay_abs=1.0)

        # cheap true CE harvest (does not stop search)
        _sample_true_ces(
            w, box, stats=stats, rng=rng, n=max(4, sample_per_unknown // 4), source="pre_split", depth=depth, ss=ss
        )

        # IBP SafeSet membership / direct p2
        ibp = ibp_planar_status(w, box)
        if ibp["status"] == "proven":
            stats.proven += 1
            stats.leaves += 1
            stats.ibp_hits += 1
            if len(stats.proven_sample) < 15:
                stats.proven_sample.append(
                    {
                        **_box_dict(box),
                        "depth": depth,
                        "method": ibp.get("method"),
                        "a_lo": ibp.get("a_lo"),
                        "a_min": ss.get("a_min"),
                    }
                )
            progress("ibp_proven")
            continue
        if ibp["status"] == "conflict":
            # whole box Deploy outside SafeSet or all-collide — still need true CE if possible
            n_ce = _sample_true_ces(
                w, box, stats=stats, rng=rng, n=sample_per_unknown, source="ibp_conflict", depth=depth, ss=ss
            )
            if n_ce == 0:
                # sound conflict on action set but no sample CE yet: keep as violated-event via synthetic mid
                x_mid = 0.5 * (box.feat_lo + box.feat_hi)
                # enforce goal_dv=-v
                x_mid[6] = -x_mid[2]
                x_mid[7] = -x_mid[3]
                rel_mid = 0.5 * (box.rel_lo + box.rel_hi)
                ok, info = is_planar_safe_point(w, x_mid, rel_mid, radius=box.radius)
                if not ok:
                    _record_ce(
                        stats, box=box, x=x_mid, rel=rel_mid, info=info, source="ibp_conflict_mid", depth=depth, ss=ss
                    )
            stats.leaves += 1
            progress("ibp_conflict")
            continue

        # Marabou when pose+v/goal+threat are in the decidable band
        pose_cap = float(marabou_pose_max)
        vg_cap = float(min(marabou_vg_max, marabou_max_width))
        thr_cap = float(marabou_threat_max)
        if (
            use_marabou
            and ss.get("feasible", False)
            and _marabou_ready(box, pose_max=pose_cap, vg_max=vg_cap, threat_max=thr_cap)
            and float(ss.get("a_min", 1.0)) <= 1.0
        ):
            stats.marabou_calls += 1
            mq = marabou_safeset_subbox(
                w,
                box.feat_lo,
                box.feat_hi,
                a_min=float(ss["a_min"]),
                ay_abs=float(ss.get("ay_abs", 1.0)),
                timeout_s=marabou_timeout,
            )
            if mq["status"] == "proven":
                stats.proven += 1
                stats.leaves += 1
                stats.marabou_proven += 1
                if len(stats.proven_sample) < 15:
                    stats.proven_sample.append(
                        {
                            **_box_dict(box),
                            "depth": depth,
                            "method": "marabou_safeset",
                            "a_min": ss.get("a_min"),
                            "marabou_s": mq.get("elapsed_s"),
                        }
                    )
                progress("marabou_unsat")
                continue
            if mq["status"] == "sat_candidate" and mq.get("x") is not None:
                x = np.asarray(mq["x"], dtype=np.float64)
                # lock goal_dv if solver ignored equality (box already has bounds)
                if x.size >= 8:
                    x[6] = -float(x[2])
                    x[7] = -float(x[3])
                a = deploy_action_np(w, x)
                # pick worst-case rel corner-ish: sample several rel
                found = False
                for _ in range(16):
                    rel = box.rel_lo + rng.random(4) * (box.rel_hi - box.rel_lo)
                    ok, info = is_planar_safe_point(w, x, rel, radius=box.radius)
                    if not ok:
                        _record_ce(
                            stats,
                            box=box,
                            x=x,
                            rel=rel,
                            info=info,
                            source="marabou_sat_true",
                            depth=depth,
                            ss=ss,
                        )
                        found = True
                        break
                if not found and not action_in_safeset(a, ss):
                    # out of safeset but still planar-safe on samples — false-ish sat for template
                    stats.marabou_false_sat += 1
                elif found:
                    progress("marabou_ce")
                    # continue splitting neighborhood for more CEs
                    if depth < max_depth and float(np.max(box.width_feat())) > min_width:
                        stats.splits += 1
                        L, R = box.split()
                        # DFS: push right then left so left is processed first
                        q.append((R, depth + 1))
                        q.append((L, depth + 1))
                    else:
                        stats.leaves += 1
                    continue
            if mq["status"] in ("timeout", "unknown", "error"):
                stats.marabou_timeouts += 1
                progress(f"marabou_{mq['status']}")
                # On timeout: keep splitting if room (do not burn depth as unknown yet)

        # depth / width leaf — only when too small to refine further
        tiny = float(np.max(box.width_feat())) < min_width and float(np.max(box.width_rel())) < min_width
        if tiny:
            mark_unknown(box, depth, "width_floor", ss)
            continue
        if depth >= max_depth:
            mark_unknown(box, depth, "depth_cap", ss)
            continue

        # split; DFS deepen left child first
        stats.splits += 1
        L, Rgt = box.split()
        q.append((Rgt, depth + 1))
        q.append((L, depth + 1))
        progress("split")

    wall = round(time.time() - t0, 3)
    has_ce = len(stats.ces) > 0
    if stats.unknown == 0 and not has_ce and stats.proven > 0:
        status = "PROVEN_FULL"
    elif stats.unknown == 0 and has_ce:
        status = "VIOLATED_WITH_CES_FULLPASS"
    elif has_ce:
        status = "INCOMPLETE_WITH_CES"
    elif stats.proven > 0:
        status = "INCOMPLETE"
    else:
        status = "INCOMPLETE"

    return {
        "aggregate_status": status,
        "stats": {
            "proven": stats.proven,
            "violated_events": stats.violated,
            "n_ce": len(stats.ces),
            "n_ce_boxes": len(stats.ce_boxes),
            "unknown": stats.unknown,
            "splits": stats.splits,
            "leaves": stats.leaves,
            "ibp_hits": stats.ibp_hits,
            "marabou_calls": stats.marabou_calls,
            "marabou_proven": stats.marabou_proven,
            "marabou_timeouts": stats.marabou_timeouts,
            "marabou_false_sat": stats.marabou_false_sat,
            "incomplete_reasons": stats.incomplete_reasons,
            "wall_s": wall,
        },
        "ces": stats.ces,
        "ce_boxes": stats.ce_boxes,
        "unknown_boxes": stats.unknown_boxes,
        "proven_sample": stats.proven_sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Planar SafeSet verify (side-branch)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--d-min", type=float, default=0.118)
    ap.add_argument("--d-max", type=float, default=0.122)
    ap.add_argument("--close-lo", type=float, default=0.25)
    ap.add_argument("--close-hi", type=float, default=0.30)
    ap.add_argument("--lat-max", type=float, default=0.04)
    ap.add_argument("--max-splits", type=int, default=2000)
    ap.add_argument("--max-depth", type=int, default=24)
    ap.add_argument("--wall-limit", type=float, default=900.0)
    ap.add_argument("--sample-per-unknown", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--marabou-timeout", type=int, default=30)
    ap.add_argument("--marabou-max-width", type=float, default=0.20, help="legacy vg cap")
    ap.add_argument("--marabou-pose-max", type=float, default=0.50, help="call Marabou when pose_w≤this")
    ap.add_argument("--marabou-vg-max", type=float, default=0.20, help="call Marabou when vg_w≤this")
    ap.add_argument("--marabou-threat-max", type=float, default=0.15, help="call Marabou when threat_w≤this")
    ap.add_argument(
        "--freeze-pose",
        action="store_true",
        help="Pin ego px/py to mid of [0,4] (pose-fixed slice; still netdep on rel/v/goal)",
    )
    ap.add_argument(
        "--vg-abs",
        type=float,
        default=None,
        help="Scope vx,vy,goal_xy to [-vg,vg] (goal_dv=-v). e.g. 0.01 min cert, 0.05 enlarge",
    )
    ap.add_argument(
        "--threat-mid",
        action="store_true",
        help="Pin threat/quad features to midpoint (scoped slice; not full threat cover)",
    )
    ap.add_argument("--no-marabou", action="store_true")
    ap.add_argument("--no-update-active", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    w = load_mf_residual_weights(args.checkpoint)
    box = default_planar_netdep_claim(
        radius=args.radius,
        d_min=args.d_min,
        d_max=args.d_max,
        close_lo=args.close_lo,
        close_hi=args.close_hi,
        lat_max=args.lat_max,
        freeze_pose=bool(args.freeze_pose),
        vg_abs=args.vg_abs,
        threat_mid=bool(args.threat_mid),
    )
    phys = claim_physics_report(box)
    if phys.get("envelope_any_a_safe"):
        print("[planar-safeset] REFUSE envelope claim", flush=True)
        raise SystemExit(2)
    fc = phys.get("feat_contract") or {}
    if not fc.get("goal_dv_equals_minus_v"):
        print("[planar-safeset] REFUSE: goal_dv not synced", flush=True)
        raise SystemExit(3)

    ss_root = safeset_ax_min(box.rel_lo, box.rel_hi, radius=box.radius)
    rel_mid = 0.5 * (box.rel_lo + box.rel_hi)
    print(f"[planar-safeset] ckpt={args.checkpoint}", flush=True)
    print(
        f"[planar-safeset] CLAIM: Deploy(x) ∈ SafeSet(rel) ⇒ ||p2||≥R={args.radius}",
        flush=True,
    )
    print(
        f"[planar-safeset] root SafeSet: a_min={ss_root.get('a_min'):.6g} "
        f"feasible={ss_root.get('feasible')} ay_abs={ss_root.get('ay_abs')}",
        flush=True,
    )
    print(
        f"[planar-safeset] physics netdep={phys.get('network_dependent')} "
        f"envelope={phys.get('envelope_any_a_safe')} "
        f"p2_worst={phys.get('p2x_worst_a')} p2_best={phys.get('p2x_best_a')}",
        flush=True,
    )
    print(
        f"[planar-safeset] feat reachable_v1 goal≤comm goal_dv=-v "
        f"synced={fc.get('goal_dv_equals_minus_v')}",
        flush=True,
    )
    print(
        f"[planar-safeset] mid p2_brake={planar_p2(rel_mid, np.array([-1.,0.]))} "
        f"p2_gas={planar_p2(rel_mid, np.array([1.,0.]))}",
        flush=True,
    )
    print(
        f"[planar-safeset] search splits={args.max_splits} depth={args.max_depth} "
        f"wall={args.wall_limit}s marabou={not args.no_marabou} "
        f"pose≤{args.marabou_pose_max} vg≤{args.marabou_vg_max} thr≤{args.marabou_threat_max}",
        flush=True,
    )
    print(
        f"[planar-safeset] root pose_w={_pose_width(box):.3f} vg_w={_vg_width(box):.3f} "
        f"thr_w={_threat_width(box):.3f} crit_w={_critical_width(box):.3f}",
        flush=True,
    )

    result = verify_planar(
        w,
        box,
        max_splits=args.max_splits,
        max_depth=args.max_depth,
        wall_limit_s=args.wall_limit,
        sample_per_unknown=args.sample_per_unknown,
        seed=args.seed,
        use_marabou=not args.no_marabou,
        marabou_timeout=args.marabou_timeout,
        marabou_max_width=args.marabou_max_width,
        marabou_pose_max=args.marabou_pose_max,
        marabou_vg_max=args.marabou_vg_max,
        marabou_threat_max=args.marabou_threat_max,
    )

    report = {
        "track": "meanfield_sidebranch_planar_safeset",
        "verify_kind": "deploy_in_safeset_then_planar_L2",
        "pipeline": [
            "safeset_ax_min(rel_box) sufficient for ||p2||≥R",
            "IBP: Deploy action box ⊆ SafeSet",
            "Marabou: UNSAT of (ax < a_min) on feat box",
            "true CE: real Deploy + planar p2",
        ],
        "property": "planar_two_step_L2_NOT_radial",
        "feat_contract": "reachable_v1_goal_xy_le_comm_goal_dv_eq_minus_v",
        "checkpoint": args.checkpoint,
        "base_checkpoint": w.base_checkpoint,
        "aggregate_status": result["aggregate_status"],
        "physics": phys,
        "root_safeset": ss_root,
        "root_box": _box_dict(box),
        "stats": result["stats"],
        "config": {
            "max_splits": args.max_splits,
            "max_depth": args.max_depth,
            "wall_limit_s": args.wall_limit,
            "sample_per_unknown": args.sample_per_unknown,
            "seed": args.seed,
            "use_marabou": not args.no_marabou,
            "marabou_timeout": args.marabou_timeout,
            "marabou_max_width": args.marabou_max_width,
            "marabou_pose_max": args.marabou_pose_max,
            "marabou_vg_max": args.marabou_vg_max,
            "marabou_threat_max": args.marabou_threat_max,
            "freeze_pose": bool(args.freeze_pose),
            "vg_abs": args.vg_abs,
            "threat_mid": bool(args.threat_mid),
        },
        "ACTIVE_written": False,
        "note": (
            "Side-branch. acceptance/mf_best and ACTIVE_MF_* untouched. "
            + (
                "pose-fixed slice (px/py mid): full free-pose cover is larger BnB. "
                if args.freeze_pose
                else "full free pose [0,4]. "
            )
            + (f"vg_abs={args.vg_abs}. " if args.vg_abs is not None else "")
            + ("threat-mid slice. " if args.threat_mid else "")
        ),
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "PHYSICS.json").write_text(json.dumps(phys, indent=2) + "\n")
    (out / "ROOT_SAFESET.json").write_text(json.dumps(ss_root, indent=2) + "\n")
    (out / "counterexamples_all.json").write_text(
        json.dumps({"n": len(result["ces"]), "counterexamples": result["ces"]}, indent=2) + "\n"
    )
    (out / "ce_boxes.json").write_text(
        json.dumps({"n": len(result["ce_boxes"]), "boxes": result["ce_boxes"]}, indent=2) + "\n"
    )
    (out / "unknown_boxes.json").write_text(
        json.dumps({"n": len(result["unknown_boxes"]), "boxes": result["unknown_boxes"]}, indent=2) + "\n"
    )
    (out / "proven_sample.json").write_text(
        json.dumps({"n": len(result["proven_sample"]), "boxes": result["proven_sample"]}, indent=2) + "\n"
    )
    st = result["stats"]
    md = [
        "# Planar SafeSet verify (reachable_v1)",
        "",
        f"- status: **{result['aggregate_status']}**",
        f"- pipeline: geometry SafeSet(a) → prove Deploy ∈ SafeSet (IBP / Marabou)",
        f"- root a_min: **{ss_root.get('a_min'):.6g}** (ax≥a_min ⇒ ||p2||≥R on root rel)",
        f"- netdep={phys.get('network_dependent')} envelope={phys.get('envelope_any_a_safe')}",
        f"- proven={st['proven']} CE={st['n_ce']} unknown={st['unknown']} splits={st['splits']}",
        f"- ibp_hits={st['ibp_hits']} marabou={st['marabou_calls']} "
        f"(proven={st['marabou_proven']} timeout={st['marabou_timeouts']})",
        f"- wall={st['wall_s']}s",
        "",
        "PROVEN_FULL = every leaf proven, 0 CE.",
        "INCOMPLETE = resource limit; not a pass.",
        "VIOLATED_* = true Deploy planar CE found.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(md) + "\n")
    print(
        f"[planar-safeset] DONE {result['aggregate_status']} "
        f"proven={st['proven']} CE={st['n_ce']} unk={st['unknown']} → {out}",
        flush=True,
    )
    print("[planar-safeset] ACTIVE_MF_* not written", flush=True)


if __name__ == "__main__":
    main()
