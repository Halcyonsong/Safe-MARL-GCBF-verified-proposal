"""MF full-claim verification on the 44-D feature box (agreed design).

Claim
-----
    ∀ x ∈ X ⊂ R^{44}:  TwoStepRadialL2Safe( Deploy(x); R )

- **Domain X**: product of per-dimension bounds on network inputs
  (ego speed, goal, each quadrant min_d / close_wmean / inv_ttc / …).
  Branch-and-bound splits along free feature axes. **No** DeepSet (k,m)
  buckets and **no** primary AA/AO claim split.
- **Property**: two-step L2 radial over-approx (horizon simplification).
  d2 = d0 - α c0 - β |a|_∞ ≥ R, with d0,c0 from features.

Algorithm: IBP on property → (optional) Marabou on small boxes → true-policy
recheck → collect **all** real CEs; unknown leaves sampled for more CEs.

Strategy notes
--------------
- Prefer splitting property-critical dims (min_d, close_max) first.
- Skip Marabou on very wide boxes (IBP-first + split); call Marabou only when
  box volume / width is moderate so timeouts are rare.
- Progress lines every N leaves so long jobs are observable.

Outputs: runs/meanfield/verify/ only. DeepSet ACTIVE_* untouched.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stage_mf.domain import Box, MFClaim, claim_manifest, default_mf_claims
from stage_mf.encode import load_mf_residual_weights, min_d_indices
from stage_mf.marabou_mf import marabou_subbox
from stage_mf.property_l2 import (
    close_max_indices,
    ibp_property_status,
    is_two_step_safe_point,
)


@dataclass
class VerifyStats:
    proven: int = 0
    violated: int = 0
    false_sat: int = 0
    unknown: int = 0
    ibp_hits: int = 0
    marabou_proven: int = 0
    marabou_calls: int = 0
    marabou_timeouts: int = 0
    splits: int = 0
    ces: list = field(default_factory=list)
    unknown_boxes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "proven_leaves": self.proven,
            "violated_real": self.violated,
            "false_sat_relaxed": self.false_sat,
            "unknown_leaves": self.unknown,
            "ibp_property_hits": self.ibp_hits,
            "marabou_proven": self.marabou_proven,
            "marabou_calls": self.marabou_calls,
            "marabou_timeouts": self.marabou_timeouts,
            "splits": self.splits,
            "n_ces": len(self.ces),
            "n_unknown_boxes": len(self.unknown_boxes),
        }


def _property_split_axis(box: Box) -> int:
    """Split along the widest property-critical dim (min_d / close_max), else longest."""
    crit = min_d_indices() + close_max_indices()
    w = box.width()
    best_i, best_w = -1, -1.0
    for i in crit:
        if float(w[i]) > best_w + 1e-15:
            best_w = float(w[i])
            best_i = int(i)
    if best_i >= 0 and best_w > 1e-6:
        return best_i
    return box.longest_axis()


def _box_width_score(box: Box) -> float:
    """Scalar 'how wide is the property-relevant part'."""
    crit = min_d_indices() + close_max_indices()
    w = box.width()
    return float(max(w[i] for i in crit))


def verify_claim(
    w,
    claim: MFClaim,
    *,
    max_splits: int = 800,
    max_depth: int = 16,
    marabou_timeout: int = 20,
    min_width: float = 1e-3,
    sample_per_unknown: int = 64,
    seed: int = 0,
    use_marabou: bool = True,
    wall_limit_s: float = 900.0,
    marabou_max_width: float = 0.08,
    progress_every: int = 25,
) -> dict:
    rng = np.random.default_rng(seed)
    stats = VerifyStats()
    t0 = time.time()
    queue: deque[tuple[Box, int]] = deque([(claim.feature, 0)])
    R = float(claim.radius)
    leaves_done = 0

    def add_ce(x: np.ndarray, source: str, extra: dict | None = None) -> None:
        safe, info = is_two_step_safe_point(w, x, radius=R)
        if safe:
            return
        stats.violated += 1
        entry = {
            "source": source,
            "claim": claim.name,
            "x": np.asarray(x, dtype=float).tolist(),
            **info,
        }
        if extra:
            entry.update(extra)
        stats.ces.append(entry)

    def mark_unknown(box: Box, depth: int, reason: str) -> None:
        stats.unknown += 1
        stats.unknown_boxes.append(
            {
                "reason": reason,
                "depth": depth,
                "width_max": float(box.width().max()),
                "prop_width": _box_width_score(box),
                "split_axis": int(_property_split_axis(box)),
            }
        )
        for _ in range(sample_per_unknown):
            add_ce(box.sample(rng, 1)[0], f"unknown_{reason}")

    def progress(tag: str) -> None:
        nonlocal leaves_done
        leaves_done += 1
        if leaves_done % progress_every == 0 or tag.startswith("marabou"):
            elapsed = time.time() - t0
            print(
                f"[mf-verify] progress leaves={leaves_done} "
                f"proven={stats.proven} CE={stats.violated} unk={stats.unknown} "
                f"splits={stats.splits} marabou={stats.marabou_calls} "
                f"q={len(queue)} t={elapsed:.1f}s last={tag}",
                flush=True,
            )

    while queue:
        if time.time() - t0 > wall_limit_s:
            while queue:
                b, d = queue.popleft()
                mark_unknown(b, d, "wall_limit")
            break

        box, depth = queue.popleft()

        if stats.splits >= max_splits:
            mark_unknown(box, depth, "max_splits")
            while queue:
                b, d = queue.popleft()
                mark_unknown(b, d, "max_splits")
            break

        if depth >= max_depth or float(box.width().max()) < min_width:
            ibp = ibp_property_status(w, box.lo, box.hi, radius=R)
            if ibp["status"] == "proven":
                stats.proven += 1
                stats.ibp_hits += 1
                progress("leaf_ibp")
            else:
                # last-chance Marabou on tiny leaf
                if use_marabou and _box_width_score(box) <= marabou_max_width * 2:
                    stats.marabou_calls += 1
                    mq = marabou_subbox(
                        w, box.lo, box.hi, radius=R, timeout_s=marabou_timeout
                    )
                    if mq["status"] == "proven":
                        stats.proven += 1
                        stats.marabou_proven += 1
                        progress("leaf_marabou_unsat")
                        continue
                    if mq["status"] == "sat_relaxed" and mq.get("x") is not None:
                        x = np.asarray(mq["x"], dtype=np.float64)
                        safe, info = is_two_step_safe_point(w, x, radius=R)
                        if not safe:
                            stats.violated += 1
                            stats.ces.append(
                                {
                                    "source": "marabou_sat_true_leaf",
                                    "claim": claim.name,
                                    "x": x.tolist(),
                                    "a_relaxed": mq.get("a_relaxed"),
                                    "d2_relaxed": mq.get("d2_relaxed"),
                                    **info,
                                }
                            )
                            progress("leaf_marabou_ce")
                            continue
                        stats.false_sat += 1
                    if mq["status"] in ("timeout", "unknown", "error"):
                        stats.marabou_timeouts += 1
                mark_unknown(box, depth, "max_depth_or_min_width")
                progress("leaf_unknown")
            continue

        # IBP first (cheap)
        ibp = ibp_property_status(w, box.lo, box.hi, radius=R)
        if ibp["status"] == "proven":
            stats.proven += 1
            stats.ibp_hits += 1
            progress("ibp")
            continue

        prop_w = _box_width_score(box)
        # Marabou only on moderately small property boxes
        if use_marabou and prop_w <= marabou_max_width:
            stats.marabou_calls += 1
            mq = marabou_subbox(
                w, box.lo, box.hi, radius=R, timeout_s=marabou_timeout
            )
            if mq["status"] == "proven":
                stats.proven += 1
                stats.marabou_proven += 1
                progress("marabou_unsat")
                continue
            if mq["status"] == "sat_relaxed" and mq.get("x") is not None:
                x = np.asarray(mq["x"], dtype=np.float64)
                safe, info = is_two_step_safe_point(w, x, radius=R)
                if not safe:
                    stats.violated += 1
                    stats.ces.append(
                        {
                            "source": "marabou_sat_true",
                            "claim": claim.name,
                            "x": x.tolist(),
                            "a_relaxed": mq.get("a_relaxed"),
                            "d2_relaxed": mq.get("d2_relaxed"),
                            "marabou_elapsed": mq.get("elapsed_s"),
                            **info,
                        }
                    )
                    # still refine to harvest more CEs in neighborhood
                    if depth < max_depth and stats.splits < max_splits:
                        stats.splits += 1
                        left, right = box.split(_property_split_axis(box))
                        queue.append((left, depth + 1))
                        queue.append((right, depth + 1))
                    progress("marabou_ce")
                    continue
                stats.false_sat += 1
            if mq["status"] in ("timeout", "unknown", "error"):
                stats.marabou_timeouts += 1
                progress(f"marabou_{mq['status']}")
            # else fall through to split

        # split along property-critical free feature dim
        stats.splits += 1
        left, right = box.split(_property_split_axis(box))
        queue.append((left, depth + 1))
        queue.append((right, depth + 1))
        progress("split")

    elapsed = round(time.time() - t0, 3)
    if stats.unknown == 0 and stats.violated == 0 and stats.proven > 0:
        status = "PROVEN_FULL"
    elif stats.violated > 0:
        status = "VIOLATED"
    elif stats.proven > 0:
        status = "PARTIAL"
    else:
        status = "UNKNOWN"

    return {
        "claim": claim.name,
        "status": status,
        "manifest": claim_manifest(claim),
        "stats": stats.as_dict(),
        "wall_time_s": elapsed,
        "counterexamples": stats.ces,
        "unknown_boxes_preview": stats.unknown_boxes[:40],
        "n_unknown_boxes_total": len(stats.unknown_boxes),
        "design": {
            "query_domain": "44D_feature_box_per_free_dim",
            "property": "two_step_radial_L2",
            "not_deepset_km_buckets": True,
            "not_primary_aa_ao_split": True,
            "split_priority": "min_d_and_close_max",
            "marabou_max_prop_width": marabou_max_width,
        },
        "soundness": (
            "IBP/Marabou UNSAT proves the radial two-step property on the feature "
            "sub-box (sufficient for planar collision under same d0/c0). "
            "Real CEs re-checked with true deploy policy on the same property. "
            "PARTIAL = resource-bounded incomplete coverage of X, not a full pass."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MF 44-D feature-box full verification + two-step L2 property"
    )
    ap.add_argument("--checkpoint", default="")
    ap.add_argument(
        "--mode",
        choices=["smoke", "near", "near_plus", "full"],
        default="smoke",
        help="smoke | near | near_plus (bounded acceptance) | full",
    )
    ap.add_argument("--max-splits", type=int, default=800)
    ap.add_argument("--max-depth", type=int, default=14)
    ap.add_argument("--marabou-timeout", type=int, default=20)
    ap.add_argument("--marabou-max-width", type=float, default=0.08,
                    help="only call Marabou when max(min_d,close) width ≤ this")
    ap.add_argument("--no-marabou", action="store_true")
    ap.add_argument("--wall-limit", type=float, default=900.0)
    ap.add_argument("--sample-per-unknown", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-min", type=float, default=None,
                    help="override near/full min_d lo (default: mode preset)")
    ap.add_argument("--d-max", type=float, default=None,
                    help="override near/full min_d hi (default: mode preset)")
    ap.add_argument("--close-hi", type=float, default=None,
                    help="override |close| upper (default: mode preset)")
    ap.add_argument("--radius", type=float, default=0.10, help="two-step R (default AA)")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    ckpt = args.checkpoint.strip()
    if not ckpt:
        ptr = pathlib.Path("runs/meanfield/ACTIVE_MF_RESIDUAL.txt")
        if not ptr.exists():
            raise SystemExit("no --checkpoint and no ACTIVE_MF_RESIDUAL.txt")
        ckpt = ptr.read_text().strip()

    w = load_mf_residual_weights(ckpt)
    claims = default_mf_claims(
        d_min=args.d_min,
        d_max=args.d_max,
        mode=args.mode,
        radius=args.radius,
        close_hi=args.close_hi,
    )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = (
        pathlib.Path(args.out_dir)
        if args.out_dir
        else pathlib.Path(f"runs/meanfield/verify/mf_feat44_{args.mode}_{stamp}")
    )
    out.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "track": "meanfield",
        "verify_kind": "feature44_box_full_coverage_two_step_L2",
        "checkpoint": ckpt,
        "base_checkpoint": w.base_checkpoint,
        "mode": args.mode,
        "design": {
            "query_domain": "each free dim of 44-D MF feature",
            "property": "two_step_radial_L2",
            "deepset_buckets": False,
            "aa_ao_primary_split": False,
        },
        "config": {
            "max_splits": args.max_splits,
            "max_depth": args.max_depth,
            "marabou_timeout": args.marabou_timeout,
            "marabou_max_width": args.marabou_max_width,
            "use_marabou": not args.no_marabou,
            "wall_limit_s": args.wall_limit,
            "d_min": args.d_min,
            "d_max": args.d_max,
            "close_hi": args.close_hi,
            "radius": args.radius,
        },
        "claims": [],
    }

    all_ces: list[dict] = []
    print(f"[mf-verify] ckpt={ckpt}", flush=True)
    print(
        f"[mf-verify] mode={args.mode} claims={[c.name for c in claims]} "
        f"R={args.radius} marabou_max_w={args.marabou_max_width} → {out}",
        flush=True,
    )

    for claim in claims:
        print(f"[mf-verify] === {claim.name} (44-D feature box) ===", flush=True)
        (out / f"{claim.name}_manifest.json").write_text(
            json.dumps(claim_manifest(claim), indent=2) + "\n"
        )
        res = verify_claim(
            w,
            claim,
            max_splits=args.max_splits,
            max_depth=args.max_depth,
            marabou_timeout=args.marabou_timeout,
            sample_per_unknown=args.sample_per_unknown,
            seed=args.seed,
            use_marabou=not args.no_marabou,
            wall_limit_s=args.wall_limit,
            marabou_max_width=args.marabou_max_width,
        )
        report["claims"].append(
            {k: v for k, v in res.items() if k != "counterexamples"}
        )
        ces = res.get("counterexamples") or []
        all_ces.extend(ces)
        (out / f"{claim.name}.json").write_text(json.dumps(res, indent=2) + "\n")
        st = res["stats"]
        print(
            f"[mf-verify] {claim.name}: {res['status']} "
            f"proven={st['proven_leaves']} CE={st['violated_real']} "
            f"unknown={st['unknown_leaves']} marabou={st['marabou_calls']} "
            f"splits={st['splits']} wall={res['wall_time_s']}s",
            flush=True,
        )

    ce_path = out / "counterexamples_all.json"
    ce_path.write_text(
        json.dumps(
            {
                "n": len(all_ces),
                "counterexamples": all_ces,
                "note": "All real property violations found (true deploy policy).",
            },
            indent=2,
        )
        + "\n"
    )
    report["n_counterexamples_total"] = len(all_ces)
    statuses = [c["status"] for c in report["claims"]]
    if statuses and all(s == "PROVEN_FULL" for s in statuses):
        report["aggregate_status"] = "PROVEN_FULL"
    elif any(s == "VIOLATED" for s in statuses):
        report["aggregate_status"] = "VIOLATED_WITH_CES"
    elif any(s == "PARTIAL" for s in statuses):
        report["aggregate_status"] = "PARTIAL"
    else:
        report["aggregate_status"] = "UNKNOWN"

    (out / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")

    md = [
        "# MF verification — 44-D feature box + two-step L2",
        "",
        f"- checkpoint: `{ckpt}`",
        f"- mode: **{args.mode}**",
        f"- aggregate: **{report['aggregate_status']}**",
        f"- real CEs: **{len(all_ces)}**",
        "",
        "## Design (agreed)",
        "",
        "| | |",
        "|---|---|",
        "| Query domain | **Each free dim of 44-D MF features** (speed, goal, quad min_d / wmean / inv_ttc, …) |",
        "| Coverage | Branch-and-bound on feature box → aim **PROVEN_FULL** |",
        "| Property | **Two-step** radial L2 (horizon simplification) |",
        "| Not used | DeepSet (k,m) buckets; primary AA/AO claim split |",
        "",
        "## Per-claim",
        "",
        "| claim | status | proven | CE | unknown | marabou | splits | time |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in report["claims"]:
        st = c["stats"]
        md.append(
            f"| {c['claim']} | {c['status']} | {st['proven_leaves']} | "
            f"{st['violated_real']} | {st['unknown_leaves']} | {st['marabou_calls']} | "
            f"{st['splits']} | {c['wall_time_s']}s |"
        )
    md += [
        "",
        "PROVEN_FULL = full coverage of declared X, 0 real CE, 0 unknown.",
        "PARTIAL = unfinished leaves under resource limits (CEs still harvested).",
        "VIOLATED = ≥1 true-policy CE in `counterexamples_all.json`.",
        "",
        "DeepSet `runs/phase2/ACTIVE_*` untouched.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(md))
    # Always record latest run; only promote ACTIVE on PROVEN_FULL so
    # exploratory full/PARTIAL/VIOLATED jobs do not clobber the primary badge.
    latest = pathlib.Path("runs/meanfield/LATEST_MF_VERIFY.txt")
    latest.write_text(str(out) + "\n")
    agg = str(report.get("aggregate_status", ""))
    if agg == "PROVEN_FULL":
        ptr = pathlib.Path("runs/meanfield/ACTIVE_MF_VERIFY.txt")
        ptr.write_text(str(out) + "\n")
        print(f"[mf-verify] ACTIVE_MF_VERIFY → {out} (PROVEN_FULL)", flush=True)
    else:
        print(
            f"[mf-verify] ACTIVE_MF_VERIFY unchanged "
            f"(aggregate={agg}; see LATEST_MF_VERIFY.txt)",
            flush=True,
        )
    print(
        f"[mf-verify] DONE aggregate={agg} "
        f"ces={len(all_ces)} → {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
