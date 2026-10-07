#!/usr/bin/env python3
"""Compare multi-seed Stage1 redistill runs and recommend a new DeepSet base.

Reads each seed's validate_32epi.txt, summary JSON, optional d_hit JSON.
Does **not** overwrite Phase-1 frozen; writes SELECTION.md + ranking.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_validate_txt(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    for key, pat in (
        ("safe_rate", r"safe_rate=([0-9.]+)%"),
        ("finish_rate", r"finish_rate=([0-9.]+)%"),
        ("success_rate", r"success_rate=([0-9.]+)%"),
        ("reward", r"reward=([-\d.]+)"),
        ("cost", r"cost=([-\d.]+)"),
    ):
        m = re.search(pat, text)
        if m:
            v = float(m.group(1))
            if key.endswith("_rate"):
                v /= 100.0
            out[key] = v
    return out


def _d_hits(path: Path) -> dict:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    pols = d.get("policies") or {}
    # first policy buckets
    for _name, pol in pols.items():
        buckets = pol.get("buckets") or {}
        return {
            k: float(v.get("hit_rate_on_D", 0.0))
            for k, v in buckets.items()
            if isinstance(v, dict)
        }
    return {}


def _score(row: dict) -> tuple:
    """Higher better: success_32, -mean_D_hit, -cost."""
    succ = float(row.get("success_32", 0.0) or 0.0)
    hits = row.get("d_hit") or {}
    mean_d = None
    if hits:
        mean_d = sum(hits.values()) / len(hits)
    cost = float(row.get("cost_32", 0.0) or 0.0)
    # missing D → neutral
    md = mean_d if mean_d is not None else 0.5
    return (succ, -md, -cost)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        required=True,
        help="method/deepset/output/stage1/redistill/<run_id>",
    )
    parser.add_argument(
        "--phase1-frozen",
        default="method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl",
    )
    args = parser.parse_args()
    root = Path(args.run_dir)
    if not root.is_dir():
        raise SystemExit(f"missing {root}")

    rows = []
    skipped = []
    for seed_dir in sorted(root.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        # Incomplete / aborted seeds: no final 32-epi validate → do not rank
        if (seed_dir / "ABORTED.md").exists() or not (seed_dir / "validate_32epi.txt").exists():
            skipped.append(
                {
                    "seed_dir": str(seed_dir),
                    "reason": (
                        "ABORTED.md"
                        if (seed_dir / "ABORTED.md").exists()
                        else "missing validate_32epi.txt"
                    ),
                }
            )
            continue
        mpath = seed_dir / "MANIFEST.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        summary = {}
        sj = seed_dir / "deepset_actor512_student.json"
        if sj.exists():
            summary = json.loads(sj.read_text())
        val = _parse_validate_txt(seed_dir / "validate_32epi.txt")
        dhit = _d_hits(seed_dir / "d_hit_seed123.json")
        row = {
            "seed_dir": str(seed_dir),
            "train_seed": manifest.get("train_seed"),
            "best_checkpoint": manifest.get("best_checkpoint")
            or str(seed_dir / "deepset_actor512_student_best.pkl"),
            "success_32": val.get("success_rate"),
            "safe_32": val.get("safe_rate"),
            "finish_32": val.get("finish_rate"),
            "reward_32": val.get("reward"),
            "cost_32": val.get("cost"),
            "best_success_short": summary.get("best_success_short_eval"),
            "best_step": summary.get("best_step"),
            "best_mae_seen": summary.get("best_mae_seen"),
            "wall_time_s": summary.get("wall_time_total_s"),
            "d_hit": dhit,
            "mean_d_hit": (sum(dhit.values()) / len(dhit)) if dhit else None,
        }
        rows.append(row)

    if not rows:
        raise SystemExit(f"no complete seed* under {root} (skipped={skipped})")

    ranked = sorted(rows, key=_score, reverse=True)
    best = ranked[0]

    # Phase-1 frozen reference numbers (from manifest if present)
    phase1_ref = {
        "checkpoint": args.phase1_frozen,
        "success_32_ref": 0.87109,
        "safe_32_ref": 0.94141,
        "note": "Phase-1 frozen independent_eval_32epi from BASELINE_MANIFEST",
    }

    ranking_path = root / "ranking.json"
    ranking_path.write_text(
        json.dumps(
            {
                "run_dir": str(root),
                "phase1_ref": phase1_ref,
                "ranked": ranked,
                "recommended": best,
                "skipped_incomplete": skipped,
            },
            indent=2,
        )
    )

    lines = [
        f"# Stage1 redistill selection — `{root.name}`",
        "",
        f"Teacher-side distill only. **Does not** overwrite `{args.phase1_frozen}`.",
        "",
        "## Ranking (success_32 ↓, then lower mean D-hit, then lower cost)",
        "",
        "| rank | seed | success_32 | safe_32 | mean_D_hit | best_step | wall_s | ckpt |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(ranked, 1):
        md = r.get("mean_d_hit")
        md_s = f"{md*100:.1f}%" if md is not None else "—"
        succ = r.get("success_32")
        safe = r.get("safe_32")
        lines.append(
            f"| {i} | {r.get('train_seed')} | "
            f"{(succ or 0)*100:.2f}% | {(safe or 0)*100:.2f}% | {md_s} | "
            f"{r.get('best_step')} | {r.get('wall_time_s')} | "
            f"`{r.get('best_checkpoint')}` |"
        )

    lines += [
        "",
        "## Recommended",
        "",
        f"- **checkpoint:** `{best.get('best_checkpoint')}`",
        f"- **train_seed:** {best.get('train_seed')}",
        f"- **success_32:** {(best.get('success_32') or 0)*100:.2f}% "
        f"(Phase-1 frozen ref ≈ {phase1_ref['success_32_ref']*100:.1f}%)",
        f"- **safe_32:** {(best.get('safe_32') or 0)*100:.2f}%",
        "",
    ]
    if skipped:
        lines += [
            "## Incomplete / aborted (not ranked)",
            "",
        ]
        for s in skipped:
            lines.append(f"- `{s['seed_dir']}` — {s['reason']}")
        lines.append("")
    lines += [
        "## Promote? (manual)",
        "",
        "Only after you review curves + D-hit:",
        "",
        "```bash",
        f"# optional: keep Phase-1 frozen as-is; point Stage3 at new base via --base-checkpoint",
        f"NEW_BASE=\"{best.get('best_checkpoint')}\"",
        f"# or copy into a *new* frozen slot (do not clobber Phase-1 name unless intentional):",
        f"# cp \"$NEW_BASE\" method/deepset/output/stage1/frozen/deepset_actor512_redistill_best.pkl",
        "```",
        "",
        f"Full JSON: `{ranking_path}`",
        "",
    ]
    sel = root / "SELECTION.md"
    sel.write_text("\n".join(lines))
    print(sel.read_text())
    print(f"wrote {ranking_path}")


if __name__ == "__main__":
    main()
