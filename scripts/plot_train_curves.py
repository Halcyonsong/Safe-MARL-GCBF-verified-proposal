#!/usr/bin/env python3
"""Plot Stage3 training curves from eval CSV (and optional JSON history).

Examples:
  python scripts/plot_train_curves.py \\
    --csv method/deepset/output/stage3/repair/deepset_residual_gate_v2.csv \\
    --out-dir method/deepset/output/figures/curves

  python scripts/plot_train_curves.py \\
    --glob 'method/deepset/output/phase2/phase2_*/seed*/V2d/repair.csv' \\
    --out-dir method/deepset/output/phase2/latest/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _load_json_history(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    hist = d.get("eval_history") or []
    rows = []
    for h in hist:
        hit = h.get("hit_D") or {}
        rows.append(
            {
                "step": h.get("step"),
                "loss": h.get("loss"),
                "mae": h.get("mae"),
                "mean_hit_D": h.get("mean_hit_D"),
                "hit_aa": hit.get("l2_aa_near"),
                "hit_ao": hit.get("l2_ao_near"),
                "hit_11": hit.get("l2_11_near"),
                "safe_rate": h.get("safe_rate"),
                "success_rate": h.get("success_rate"),
                "eligible": h.get("eligible"),
                "is_best": h.get("is_best"),
                "wall_time_s": h.get("wall_time_s"),
            }
        )
    return rows


def _f(row: dict, key: str):
    v = row.get(key)
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def plot_one(rows: list[dict], title: str, out_base: Path) -> None:
    import matplotlib.pyplot as plt

    steps = [int(_f(r, "step") or 0) for r in rows]
    if not steps:
        print(f"  skip empty {title}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle(title, fontsize=12)

    def series(key):
        return [_f(r, key) for r in rows]

    ax = axes[0, 0]
    loss = series("loss")
    mae = series("mae")
    if any(v is not None for v in loss):
        ax.plot(steps, loss, "o-", color="#1f77b4", label="loss")
    if any(v is not None for v in mae):
        ax.plot(steps, mae, "s--", color="#ff7f0e", label="mae", alpha=0.8)
    ax.set_title("distill loss / mae")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    hit_keys = (
        ("hit_aa", "AA", "#d62728"),
        ("hit_ao", "AO", "#ff7f0e"),
        ("hit_11", "(1,1)", "#9467bd"),
        ("mean_hit_D", "mean", "k"),
    )
    plotted_hit = False
    for key, lab, c in hit_keys:
        ys = series(key)
        if any(v is not None for v in ys):
            ax.plot(steps, ys, "o-" if key != "mean_hit_D" else "s--", label=lab, color=c, alpha=0.9)
            plotted_hit = True
    if plotted_hit:
        ax.set_title("D-hit (eval resample)")
        ax.legend(fontsize=8)
    else:
        # Stage1: no D-hit in-loop — show reward/cost if present
        rew = series("reward")
        cost = series("cost")
        if any(v is not None for v in rew):
            ax.plot(steps, rew, "o-", color="#2ca02c", label="reward")
        if any(v is not None for v in cost):
            ax2 = ax.twinx()
            ax2.plot(steps, cost, "s--", color="#d62728", label="cost")
            ax2.set_ylabel("cost")
        ax.set_title("rollout reward (Stage1)")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, lab, c in (("success_rate", "success", "#2ca02c"), ("safe_rate", "safe", "#1f77b4")):
        ys = series(key)
        if any(v is not None for v in ys):
            ax.plot(steps, ys, "o-", label=lab, color=c)
    ax.set_title("rollout (short eval)")
    ax.set_xlabel("step")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    # mark eligible / best
    elig = [bool(_f(r, "eligible")) for r in rows]
    best = [bool(_f(r, "is_best")) for r in rows]
    mean_h = series("mean_hit_D")
    succ = series("success_rate")
    if any(v is not None for v in mean_h) and any(v is not None for v in succ):
        ax.plot(mean_h, succ, "-", color="0.7", alpha=0.5)
        for i, r in enumerate(rows):
            x, y = mean_h[i], succ[i]
            if x is None or y is None:
                continue
            color = "#2ca02c" if elig[i] else "#d62728"
            marker = "*" if best[i] else "o"
            ax.scatter([x], [y], c=color, marker=marker, s=80 if best[i] else 40, zorder=3)
            ax.annotate(str(steps[i]), (x, y), fontsize=7, alpha=0.7)
    ax.set_xlabel("mean D-hit")
    ax.set_ylabel("success")
    ax.set_title("Pareto (green=eligible, *=best)")
    ax.grid(True, alpha=0.3)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150)
        print(f"  wrote {p}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage3 eval CSV/JSON curves")
    parser.add_argument("--csv", action="append", default=[], help="path to repair.csv")
    parser.add_argument("--json", action="append", default=[], help="path to repair.json (eval_history)")
    parser.add_argument("--glob", default="", help="glob for CSV files")
    parser.add_argument("--out-dir", default="method/deepset/output/figures/curves")
    args = parser.parse_args()

    paths: list[Path] = []
    for c in args.csv:
        paths.append(Path(c))
    for j in args.json:
        paths.append(Path(j))
    if args.glob:
        paths.extend(sorted(Path().glob(args.glob)))

    # default: pilot residual v2 if present
    if not paths:
        cand = [
            Path("method/deepset/output/stage3/repair/deepset_residual_gate_v2.csv"),
            Path("method/deepset/output/stage3/repair/deepset_residual_gate_v2.json"),
            Path("method/deepset/output/stage3/repair/deepset_l2_safe_repair.json"),
        ]
        paths = [p for p in cand if p.exists()]

    if not paths:
        raise SystemExit("no CSV/JSON inputs found")

    out_dir = Path(args.out_dir)
    for p in paths:
        if p.suffix == ".csv":
            rows = _load_csv(p)
        elif p.suffix == ".json":
            rows = _load_json_history(p)
        else:
            print(f"  skip {p}")
            continue
        title = p.stem
        plot_one(rows, title=title, out_base=out_dir / title)
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
