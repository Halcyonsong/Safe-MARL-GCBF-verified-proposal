#!/usr/bin/env python3
"""Plot accumulative Stage1 compare rows (GCBF+ vs DeepSet bases)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(compare_dir: Path) -> list[dict]:
    p = compare_dir / "rows.jsonl"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    root = Path(args.compare_dir)
    out_dir = Path(args.out_dir or (root / "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    task = [r for r in rows if r.get("metric_kind", "task") == "task"]
    dhit = [r for r in rows if r.get("metric_kind") == "d_hit"]

    # --- task bars: group by episodes ---
    epis = sorted({int(r["episodes"]) for r in task if r.get("episodes") is not None})
    policies = []
    for r in task:
        pid = r.get("policy_id")
        if pid not in policies:
            policies.append(pid)
    colors = {
        "gcbfplus_teacher": "#2ca02c",
        "deepset_phase1_frozen": "#1f77b4",
        "deepset_redistill_seed0": "#ff7f0e",
        "residual_v2d_seed0base": "#d62728",
    }
    labels = {
        "gcbfplus_teacher": "GCBF+ GNN",
        "deepset_phase1_frozen": "Phase-1 frozen",
        "deepset_redistill_seed0": "redistill seed0",
        "residual_v2d_seed0base": "V2d residual",
    }

    metrics = [("success_rate", "success"), ("safe_rate", "safe"), ("finish_rate", "finish")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (key, title) in zip(axes, metrics):
        x = np.arange(len(epis))
        width = 0.8 / max(len(policies), 1)
        for i, pid in enumerate(policies):
            ys, yerr = [], []
            for e in epis:
                cand = [r for r in task if r.get("policy_id") == pid and int(r.get("episodes", -1)) == e]
                if not cand:
                    ys.append(np.nan)
                    yerr.append(0.0)
                    continue
                # latest row if duplicates
                r = cand[-1]
                ys.append(100.0 * float(r.get(key) or 0.0))
                se_key = "success_se_agent" if key == "success_rate" else "safe_se_agent"
                se = r.get(se_key)
                yerr.append(100.0 * float(se) if se is not None and key in ("success_rate", "safe_rate") else 0.0)
            ax.bar(
                x + i * width - 0.4 + width / 2,
                ys,
                width,
                yerr=yerr,
                label=labels.get(pid, pid),
                color=colors.get(pid, None),
                alpha=0.9,
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(e) for e in epis])
        ax.set_xlabel("episodes")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Stage1 compare — {root.name}")
    fig.savefig(out_dir / "task_bars.png", dpi=140)
    fig.savefig(out_dir / "task_bars.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'task_bars.png'}")

    # --- D-hit grouped bars ---
    if dhit:
        buckets = ["l2_aa_near", "l2_ao_near", "l2_11_near"]
        b_labs = ["AA", "AO", "(1,1)"]
        # unique policy order
        pids = []
        for r in dhit:
            pid = r.get("policy_id")
            if pid not in pids:
                pids.append(pid)
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        x = np.arange(len(buckets))
        width = 0.8 / max(len(pids), 1)
        for i, pid in enumerate(pids):
            cand = [r for r in dhit if r.get("policy_id") == pid]
            r = cand[-1]
            hits = r.get("d_hit") or {}
            ys = [100.0 * float(hits.get(b, np.nan)) for b in buckets]
            ax.bar(
                x + i * width - 0.4 + width / 2,
                ys,
                width,
                label=labels.get(pid, pid),
                color=colors.get(pid, None),
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(b_labs)
        ax.set_ylabel("D-hit rate (%)")
        ax.set_title("Same-std L2-D hit (lower better)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / "d_hit_bars.png", dpi=140)
        fig.savefig(out_dir / "d_hit_bars.pdf")
        plt.close(fig)
        print(f"wrote {out_dir / 'd_hit_bars.png'}")
    else:
        print("no d_hit rows yet — skip d_hit plot")

    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
