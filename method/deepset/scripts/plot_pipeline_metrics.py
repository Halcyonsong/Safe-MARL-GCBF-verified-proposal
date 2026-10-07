#!/usr/bin/env python3
"""Plot full-pipeline metrics for Safe-MARL GCBF-verified proposal.

Reads existing JSON artifacts (no re-training) and writes PNG/PDF figures under
``method/deepset/output/figures/`` (or ``--out-dir``).

Figures
-------
1. pipeline_d_hit_bars      — AA/AO/(1,1) D-hit: GCBF+ / frozen / v2 / r1
2. pipeline_task_bars       — 32-epi safe / finish / success
3. pipeline_pareto          — success vs mean D-hit scatter (Pareto view)
4. residual_v2_curves       — loss, D-hit, success, gate over residual v2 steps
5. residual_cegar_compare   — v2 vs r2 training curves
6. repair_family_bars       — before/after D-hit for main repair families
7. hybrid_lambda_bars       — hybrid λ sweep D-hit + short success
8. formal_verdict_stack     — Stage2 sampling/formal verdict counts (if present)

All plots are empirical visualization aids — not formal proofs.

Note (2026-10-07): Phase-1 ablation JSON under archive/ were deleted;
panels that depended on joint/hybrid/r2 dumps will be empty/skipped.
Active Stage3 metrics under method/deepset/output/stage3/ still load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

BUCKET_KEYS = ("l2_aa_near", "l2_ao_near", "l2_11_near")
BUCKET_LABELS = ("AA", "AO", "(1,1)")
COLORS = {
    "gcbfplus": "#1f77b4",
    "frozen": "#7f7f7f",
    "v2": "#2ca02c",
    "r1": "#d62728",
    "joint": "#ff7f0e",
    "r2": "#9467bd",
    "hybrid": "#8c564b",
    "v1": "#17becf",
}


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _hit(block: dict | None, key: str) -> float | None:
    if not block:
        return None
    b = block.get(key) or block.get(key.replace("l2_", ""))
    if b is None:
        return None
    if isinstance(b, dict):
        return float(b.get("hit_rate_on_D", b.get("hit", float("nan"))))
    return float(b)


def _pct(x: float | None) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    return 100.0 * float(x)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(fig, out_dir: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("png", "pdf"):
        fp = out_dir / f"{name}.{ext}"
        fig.savefig(fp, bbox_inches="tight")
        paths.append(fp)
    plt.close(fig)
    print(f"  wrote {out_dir / name}.{{png,pdf}}")
    return paths


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def assemble_main_table(root: Path) -> dict:
    """Static end-point metrics used in overview plots."""
    manifest = _load(root / "method/deepset/output/stage1/frozen/BASELINE_MANIFEST.json") or {}
    gcbf_cmp = _load(root / "method/deepset/output/stage2/l2_gcbfplus_vs_student_sampling_report.json") or {}
    policies = (gcbf_cmp.get("policies") or {})

    def from_policy(name: str):
        b = (policies.get(name) or {}).get("buckets") or {}
        return {k: _hit(b, k) for k in BUCKET_KEYS}

    # Prefer same-seed GCBF comparison report; fall back to known log numbers.
    gcbf_d = from_policy("gcbfplus_teacher")
    frozen_d = from_policy("deepset_frozen")
    v2_d = from_policy("residual_v2_best")
    r1_d = from_policy("r1_l2_safe")

    # Task metrics (32-epi) from manifest + experiment log constants
    gcbf_task = manifest.get("gcbfplus_teacher_eval_32epi_ref") or {
        "safe_rate": 0.95703,
        "finish_rate": 0.95312,
        "success_rate": 0.91406,
    }
    frozen_task = manifest.get("independent_eval_32epi") or {
        "safe_rate": 0.94141,
        "finish_rate": 0.92969,
        "success_rate": 0.87109,
    }
    # residual / r1 from summaries + known 32-epi
    v2_sum = _load(root / "method/deepset/output/stage3/repair/deepset_residual_gate_v2.json") or {}
    r1_sum = _load(root / "method/deepset/output/stage3/repair/deepset_l2_safe_repair.json") or {}
    joint_sum = _load(root / "REMOVED_archive/runs_phase1/phase1_artifacts/stage3_repair_ablations/deepset_l2_joint_safe_repair.json") or {}

    # Hard-coded 32-epi from EXPERIMENT_LOG when not in json
    v2_task = {"safe_rate": 0.934, "finish_rate": None, "success_rate": 0.848}
    r1_task = {"safe_rate": 0.980, "finish_rate": None, "success_rate": 0.711}
    joint_task = {"safe_rate": 0.94, "finish_rate": None, "success_rate": 0.773}

    # Fill D from repair jsons if comparison report missing a policy
    def fill_from_repair(d_map, summary, tag="after_best_D"):
        src = summary.get(tag) or summary.get("after_last_D") or {}
        for k in BUCKET_KEYS:
            if d_map.get(k) is None and k in src:
                d_map[k] = _hit(src, k)
        return d_map

    if all(v is None for v in v2_d.values()):
        v2_d = fill_from_repair({}, v2_sum)
    if all(v is None for v in r1_d.values()):
        r1_d = fill_from_repair({}, r1_sum)
    joint_d = fill_from_repair({}, joint_sum)

    return {
        "GCBF+": {"D": gcbf_d, "task": gcbf_task, "color": COLORS["gcbfplus"]},
        "frozen": {"D": frozen_d, "task": frozen_task, "color": COLORS["frozen"]},
        "residual v2": {"D": v2_d, "task": v2_task, "color": COLORS["v2"]},
        "r1": {"D": r1_d, "task": r1_task, "color": COLORS["r1"]},
        "joint min-int": {"D": joint_d, "task": joint_task, "color": COLORS["joint"]},
    }


def plot_d_hit_bars(table: dict, out_dir: Path) -> None:
    names = list(table.keys())
    x = np.arange(len(BUCKET_LABELS))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, name in enumerate(names):
        vals = [_pct(table[name]["D"].get(k)) for k in BUCKET_KEYS]
        ax.bar(
            x + (i - (len(names) - 1) / 2) * width,
            vals,
            width,
            label=name,
            color=table[name]["color"],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_ylabel("Hit rate on preventable D (%)")
    ax.set_xlabel("L2 threat bucket")
    ax.set_title("Same-standard L2-D collision hit rate (sampling, 2000/bucket)")
    ax.legend(ncol=min(3, len(names)), loc="upper right")
    ax.set_ylim(0, max(50, ax.get_ylim()[1] * 1.05))
    ax.axhline(0, color="k", lw=0.5)
    fig.text(
        0.01,
        0.01,
        "Empirical sampling on D only — not a formal proof.",
        fontsize=7,
        color="0.4",
    )
    save(fig, out_dir, "01_pipeline_d_hit_bars")


def plot_task_bars(table: dict, out_dir: Path) -> None:
    names = list(table.keys())
    metrics = [("safe_rate", "safe"), ("success_rate", "success")]
    x = np.arange(len(metrics))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for i, name in enumerate(names):
        vals = [_pct(table[name]["task"].get(m)) for m, _ in metrics]
        ax.bar(
            x + (i - (len(names) - 1) / 2) * width,
            vals,
            width,
            label=name,
            color=table[name]["color"],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("32-episode rate (%)")
    ax.set_title("Official-style 32-epi task / safety rates")
    ax.legend(ncol=min(3, len(names)))
    ax.set_ylim(60, 100)
    fig.text(
        0.01,
        0.01,
        "Rollout metrics (max_step=256). Not formal verification.",
        fontsize=7,
        color="0.4",
    )
    save(fig, out_dir, "02_pipeline_task_bars")


def plot_pareto(table: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for name, block in table.items():
        hits = [block["D"].get(k) for k in BUCKET_KEYS]
        hits = [h for h in hits if h is not None]
        if not hits:
            continue
        mean_d = 100.0 * float(np.mean(hits))
        succ = _pct(block["task"].get("success_rate"))
        ax.scatter(
            mean_d,
            succ,
            s=120,
            color=block["color"],
            zorder=3,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.annotate(name, (mean_d, succ), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Mean D-hit AA/AO/(1,1) (%)  → safer left")
    ax.set_ylabel("32-epi success (%)  → better up")
    ax.set_title("Safety–task Pareto (empirical)")
    # Ideal corner guide
    ax.axhline(80, color="0.6", ls="--", lw=0.8, label="success floor 80%")
    ax.legend(loc="lower left")
    # Invert x so left = safer
    ax.invert_xaxis()
    fig.text(
        0.01,
        0.01,
        "Upper-left is better (high success, low D-hit). Sampling D-hit ≠ proof.",
        fontsize=7,
        color="0.4",
    )
    save(fig, out_dir, "03_pipeline_pareto")


def plot_residual_curves(root: Path, out_dir: Path) -> None:
    v2 = _load(root / "method/deepset/output/stage3/repair/deepset_residual_gate_v2.json")
    if not v2 or not v2.get("eval_history"):
        print("  skip residual curves (no eval_history)")
        return
    hist = v2["eval_history"]
    steps = [h["step"] for h in hist]
    loss = [h["loss"] for h in hist]
    succ = [_pct(h.get("success_rate")) for h in hist]
    safe = [_pct(h.get("safe_rate")) for h in hist]
    mean_d = [_pct(h.get("mean_hit_D")) for h in hist]
    aa = [_pct(h.get("hit_D", {}).get("l2_aa_near")) for h in hist]
    ao = [_pct(h.get("hit_D", {}).get("l2_ao_near")) for h in hist]
    d11 = [_pct(h.get("hit_D", {}).get("l2_11_near")) for h in hist]
    g_aa = [100 * h.get("mean_gate_on_D", {}).get("l2_aa_near", float("nan")) for h in hist]
    g_ao = [100 * h.get("mean_gate_on_D", {}).get("l2_ao_near", float("nan")) for h in hist]
    g_11 = [100 * h.get("mean_gate_on_D", {}).get("l2_11_near", float("nan")) for h in hist]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    ax = axes[0, 0]
    ax.plot(steps, loss, "o-", color="#1f77b4", label="train loss")
    ax.set_ylabel("Loss")
    ax.set_title("Residual v2: loss")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, aa, "o-", label="AA", color="#d62728")
    ax.plot(steps, ao, "s-", label="AO", color="#ff7f0e")
    ax.plot(steps, d11, "^-", label="(1,1)", color="#9467bd")
    ax.plot(steps, mean_d, "k--", label="mean", alpha=0.7)
    ax.set_ylabel("D-hit (%)")
    ax.set_title("D-hit during residual v2 repair")
    ax.legend(ncol=2)

    ax = axes[1, 0]
    ax.plot(steps, succ, "o-", color="#2ca02c", label="success")
    ax.plot(steps, safe, "s-", color="#1f77b4", label="safe")
    ax.axhline(80, color="0.5", ls="--", lw=0.8)
    ax.set_ylabel("Short-eval rate (%)")
    ax.set_xlabel("Train step")
    ax.set_title("Task metrics (short eval)")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, g_aa, "o-", label="g AA")
    ax.plot(steps, g_ao, "s-", label="g AO")
    ax.plot(steps, g_11, "^-", label="g (1,1)")
    ax.set_ylabel("Mean gate on D (%)")
    ax.set_xlabel("Train step")
    ax.set_title("Danger gate activation")
    ax.legend()

    # mark best if known
    best = v2.get("best_checkpoint")
    if best and "step" in str(best):
        pass
    # eligible markers
    for h in hist:
        if h.get("eligible"):
            for a in axes.ravel():
                a.axvline(h["step"], color="#2ca02c", alpha=0.25, lw=2)

    fig.suptitle("Residual-gate v2 training (frozen base, train gate+δ only)", y=1.01)
    fig.tight_layout()
    save(fig, out_dir, "04_residual_v2_curves")


def plot_cegar_compare(root: Path, out_dir: Path) -> None:
    v2 = _load(root / "method/deepset/output/stage3/repair/deepset_residual_gate_v2.json")
    r2 = _load(root / "REMOVED_archive/runs_phase1/phase1_artifacts/stage3_repair_ablations/deepset_residual_gate_v2_r2.json")
    if not v2 or not r2:
        print("  skip cegar compare")
        return

    def series(doc, key_fn):
        hist = doc.get("eval_history") or []
        return [h["step"] for h in hist], [key_fn(h) for h in hist]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    # mean D-hit
    ax = axes[0]
    for doc, name, c in (
        (v2, "v2", COLORS["v2"]),
        (r2, "CEGAR r2", COLORS["r2"]),
    ):
        s, y = series(doc, lambda h: _pct(h.get("mean_hit_D")))
        ax.plot(s, y, "o-", label=name, color=c)
    ax.set_title("Mean D-hit")
    ax.set_xlabel("step")
    ax.set_ylabel("%")
    ax.legend()

    ax = axes[1]
    for doc, name, c in ((v2, "v2", COLORS["v2"]), (r2, "r2", COLORS["r2"])):
        s, y = series(doc, lambda h: _pct(h.get("success_rate")))
        ax.plot(s, y, "o-", label=name, color=c)
    ax.axhline(80, ls="--", color="0.5", lw=0.8)
    ax.set_title("Short-eval success")
    ax.set_xlabel("step")
    ax.set_ylabel("%")
    ax.legend()

    ax = axes[2]
    # endpoint bar: before-after from full resample if present
    labels = ["AA", "AO", "(1,1)"]
    v2b = v2.get("after_best_D") or v2.get("after_last_D") or {}
    r2b = r2.get("after_last_D") or r2.get("after_best_D") or {}
    init = v2.get("before_D_base") or {}
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, [_pct(_hit(init, k)) for k in BUCKET_KEYS], w, label="frozen", color=COLORS["frozen"])
    ax.bar(x, [_pct(_hit(v2b, k)) for k in BUCKET_KEYS], w, label="v2 best", color=COLORS["v2"])
    ax.bar(x + w, [_pct(_hit(r2b, k)) for k in BUCKET_KEYS], w, label="r2 last", color=COLORS["r2"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("End-point D-hit")
    ax.set_ylabel("%")
    ax.legend(fontsize=7)

    fig.suptitle("Residual CEGAR: v2 → collect CE → r2 (gate/δ only)", y=1.05)
    fig.tight_layout()
    save(fig, out_dir, "05_residual_cegar_compare")


def plot_repair_family(root: Path, out_dir: Path) -> None:
    """Before/after grouped bars for main full-net + residual families."""
    specs = [
        ("frozen→r1", "method/deepset/output/stage3/repair/deepset_l2_safe_repair.json", COLORS["r1"]),
        ("frozen→joint", "REMOVED_archive/runs_phase1/phase1_artifacts/stage3_repair_ablations/deepset_l2_joint_safe_repair.json", COLORS["joint"]),
        ("frozen→v2", "method/deepset/output/stage3/repair/deepset_residual_gate_v2.json", COLORS["v2"]),
        ("v2→r2", "REMOVED_archive/runs_phase1/phase1_artifacts/stage3_repair_ablations/deepset_residual_gate_v2_r2.json", COLORS["r2"]),
    ]
    rows = []
    for label, rel, color in specs:
        d = _load(root / rel)
        if not d:
            continue
        before = d.get("before_D") or d.get("before_D_base") or d.get("before_D_init_residual") or {}
        after = d.get("after_best_D") or d.get("after_last_D") or {}
        rows.append((label, before, after, color))
    if not rows:
        print("  skip repair family")
        return

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), sharey=True)
    for ax, bkey, blab in zip(axes, BUCKET_KEYS, BUCKET_LABELS):
        labs, befs, afts, cols = [], [], [], []
        for label, before, after, color in rows:
            labs.append(label)
            befs.append(_pct(_hit(before, bkey)))
            afts.append(_pct(_hit(after, bkey)))
            cols.append(color)
        x = np.arange(len(labs))
        ax.bar(x - 0.2, befs, 0.4, label="before", color="0.75", edgecolor="white")
        ax.bar(x + 0.2, afts, 0.4, label="after", color=cols, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=20, ha="right")
        ax.set_title(blab)
        if ax is axes[0]:
            ax.set_ylabel("D-hit (%)")
            ax.legend(loc="upper right")
    fig.suptitle("Repair families: before vs after D-hit", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "06_repair_family_before_after")


def plot_hybrid_lambda(root: Path, out_dir: Path) -> None:
    lams = []
    for lam in ("03", "05", "07"):
        d = _load(root / f"REMOVED_archive/runs_phase1/phase1_artifacts/stage3_repair_ablations/deepset_l2_hybrid_lam{lam}_repair.json")
        if d:
            lams.append((float(d.get("hybrid_lam", int(lam) / 10)), d))
    if not lams:
        print("  skip hybrid lambda")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    xs = [L for L, _ in lams]
    for bkey, blab, marker in zip(BUCKET_KEYS, BUCKET_LABELS, ("o", "s", "^")):
        ys = [_pct(_hit(d.get("after_best_D"), bkey)) for _, d in lams]
        ax.plot(xs, ys, marker + "-", label=blab)
    ax.set_xlabel("hybrid λ")
    ax.set_ylabel("D-hit after (%)")
    ax.set_title("Hybrid λ sweep — D-hit")
    ax.legend()

    ax = axes[1]
    ys = [_pct(d.get("best_success_short_eval")) for _, d in lams]
    ax.plot(xs, ys, "o-", color=COLORS["hybrid"])
    ax.axhline(80, ls="--", color="0.5")
    ax.set_xlabel("hybrid λ")
    ax.set_ylabel("Short-eval success (%)")
    ax.set_title("Hybrid λ sweep — success")
    fig.suptitle("Hybrid label scan (full-net)", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "07_hybrid_lambda_sweep")


def plot_formal_verdicts(root: Path, out_dir: Path) -> None:
    reports = []
    search_dirs = [
        root / "method/deepset/output/stage2",
        root / "REMOVED_archive/runs_phase1/phase1_artifacts/stage2_reports_old",
    ]
    seen = set()
    for ddir in search_dirs:
        if not ddir.exists():
            continue
        for p in sorted(ddir.glob("*report*.json")):
            if p.name in seen:
                continue
            seen.add(p.name)
            d = _load(p)
            if not d:
                continue
            vc = (d.get("summary") or {}).get("verdict_counts")
            if not vc:
                buckets = d.get("buckets")
                if isinstance(buckets, list):
                    from collections import Counter

                    c = Counter()
                    for b in buckets:
                        if isinstance(b, dict):
                            for q in b.get("queries", []) if "queries" in b else [b]:
                                v = q.get("verdict") or q.get("result")
                                if v:
                                    c[v] += 1
                    if c:
                        vc = dict(c)
            if vc:
                reports.append((p.stem.replace("l2_", "")[:28], vc))
    if not reports:
        print("  skip formal verdicts (no verdict_counts)")
        return
    # stack
    all_keys = sorted({k for _, vc in reports for k in vc})
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(reports) + 2), 4.5))
    x = np.arange(len(reports))
    bottom = np.zeros(len(reports))
    cmap = {
        "proven": "#2ca02c",
        "proven_ibp": "#98df8a",
        "violated": "#d62728",
        "timeout": "#ffbb78",
        "unknown": "#c7c7c7",
        "error": "#843c39",
    }
    for k in all_keys:
        vals = [vc.get(k, 0) for _, vc in reports]
        ax.bar(x, vals, bottom=bottom, label=k, color=cmap.get(k, None))
        bottom = bottom + np.asarray(vals, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in reports], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("# bucket queries")
    ax.set_title("Stage2 formal/sampling verdict counts (available reports)")
    ax.legend(ncol=3, fontsize=7)
    fig.text(
        0.01,
        0.01,
        "Most main-box results are violated/timeout — proven still sparse.",
        fontsize=7,
        color="0.4",
    )
    fig.tight_layout()
    save(fig, out_dir, "08_formal_verdict_stack")


def plot_catalog_card(out_dir: Path) -> None:
    """One-page text catalog of what was / can be plotted."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.axis("off")
    text = """
Safe-MARL pipeline — what can be plotted
========================================
[Stage0] Official GCBF+ teacher
  - 32-epi safe / finish / success          [yes] fig 02
  - L2-D AA/AO/(1,1) hit (same standard)    [yes] fig 01, 03
  - Train loss/reward curves                [no]  teacher log has models only

[Stage1] DeepSet distillation
  - End loss / short-eval success           [yes] snapshot in manifest
  - Full step-wise distill curve            [no]  not saved as time series
  - 32-epi vs teacher                       [yes] fig 02

[Stage2] Verification
  - Sampling hit on D                       [yes] fig 01
  - Formal verdict stack (proven/violated)  [yes] fig 08 (where reports exist)
  - CE counts / mean delta-a                [json; optional]

[Stage3] Repair
  - r1 / joint / hybrid before->after D-hit [yes] fig 06, 07
  - Residual v2 loss, D-hit, success, gate  [yes] fig 04
  - CEGAR r2 vs v2                          [yes] fig 05
  - Safety-task Pareto                      [yes] fig 03

Claim boundary: all figures are empirical. D-hit != Marabou proof.
""".strip()
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)
    save(fig, out_dir, "00_plot_catalog")


def write_index_md(out_dir: Path, table: dict) -> None:
    lines = [
        "# Pipeline figures",
        "",
        "Generated by `scripts/plot_pipeline_metrics.py`. Empirical only.",
        "",
        "## Overview table (endpoints)",
        "",
        "| policy | AA | AO | (1,1) | success |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, b in table.items():
        aa = _pct(b["D"].get("l2_aa_near"))
        ao = _pct(b["D"].get("l2_ao_near"))
        d11 = _pct(b["D"].get("l2_11_near"))
        succ = _pct(b["task"].get("success_rate"))
        lines.append(
            f"| {name} | {aa:.1f}% | {ao:.1f}% | {d11:.1f}% | {succ:.1f}% |"
        )
    lines += [
        "",
        "## Files",
        "",
        "| # | file | meaning |",
        "|---|---|---|",
        "| 00 | `00_plot_catalog` | what can be plotted |",
        "| 01 | `01_pipeline_d_hit_bars` | D-hit vs GCBF+/students |",
        "| 02 | `02_pipeline_task_bars` | 32-epi safe/success |",
        "| 03 | `03_pipeline_pareto` | success vs mean D-hit |",
        "| 04 | `04_residual_v2_curves` | residual train curves |",
        "| 05 | `05_residual_cegar_compare` | v2 vs CEGAR r2 |",
        "| 06 | `06_repair_family_before_after` | repair families |",
        "| 07 | `07_hybrid_lambda_sweep` | hybrid λ |",
        "| 08 | `08_formal_verdict_stack` | Stage2 verdicts |",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))
    print(f"  wrote {out_dir / 'README.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(_ROOT))
    parser.add_argument("--out-dir", default="method/deepset/output/figures")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))

    print(f"plotting into {out_dir}")
    plot_catalog_card(out_dir)
    table = assemble_main_table(root)
    plot_d_hit_bars(table, out_dir)
    plot_task_bars(table, out_dir)
    plot_pareto(table, out_dir)
    plot_residual_curves(root, out_dir)
    plot_cegar_compare(root, out_dir)
    plot_repair_family(root, out_dir)
    plot_hybrid_lambda(root, out_dir)
    plot_formal_verdicts(root, out_dir)
    write_index_md(out_dir, table)
    print("done.")


if __name__ == "__main__":
    main()
