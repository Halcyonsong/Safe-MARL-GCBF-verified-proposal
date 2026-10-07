#!/usr/bin/env python3
"""Generate paper figures from paper_figure_data.json (clean academic style)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "paper_figure_data.json").read_text())
OUT = ROOT / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Soft academic palette (colorblind-friendly-ish)
C = {
    "MF-r2": "#1F4E79",
    "DS-r2": "#5B9BD5",
    "GCBF+": "#2E7D32",
    "GCBF": "#81C784",
    "HandCBF": "#7B1FA2",
    "MAPPO-Lag": "#EF6C00",
    "InforMARL": "#C62828",
    "base": "#90A4AE",
    "r1": "#5B9BD5",
    "r2": "#1F4E79",
    "success": "#5B9BD5",
    "safe": "#1F4E79",
    "finish": "#43A047",
    "accent": "#D84315",
    "grid": "#ECEFF1",
    "text": "#263238",
    "muted": "#78909C",
}


def apply_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.labelsize": 10,
            "axes.labelcolor": C["text"],
            "axes.edgecolor": "#B0BEC5",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": C["text"],
            "ytick.color": C["text"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CFD8DC",
            "legend.fancybox": True,
            "axes.grid": True,
            "grid.color": C["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
        }
    )


def save(fig, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def fig1_repair_ce_hit():
    d = DATA["fig1_repair_ce_hit"]
    stages = d["stages"]
    labels = [s["label"] for s in stages]
    vals = [100 * s["hit_on_danger"] for s in stages]
    colors = [C["base"], C["r1"], C["r2"]]

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.58, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for b, v, s in zip(bars, vals, stages):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max(vals) * 0.025,
            f"{v:.2f}%\n(n={s['n_hit']})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=C["text"],
            linespacing=1.15,
        )

    # bracket-style reduction callout above the tallest bar
    y_br = max(vals) * 1.12
    ax.plot([0, 2], [y_br, y_br], color=C["accent"], lw=1.3, solid_capstyle="round")
    ax.plot([0, 0], [y_br - 0.12, y_br], color=C["accent"], lw=1.3)
    ax.plot([2, 2], [y_br - 0.12, y_br], color=C["accent"], lw=1.3)
    ax.text(
        1.0,
        y_br + 0.12,
        "≈12.7× reduction",
        ha="center",
        va="bottom",
        fontsize=9,
        color=C["accent"],
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Two-step L2 hit on danger (%)")
    ax.set_title("Empirical CE repair trajectory")
    ax.set_ylim(0, max(vals) * 1.42)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    save(fig, "fig1_repair_ce_hit")


def fig2_repair_task_metrics():
    d = DATA["fig2_repair_task_metrics"]
    stages = d["stages"]
    labels = [s["label"] for s in stages]
    metrics = ["success", "safe", "finish"]
    metric_labels = ["Success", "Safe", "Finish"]
    metric_colors = [C["success"], C["safe"], C["finish"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [2.15, 1.0]})

    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.24
    for i, (m, ml, c) in enumerate(zip(metrics, metric_labels, metric_colors)):
        vals = [100 * s[m] for s in stages]
        ax.bar(x + (i - 1) * width, vals, width, label=ml, color=c, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(72, 100)
    ax.set_title("Task rates (@32-epi)")
    ax.legend(loc="lower left", ncol=3, columnspacing=0.9, handlelength=1.1, borderpad=0.35)

    ax = axes[1]
    costs = [s["cost"] for s in stages]
    colors = [C["base"], C["r1"], C["r2"]]
    bars = ax.bar(x, costs, width=0.52, color=colors, edgecolor="white")
    for b, v in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, v + max(costs) * 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cost ↓")
    ax.set_title("Episode cost")
    ax.set_ylim(0, max(costs) * 1.22)

    fig.suptitle("Task metrics before / after residual repair", y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, "fig2_repair_task_metrics")


def fig3_baseline_main_bar():
    d = DATA["fig3_baseline_main_bar"]
    order = d["order"]
    methods = d["methods"]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5))
    panels = [
        ("success", "Success ↑ (%)"),
        ("safe", "Safe ↑ (%)"),
        ("cost", "Cost ↓"),
    ]
    x = np.arange(len(order))
    for ax, (key, ylab) in zip(axes, panels):
        scale = 100 if key != "cost" else 1
        vals = [methods[m][key] * scale for m in order]
        colors = [C[m] for m in order]
        bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.70, zorder=3)
        # emphasize MF without hatch clutter: thicker edge + slight alpha on others
        for i, b in enumerate(bars):
            if order[i] == "MF-r2":
                b.set_edgecolor("#0D47A1")
                b.set_linewidth(1.6)
                b.set_zorder(4)
            else:
                b.set_alpha(0.88)
        for b, v in zip(bars, vals):
            txt = f"{v:.1f}" if key != "cost" else f"{v:.2f}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.02,
                txt,
                ha="center",
                va="bottom",
                fontsize=7,
                color=C["text"],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=32, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.20)
        if key == "safe":
            ax.axhline(95, color="#B0BEC5", ls="--", lw=0.9, zorder=1)
            ax.text(len(order) - 0.1, 95.6, "95%", ha="right", va="bottom", fontsize=7, color=C["muted"])
        if key == "success":
            ax.text(
                0,
                vals[0] + max(vals) * 0.10,
                "★ formal L2",
                ha="center",
                fontsize=7.5,
                color=C["MF-r2"],
                fontweight="bold",
            )

    fig.suptitle("Main-protocol comparison (@32-epi, N=8)", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_baseline_main_bar")


def fig4_transfer_n():
    d = DATA["fig4_transfer_n"]
    n_list = d["n_list"]
    methods = d["methods"]
    order = ["GCBF+", "DS-r2", "HandCBF", "MF-r2", "MAPPO-Lag", "InforMARL"]
    markers = {"GCBF+": "o", "DS-r2": "s", "HandCBF": "D", "MF-r2": "*", "MAPPO-Lag": "^", "InforMARL": "v"}
    lw = {"MF-r2": 2.6, "GCBF+": 1.9, "DS-r2": 1.6}

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharex=True)
    for ax, metric, title in zip(axes, ["success", "safe"], ["Success ↑", "Safe ↑"]):
        for m in order:
            ys = [100 * methods[m][str(n)][metric] for n in n_list]
            ax.plot(
                n_list,
                ys,
                marker=markers[m],
                markersize=11 if m == "MF-r2" else 6.5,
                lw=lw.get(m, 1.35),
                color=C[m],
                label=m,
                zorder=6 if m == "MF-r2" else 3,
                markeredgecolor="white",
                markeredgewidth=0.6 if m == "MF-r2" else 0.3,
            )
        ax.set_xticks(n_list)
        ax.set_xlabel("Number of agents N")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title)
        ax.set_ylim(18, 105)
        ax.set_xlim(6.5, 34)

    handles, labels = axes[0].get_legend_handles_labels()
    # put legend ABOVE plots so it never overlaps axes
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=8,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.6,
    )
    fig.suptitle("Zero-shot transfer to more agents (trained at N=8)", y=1.14, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "fig4_transfer_n")


def fig5_latency():
    d = DATA["fig5_latency"]
    methods = list(d["methods"].keys())
    med = np.array([d["methods"][m]["median_ms"] for m in methods])
    p95 = np.array([d["methods"][m]["p95_ms"] for m in methods])
    colors = [C.get(m, "#546E7A") for m in methods]

    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    x = np.arange(len(methods))
    bars = ax.bar(x, med, width=0.52, color=colors, edgecolor="white", label="median", zorder=3)
    # p95 as thin markers above bars (cleaner than errorbar spanning into labels)
    ax.scatter(x, p95, marker="_", s=280, color="#37474F", linewidths=2.2, zorder=5, label="p95")
    for xi, m, p in zip(x, med, p95):
        ax.text(xi, p + 0.35, f"{m:.2f} / {p:.2f}", ha="center", va="bottom", fontsize=7.5, color=C["text"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Latency (ms / control call)")
    ax.set_title("Control compute latency")
    ax.set_ylim(0, max(p95) * 1.32)
    ax.legend(loc="upper left", frameon=True)
    ax.text(0.98, 0.02, "median / p95 (ms)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=C["muted"])
    save(fig, "fig5_latency")


def fig6_pressure_stability():
    d = DATA["fig6_pressure_stability"]
    pts = d["points"]
    epis = [p["epi"] for p in pts]

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    series = [
        ("success", "Success", C["success"], "o"),
        ("safe", "Safe", C["safe"], "s"),
        ("finish", "Finish", C["finish"], "^"),
    ]
    for key, label, color, mk in series:
        ys = [100 * p[key] for p in pts]
        ax.plot(epis, ys, marker=mk, markersize=7.5, lw=2.1, color=color, label=label, markeredgecolor="white", markeredgewidth=0.5)
        # place labels with slight vertical offsets to avoid overlap
        offsets = {"success": -1.3, "safe": 0.7, "finish": 0.7}
        for x, y in zip(epis, ys):
            ax.text(x, y + offsets[key], f"{y:.1f}", ha="center", va="bottom" if offsets[key] > 0 else "top", fontsize=7.5, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(epis)
    ax.set_xticklabels([str(e) for e in epis])
    ax.minorticks_off()
    ax.set_xlabel("Evaluation episodes")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(81, 99)
    ax.set_title("MF-r2 stability under longer evaluation")
    ax.legend(loc="lower right", frameon=True)
    save(fig, "fig6_pressure_stability")


def fig7_pareto_safe_success():
    d = DATA["fig7_pareto_safe_success"]
    pts = d["points"]

    # manual label offsets (points) to avoid collisions
    offsets = {
        "MF-r2": (-55, -18),
        "DS-r2": (8, -14),
        "GCBF+": (-8, 10),
        "GCBF": (8, -6),
        "HandCBF": (-8, -16),
        "MAPPO-Lag": (8, 6),
        "InforMARL": (8, 6),
    }

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for p in pts:
        x, y = 100 * p["success"], 100 * p["safe"]
        c = C[p["id"]]
        if p.get("formal_l2"):
            ax.scatter([x], [y], s=260, c=c, marker="*", zorder=5, edgecolors="white", linewidths=0.9)
        else:
            ax.scatter([x], [y], s=78, c=c, marker="o", zorder=4, edgecolors="white", linewidths=0.7)
        dx, dy = offsets[p["id"]]
        ax.annotate(
            p["id"],
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.5,
            color=c,
            fontweight="bold" if p.get("formal_l2") else "normal",
            arrowprops=dict(arrowstyle="-", color=c, lw=0.7, shrinkA=0, shrinkB=4) if abs(dx) > 20 or abs(dy) > 12 else None,
        )

    ax.set_xlabel("Success ↑ (%)")
    ax.set_ylabel("Safe ↑ (%)")
    ax.set_xlim(52, 98)
    ax.set_ylim(58, 103)
    ax.axhline(95, color="#CFD8DC", ls="--", lw=0.9)
    ax.axvline(80, color="#CFD8DC", ls="--", lw=0.9)
    ax.set_title("Safe vs success (@32-epi)")
    handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C["MF-r2"], markersize=15, label="MF-r2 (formal L2)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#78909C", markersize=8, label="Baselines / refs"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    save(fig, "fig7_pareto_safe_success")


def fig8_distill_train():
    d = DATA["fig8_distill_train"]
    loss = d["loss"]
    ev = d["eval"]
    deployed = d.get("deployed_step", 9000)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # left: distill loss + MAE
    ax = axes[0]
    steps = [p["step"] for p in loss]
    ax.plot(steps, [p["loss"] for p in loss], color=C["MF-r2"], lw=1.6, label="distill loss")
    ax.plot(steps, [p["mae"] for p in loss], color=C["DS-r2"], lw=1.3, ls="--", label="MAE")
    ax.axvline(deployed, color=C["accent"], ls=":", lw=1.2, label=f"frozen base @{deployed}")
    ax.set_xlabel("Train step")
    ax.set_ylabel("Loss / MAE")
    ax.set_title("Distillation objective")
    ax.set_xlim(0, max(steps) * 1.02)
    ax.legend(loc="upper right", fontsize=8)

    # right: periodic short eval
    ax = axes[1]
    es = [p["step"] for p in ev]
    ax.plot(es, [100 * p["success"] for p in ev], color=C["success"], marker="o", markersize=3.5, lw=1.5, label="Success")
    ax.plot(es, [100 * p["safe"] for p in ev], color=C["safe"], marker="s", markersize=3.5, lw=1.5, label="Safe")
    ax.plot(es, [100 * p["finish"] for p in ev], color=C["finish"], marker="^", markersize=3.5, lw=1.3, label="Finish")
    ax.axvline(deployed, color=C["accent"], ls=":", lw=1.2)
    # mark short-eval peak success
    best = max(ev, key=lambda p: p["success"])
    ax.scatter([best["step"]], [100 * best["success"]], s=60, c=C["accent"], zorder=5, marker="*")
    ax.annotate(
        f"short-eval peak\n{100*best['success']:.0f}% @{best['step']}",
        xy=(best["step"], 100 * best["success"]),
        xytext=(best["step"] - 2200, 100 * best["success"] - 18),
        fontsize=7.5,
        color=C["accent"],
        arrowprops=dict(arrowstyle="->", color=C["accent"], lw=0.9),
    )
    ax.set_xlabel("Train step")
    ax.set_ylabel("Short-eval rate (%)")
    ax.set_title("Periodic short eval (noisy)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(es) * 1.02)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(d["title"], y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig8_distill_train")


def fig9_repair_train():
    d = DATA["fig9_repair_train"]
    r1, r2 = d["r1"], d["r2"]
    fin = d["final_acceptance"]

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.35))

    # panel 0: loss
    ax = axes[0]
    ax.plot([p["step"] for p in r1], [p["loss"] for p in r1], color=C["r1"], marker="o", lw=1.8, label="r1", markersize=6)
    ax.plot([p["step"] for p in r2], [p["loss"] for p in r2], color=C["r2"], marker="s", lw=1.8, label="r2", markersize=6)
    ax.set_xlabel("Repair step")
    ax.set_ylabel("Training loss")
    ax.set_title("Residual loss")
    ax.legend(loc="upper right")

    # panel 1: short-eval safe / success
    ax = axes[1]
    for series, name, color, mk in [(r1, "r1", C["r1"], "o"), (r2, "r2", C["r2"], "s")]:
        ax.plot([p["step"] for p in series], [100 * p["safe"] for p in series], color=color, marker=mk, lw=1.8, ls="-", label=f"{name} safe", markersize=6)
        ax.plot([p["step"] for p in series], [100 * p["success"] for p in series], color=color, marker=mk, lw=1.3, ls="--", label=f"{name} suc", markersize=5, alpha=0.85)
    # mark BEST steps
    for series, color in [(r1, C["r1"]), (r2, C["r2"])]:
        for p in series:
            if p.get("is_best"):
                ax.scatter([p["step"]], [100 * p["safe"]], s=90, facecolors="none", edgecolors=color, linewidths=1.6, zorder=5)
    ax.set_xlabel("Repair step")
    ax.set_ylabel("Short-eval rate (%)")
    ax.set_title("Short-eval (○ best ckpt)")
    ax.set_ylim(80, 100)
    ax.legend(loc="lower right", fontsize=7.5, ncol=2, columnspacing=0.8)

    # panel 2: final @32 acceptance bars (post-train)
    ax = axes[2]
    metrics = ["success", "safe", "finish"]
    metric_labels = ["Suc", "Safe", "Fin"]
    x = np.arange(len(metrics))
    w = 0.35
    v1 = [100 * fin["r1"][m] for m in metrics]
    v2 = [100 * fin["r2"][m] for m in metrics]
    ax.bar(x - w / 2, v1, w, color=C["r1"], edgecolor="white", label="r1 @32")
    ax.bar(x + w / 2, v2, w, color=C["r2"], edgecolor="white", label="r2 @32")
    for xi, a, b in zip(x, v1, v2):
        ax.text(xi - w / 2, a + 0.4, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w / 2, b + 0.4, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Final @32-epi (acceptance)")
    ax.set_ylim(75, 100)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(d["title"], y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig9_repair_train")


def fig_optional_midF_fringe():
    opt = DATA["optional_not_primary_figures"]["midF_ce_fringe_fail"]
    rows = opt["rows"]
    labels = [r["policy"] for r in rows]
    vals = [r["fail_pct"] for r in rows]
    cmap = {
        "coast": "#B0BEC5",
        "mf_base": "#90A4AE",
        "mf_r2": C["MF-r2"],
        "mappo_lag": C["MAPPO-Lag"],
        "gcbfplus": C["GCBF+"],
        "informarl": C["InforMARL"],
    }
    colors = [cmap.get(lb, "#90A4AE") for lb in labels]

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.62)
    for b, v, r in zip(bars, vals, rows):
        ax.text(
            b.get_x() + b.get_width() / 2,
            (v + 0.12) if v > 0 else 0.12,
            f"{v:.1f}% ({r['n_fail']})",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Binary radial-L2 fail (%)")
    ax.set_title("mid-F CE-fringe fails  [appendix only]")
    ax.set_ylim(0, max(vals + [1.0]) * 1.55)
    ax.text(
        0.98,
        0.95,
        "Formal mid-F still VIOLATED\n(not dual-gate main claim)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=C["accent"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFEBEE", edgecolor="#FFCDD2", linewidth=0.8),
    )
    save(fig, "figA_midF_ce_fringe_appendix")


def main():
    apply_style()
    print("Generating paper figures →", OUT)
    fig1_repair_ce_hit()
    fig2_repair_task_metrics()
    fig3_baseline_main_bar()
    fig4_transfer_n()
    fig5_latency()
    fig6_pressure_stability()
    fig7_pareto_safe_success()
    fig8_distill_train()
    fig9_repair_train()
    fig_optional_midF_fringe()
    print("done.")


if __name__ == "__main__":
    main()
