#!/usr/bin/env python3
"""Generate planar-netdep acceptance figures (MF-r2 only; no DeepSet).

Data: paper_figure_data.json  (aligned with part/part5_experiments.md)
Out:  this directory (pdf + png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "paper_figure_data.json").read_text())
OUT = ROOT
OUT.mkdir(parents=True, exist_ok=True)

C = {
    "MF-r2": "#1F4E79",
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
    "m1": "#90CAF9",
    "m2": "#42A5F5",
    "m3": "#1F4E79",
}


def apply_style() -> None:
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


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def fig1_repair_ce_hit() -> None:
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

    y_br = max(vals) * 1.12
    ax.plot([0, 2], [y_br, y_br], color=C["accent"], lw=1.3, solid_capstyle="round")
    ax.plot([0, 0], [y_br - 0.12, y_br], color=C["accent"], lw=1.3)
    ax.plot([2, 2], [y_br - 0.12, y_br], color=C["accent"], lw=1.3)
    ax.text(
        1.0,
        y_br + 0.12,
        d.get("reduction_label", "≈12.7× reduction"),
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


def fig2_repair_task_metrics() -> None:
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


def fig3_baseline_main_bar() -> None:
    d = DATA["fig3_baseline_main_bar"]
    order = d["order"]
    methods = d["methods"]
    badge = d.get("formal_badge", "★ formal")

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.5))
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
        ax.set_xticklabels(order, rotation=28, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.22)
        if key == "safe":
            ax.axhline(95, color="#B0BEC5", ls="--", lw=0.9, zorder=1)
            ax.text(len(order) - 0.15, 95.6, "95%", ha="right", va="bottom", fontsize=7, color=C["muted"])
        if key == "success":
            ax.text(
                0,
                vals[0] + max(vals) * 0.11,
                badge,
                ha="center",
                fontsize=7.5,
                color=C["MF-r2"],
                fontweight="bold",
            )

    fig.suptitle("Main-protocol comparison (@32-epi, N=8)  ·  no DS", y=1.02, fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_baseline_main_bar")


def fig4_transfer_n() -> None:
    d = DATA["fig4_transfer_n"]
    n_list = d["n_list"]
    methods = d["methods"]
    order = ["GCBF+", "HandCBF", "MF-r2", "MAPPO-Lag", "InforMARL"]
    markers = {"GCBF+": "o", "HandCBF": "D", "MF-r2": "*", "MAPPO-Lag": "^", "InforMARL": "v"}
    lw = {"MF-r2": 2.6, "GCBF+": 1.9, "HandCBF": 1.5}

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharex=True)
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
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=8,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.6,
    )
    fig.suptitle("Zero-shot transfer to more agents (trained at N=8)  ·  no DS", y=1.14, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "fig4_transfer_n")


def fig5_latency() -> None:
    d = DATA["fig5_latency"]
    methods = list(d["methods"].keys())
    med = np.array([d["methods"][m]["median_ms"] for m in methods])
    p95 = np.array([d["methods"][m]["p95_ms"] for m in methods])
    colors = [C.get(m, "#546E7A") for m in methods]

    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    x = np.arange(len(methods))
    ax.bar(x, med, width=0.52, color=colors, edgecolor="white", label="median", zorder=3)
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


def fig6_pressure_stability() -> None:
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


def fig7_pareto_safe_success() -> None:
    d = DATA["fig7_pareto_safe_success"]
    pts = d["points"]

    offsets = {
        "MF-r2": (-55, -18),
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
        if p.get("formal"):
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
            fontweight="bold" if p.get("formal") else "normal",
            arrowprops=dict(arrowstyle="-", color=c, lw=0.7, shrinkA=0, shrinkB=4) if abs(dx) > 20 or abs(dy) > 12 else None,
        )

    ax.set_xlabel("Success ↑ (%)")
    ax.set_ylabel("Safe ↑ (%)")
    ax.set_xlim(52, 98)
    ax.set_ylim(58, 103)
    ax.axhline(95, color="#CFD8DC", ls="--", lw=0.9)
    ax.axvline(80, color="#CFD8DC", ls="--", lw=0.9)
    ax.set_title("Safe vs success (@32-epi)  ·  no DS")
    handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C["MF-r2"], markersize=15, label="MF-r2 (planar formal)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#78909C", markersize=8, label="Baselines / refs"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True)
    save(fig, "fig7_pareto_safe_success")


def fig8_formal_m_ladder() -> None:
    """Primary formal contribution figure: M1–M3 wall time + ax margin."""
    d = DATA["fig8_formal_m_ladder"]
    levels = d["levels"]
    labels = [lv["label"] for lv in levels]
    walls = [lv["wall_s"] for lv in levels]
    axmins = [lv["ax_min"] for lv in levels]
    colors = [C["m1"], C["m2"], C["m3"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

    # left: wall time
    ax = axes[0]
    x = np.arange(len(levels))
    bars = ax.bar(x, walls, width=0.55, color=colors, edgecolor="white", zorder=3)
    for b, w, lv in zip(bars, walls, levels):
        if w >= 120:
            txt = f"{w/60:.1f} min"
        else:
            txt = f"{w:.0f} s"
        ax.text(b.get_x() + b.get_width() / 2, w + max(walls) * 0.02, txt, ha="center", va="bottom", fontsize=8.5, color=C["text"])
        ax.text(b.get_x() + b.get_width() / 2, max(walls) * 0.06, "PROVEN", ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock verify time (s)")
    ax.set_title("Formal verify cost (Marabou ladder)")
    ax.set_ylim(0, max(walls) * 1.18)

    # right: sample ax_min margin
    ax = axes[1]
    bars = ax.bar(x, axmins, width=0.55, color=colors, edgecolor="white", zorder=3)
    ax.axhline(0.0, color=C["accent"], ls="--", lw=1.0, label="SafeSet a_min ≈ 0")
    for b, a in zip(bars, axmins):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.015, f"{a:.3f}", ha="center", va="bottom", fontsize=8.5, color=C["text"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sample Deploy a_x min")
    ax.set_title("Safety margin inside SafeSet")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)

    tc = d.get("task_constant", {})
    fig.suptitle(
        f"Planar netdep M1–M3 on MF-r2  ·  task fixed @32: "
        f"suc {100*tc.get('success',0):.1f}% · safe {100*tc.get('safe',0):.1f}% · cost {tc.get('cost',0):.3f}",
        y=1.02,
        fontsize=10.5,
    )
    fig.tight_layout()
    save(fig, "fig8_formal_m_ladder")


def main() -> None:
    apply_style()
    print(f"[figs] package={DATA['meta']['package']}")
    print(f"[figs] out={OUT}")
    print("[figs] NO DeepSet / DS-r2")
    fig1_repair_ce_hit()
    fig2_repair_task_metrics()
    fig3_baseline_main_bar()
    fig4_transfer_n()
    fig5_latency()
    fig6_pressure_stability()
    fig7_pareto_safe_success()
    fig8_formal_m_ladder()
    print("[figs] done")


if __name__ == "__main__":
    main()
