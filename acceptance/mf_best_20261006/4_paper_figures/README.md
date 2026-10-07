# 论文出图包

> 从验收数据筛出的**可作主文/附录图**的子集。  
> 数据：`paper_figure_data.json` · 图：`figs/*.pdf`（矢量）+ `*.png`（预览）  
> 重绘：`python make_figures.py`

---

## 推荐主文图

| 图 | 文件 | 讲什么 | 注意 |
|---|---|---|---|
| Fig.1 | `fig1_repair_ce_hit` | CE 修复：danger 两步 L2 hit ↓12.7× | 经验修复，非形式化 |
| Fig.2 | `fig2_repair_task_metrics` | 修复前后任务率 + cost | finish 略降 = 更保守 |
| Fig.3 | `fig3_baseline_main_bar` | @32 主对照（suc / safe / cost） | MF ★ = 有 L2 证书，非 suc 冠军 |
| Fig.4 | `fig4_transfer_n` | 零样本 N=8→16→32 | 证书**不**随 N 外推 |
| Fig.5 | `fig5_latency` | 控制延迟 median / p95 | MF ≈ GCBF+ ≪ QP |
| Fig.6 | `fig6_pressure_stability` | MF @32/64/256 稳定性 | 辅；证明非过拟合 32-epi |
| Fig.7 | `fig7_pareto_safe_success` | Safe–Success Pareto | ★ = formal L2 |
| Fig.8 | `fig8_distill_train` | Base 蒸馏 loss/MAE + 短评测曲线 | 部署 base=step9000（冻结），≠短评测峰 |
| Fig.9 | `fig9_repair_train` | r1/r2 残差修复训练 + 最终 @32 | 短评测有噪声；验收看右栏 @32 |

## 附录可选（慎用）

| 图 | 文件 | 注意 |
|---|---|---|
| Fig.A | `figA_midF_ce_fringe_appendix` | mid-F fringe 上 MAPPO/GCBF+ 有经验 fail；**MF 形式化 mid-F 仍 VIOLATED**，勿当双门主 claim |

---

## 目录

```text
4_paper_figures/
├── README.md                 ← 本页
├── paper_figure_data.json    ← ★ 出图用数据（与 EVAL / 训练 CSV 同源）
├── make_figures.py           ← 重绘脚本
├── data/                     ← distill / repair_r1 / repair_r2 CSV 副本
└── figs/
    ├── fig1_…fig9_….pdf/.png
    └── figA_….pdf/.png
```

## 叙事边界（出图时也要守）

1. 主卖点 = **有界盒 ∀ 证书 + 高 safe / 低 cost**，不是任务 success 全面超 GCBF+。  
2. 延展图只写经验；**不要**写「大 N 已形式化」。  
3. Fig.A 仅附录；正式 mid-F 双门未达成（见 `NOTES.md`）。

来源全表：`../EVAL_AND_REPAIR.md` · `../RECORD.md`
