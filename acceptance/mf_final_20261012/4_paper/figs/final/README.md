# final — 最终验收图（对齐 part/ · MF-r2 · 无 DS）

**主方法：** MF residual r2  
**形式化：** planar netdep M1–M3  
**不含：** DeepSet / DS-r2  

## 生成

```bash
cd acceptance/mf_final_20261012/5_frozen_sources/mf_planar_netdep_20261011/figs/final
python make_figures.py
```

## 清单

| 文件 | 含义 |
|------|------|
| fig1_repair_ce_hit | 修复 CE hit |
| fig2_repair_task_metrics | 修复任务指标 |
| fig3_baseline_main_bar | 主对照（无 DS） |
| fig4_transfer_n | N 延展（无 DS） |
| fig5_latency | 控制延迟 |
| fig6_pressure_stability | @32/64/256 |
| fig7_pareto_safe_success | safe–success（无 DS） |
| fig8_formal_m_ladder | M1–M3 形式化阶梯 |

数据：`paper_figure_data.json` · 脚本：`make_figures.py`
