# 形式化证据索引

## A. 推荐主引用（论文 Deploy）— **mf_best r2 · 同 X**

| 级 | 路径 | 状态 | 说明 |
|----|------|------|------|
| M1 | `mf_best_r2_same_X/M1_vg001/` | **PROVEN_FULL** | root Marabou ~28s |
| M2 | `mf_best_r2_same_X/M2_vg005/` | **PROVEN_FULL** | vg 二分 ~19s+44s |
| **M3** | `mf_best_r2_same_X/M3_vg010/` | **PROVEN_FULL** | vg 二分 ~150s+563s |
| 摘要 | `mf_best_r2_same_X/LADDER_SUMMARY.json` | — | |
| 说明 | `mf_best_r2_same_X/README.md` | — | |

**权重：** `01_models/residual_mf_best_r2_ref/repair_best.pkl`  
（同源：`acceptance/mf_final_20261012/5_frozen_sources/mf_best_20261006/.../residual_r2/repair_best.pkl`）

**任务：** safe@32 **96.5%** · cost **0.31**（`03_task_eval/mf_best_r2_reference/`）

## B. 侧枝对照 — netdep r3

| 级 | 路径 | 状态 |
|----|------|------|
| M1 | `M1_vg001/` | PROVEN_FULL |
| M2 | `M2_vg005/` | PROVEN_FULL |
| M3 | `M3_vg010/` | PROVEN_FULL |
| ladder | `ladder/` | 原侧枝阶梯说明 |

**权重：** `01_models/residual_netdep_r3/repair_manual_pick.pkl`  
**任务：** safe@32 95.7% · cost 0.69

## 公共 claim 盒（A 与 B 相同）

- \(X_{\mathrm{rel}}\)：dx∈[0.118,0.122], |dy|≤0.04, dvx∈[−0.30,−0.25], |dvy|≤0.05, R=0.10  
- pose px=py=2；threat 特征中点；vg = ±0.01 / ±0.05 / ±0.10  
- netdep=True · envelope=False · McCormick 已修

## 勿引用

- `3_appendix/bnb_INCOMPLETE_M2`  
- 任何 pre-McCormick 证书  
