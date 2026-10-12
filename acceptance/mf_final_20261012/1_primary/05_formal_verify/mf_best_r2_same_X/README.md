# mf_best r2 · 同一 planar X（推荐主形式化证据）

**验收角色：** 本包 **推荐** 对外引用的 Deploy + M1–M3 证书。  
**ACTIVE：** 未改写。

---

## 问题与结论

| 问题 | 结论 |
|---|---|
| 更强经验的 mf_best r2 能否过当前 planar 验收链路？ | **能。M1/M2/M3 全部 PROVEN_FULL。** |
| 是否必须用 netdep_r3？ | **否。** r3 为侧枝对照。 |

---

## 阶梯结果

| 级 | vg | 状态 | 证明方式 | wall |
|----|-----|------|----------|------|
| M1 | ±0.01 | **PROVEN_FULL** | root Marabou UNSAT ~27.7 s | ~65 s |
| M2 | ±0.05 | **PROVEN_FULL** | root 600 s TO → vg 二分 lo/hi ~18.8 s + 43.9 s | ~713 s |
| M3 | ±0.10 | **PROVEN_FULL** | root 900 s TO → vg 二分 ~150 s + 563 s | ~1704 s |

**权重：** `01_models/residual_mf_best_r2_ref/repair_best.pkl`  
（同源：`acceptance/mf_final_20261012/5_frozen_sources/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl`）

**任务：** @32 safe **96.5%** · cost **0.31**（`03_task_eval/mf_best_r2_reference/`）

---

## 与 netdep_r3 同盒对照

| | **r2（本目录）** | netdep_r3（`../M{1,2,3}_*`） |
|--|------------------|------------------------------|
| 任务 @32 safe | **96.5%** | 95.7% |
| cost @32 | **0.31** | 0.69 |
| M1–M3 planar | **PROVEN_FULL** | PROVEN_FULL |
| sample ax_min (M1) | ~0.874 | ~0.716 |

---

## 门控 sanity（各级均过）

| 检查 | 结果 |
|---|---|
| network_dependent | True |
| envelope_any_a_safe | False |
| encode unconstrained | SAT |
| sample CE | 0 |
| ax≥0.99 counterfactual | sat_candidate（非真空 proven） |
| McCormick | 修复后编码 |

---

## 文件

| 文件 | 内容 |
|---|---|
| `LADDER_SUMMARY.json` | 三档状态与时间 |
| `M1_vg001/REPORT.json\|md` | M1 全量报告 |
| `M2_vg005/REPORT.json\|md` | M2 + vg 二分叶 |
| `M3_vg010/REPORT.json\|md` | M3 + vg 二分叶 |

公共 claim 盒定义见包证书 `00_acceptance/ACCEPTANCE_CERTIFICATE.md` §2。
