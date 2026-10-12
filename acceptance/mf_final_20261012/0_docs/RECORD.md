# 实验账本 — planar netdep 2026-10-11

## 1. 目标

- 用 **网络依赖的平面两步** 形式化链路，替代旧 **包络径向** 主叙事。  
- 交付 **新增验收包**（不改 `mf_best_20261006`、不写 ACTIVE）。  
- **补证：** 更强经验 **mf_best r2** 在同一 X 上 M1–M3 亦 PROVEN → 论文可任务+形式化双挂 r2。

## 2. 策略与权重

| 角色 | 路径 / 值 |
|---|---|
| **推荐 Deploy** | `1_primary/01_models/residual_mf_best_r2_ref/repair_best.pkl` |
| 侧枝 Deploy | `1_primary/01_models/residual_netdep_r3/repair_manual_pick.pkl` |
| base | v5 distill step9000（`01_models/base_v5/`） |
| 推荐形式化 | `05_formal_verify/mf_best_r2_same_X/`（M1–M3） |
| 侧枝形式化 | `05_formal_verify/M{1,2,3}_*` |
| r2 @32 | suc 82.0% · safe **96.5%** · fin 84.0% · cost **0.31** |
| r3 @32 | suc 82.0% · safe 95.7% · fin 85.2% · cost 0.69 |

## 3. 形式化实验表

### 3.1 公共设置

- 性质：planar two-step L2，R=0.10  
- \(X_{\mathrm{rel}}\)：dx∈[0.118,0.122], \|dy\|≤0.04, dvx∈[−0.30,−0.25], \|dvy\|≤0.05  
- pose：px=py=2；threat：特征中点  
- 后端：Marabou SafeSet（Deploy.ax ≥ a_min）；McCormick **已修**  
- 环境：CPU · `safe_marl_verify`

### 3.2 mf_best r2（推荐 · 2026-10-11）

| ID | vg | aggregate | root | cover | wall_s | n_CE | ax_min | encode | hard≥0.99 |
|---|---:|---|---|---|---:|---:|---:|---|---|
| M1 | 0.01 | PROVEN_FULL | proven 27.7 s | 单盒 | 65.0 | 0 | 0.874 | SAT | sat_candidate |
| M2 | 0.05 | PROVEN_FULL | timeout 600 s | lo 18.8 + hi 43.9 | 713 | 0 | 0.850 | SAT | sat_candidate |
| M3 | 0.10 | PROVEN_FULL | timeout 900 s | lo 150 + hi 563 | 1704 | 0 | 0.819 | SAT | sat_candidate |

### 3.3 netdep r3（侧枝）

| ID | vg | aggregate | root Marabou | cover | wall_s | n_CE | ax_min |
|---|---:|---|---|---|---:|---:|---:|
| M1 | 0.01 | PROVEN_FULL | proven 25.0 s | 单盒 | 52.6 | 0 | 0.716 |
| M2 | 0.05 | PROVEN_FULL | proven 463.6 s | 单盒 | 515.5 | 0 | 0.688 |
| M3 | 0.10 | PROVEN_FULL | timeout 900 s | lo 372 + hi 145 | 1524 | 0 | 0.650 |

### 3.4 物理诊断（公共 rel）

| 量 | 值 |
|---|---|
| a_min (SafeSet) | ≈ 0 |
| p2x_worst (a=−1) | 0.091 |
| p2x_best (a=+1) | 0.109 |
| network_dependent | True |
| envelope_any_a_safe | False |

### 3.5 旁路（不计入 PASS）

| 项 | 说明 |
|---|---|
| M2 通用 DFS BnB | `3_appendix/bnb_INCOMPLETE_M2` — **不采纳** |
| 修前 McCormick | 假 PROVEN；见附录 INVALIDATED |

## 4. 源码快照（包内）

`1_primary/07_source_snapshot/`：`property_planar_l2.py` · `marabou_mf.py` · `verify_planar.py` · `run_mf_verify_planar.sh` · …

## 5. 任务资产

| 项 | 位置 |
|---|---|
| r2 @32/64/256 + N 延展 | `03_task_eval/mf_best_r2_reference/` |
| netdep 链 @32 | `03_task_eval/chain_32epi/` |
| r3 @64/@256 / N / 延迟 | `pressure/` · `transfer_n/` · `2_baselines/latency_mf_best_era/` |
| 台账 | `03_task_eval/TASK_EVAL_LEDGER.md` |
| 主文 | `0_docs/EVAL_AND_REPAIR.md` · `0_docs/DUAL_WEIGHT_FORMAL.md` |

## 6. 打包清单动作

- 复制：r2/r3 模型、形式化 REPORT、任务 JSON、SELECTION、源码  
- 写入：证书（双权重）、指标、NOTES、INDEX、指针、MANIFEST、SHA256  
- **未**执行：改 ACTIVE、改 mf_best 树、git commit（用户门控）

## 7. 后续（未纳入本包 PASS）

- threat 轴扩展、goal 与 v 解耦、自由 pose  
- 外源基线全量重跑并表（可选）  
- 显式 promote 替换旧 near_plus ACTIVE

## 8. mf_best r2 × planar 同盒（入包验收项）

| 项 | 结果 |
|---|---|
| 动机 | 旧 r2 任务更强；验证「是否只有 netdep_r3 才能证」 |
| M1–M3 | **全部 PROVEN_FULL** |
| 产物 | `05_formal_verify/mf_best_r2_same_X/` + 模型副本 + 任务引用 |
| 叙事 | **推荐** 任务+形式化均挂 r2；r3 侧枝对照 |
| ACTIVE | 未改 |
