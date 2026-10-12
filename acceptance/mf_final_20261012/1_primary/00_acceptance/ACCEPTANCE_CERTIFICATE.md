# 平面两步 netdep 形式化验收证书（最终合并包）

| 字段 | 值 |
|---|---|
| 轨道 | Mean-field **最终主贡献**：网络依赖平面两步 |
| **推荐 Deploy（论文）** | **mf_best residual r2** `repair_best` + 冻结 v5 base |
| 侧枝对照 Deploy | netdep residual r3 `repair_manual_pick` |
| 证书状态 | **通过（PASS）** — 两权重在**同一 X** 上 M1–M3 均 **PROVEN_FULL** |
| 主层（推荐引用） | **r2 · M3** vg±0.10 · **PROVEN_FULL** |
| 日期 | 2026-10-12（合并最终包；形式化跑次 2026-10-11） |
| 包 | `acceptance/mf_final_20261012/` |
| 相对源包 | 合并 planar 增量 + mf_best 基线/历史切片；**不**改 ACTIVE；源包冻结保留 |

---

## 0. 一页结论

| | **mf_best r2（推荐）** | netdep r3（侧枝） |
|--|------------------------|-------------------|
| 形式化 M1–M3 同 X | **PROVEN_FULL** | PROVEN_FULL |
| 任务 @32 suc/safe/fin/cost | **82.0 / 96.5 / 84.0 / 0.31** | 82.0 / 95.7 / 85.2 / 0.69 |
| 证据目录 | `05_formal_verify/mf_best_r2_same_X/` | `05_formal_verify/M{1,2,3}_*` |
| 权重 | `01_models/residual_mf_best_r2_ref/repair_best.pkl` | `01_models/residual_netdep_r3/repair_manual_pick.pkl` |

**主 claim：** 平面两步 \(\|p_2\|_2\ge R\)，**非**径向包络；Deploy∈SafeSet 经 Marabou。  
**对外：** 任务 + 形式化均挂 **mf_best r2**；r3 = 侧枝对照。

---

## 1. 验收声明

```text
∀ x ∈ X_feat,  ∀ rel ∈ X_rel:
    || p₂(rel, Deploy(x)) ||₂  ≥  R = 0.10
```

**证书定级：** 去中心化部署策略在**声明 feat–rel 关系**上的**局部两步 pairwise 动作安全**（单威胁）；**不是**多智能体闭环整局。\(X_D\) 为 constrained subdomain，非裸 44 维超矩形。

### 充分证明路径

```text
1) 几何：SafeSet → a_x ≥ a_min(I_rel)（|a_y|≤1 常由 clip 蕴含）
2) 网络：Marabou 编码 Deploy；查询 a_x < a_min → UNSAT ⇒ ∀x Deploy ∈ SafeSet
3) 复核：真 numpy Deploy + planar p₂；无约束须 SAT；反事实 a_x≥0.99 非真空 proven
```

---

## 2. 声明域（M3 主引用）

| 量 | 范围 |
|---|---|
| dx | [0.118, 0.122] |
| dy | [−0.04, 0.04] |
| dvx | [−0.30, −0.25] |
| dvy | [−0.05, 0.05] |
| R | **0.10** |
| pose | px=py=**2.0** |
| threat | 特征**中点** |
| vg 阶梯 | ±0.01 / ±0.05 / ±**0.10** |
| goal_dv | ≡ −v |

→ **network_dependent = True**，**envelope_any_a_safe = False**。

---

## 3. 形式化结果（r2 推荐）

| ID | 路径 | 状态 | wall |
|---|---|---|---:|
| M1 | `mf_best_r2_same_X/M1_vg001/` | **PROVEN_FULL** | **65 s** |
| M2 | `…/M2_vg005/` | **PROVEN_FULL** | **713 s** |
| **M3** | `…/M3_vg010/` | **PROVEN_FULL** | **1704 s** |

侧枝 r3 同 X 亦 PROVEN_FULL（见 `M1_vg001`…`M3_vg010`）。  
McCormick 修复后编码；pre-fix 作废见 `3_appendix/INVALIDATED_pre_mccormick.md`。

---

## 4. 部署对象

| 角色 | 路径 |
|---|---|
| **推荐 r2** | `01_models/residual_mf_best_r2_ref/repair_best.pkl` |
| 侧枝 r3 | `01_models/residual_netdep_r3/repair_manual_pick.pkl` |
| base | `01_models/base_v5/meanfield_mlp512_student_step9000.pkl` |

---

## 5. 通过条件

- [x] 平面两步 L2 netdep，非径向包络  
- [x] r2 与 r3 同 X 上 M1–M3 PROVEN_FULL  
- [x] McCormick 修复后健全性  
- [x] 未改 ACTIVE；源包冻结  
- [x] 完整基线并入 `2_baselines/`  
- [x] 论文 part/LaTeX/figs 在 `4_paper/`  

**结论：最终 planar netdep scoped 形式化验收 — PASS。**
