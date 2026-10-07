# 项目介绍：可验证安全多智能体控制（DeepSet 主路径 + Mean-field 并行）

> 增量式项目介绍，按阶段 / 轨道逐步补充。  
> DeepSet = 任务与经验修复主路径；Mean-field = 验证友好架构对照（隔离命名空间）。  
> DeepSet `method/deepset/output/phase2/ACTIVE_*` **永不被 MF 覆盖**。

## Stage 0：前置基线 GCBF / GCBF+

### 0.1 研究任务背景

多智能体避障控制：多个智能体在含障碍的二维空间内同时导航到各自目标，须在到达的同时避免相互碰撞与撞障。本项目固定实验为 `DoubleIntegrator`（双积分器）环境：

| 项 | 值 |
|---|---|
| 智能体数 | 8 |
| 区域 | 4.0 × 4.0 |
| 障碍 | 4 个随机矩形 |
| LiDAR 射线 | 32 |
| 回合步数 | 256 |
| `dt` / 质量 `m` | 0.03 / 0.1 |
| 动作 | `[-1,1]²`（环境侧 clip） |
| `car_radius` / 碰撞距离 | 0.05 / 0.10 |
| 通信半径 | 0.50 |

动力学（欧拉，与部署一致）：

```
p' = p + v·dt
v' = clip(v + (a/m)·dt, ±0.5)
```

研究目标：在官方 GCBF+ 对照基线之上，构造**验证友好**的 DeepSet 策略，并对其做有界形式化验证 + 经验反例修复。核心约束是 *可部署低延迟策略 ↔ 与部署完全一致的验证对象*，而非把变结构 GNN 交给验证器。

### 0.2 前置知识：GCBF / GCBF+

传统 CBF 依赖**人工构造**的 barrier 函数 `h(x)`，难以适应多智能体变拓扑。GCBF 系列用**图神经网络从数据学习 `h`**，使其自动适配动态邻居与障碍拓扑。

**GCBF（神经图 CBF 框架）** 由两个共享注意力 GNN 骨干的网络组成：

- **CBF 网络**：输出 `h_i ∈ (-1,1)`，作为学习的安全证书（`h>0` 安全 / `h<0` 危险）；
- **Actor 网络**：输出动作 `a_i ∈ (-1,1)`，部署时 `2·a + u_ref`。

`h` 的标签**不是人为打分**，而是环境几何派生的二值掩码：`unsafe_mask`（已碰撞或朝近邻/障碍的危险方向）/ `safe_mask`（不撞且不在障碍内）。损失用 hinge 只塑 `h` 的符号，幅度无显式标签。训练四损失联合优化两网：危险态压 `h<0`、安全态推 `h>0`、前向不变性 `ḣ + α·h > 0`、Actor 跟踪参考控制 `u_ref`。**GCBF 没有 QP**，安全只靠损失间接塑造，部署无在线安全过滤器。`u_ref` 是 LQR 反馈控制（朝目标的最优标称动作，纯任务、无安全考量）。

**GCBF+（加 QP 安全过滤器）** 继承 GCBF，核心增量是引入 CBF-QP 安全过滤器：用学习到的 `h` 及其 Lie 导数解 QP，得到 `u_qp = u_ref + 最小安全纠正`（在满足前向不变性约束下离 `u_ref` 最近的动作）。Actor 监督目标由 `u_ref` 改为 `u_qp`，即蒸馏 QP 纠正后的动作。其余增量：CBF 目标网络稳定 QP 教师标号、`MaskedReplayBuffer` 优先回放危险样本、Horizon-based `safe_mask` 提供更锐监督。

**概念差一句话**：GCBF 只学证书形状 + Actor 跟 `u_ref`，部署无安全过滤器；GCBF+ 把学习 CBF 变成训练期 QP 安全过滤器纠正 `u_ref→u_qp`，Actor 蒸馏该纠正。

**部署期与安全边界**：GCBF+ 部署只跑 GNN Actor，**不跑 QP**（实时）；QP 仅训练期当教师标号。CBF 是学习得来、未被证明是合法 barrier，部署期 Actor 是 QP 解的近似，安全关键状态可能偏离 `u_qp` 且无在线纠正——这是部署期安全弱化的根源。冻结教师为单种子（`seed=0`）单步（`step=1000`）run，仅作上界参考。

### 0.3 效果测评

冻结教师 checkpoint：`baselines/gcbfplus/log/gcbf+/seed0_20261001232611`（step 1000）。

**任务测评**（`eval_seed=1234`，`max_step=256`）：

| 规模 | safe↑ | finish↑ | success↑ | cost | reward | success SE |
|---|---:|---:|---:|---:|---:|---:|
| 32-epi | 95.70% | 95.31% | **91.41%** | 0.441 | -4.409 | 0.0175 |
| 64-epi（压力） | 95.90% | 96.09% | 92.19% | 0.453 | -4.833 | 0.0119 |

**L2 preventable D-hit**（同标 2000 in-D 采样/桶，`seed=123`，两步欧拉）：

| 桶 | AA | AO | (1,1) | mean |
|---|---:|---:|---:|---:|
| D-hit | 22.7% | 33.6% | 26.8% | 27.7% |

**结论**：GCBF+ 经验性能最强（success≈91.4%、safe≈95.7%），作为对照上界与 Stage1 蒸馏教师合理。其 D-hit 在 AO 桶偏高（33.6%），说明即便是教师，在近障单威胁域上仍有可观的"可预防碰撞"残留——这正是后续 Stage2/3 要针对的修复空间。所有指标均为经验 rollout / 采样，**非形式化验证**。

---

## Stage 1：DeepSet 学生蒸馏

### 1.1 蒸馏方式

冻结 GCBF+ 的 GNN Actor 跑环境 rollout 探索，采集每步 `(状态, GNN Actor 动作)`；DeepSet 学生作为回归器，**纯监督模仿 GNN Actor 动作**，损失为 danger-weighted MSE（近邻/近障样本权重更高）。DS 不自行探索或与环境交互。

### 1.2 为什么不直接模仿 u_qp

试过：直接 QP 蒸馏（`hidden=512, steps=8000`，对齐 Actor512，目标改为 `u_qp`）。结果 Actor 蒸馏明显优于直接 QP：

| 策略 | safe | success | cost |
|---|---:|---:|---:|
| Actor512（模仿 GNN Actor） | 94.1% | 87.1% | 0.95 |
| QP512（直接模仿 u_qp） | 70.3% | 65.6% | 5.34 |

原因：① QP 标签在危险边界陡峭/不连续，DS 回归困难；GNN Actor 已对 QP 做一次平滑近似，DS 模仿更易学。② `u_qp` 与"两步不撞"的 L2-safe 标签几乎正交（`MAE≈1.02`），作为安全修复标签不对症。故不采用，仅留 `method/deepset/output/stage1/qp/` 作消融。

### 1.3 网络输入输出

**输入**：

- `ego ∈ R⁸`：`[px, py, vx, vy, goal_dx, goal_dy, goal_dvx, goal_dvy]`
- `entities ∈ R^{40×7}`：8 邻居槽 + 32 lidar 障碍 hit 槽；每行 `[dx, dy, dvx, dvy, type_agent, type_obs, presence]`；邻居 `type_agent=1`，障碍 `type_obs=1`，空槽全 0 且 `presence=0`；相对量 = `ego − other`。

**输出**：`a ∈ R²`（`[fx, fy]` 力）。**无 tanh**，直接 `Dense(2)`，部署靠 `clip` 截断到 `[-1,1]`。

**结构**（`hidden=512`）：`φ = Dense(512)→ReLU×3`（共享，逐 entity）→ masked sum（置换近似不变）→ `ρ = [ego; pooled]→Dense(512)→ReLU×3→Dense(2)`。参数量约 1.32M。

### 1.4 效果测评

Phase-2 redistill seed0 产物（`method/deepset/output/stage1/redistill/stage1_redistill_20261004_183322/seed0/deepset_actor512_student_best.pkl`）。

**任务测评**（`eval_seed=1234`，`max_step=256`）：

| 规模 | safe↑ | finish↑ | success↑ | cost | reward | success SE |
|---|---:|---:|---:|---:|---:|---:|
| 32-epi | 93.75% | 94.53% | **89.06%** | 1.043 | -11.354 | 0.0195 |
| 64-epi（压力） | 93.16% | 94.92% | 88.67% | 1.119 | -11.336 | 0.0140 |

**L2 preventable D-hit**（同标 2000 in-D 采样/桶，`seed=123`，两步欧拉）：

| 桶 | AA | AO | (1,1) | mean |
|---|---:|---:|---:|---:|
| D-hit | 35.95% | 43.25% | 37.85% | 39.02% |

**结论**：Phase-2 seed0 基线 success≈89.1%、safe≈93.8%，略低于 GCBF+ 教师（91.4%/95.7%）但任务性能可用；D-hit 全面偏高（尤其 AO 43.3%），是后续 Stage2/3 的修复空间。DS 模仿的是 GNN Actor（已被 QP 平滑），非直接 QP；经验指标，非形式化验证。

> 注：Phase-1 frozen（Actor512 @ step8000）为 canon 只读基线；此处记录 Phase-2 多种子基线之一（seed0），后续阶段测评数据均记 Phase-2。

---

## Stage 2：有界形式化验证

### 2.1 验证对象与契约

对象为**冻结 base DeepSet**（非 `base+δ`），与部署一致：`a_deploy = clip(DeepSet(x), -1, 1)`。两步 **held-action 欧拉**（单步位置与当前动作无关，需两步让加速度进入位置）。

### 2.2 三桶与指标

| 桶 | 威胁 | 假设 | 验证指标（阈值） |
|---|---|---|---|
| L2-AA (1,0) | 邻居智能体 | 邻居 coast（加速度=0） | 两步后 ego-邻居 `‖Δp'‖ ≥ 0.10` |
| L2-AO (0,1) | 障碍 lidar hit | hit 静止 | 两步后 ego-hit `‖Δp'‖ ≥ 0.05`（hit 球代理） |
| (1,1) | 双威胁 | 邻居 coast + hit 静止 | **两者都要避开** |

验证的是**局部、单次、两步**性质——声明 D 域（初始安全 + 可撞 + 可避）内不进碰撞集，不等于无限时域全局安全。

### 2.3 三层递进与现状

每桶按 **采样攻击 → IBP → Marabou** 递进：采样找 CE（可反驳不可证）；IBP 界传播（可靠不完备）；Marabou ONNX 完备查询（SAT→violated+CE，UNSAT→proven）。

| 项 | 现状 |
|---|---|
| L1 裸网界 | smoke proven；主盒 violated（裸输出越界） |
| L2 smoke 极小盒 | Marabou UNSAT 6–7s（工具链可用） |
| L2 主桶 AA/AO near | 采样阶段即 violated（未跑形式化） |
| L2 主桶 (1,1) near | dual IBP unknown；**dual Marabou timeout**（两个 outer-square 查询均 timeout） |

**形式化验证未达全量**：主桶（aa/ao/11 near）的 IBP/Marabou proven 均未完成；当前"验证"产物仅采样 D-hit（经验，非形式化）。这是 Phase-2 协议把 formal proven 排除在验收门槛外的原因。

### 2.4 全量形式化验证可行性

- **单威胁 AA/AO**：~3072 ReLU，修复到无 CE 后 IBP 秒级 + Marabou 可能分钟级，**可尝试**。
- **双威胁 (1,1)**：~4608 ReLU，当前直接 timeout。但 Marabou 对纯 ReLU 网络线性约束**完备可判定**（LP+ReLU 分支定界必终止），分治切子盒 + 不设超时**理论上必然完成**；实际几小时-几天，需 CROWN 预筛 / 并行 / 降规模加速。
- **前置依赖**：网络须先修复到采样无 CE，否则直接 violated。

### 2.5 效果测评（采样 D-hit，base 起点）

base DeepSet（redistill seed0）D-hit（`seed=123`, 2000/桶）：AA 35.95% / AO 43.25% / (1,1) 37.85% / mean 39.02%——这是 Stage3 修复的起点。

---

## Stage 3：反例引导反馈修复

### 3.1 方法

Residual gate 分解部署，冻结 base 只训修正头：

```
a_deploy(x) = clip( π_base(x) + g(x) · δ(x) )
```

- `π_base`：冻结 DeepSet（不更新），远离危险 `g→0` 退化为基座；
- `g(x)`：hybrid 危险门控（物理距离包络）；
- `δ(x)`：小型 DeepSet 修正头（可训练）。

CE 采集 → `min_intervention` 标签（让 `a_deploy≈a_l2_safe`）→ 训练 `gate+δ`；正常轨迹 `idle loss` 使 `a_deploy≈π_base`。形式化验证可拆成 base 在 gate 关闭区 + `base+δ` 在 gate 开启区，但 v1 未实现，**不声称 residual 已 proven**。

### 3.2 Phase-2 V2d 第一轮（seed0 base）

产物：`method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d/`。

| 项 | 值 |
|---|---|
| base | redistill seed0 best（`method/deepset/output/stage1/redistill/.../seed0/deepset_actor512_student_best.pkl`） |
| 方法 | V2d residual gate（冻结 base + gate/δ） |
| CE | 5082（AA+AO+(1,1)），label=min_intervention |
| steps | 1400（延长自默认 800） |
| 超参 | ce_weight=2.5, ao_ce_weight=1.75, dual_ce_weight=1.75, idle=1.5, lr=8e-5 |
| 网络结构 | gate_hidden=64, correction_hidden=128, gate_mode=hybrid |
| best | step 1350（eligible），wall ≈ 58 min |

### 3.3 效果测评

**任务测评**（32-epi, `eval_seed=1234`, best checkpoint）：

| safe↑ | finish↑ | success↑ | cost | reward |
|---:|---:|---:|---:|---:|
| 93.75% | 94.14% | **88.28%** | 0.906 | -12.761 |

**L2 preventable D-hit**（`seed=123`, 2000/桶，best vs base）：

| 桶 | base | V2d best | 改善 |
|---|---:|---:|---:|
| AA | 35.95% | 18.65% | −17.3 pt |
| AO | 43.25% | 32.85% | −10.4 pt |
| (1,1) | 37.85% | 28.25% | −9.6 pt |
| mean | 39.02% | 26.58% | −12.4 pt |

### 3.4 验收对照（Phase-2 平衡轨门槛）

| 门槛 | 要求 | V2d seed0 | 结果 |
|---|---|---:|---|
| success ≥ 80% | 80% | 88.28% | ✅ |
| AA ≤ 0.5×frozen | ≤ 17.98% | 18.65% | ⚠️ 略超 |
| AO ≤ 0.5×frozen | ≤ 21.63% | 32.85% | ❌ 超 |
| (1,1) ≤ frozen | ≤ 37.85% | 28.25% | ✅ |

**结论**：Phase-2 V2d 第一轮（单种子 seed0）success 与 (1,1) 达标，AA/AO D-hit 未达 0.5×frozen 门槛（尤其 AO 仍 32.85%）。D-hit 全面下降但幅度有限，AO/双威胁仍是瓶颈。属首轮基线。所有指标为经验 residual 修复，**非形式化安全证明**。

### 3.5 CEGAR r2（V2d 第二轮，经验修复）

产物：`method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/`（当前 active residual）。

| 项 | 值 |
|---|---|
| 输入 | V2d r1 `repair_best.pkl` |
| 流程 | residual 采样验证 → 残差 CE → warm-start 短修（600 step）→ 再验证 |
| best | `repair_r2_best.pkl` |

**任务 / D-hit 对比**（32-epi `eval_seed=1234`；D-hit `seed=123`, 2000/桶）：

| 策略 | success@32 | AA | AO | (1,1) | mean D |
|---|---:|---:|---:|---:|---:|
| GCBF+ GNN | ~91.4% | 22.7% | 33.6% | 26.8% | 27.7% |
| seed0 base | ~89.5% | 36.0% | 43.3% | 37.9% | 39.0% |
| V2d r1 | 88.3% | 18.7% | 32.9% | 28.3% | 26.6% |
| **V2d r2** | **87.9%** | **10.9%** | **24.9%** | **22.1%** | **19.3%** |

r2 相对 r1：D-hit 再降约 7 pt，三桶均低于 GNN；success 再降约 0.4 pt（仍 >85% floor）。

**完整声明域采样诊断**（4k in-D/桶，`verify_r2_residual_sampling.json`）——**仍未通过**：

| 桶 | verdict | hit_rate_on_D |
|---|---|---:|
| AA | **violated** | 10.45%（418/4000） |
| AO | **violated** | 24.60%（984/4000） |
| (1,1) | **violated** | 22.65%（906/4000） |

→ 声明「preventable-D 上两步不撞」在 r2 上仍有具体 CE；`sampling_clear ≠ proven`，此处连 sampling clear 都未达到。

### 3.6 Residual Marabou 可行性 PoC（2026-10-05）

**目的**：回答 residual 部署路径能否进入现有 Marabou，**不是**全声明域安全证明。

编码：`base + correction + g·δ + clip + 两步动力学 + square collision proxy`。  
Gate 含 sigmoid/范数/physics，未直接编码；用保守放宽 `g∈[0,1]` + McCormick 包络 `z≈g·δ`（真实 residual ⊆ 放宽族 → **UNSAT 可推出真实策略局部安全**；SAT 仅针对放宽；TIMEOUT 不算通过）。

实现：`stage2/verify_residual_marabou.py`；记录：`.../V2d_r2/RESIDUAL_MARABOU_POC.md`。

| 局部盒 | AA | AO |
|---|---|---|
| center, dist=0.15, radius≤0.05 | **UNSAT** ~0.5s | **UNSAT** ~0.5s |
| center, dist=0.08, radius=0.01 | **TIMEOUT** | **UNSAT** ~0.5s |
| 完整 `l2_*_near` + preventable-D | **未证明**（采样仍 3×violated） | 同左 |

**正确读法**：

```text
局部小盒 UNSAT  ≠  全域 claim 通过
采样仍有大量 CE  ≠  与 PoC 矛盾（CE 不在这些极小盒里）
```

PoC 结论：residual **验证链路已打通**；远离危险边界的局部盒可给出保守证明；近 AA 危险边界仍难求；**全声明域形式化安全仍未达成**。

---

## 链路现状

```text
Stage0 GCBF+ 教师
  → Stage1 DeepSet base（redistill seed0）
  → Stage2 采样 D-hit / 形式化（base；主桶未 proven）
  → Stage3 V2d r1 → CEGAR r2（经验 D-hit 改善；4k 采样仍 3×violated）
  → Residual Marabou PoC（局部盒可 UNSAT；全域未证）
```

| 层级 | 状态 |
|---|---|
| 经验任务 success | r2 ≈ 87.9%（可用，低于教师） |
| 经验 D-hit | r2 三桶均 < GNN；相对 base 大幅下降 |
| 完整 D 采样 claim | **仍 violated 3/3** |
| Residual 形式化 | 链路可行；**仅局部小盒**；非全域 proven |
| Base 形式化主桶 | 仍未达 IBP/Marabou proven |

后续 DeepSet 侧：gate 分区验证或证明友好 residual；不宜盲跑 CEGAR r3。  
并行侧：Mean-field 已落地（见下 **MF 并行轨道**）。

### 文件归类（双轨）

| Track | 代码 | 产物 | 总索引 |
|---|---|---|---|
| **DeepSet（主路径，冻结参考）** | `stage1/` `stage2/` `stage3/` | `method/deepset/output/stage1/` `method/deepset/output/phase2/` | [`docs/DEEPSET_PATH_CATALOG.md`](docs/DEEPSET_PATH_CATALOG.md) |
| **Mean-field（并行）** | `stage_mf/` | `method/meanfield/output/` | [`docs/MEANFIELD_PARALLEL.md`](docs/MEANFIELD_PARALLEL.md) · [`docs/MEANFIELD_VERIFY.md`](docs/MEANFIELD_VERIFY.md) |

- DeepSet **不物理搬迁**；catalog 按 Stage0–3 / 配置 / 脚本 / checkpoint / 指标归类。  
- MF **隔离命名空间**：`ACTIVE_MF_BASE.txt` / `ACTIVE_MF_RESIDUAL.txt` / `ACTIVE_MF_VERIFY.txt`；**不**改写 DeepSet `ACTIVE_*`；不改 `stage1/student.py`。  
- 布局日更：`docs/ACTIVE_LAYOUT.md`；Phase-2：`method/deepset/output/phase2/INDEX.md`。

完整设计见 `docs/DESIGN_PROPOSAL.md`；Phase-2 协议见 `docs/FULL_PIPELINE_EVAL_PROTOCOL.md`。

---

## MF 并行轨道：Mean-field 验证友好对照

> 与 DeepSet **并行**，不是替换。同一教师、同一环境与任务协议；架构改为固定维 MLP，形式化按 **44 维特征盒全覆盖**，**不按** DeepSet (k,m) 桶验收。

### MF.0 动机与隔离

| 项 | DeepSet 主路径 | Mean-field 并行 |
|---|---|---|
| 输入 | 变长实体槽（k 邻 + m 障） | **固定 44 维**象限聚合特征 |
| 网络 | φ/ρ DeepSet，~(k,m) 决定 ReLU 规模 | **MLP512**（base）+ 小 δ 头 |
| 验证域 | 必须 (k,m) 分桶 | **每个自由维**区间积，B&B 切盒 |
| 主指标 | (1,0)/(0,1) proven% / D-hit | **PROVEN_FULL / PARTIAL / CE 列表** |
| 产物根 | `method/deepset/output/phase2/` | `method/meanfield/output/` only |

**禁止**：用 MF 结果改写 `method/deepset/output/phase2/ACTIVE_*`；在 `stage2/verify_l2.py` 混入 MF 分支。

### MF.1 特征与网络（v5）

**输入 dim=44**（`FEATURE_VERSION=5`）：

| 段 | 维 | 内容 |
|---|---|---|
| ego | 8 | `px,py,vx,vy, goal_dx,dy,dvx,dvy` |
| agent 4 象限 | 20 | 每象限 `(n, min_d, close_max, close_wmean, inv_ttc_max)` |
| obs 4 象限 | 16 | 每象限 `(n, min_d, close_max, inv_ttc_max)` |

- 空象限：`n=0`，`min_d=d_empty=comm_radius`，close/inv_ttc=0。  
- `close_wmean`：距离加权 `w=exp(-d/(0.5·comm_radius))`。  
- `inv_ttc_max`：象限内最近威胁的 `max(close,0)/max(min_d,ε)`，clip 到 `[0, INV_TTC_MAX]`。

**Base 结构**：`x∈R⁴⁴ → Dense(512)→ReLU ×3 → Dense(2)`，部署 `clip(a,-1,1)`。无 tanh。  
**教师**：与 DeepSet 相同 — 冻结 GCBF+ GNN Actor（`seed0_20261001232611` @ step1000）。蒸馏：rollout 监督模仿 Actor 动作（danger-weighted MSE）。

**演进**：早期 22-D / q4 / v4 已弃用；**active base = v5 step9000**。

### MF.2 Base 蒸馏效果

产物：`method/meanfield/output/distill/mf_distill_20261005_v5/seed0/`  
指针：`method/meanfield/output/ACTIVE_MF_BASE.txt` → `.../meanfield_mlp512_student_step9000.pkl`

**任务测评**（`eval_seed=1234`，`max_step=256`；step9000 / best 同源量级）：

| 规模 | safe↑ | finish↑ | success↑ | 备注 |
|---|---:|---:|---:|---|
| 32-epi | 80.86% | 94.14% | **77.73%** | best ckpt 记录 |
| 64-epi（压力） | — | — | **~78.7%** | distill 选点参考 |

| 对照 | success@32 |
|---|---:|
| GCBF+ 教师 | ~91.4% |
| DeepSet seed0 base | ~89.1% |
| MF v5 base | **~77.7%** |
| MF 早期 22-D 首轮 | ~58.6%（已弃） |

**经验 danger 两步 L2 hit**（rollout 近危采样，非 DeepSet 分桶 D-hit）：base **5.95%**（542/9108）。  
**结论**：v5 相对 22-D 大幅抬升任务性能，仍低于 DeepSet ~10+ pt；**不追求追平 DeepSet success**，目标是验证友好 + 可修 CE。经验指标，非形式化。

### MF.3 Residual 修复（physics gate）

部署分解（与 DeepSet residual 同形，门控不同）：

```text
a_deploy(x) = clip( π_base(x) + g_phys(x) · clip(δ(x)) )
```

| 部件 | 含义 |
|---|---|
| `π_base` | 冻结 MF MLP512 |
| `g_phys` | **物理 ReLU 门**（非 DeepSet hybrid sigmoid）：距离包络 open=3×R / close=1×R + inv_ttc 支路，`gate_mix=0.5` |
| `δ` | 小 MLP `44→128→128→2`（~22k 参），`delta_clip` |

流程：向量化 CE 采集（两步 L2，邻 coast / hit 静）→ 安全标签 → 只训 δ（+idle 保任务）。

#### r1

产物：`method/meanfield/output/repair/mf_residual_v5_20261005/seed0/`

| 项 | base | r1 |
|---|---:|---:|
| hit_on_danger | 5.95% | **1.31%** |
| success@32 | ~77.7% | **83.2%** |
| safe@32 | ~80.9% | **93.0%** |
| finish@32 | ~94% | **87.9%** |

#### r2（当前 active residual）

产物：`method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/`  
指针：`method/meanfield/output/ACTIVE_MF_RESIDUAL.txt` → `.../repair_best.pkl`

| 项 | 值 |
|---|---|
| 输入 | r1 best warm-start |
| 超参 | lr=5e-5，idle_w=2.5，delta_l2=2e-3，steps=500 |
| CE 池 | merge(r1 post + fresh + base)，262 labeled-safe / 736 raw |

**CE hit_on_danger（两步 L2，经验）**：

| phase | hit_on_danger | n_hit / n_danger |
|---|---:|---|
| base v5 | 5.95% | 542 / 9108 |
| residual r1 | 1.31% | 128 / 9783 |
| **residual r2** | **0.47%** | **33 / 7089** |

相对 base **≈12.7×** 下降。r2 残留 CE 以 **AO** 为主（33/33）。

**任务 32-epi**（`eval_seed=1234`）：

| 策略 | success↑ | safe↑ | finish↑ |
|---|---:|---:|---:|
| base step9000 | ~77.7% | ~80.9% | ~94.1% |
| residual r1 | 83.20% | 92.97% | 87.89% |
| **residual r2** | **82.03%** | **96.48%** | **83.98%** |
| DeepSet V2d_r2（对照） | ~87.9% | — | — |

**结论**：r2 进一步压碰撞、抬 safe（~96.5%）；finish 下滑、success 相对 r1 略降仍 >80%。**经验修复 only** — `sampling_clear ≠ proven`；r2 后 hit&lt;1%，进入形式化。

### MF.4 形式化验证（定稿）

完整约定：[`docs/MEANFIELD_VERIFY.md`](docs/MEANFIELD_VERIFY.md)。

#### 声明

```text
∀ x ∈ X ⊂ R^{44}:   TwoStepRadialL2Safe( Deploy(x); R )
```

| 部分 | 含义 |
|---|---|
| **查询域 X** | 网络输入**每个自由维**的轴对齐区间积（速度、目标、各象限 min_d / wmean / inv_ttc …） |
| **覆盖** | IBP →（小盒）Marabou 分治；目标 **PROVEN_FULL**；真实 CE **全部收集** |
| **性质** | **两步**径向 L2（地平线简化，与 DeepSet L2 同思想） |
| **不做** | DeepSet (k,m) 桶表；主验收 AA/AO 分表 |

性质公式（占用象限 n≥0.5 才计入 d0/c0）：

```text
d0 = min occupied min_d
c0 = max(0, max occupied close_max)
α  = n_steps·dt = 0.06
β  = (dt²/mass)·n(n-1)/2 = 0.009
d2 = d0 − α c0 − β |a|_∞
Safe ⇔ d2 ≥ R     (默认 R = COLLISION_AA = 0.10)
```

- 径向头对头过近似：对平面碰撞 **证明安全充分**；SAT 可能虚警 → 真策略再检。  
- **分层可避免带**（排除 a=0 已撞点，否则 CE 淹没）：

| 层 | mode | min_d | \|close\| | 最坏 \(d_2\) | 角色 |
|---|---|---|---|---|---|
| L1 | near | [0.15, 0.35] | ≤0.25 | 0.126 | 基线可避免带 |
| **L2** | **near_plus** | **[0.14, 0.40]** | **≤0.30** | **0.113** | **有界主验收** |
| L3 | full | [0.12, 0.50] | ≤0.50 | 0.081\<R | 宽盒附录（不冲 PROVEN） |

  共用：agent Q0 强制占用，inv_ttc≤8（near/plus），R=0.10。

#### 工具链

| 模块 | 路径 |
|---|---|
| 域 / claim / manifest | `stage_mf/domain.py` |
| 两步性质 + IBP | `stage_mf/property_l2.py` |
| 权重 / 部署 | `stage_mf/encode.py` |
| 动作 IBP | `stage_mf/ibp_mf.py` |
| Marabou（base+gate+δ + d2\<R） | `stage_mf/marabou_mf.py` |
| 分治入口 | `stage_mf/verify_full.py` |
| 启动 | `scripts/run_mf_verify_full.sh [ckpt] [smoke\|near\|near_plus\|full]` |

状态语义：

| 状态 | 含义 |
|---|---|
| **PROVEN_FULL** | 声明 X 全覆盖 proven，0 真实 CE，0 unknown |
| **PARTIAL** | 资源内未盖满；unknown 上已采 CE |
| **VIOLATED** | ≥1 真实策略 CE → `counterexamples_all.json` |

#### 当前形式化结果（r2 residual，2026-10-05）

对象：`ACTIVE_MF_RESIDUAL` = r2 `repair_best.pkl`。  
产物指针：`method/meanfield/output/ACTIVE_MF_VERIFY.txt` → `method/meanfield/output/verify/mf_feat44_near_plus_r2/`

| mode | 声明盒 | 结果 | 方法 / 备注 |
|---|---|---|---|
| smoke | 点邻域（Q0 占用 min_d≈0.25） | **PROVEN_FULL** | IBP；Marabou 交叉 UNSAT ~0.5s |
| near | min_d∈[0.15,0.35]，\|close\|≤0.25，Q0 occupied | **PROVEN_FULL** | 整盒 IBP（d2_lo=0.126≥R），0 CE，~2ms |
| **near_plus** | **min_d∈[0.14,0.40]，\|close\|≤0.30，Q0 occupied** | **PROVEN_FULL** | **整盒 IBP（d2_lo=0.113≥R），0 CE，~2ms** ← **有界主验收** |
| full（L3 附录） | min_d∈[0.12,0.50]，\|close\|≤0.5 | **VIOLATED_WITH_CES** | 物理 d2_lo=0.081\<R；**11260 CE**，d0 全∈[0.12,0.14)；**100% 在 near_plus 外** → 支持 L2 切界 |

**正确读法**：

```text
near_plus PROVEN_FULL  =  有界主验收：声明可避免 44 维特征盒全覆盖
L3 full VIOLATED       =  预期附录：宽盒下缘 CE；不否定 L2
                       ≠  无限时域部署零间隙
                       ≠  DeepSet preventable-D 三桶 proven
全量验证能力（附录）   =  固定维 MLP 对任意声明盒可 IBP+B&B+Marabou 完备分治
```

Soundness：特征盒 ⊇ 可实现场景的过近似；径向 safe ⇒ 同 d0/c0 下平面两步 safe（充分）；Marabou UNSAT 对占用感知性质仍 sound（编码侧 d0/c0 更松）。

### MF.5 链路现状与对照

```text
同一 GCBF+ 教师
  ├─ DeepSet: base → V2d r1 → r2（任务主路径；全域 formal 未达成）
  └─ Mean-field: v5 base → residual r1 → r2
                 → 44-D 特征盒 formal：smoke / near / near_plus PROVEN_FULL
```

| 层级 | DeepSet V2d_r2 | MF residual r2 |
|---|---|---|
| 任务 success@32 | ~87.9% | **82.0%** |
| 经验近危碰撞 | D-hit mean ~19.3%（分桶） | hit_on_danger **0.47%**（rollout） |
| 完整声明域采样 | 4k/桶仍 3×violated | —（指标不同，不直接比） |
| 形式化 | 局部 residual PoC UNSAT；主桶未 proven | **near_plus 特征盒 PROVEN_FULL**（IBP） |
| 隔离 | `phase2/ACTIVE_*` | `meanfield/ACTIVE_MF_*` |

**不把 MF success 追到 DeepSet ~89% 作为验收**；MF 价值在固定维 + 可全覆盖声明盒。  
L3 full 附录已收 CE（见 `verify/mf_feat44_full_r2/APPENDIX_L3.md`）。后续可选：r3 仅在需抬 finish 时；平面几何加强性质（更重）。

### MF.6 Active 指针一览

| 指针 | 指向 |
|---|---|
| `method/meanfield/output/ACTIVE_MF_BASE.txt` | v5 step9000 base |
| `method/meanfield/output/ACTIVE_MF_RESIDUAL.txt` | r2 `repair_best.pkl` |
| `method/meanfield/output/ACTIVE_MF_VERIFY.txt` | `verify/mf_feat44_near_plus_r2` |

```bash
# 任务 / 修复
bash scripts/run_mf_residual_repair.sh
# 形式化
bash scripts/run_mf_verify_full.sh $(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt) smoke
bash scripts/run_mf_verify_full.sh $(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt) near
bash scripts/run_mf_verify_full.sh $(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt) near_plus
```

---

*最近更新：2026-10-05 — MF v5 + r2；**L2 near_plus PROVEN_FULL**（主验收）；**L3 full VIOLATED 附录**（CE 全在 L2 外）；DeepSet ACTIVE_* 未改。*
