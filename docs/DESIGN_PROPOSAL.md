# 设计方案提案：可验证 DeepSet 安全多智能体控制

> 版本：2026-10-03（**历史设计提案**）  
> 项目：`Safe-MARL-GCBF-verified-proposal`  
> 目标：在官方 GCBF+ 对照基线之上，构造**验证友好**的 DeepSet 策略，并对其做有界形式化验证。
>
> **状态（2026-10-07）**：本文保留作 DeepSet 轨设计溯源。  
> **当前主验收** = Mean-field r2 L2 `near_plus` → [`acceptance/mf_best_20261006/`](../acceptance/mf_best_20261006/)。  
> **目录 / 命令** → [`ACTIVE_LAYOUT.md`](ACTIVE_LAYOUT.md) · 仓库根 [`README.md`](../README.md)。  
> 下文 §6 布局表为提案时快照，物理路径已迁入 `method/deepset/` 等。

---

## 1. 问题与动机

多智能体避障控制中，官方 GCBF+ 使用 GNN Actor + 候选 GCBF + CBF-QP，经验性能强，但：

1. **GNN 变长图结构**难以直接交给 Marabou 等 DNN 验证器；
2. **在线 QP** 部署代价高，且不是最终要验证的对象；
3. 需要一条链路：  
   **可部署低延迟策略** ↔ **与部署完全一致的验证对象**。

本方案因此拆成：

```text
对照基线：官方 GCBF+（GNN + QP 教师）
方法 Stage1：蒸馏到固定结构 DeepSet（可部署 / 可验证）
方法 Stage2：对冻结 DeepSet 做有界形式化验证
方法 Stage3：反例采集 + 反馈修复（经验闭环；非正式证明）
```

---

## 2. 总体架构

```text
┌─────────────────────────────────────────────┐
│ Baseline: Official GCBF+ (MIT-REALM JAX)    │
│  Candidate CBF → CBF-QP (u_qp)              │
│  GNN Actor ≈ u_qp                           │
│  用途：对照实验 + Stage1 蒸馏教师            │
└──────────────────────┬──────────────────────┘
                       │ rollout actions / (可选) u_qp
                       ▼
┌─────────────────────────────────────────────┐
│ Stage1 (方法): DeepSet Student              │
│  ego(8) + entities(40×7)                    │
│  φ (共享 ReLU) → masked sum → ρ (ReLU) → a  │
│  冻结产物供部署与 Stage2 验证                │
└──────────────────────┬──────────────────────┘
                       │ exact checkpoint
                       ▼
┌─────────────────────────────────────────────┐
│ Stage2 (方法): Formal Verification          │
│  按 (k,m) presence 分桶 + 连续有界盒         │
│  L1: 裸网 a∈[-1,1]²（已跑；大盒越界）        │
│  L2: clip∘π 两步 held-action 局部不碰撞     │
└──────────────────────┬──────────────────────┘
                       │ CEs / D metrics
                       ▼
┌─────────────────────────────────────────────┐
│ Stage3 (方法): CE-Guided Repair             │
│  采 D 上反例 → L2-safe / min-intervention   │
│  混冻结 on-policy 锚定 → 微调 DeepSet       │
│  产物: method/deepset/output/stage3/{ce,repair}/             │
└─────────────────────────────────────────────┘
```

### 2.1 命名边界（重要）

| 名称 | 含义 | 是否方法主线 |
|---|---|---|
| `baseline/gcbfplus` | 官方 GCBF+ GNN 教师 | 否（对照 / 教师） |
| `stage1/` DeepSet | 验证友好学生策略 | **是** |
| `stage2/` | 对 Stage1 DeepSet 的形式化验证 | **是** |
| `stage3/` | 反例引导反馈修复 | **是**（经验；≠ proven） |

**禁止混淆：**

```text
验证 Stage1 DeepSet 成功
  ≠  官方 GNN Actor 已形式化安全
```

---

## 3. 环境与物理契约

### 3.1 任务设定（当前实验）

```text
Env:          DoubleIntegrator
area_size:    4.0
num_agents:   8
obstacles:    4 random rectangles
n_rays:       32
episode:      256 steps（标记碰撞/到达，不提前终止）
dt:           0.03
action:       [-1, 1]²（环境侧 clip）
```

### 3.2 安全相关常量

```text
car_radius         = 0.05
collision_distance = 0.10   # 2 * car_radius
communication_radius = 0.50
```

离散动力学：官方 `step` 使用 **欧拉** `agent_step_euler`（`accel=a/mass`，`mass=0.1`）：

```text
p' = p + v·dt
v' = clip(v + (a/m)·dt, ±0.5)
```

注意：单步欧拉**不把当前加速度写入位置**。因此碰撞相关验证使用 **保持同一 clip 动作的两步欧拉**，使加速度进入位置。`agent_step_exact` 的二次式与部署不一致，验证不用。

---

## 4. Stage1：DeepSet 设计

### 4.1 为什么用 DeepSet

相对 GNN：

- **固定输入维**：`ego + 固定槽位`，便于验证器编码；
- **置换近似不变**：共享 φ + sum；
- **空槽可屏蔽**：`presence=0` 且特征置零；
- **纯 ReLU MLP**：与 Marabou / IBP 兼容。

### 4.2 输入张量

对每个 ego agent：

```text
ego ∈ R^8
  [px, py, vx, vy, goal_dx, goal_dy, goal_dvx, goal_dvy]

entities ∈ R^{40×7}
  行 0..7  : 邻居智能体槽（最多 8，含自环被 mask）
  行 8..39 : lidar 障碍 hit 槽（最多 32）

每行:
  [dx, dy, dvx, dvy, type_agent, type_obs, presence]
```

**智能体与障碍必须分类型：**

| 实体 | type_agent | type_obs | 典型速度 |
|---|---:|---:|---|
| 邻居 | 1 | 0 | 有相对速度 |
| 障碍 hit | 0 | 1 | 常为 0 |
| 空槽 | 0 | 0 | 全 0，presence=0 |

不是“同一类只改速度”；type 是显式类别特征。φ 权重共享，但靠 type 区分语义。

### 4.3 网络结构

```text
h_i = ReLU∘Dense³_φ (entity_i)     # 共享 φ，hidden=512
pooled = Σ_i h_i * presence_i      # masked sum
a = Dense∘ReLU∘Dense³_ρ ([ego; pooled])  # → R²
```

参数量约 **1.32M**（hidden=512）。

### 4.4 蒸馏路径

#### 主路径 A：Actor 蒸馏（当前冻结基线）

```text
冻结 GCBF+ GNN Actor
  → rollout 得状态与 Actor 动作
  → DeepSet 模仿 Actor 动作
  → danger-weighted MSE（近邻/近障加权）
```

优点：标签便宜、与教师部署行为接近。  
代价：相对 QP 是**二次蒸馏**（QP→Actor→DeepSet）。

#### 实验路径 B：直接 QP 蒸馏

```text
用 Actor（或策略）采集状态分布
  → 每状态解官方 CBF-QP 得 u_qp
  → DeepSet ≈ u_qp
```

优点：避免二次误差放大。  
代价：QP 标号昂贵；标签在危险边界更“尖”，拟合更难。

#### 实验路径 C：Danger+QP 精修

在 Actor 学生上，对近险/合成近险状态用 `u_qp` 精修。  
早期 256 维实验未超过 Actor 基线，故未并入冻结模型。

### 4.5 训练与选择协议

```text
优化：Adam, lr=1e-4
buffer + minibatch
周期性短评估（success）保存 best
最终独立 32-episode 经验验证（非正式证明）
```

冻结规则：

```text
只冻结“干净”的主结果，便于后续公平对比
当前冻结：Actor512 @ step8000
  method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl
```

---

## 5. Stage2：形式化验证设计

### 5.1 验证对象

**只能是**与部署相同的冻结 DeepSet checkpoint：

```text
method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl
```

部署契约（与 `DoubleIntegrator.step` 一致）：

```text
a_deploy(x) := clip( DeepSet(x), -1, 1 )
```

L1 已验证**裸输出** `a_raw`；大盒上存在 `|a_raw|>1`。  
**不单独做 L1-clip**：`clip` 后动作界由定义恒真，对安全无新信息。  
`clip` 只作为 L2 动力学输入的一部分编入验证图。

### 5.2 为什么按 (k,m) 分桶

- 全局场景固定 8 agents / 4 rectangles；
- 网络看到的是**局部有效槽位数**：`k` 邻居、`m` 障碍 lidar hit；
- 连续相对量用**有界盒**全称量化。

### 5.3 方法栈

| 方法 | 角色 |
|---|---|
| 随机采样攻击 | 可反驳，不可证明 |
| IBP | 可靠但不完备的界传播 |
| Marabou（ONNX） | 完备查询；大网大盒易 TIMEOUT |

交付物仍是分桶矩阵：`proven_* / violated / timeout / unknown`。

### 5.4 验证层级（更新）

```text
L1       裸网动作界 a_raw∈[-1,1]²     ← 已做（smoke proven；主盒 violated）
L1-clip  （跳过）定义恒真，不单独交付
L2       局部两步不碰撞（Stage2）     ← 已实现采样/IBP/Marabou 接口
Stage3   反例重训闭环（经验修复）     ← collect + retrain；≠ formal proven
L3       可选小证书 h_ver / DCBF（后续）
```

---

### 5.5 L2：局部一步碰撞验证（设计）

#### 5.5.1 要回答的问题

在**声明的局部输入盒**上，部署策略走**环境同款一步动力学**后，是否进入碰撞集？

```text
∀ x ∈ Box(k,m):
  a = clip(DeepSet(x))
  s' = f_euler(s, a)          # 与 env.step 使用的欧拉一致
  ⇒  ¬ Collision(s')
```

这是**有限域、单步、局部**性质；**不**等于无限时域全局无碰撞。

#### 5.5.2 动力学契约（必须与代码一致）

官方 `step` 调用的是 `agent_step_euler`（不是 `agent_step_exact`）：

```text
mass m = 0.1
accel = a_deploy / m = 10 · a_deploy
p' = p + v · dt
v' = clip(v + accel · dt, ±0.5)
dt  = 0.03
```

**单步位置与当前动作无关**，故 L2 验证 **两步 hold-action 欧拉**（与连续两次 `env.step` 且动作相同一致）。误用 exact 二次式会与部署不一致。

动作截断编码（Marabou / ONNX，纯 ReLU）：

```text
clip(a,-1,1) = -1 + ReLU(a+1) - ReLU(a-1)
```

#### 5.5.3 碰撞定义

与环境 `get_cost` / `unsafe_mask` 对齐：

```text
agent–agent:  ‖p_i - p_j‖ < 2·car_radius = 0.10
agent–障碍:   inside_obstacles(p_i, rectangles, r=car_radius=0.05)
```

局部特征里障碍以 **lidar hit 点**出现（相对向量 `dx,dy`），不是完整矩形。因此 L2 声明必须写清几何代理（见下）。

#### 5.5.4 多智能体难点与假设

DeepSet 只输出 **ego** 的动作；邻居下一步动作在局部特征中不可直接得。L2 v1 采用显式假设（写进报告）：

| 子性质 | 场景 | 邻居/障碍运动假设 | 碰撞代理 |
|---|---|---|---|
| **L2-AA** | `k≥1, m=0` 近邻盒 | 邻居**匀速外推**（其加速度=0）：`p_j'=p_j+v_j·dt` | ‖p_i'-p_j'‖ ≥ 0.10 |
| **L2-AO** | `k=0, m≥1` 近障盒 | 障碍 / hit **静止** | ‖p_i'-hit‖ ≥ 0.05（hit 球代理） |
| **L2-AA-WC**（可选） | 同 AA | 邻居加速度在 `[-1,1]²` 最坏 | 更强，但更难证 |

相对量约定（与 `extract_local_sets` 一致）：

```text
entity[dx,dy,dvx,dvy] = (s_ego - s_other)
⇒ p_ego - p_other = (dx, dy),  v_ego - v_other = (dvx, dvy)
```

在 AA + 邻居 coast 假设下，相对下一步可写成 ego 状态与 `a_deploy` 的仿射函数（便于 IBP / Marabou 线性约束）。

**L2-AO 声明边界：**  
`‖p'-hit‖≥r` 只保证「不落入该 hit 的 r-球」；**不能**单独推出「相对整块矩形障碍安全」（未观测到的边/角仍可能撞）。v1 诚实标注为 **observed-hit 局部安全**；完整矩形几何留待后续把障碍参数纳入输入盒。

#### 5.5.5 首版桶（近险优先）

先做小、近、可攻击的盒（证明或找 CE），不做全空间：

```text
l2_aa_near_smoke   k=1,m=0  极小盒     工具链 smoke
l2_aa_near         k=1,m=0  dist≤0.20  邻居一步不撞（coast）
l2_ao_near_smoke   k=0,m=1  极小盒
l2_ao_near         k=0,m=1  dist≤0.20  hit 球一步不穿入
```

速度界首版可收紧到与 `state_lim` 一致的 `|v|≤0.5`（比 L1 的 1.5 更贴环境），降低盒体积。

#### 5.5.6 证明 / 反驳流程

每个桶：

```text
1. Sampling
   x ~ Box; a=clip(π(x)); s'=f_euler(...); 若 Collision → violated + CE

2. IBP（不完备）
   传播 a_raw 区间 → clip → s' 区间盒
   若下一步包围盒与碰撞集不相交 → proven_ibp（过近似安全）
   若相交 → 不结论，交 Marabou

3. Marabou
   ONNX: DeepSet + clip(+ 可选仿射动力学层)
   查询否定：∃x∈Box 使 CollisionProxy(s'(x))
   UNSAT → proven；SAT → violated + CE；否则 timeout
```

碰撞的欧氏球约束非凸。v1 实用策略：

- **找 CE（反驳）**：采样用真欧氏距离；Marabou 可用「进入轴对齐危险相对盒」等**充分碰撞条件**（SAT 仍是真撞的充分证据，若代理保守）。
- **证明安全**：优先 IBP 下一步盒与危险集分离；或 Marabou 上对「不安全过近似」求 UNSAT（证的是更强性质：连过近似危险区都不进）。

#### 5.5.7 Clip 与 Marabou

| 问题 | 结论 |
|---|---|
| 要单独验 L1-clip 吗？ | **否** |
| L2 要把 clip 编进 Marabou 吗？ | **是**（或等价：查询里约束部署动作） |
| 推荐编码 | ONNX 末尾 `clip = -1+ReLU(a+1)-ReLU(a-1)`，与采样/IBP 共用 |

#### 5.5.8 工程落地顺序（实现前清单）

1. `export_onnx.py`：可选 `with_clip=True`  
2. `stage2/dynamics.py`：欧拉一步 + mass + 可选速度 clip（numpy，供采样/IBP）  
3. `stage2/collision.py`：AA 距离 / AO hit 球；记录假设  
4. `stage2/verify_l2.py` + `scripts/run_stage2_l2.sh`：采样 → IBP →（小盒）Marabou  
5. 报告：`method/deepset/output/stage2/l2_*_report.json`  
6. 有碰撞 CE 后再设计 L4 反例重训

#### 5.5.9 可声称 / 不可声称

**可声称（在报告写明盒、假设、动力学后）：**

- 某桶上采样未找到 / 找到一步碰撞 CE；
- 某桶 IBP 或 Marabou 对声明的 CollisionProxy 给出 proven / violated。

**不可声称：**

- 跳过 L2 仅凭 L1 越界谈碰撞；
- 局部 hit 球安全 ⇒ 全局矩形障碍安全；
- 邻居 coast 假设下的 AA 安全 ⇒ 任意邻居策略下安全；
- 单步安全 ⇒ 回合级 / 无限时域安全；
- DeepSet 结论 ⇒ 原 GNN Actor 形式化安全。

---

## 6. 工程结构

```text
# 提案时布局 → 现路径（见 ACTIVE_LAYOUT.md）
core/gcbfplus/                      官方 GCBF+ 源码（未变）
baselines/gcbfplus/{code,log,...}   基线
method/deepset/stage{1,2,3}/        DeepSet 蒸馏 / 验证 / CE 修复
method/deepset/{config,scripts,output}/
method/meanfield/                   ★ 当前主验收轨（本文提案之后新增）
acceptance/mf_best_20261006/        当前交付包
docs/                               精简后的活动文档
```

关键命令（重整后）：

```bash
# 基线
bash baselines/gcbfplus/scripts/validate_baseline_gcbfplus.sh \
  baselines/gcbfplus/log/gcbf+/seed0_20261001232611 1000 32

# DeepSet Stage1
bash method/deepset/scripts/run_stage1.sh \
  baselines/gcbfplus/log/gcbf+/seed0_20261001232611 8000 16 1000 512
bash method/deepset/scripts/validate_stage1.sh \
  method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl 32

# DeepSet Stage2
bash method/deepset/scripts/run_stage2_l1.sh 30 3000

# MF 主验收（当前）
bash method/meanfield/scripts/run_mf_verify_full.sh \
  "$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt)" near_plus
```

---

## 7. 评价与声明边界

### 7.1 经验指标（Stage1）

```text
safe_rate / finish_rate / success_rate / cost / reward
独立 32-episode rollout
```

**经验指标 ≠ 形式化安全证明。**

### 7.2 形式化指标（Stage2）

对每个声明桶记录：

```text
proven_ibp | proven_marabou | violated | timeout | unknown
```

仅当某桶在**同一网络版本 + 同一输入域 + 同一性质**下得到 UNSAT（或可靠 IBP 内含），才可声称该桶性质成立。

### 7.3 不可声称的内容

- 未验证的 `(k,m)` / 界外状态；
- 由 DeepSet 验证结论外推到 GNN Actor；
- TIMEOUT / UNKNOWN 当作通过；
- 无限时域全局安全（除非另行编码并证明）。

---

## 8. 当前决策摘要

1. **对照基线**保留官方 GCBF+（success≈91.4%）。  
2. **方法 Stage1 冻结基线**为 Actor 蒸馏 DeepSet512（success≈87.1%）；**不**换成直接 QP（32-epi success≈65.6%）。  
3. **直接 QP512** 仅作消融对照，产物留在 `method/deepset/output/stage1/qp/`。  
4. **Stage2 L1**：裸网大盒越界；**跳过独立 L1-clip**；下一步实现 **L2 局部一步碰撞**（§5.5）。  
5. 危险集精修暂不并入冻结基线，以保持对照干净。

### 8.1 r1 的双威胁增量消融

`r1` 仅使用 `(1,0)/(0,1)` 的 CE 修复；在 `(1,1)` 上约 13% 的 D-hit
属于迁移指标。新增 `r1→11` 路线从 r1 初始化，重采 r1 的双威胁反例，
正常样本锚定 r1，并在旧 AA/AO CE 输入上回放 r1 动作以限制遗忘。
此路线是独立增量消融，保留此前从 frozen 独立修复 `(1,1)` 的对照结果。

Stage2 双威胁 IBP 分别传播 AA/AO 的下一状态区间，只有两者都避开对应
碰撞球时才可证明。双威胁 Marabou 对 AA/AO 的危险外接方形分别查询，
只有两者都 UNSAT 时才可证明编码相对动力学下的圆盘避免。
查询在外盒上进行；SAT 需回放欧氏碰撞并过滤 D，timeout 不是证明。
当前相对动力学忽略速度裁剪、采用 held-action、邻居 coast 和 hit-ball
假设；结论仅限这个模型，不是部署闭环的完整环境安全保证。

### 8.2 Residual-gate 分解修复

全网微调会在已收敛安全模式上产生灾难性遗忘。Stage3 增加分解部署：

```text
a_deploy(x) = clip( π_base(x) + g(x) · δ(x) )
```

- `π_base`：Stage1 冻结蒸馏 DeepSet（不更新）
- `g(x)`：危险门控（物理距离包络 × 可选学习 soft gate）
- `δ(x)`：小型 DeepSet 修正头（可训练）

远离危险时 `g→0`，策略恒等于基座，因此正常任务轨迹不被改写。
CE 上监督 `a_deploy ≈ a_l2_safe`；正常轨迹上 idle loss 使
`a_deploy ≈ π_base`。形式化验证可拆成：

1. 基座在“gate 关闭区域”的性质（退化为 Stage2 对 base 的查询）；
2. gate 开启区域上对 `base + δ` 的局部查询（更小、更贵）。

v1 只做经验 CE 修复与 32-epi 评估；不声称 residual 已 proven。


---

## 9. 参考文献（项目内）

- GCBF+：`refer/0-CBF-GCBF+ ...pdf`
- KRTJ4225：`refer/KRTJ4225.pdf`
- Zhang et al. AAAI-25（Marabou / 证书学生 / 反例重训）：`refer/06731-ZhangM.pdf`

本项目是上述工作的**组合实现方案**，不宣称任一论文已完整覆盖本链路全部环节。
