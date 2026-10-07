# Safe-MARL · 可验证平均场安全多智能体控制

**一句话：** 用固定维 Mean-field 特征 + 残差门控修复，在声明盒上为 **Deploy 策略** 给出两步径向 L2 的 **∀ 形式化证书**；任务评测支撑可部署性。  
**不是：** 仿真全程零碰撞证明，也不是无限时域闭环安全证明。

| | |
|---|---|
| **主验收** | residual **r2** · L2 `near_plus` · **PROVEN_FULL** |
| **交付包** | [`acceptance/mf_best_20261006/`](acceptance/mf_best_20261006/) |
| **任务 @32** | success **82.0%** · safe **96.5%** · finish **84.0%** · cost **0.30** |
| **声明域** | min_d∈[0.14, 0.40]，\|close\|≤0.30，R=0.10（44-D 特征盒） |
| **论文基线** | GCBF+ · MAPPO-Lag · InforMARL（coast / mf_base = 消融） |

中文长介绍 → [`PROJECT_INTRO.md`](PROJECT_INTRO.md) · 目录地图 → [`docs/ACTIVE_LAYOUT.md`](docs/ACTIVE_LAYOUT.md)

---

## 1. 要解决什么问题

多智能体避障（本仓库：DoubleIntegrator，N=8）上，学习型控制器（GCBF+、MAPPO-Lag、InforMARL 等）**任务强、但缺少同级形式化证书**。障碍主要有三：

1. **变结构 GNN** 输入维随邻居数变化，主流验证器难直接吃。  
2. **部署策略 ≠ 验证对象** 时，证书说不清「上线跑的是不是证过的那个」。  
3. **全域 / 无限时域** 证明过重；需要可交付的 **有界、可复现** 声明。

本项目的答案：**固定 44 维平均场特征 → ReLU-MLP → physics-gated 残差 Deploy**；在显式特征盒上验证 **两步径向 L2**，证书对象就是 Deploy。

---

## 2. 方法（实验管线）

```text
GCBF+ (GNN teacher)
    │  distill
    ▼
MF base π_base   (44-D → MLP512 → a∈[-1,1]²)     ← ACTIVE_MF_BASE
    │  sample CE on danger / two-step L2
    │  label safe actions · keep task samples
    ▼
residual δ       (physics gate g_phys · clip δ)   ← r1 → r2
    │
    ▼
Deploy:  a = clip( π_base(x) + g_phys(x)·clip(δ(x)) )
    │
    ▼
Formal:  ∀ x ∈ X_near_plus:  TwoStepRadialL2Safe(Deploy(x); R=0.10)
```

**特征（v5，44-D）**

| 块 | 内容 |
|---|---|
| Ego (8) | px, py, vx, vy, goal_dx, goal_dy, goal_dvx, goal_dvy |
| Agent ×4 象限 (20) | n, min_d, close_max, close_wmean, inv_ttc_max |
| Obs ×4 象限 (16) | n, min_d, close_max, inv_ttc_max |

**动力学：** 官方 GCBF+ Euler；L2 = 两步持有动作后的径向距离下界。  
**验证：** 区间传播（IBP）；本主盒一次整盒 PROVEN（~2 ms，0 CE）。约定见 [`docs/MEANFIELD_VERIFY.md`](docs/MEANFIELD_VERIFY.md)。

**并行轨（非主验收）：** DeepSet（变长集合 → 固定学生）作任务对照；Phase-2 `ACTIVE_*` **永不被 MF 覆盖**。

---

## 3. 主验收结果（可写 / 不可写）

### 可写（主文）

1. r2 在 \(X_{\mathrm{near\_plus}}\) 上两步径向 L2（R=0.10）**PROVEN_FULL**（有界盒 ∀ 证书）。  
2. MAPPO / GCBF+ / InforMARL **无同协议同级证书**（非对称 claim）。  
3. 任务评测：@32 safe **96.5%**、cost **0.30**；@64/@256 仍稳定（见验收包）。  
4. 经验修复：danger 两步 hit **5.95% → 0.47%**（≈12.7×）。

### 不可写

1. mid-F / 更紧网络依赖带「已证满」——**未证满**（附录仍 VIOLATED）。  
2. 「主验收盒上基线必有两步 L2 反例」——包络 MC 下 binary fail≈0。  
3. coast / mf_base 当作论文基线——它们是**消融**。  
4. 无限时域 / 仿真零碰撞 = 形式化结论。

细节与附录结论：[`acceptance/mf_best_20261006/NOTES.md`](acceptance/mf_best_20261006/NOTES.md)

### 任务对照（@32，摘录）

| 方法 | 角色 | suc | safe | cost |
|---|---|---:|---:|---:|
| **MF-r2** | **形式化主方法** | **82.0** | **96.5** | **0.30** |
| DeepSet-r2 | 任务并行轨 | 87.9 | 93.4 | 1.04 |
| GCBF+ | 教师 / 上界 | 91.4 | 95.7 | 0.44 |
| MAPPO-Lag | 论文基线 | 68.4 | 90.6 | 1.38 |
| InforMARL | 论文基线 | 60.9 | 62.5 | 6.93 |

全表 → [`baselines/compare/BASELINE_COMPARE.md`](baselines/compare/BASELINE_COMPARE.md) · 修复轨迹 → [`acceptance/mf_best_20261006/EVAL_AND_REPAIR.md`](acceptance/mf_best_20261006/EVAL_AND_REPAIR.md)

---

## 4. 仓库怎么读

```text
acceptance/mf_best_20261006/   ★ 唯一验收交付（证书·模型·评测·基线·论文图）
method/meanfield/              主方法代码 · config · scripts · ACTIVE 指针
method/deepset/                冻结参考轨（+ tests/）
baselines/                     gcbfplus · gcbf · handcbf · mappo_lag · informarl · compare/
docs/                          布局 / 验证约定 / 历史设计
core/                          官方 GCBF+（import gcbfplus）
scripts/                       仅共享：cuda_env.sh · plot_train_curves.py
refer/                         本地论文 PDF（默认不进 git）
```

**验收包内先读顺序：** `README` → `EVAL_AND_REPAIR.md` → `NOTES.md` → `1_primary/00_acceptance/` 证书。  
论文主稿：`acceptance/mf_best_20261006/docs/PAPER_AAMAS_CN.md`。

导入名未改（`stage_mf`、`stage2`、`baseline.mappo_lag`…）；`core.paths.ensure_import_path()` 接线。

---

## 5. 快速复现

```bash
# 环境（本机示例）
source scripts/cuda_env.sh
# /home/xiang/miniconda3/envs/safe_marl_verify/bin/python -c 'import jax; print(jax.devices())'

# 任务评测（活指针 → r2）
python method/meanfield/stage_mf/validate_residual.py

# 形式化主验证（near_plus）
bash method/meanfield/scripts/run_mf_verify_full.sh \
  "$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt)" near_plus

# 或直接用验收包权重
bash method/meanfield/scripts/run_mf_verify_full.sh \
  acceptance/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl \
  near_plus

cd acceptance/mf_best_20261006 && sha256sum -c SHA256SUMS.txt
```

| 指针 | 指向 |
|---|---|
| `method/meanfield/output/ACTIVE_MF_BASE.txt` | distill v5 step9000 |
| `method/meanfield/output/ACTIVE_MF_RESIDUAL.txt` | residual **r2** best |
| `method/meanfield/output/ACTIVE_MF_VERIFY.txt` | L2 near_plus PROVEN_FULL |
| `method/deepset/output/phase2/ACTIVE_*` | DeepSet Phase-2（勿被 MF 覆盖） |

**不要**在未显式 lock 时：覆盖 DeepSet Phase-1 frozen、升格 mid-F 旁支、改写主验收包语义。

---

## 6. 文档索引

| 文档 | 用途 |
|---|---|
| [`acceptance/mf_best_20261006/`](acceptance/mf_best_20261006/) | ★ 主验收：证书 / 指标 / 论文 |
| [`docs/ACTIVE_LAYOUT.md`](docs/ACTIVE_LAYOUT.md) | 目录地图（重整后） |
| [`docs/MEANFIELD_VERIFY.md`](docs/MEANFIELD_VERIFY.md) | MF 形式化约定与命令 |
| [`PROJECT_INTRO.md`](PROJECT_INTRO.md) | 中文项目长介绍 |
| [`WORK_ASSESSMENT.md`](WORK_ASSESSMENT.md) | 2026-10-06 评估快照 |
| [`docs/DESIGN_PROPOSAL.md`](docs/DESIGN_PROPOSAL.md) | DeepSet 早期设计（历史） |
| [`baselines/compare/BASELINE_COMPARE.md`](baselines/compare/BASELINE_COMPARE.md) | 基线对照表 |

---

## 7. Git 上传策略

[`.gitignore`](.gitignore)：**可复现研究仓**（预计 ~84 MB 进仓 vs ~1.3 GB 工作树）。

| 上传 | 跳过 |
|---|---|
| 代码 · 文档 · 配置 · 脚本 · 测试 | `__pycache__` · `.idea` · `wandb` |
| 完整 `acceptance/`（含 L3） | `refer/` PDF（版权；本地自留） |
| MF ACTIVE 权重 + near_plus 报告 | 其它 `method/*/output` 大树 · `*_step*.pkl` · `steps/` |
| DeepSet frozen + ACTIVE `V2d_r2` | redistill / 其它 phase2 |
| 基线 `output/` + GCBF+ teacher `log/` | 其余 `baselines/*/log` · `env_pack.pkl` |

`git init` / 首次推送：待你确认后再做。

---

## 8. 依赖与引用

- 环境：`requirements.txt` · 建议 conda env `safe_marl_verify`（含 JAX）。  
- 核心依赖官方 **GCBF+** 动力学与教师（`core/`）。  
- 文献 PDF 放在本地 `refer/`（默认 gitignore）。
