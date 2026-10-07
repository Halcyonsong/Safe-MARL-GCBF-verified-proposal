# 全面测评 + 反例修复记录（r2 主验收）

> 本文件把 **CE 修复轨迹**、**@32/@64/@256**、**人数延展**、**延迟**、**基线对照** 收成一份，说明「修复做了什么、为何选 r2」。  
> 形式化主声明仍见 `1_primary/00_acceptance/`；叙事边界见 `NOTES.md`。

---

## 0. 一页结论

| 轴 | 结果 |
|---|---|
| **经验 CE 修复** | danger 两步 L2 hit：**5.95% → 1.31% → 0.47%**（base→r1→r2，≈**12.7×**） |
| **任务安全** | @32 safe：**80.9% → 93.0% → 96.5%**；cost：**4.86 → 0.72 → 0.30** |
| **压力** | r2 @64 / @256 仍 suc≈82–84%、safe≈95–96%（见 §3） |
| **延展 N** | r2 safe@32-agents 仍 **90.7%**；finish↓ 致 suc↓（证书**未**声称大 N） |
| **形式化** | 修复后才具备进盒条件；L2 near_plus **PROVEN_FULL**（主验收） |
| **相对基线** | 任务不追 GCBF+/DS 上限；卖点是 **∀ 证书 + 高 safe/低 cost** |

---

## 1. 反例修复轨迹（为何需要 r1 / r2）

### 1.1 方法

- **冻结 base**：v5 MLP512 distill（`1_primary/01_models/base_v5/`）  
- **残差 δ**：physics-gated（开场/近危混合），两步径向 L2 危险步上的 min-intervention / throttle 标签  
- **r1**：在 base 的经验 CE 上训残差  
- **r2**：合并 CE 池（base baseline + r1 post + 新采 r1 CE）warm-start 再修  

产物与日志：`1_primary/02_train_logs/repair_r{1,2}/` · `1_primary/04_ce_repair/r{1,2}/`  
选型说明：`1_primary/04_ce_repair/r2/SELECTION.md`

### 1.2 经验两步 L2 · hit_on_danger

协议：rollout danger（半径 0.35）上的两步径向 L2；邻居 coast；非形式化采样。

| 阶段 | 策略 | hit_on_danger | n_hit | n_danger | 摘要路径 |
|---|---|---:|---:|---:|---|
| base | v5 step9000 | **5.95%** | 542 | 9108 | `1_primary/04_ce_repair/r1/ce_baseline_summary.json` |
| r1 | residual r1 | **1.31%** | 128 | 9783 | `…/r1/ce_post_summary.json` |
| r1 复测 | r1（seed456） | 0.77% | 66 | 8535 | `…/r2/ce_from_r1_summary.json` |
| **r2** | **residual r2** | **0.47%** | **33** | **7089** | `…/r2/ce_post_summary.json` |

累计相对 base：**5.95% → 0.47% ≈ 12.7× 下降**。  
r2 训练 CE 池：736 raw / 262 labeled-safe（`ce_train_summary.json`）。

### 1.3 任务 @32：修复前后对照

协议：DI · N=8 · area4 · obs4 · rays32 · seed=1234

| 策略 | success↑ | safe↑ | finish↑ | cost↓ | 路径 |
|---|---:|---:|---:|---:|---|
| base v5 | 77.7% | 80.9% | 94.1% | 4.86 | `1_primary/03_task_eval/base_v5_validate_32epi.json` |
| residual r1 | 83.2% | 93.0% | 87.9% | 0.72 | `…/residual_r1_validate_32epi.json` |
| **residual r2 ★** | **82.0%** | **96.5%** | **84.0%** | **0.30** | `…/residual_r2_validate_32epi.json` |

**修复作用（经验层）**

1. **碰撞相关**：danger hit ↓12.7×；safe ↑约 **+15.6pp**（80.9→96.5）；cost ↓约 **16×**。  
2. **权衡**：finish 略降（94→84）——残差更保守（径向节流）；success 相对 r1 基本持平、相对 base 仍 +4pp。  
3. **形式化入口**：经验 CE 压到 sub-1% 后启动有界盒验证；最终 L2 near_plus **PROVEN_FULL**（见 `1_primary/05_formal_verify/L2_near_plus_PRIMARY/`）。  
4. **不是**：把任务做到 DS/GCBF+ 的 success 上限；也不是无限时域零碰撞证明。

---

## 2. 基线对照 · 主表（@32）

论文基线行：GCBF+ · MAPPO-Lag · InforMARL。DS = 任务并行主线；HandCBF / GCBF mid = 同任务参照；coast/mf_base = 消融（不在主表）。

完整叙述表：`2_baselines/BASELINE_COMPARE.md`

| ID | 方法 | 角色 | suc | safe | fin | cost |
|---|---|---|---:|---:|---:|---:|
| **MF-r2** | Mean-field residual r2 | **形式化验收主方法** | **82.0%** | **96.5%** | **84.0%** | **0.305** |
| DS-r2 | DeepSet V2d_r2 | 任务并行主线 | 87.9% | 93.4% | 94.1% | 1.039 |
| GCBF+ | Official @1000 | 教师 / 同族上界 | 91.4% | 95.7% | 95.3% | 0.441 |
| GCBF | mid@270 | 同族消融 | 55.5% | 62.1% | 81.2% | 9.246 |
| HandCBF | DecShareCBF | 外源 QP | 84.8% | 100% | 84.8% | 0.000 |
| MAPPO-Lag | MAPPO+Lagrangian | 外源 | 68.4% | 90.6% | 72.7% | 1.383 |
| InforMARL | 局部图 PPO | 外源 | 60.9% | 62.5% | 95.7% | 6.934 |

**读法：** MF-r2 在 success 上不追 GCBF+/DS；**safe 与 cost 很强**，且是唯一具备同协议 **L2 near_plus ∀ 证书** 的一行。

---

## 3. 压力测评 · @64 / @256

| ID | @64 suc/safe/fin/cost | @256 suc/safe/fin/cost |
|---|---|---|
| **MF-r2** | **82.6 / 94.9 / 85.0 / 0.40** | **83.5 / 95.7 / 86.4 / 0.44** |
| DS-r2 | 88.7 / 93.8 / 94.7 / 0.98 | 88.8 / 93.8 / 94.5 / 1.00 |
| GCBF+ | 92.2 / 95.9 / 96.1 / 0.45 | 91.3 / 95.3 / 95.3 / 0.65 |
| GCBF mid | 56.6 / 61.7 / 82.8 / 8.53 | 56.7 / 62.5 / 83.8 / 7.50 |
| HandCBF | 83.4 / 99.6 / 83.8 / 0.06 | 83.4 / 99.8 / 83.5 / 0.04 |
| MAPPO-Lag | 68.8 / 90.2 / 73.6 / 1.19 | 67.6 / 88.4 / 74.7 / 1.88 |
| InforMARL | 61.3 / 62.7 / 96.3 / 6.11 | 62.8 / 64.3 / 96.3 / 5.78 |

包内 JSON：`2_baselines/epi_pressure/epi{64,256}/`  
MF 专用：`1_primary/03_task_eval/residual_r2_validate_{64,256}epi.json`

**MF-r2：** 加 epi 后 suc/fin 略升、safe 仍 ≥94.9%——主验收点稳定，非过拟合 32-epi。

---

## 4. 人数延展（零样本 N=16 / 32）

训练仍 N=8；评测加人数、不微调。完整表：`2_baselines/TRANSFER_COMPARE.md`  
JSON：`2_baselines/transfer_n/N{8,16,32}/`

格内：`suc% / safe% / fin% / cost`

| 方法 | N=8 | N=16 | N=32 | Δsuc 8→32 |
|---|---|---|---|---:|
| GCBF+ | 91.4 / 95.7 / 95.3 / 0.44 | 88.7 / 96.1 / 91.8 / 0.45 | 81.9 / 93.8 / 86.4 / 0.87 | −9.5pp |
| DS-r2 | 87.9 / 93.4 / 94.1 / 1.04 | 82.8 / 92.4 / 89.1 / 1.04 | 73.9 / 89.5 / 80.7 / 1.84 | −14.0pp |
| HandCBF | 84.8 / 100 / 84.8 / 0.00 | 76.6 / 100 / 76.6 / 0.00 | 63.7 / 97.0 / 64.8 / 0.55 | −21.1pp |
| **MF-r2** | **82.0 / 96.5 / 84.0 / 0.31** | **72.9 / 94.3 / 75.6 / 0.73** | **48.4 / 90.7 / 50.8 / 1.23** | **−33.6pp** |
| MAPPO-Lag | 68.4 / 90.6 / 72.7 / 1.38 | 57.4 / 78.3 / 69.7 / 3.60 | 39.2 / 54.0 / 62.5 / 8.33 | −29.2pp |
| InforMARL | 60.9 / 62.5 / 95.7 / 6.93 | 44.7 / 45.9 / 95.7 / 8.96 | 25.3 / 25.4 / 96.1 / 14.87 | −35.6pp |

**MF 读法：** @32-agents **safe 仍 90.7%**（修复带来的保守性在延展中保留）；掉点主要来自 **finish**（拥挤下到达变差）。**形式化证书不随 N 外推**——延展是经验辅轴。

---

## 5. 控制延迟（辅）

同硬件；JIT warm-up；ms / 一次 8-agent 控制（不含 `env.step`）。  
`2_baselines/latency/control_latency.json`

| 方法 | median↓ | mean↓ | p95↓ |
|---|---:|---:|---:|
| **MF-r2** | **1.37** | 1.54 | 1.63 |
| GCBF+ | 1.39 | 1.41 | 1.57 |
| HandCBF（在线 QP） | 4.72 | 5.20 | 8.55 |

MF 与 GCBF+ 同量级；远快于在线 QP。

---

## 6. 修复 → 形式化 → 基线：怎么串起来写

| 层次 | 证据 | 主张 |
|---|---|---|
| 经验修复 | §1 hit 5.95%→0.47%；safe↑ / cost↓ | 残差 CE 修复有效压近危两步 L2 失败 |
| 任务可用 | §2–§3 @32/64/256 | 部署指标稳定；不牺牲到 soft 退化线以下 |
| 延展辅轴 | §4 | safe 保持；finish 是大 N 瓶颈（非证书失效） |
| **形式化主** | L2 near_plus PROVEN | 有界盒 ∀ 安全；基线无同级证书 |
| 勿写 | — | 「修复后任务全面超过 GCBF+」；「大 N 已形式化」；「主盒上基线必有 binary L2 fail」 |

---

## 7. 包内路径索引

| 内容 | 路径 |
|---|---|
| 本文件 | `EVAL_AND_REPAIR.md` |
| 修复日志 / SELECTION | `1_primary/02_train_logs/repair_r{1,2}/` |
| CE 摘要 JSON | `1_primary/04_ce_repair/r{1,2}/` |
| 任务 JSON（含 64/256） | `1_primary/03_task_eval/` |
| 基线主表 | `2_baselines/BASELINE_COMPARE.md` |
| 延展表 | `2_baselines/TRANSFER_COMPARE.md` |
| 延展 / 压力 / 延迟 JSON | `2_baselines/transfer_n/` · `epi_pressure/` · `latency/` |
| 形式化证书 | `1_primary/00_acceptance/ACCEPTANCE_CERTIFICATE.md` |
| 全量数据速查 | `RECORD.md` |
