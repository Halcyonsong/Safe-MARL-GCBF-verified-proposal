# Baseline Compare（数据表）

> 只记**同协议实测数字**与空位。叙述见 `PROJECT_INTRO.md`、`baseline/README.md`。  
> 填数：改单元格 + 更新「来源」；未测保持 `—`。

---

## 0. 统一测评协议

| 项 | 值 |
|---|---|
| env | DoubleIntegrator |
| agents / area / obs / rays | **8** / **4.0** / **4** / **32** |
| eval_seed | **1234** |
| **主规模** | **32-epi**（主表 / 验收口径） |
| 压力 | **64-epi** |
| 大规模压力 | **256-epi**（辅表；不改主协议） |
| 主指标 | success↑ · safe↑ · finish↑ · **cost↓** |
| 辅指标 | **控制计算耗时**（ms/call；不含 env.step） |
| 形式化（仅 MF 验收轨） | L2 `near_plus` · 44-D · TwoStepRadialL2 · R=0.10 |

GCBF **只保留**选定对照 `mid@270`。HandCBF = 官方 `DecShareCBF`（α=1.0）。512-epi 未跑。

---

## 1. 任务性能 · 32-epi（主）

| ID | 方法 | 角色 | success↑ | safe↑ | finish↑ | cost↓ | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| MF-r2 | **Mean-field residual r2** | **形式化验收主方法** | **82.0%** | **96.5%** | **84.0%** | **0.305** | ✅ |
| DS-r2 | DeepSet V2d_r2 | 任务/修复主线（并行） | 87.9% | 93.4% | 94.1% | 1.039 | ✅ |
| GCBF+ | Official GCBF+ @1000 | 教师 / 同族上界 | **91.4%** | **95.7%** | **95.3%** | **0.441** | ✅ |
| GCBF | Official GCBF mid@270 | 同族消融 | **55.5%** | **62.1%** | **81.2%** | **9.246** | ✅ |
| HandCBF | DecShareCBF（手工 CBF-QP） | 外源同任务 | **84.8%** | **100%** | **84.8%** | **0.000** | ✅ |
| MAPPO-Lag | MAPPO + Lagrangian | 外源同任务 | **68.4%** | **90.6%** | **72.7%** | **1.383** | ✅ |
| InforMARL | 局部图 GNN + PPO | 外源同任务 | **60.9%** | **62.5%** | **95.7%** | **6.934** | ✅ |

---

## 2. 任务性能 · 64-epi（压力）

| ID | 方法 | success↑ | safe↑ | finish↑ | cost↓ | 状态 |
|---|---|---:|---:|---:|---:|---|
| MF-r2 | Mean-field residual r2 | **82.6%** | **94.9%** | **85.0%** | **0.400** | ✅ |
| DS-r2 | DeepSet V2d_r2 | 88.7% | 93.8% | 94.7% | 0.984 | ✅ |
| GCBF+ | Official GCBF+ @1000 | **92.2%** | **95.9%** | **96.1%** | **0.453** | ✅ |
| GCBF | Official GCBF mid@270 | **56.6%** | **61.7%** | **82.8%** | **8.529** | ✅ |
| HandCBF | DecShareCBF | **83.4%** | **99.6%** | **83.8%** | **0.059** | ✅ |
| MAPPO-Lag | MAPPO + Lagrangian | **68.8%** | **90.2%** | **73.6%** | **1.193** | ✅ |
| InforMARL | 局部图 GNN + PPO | **61.3%** | **62.7%** | **96.3%** | **6.109** | ✅ |

---

## 3. 任务性能 · 256-epi（大规模压力）

| ID | 方法 | success↑ | safe↑ | finish↑ | cost↓ | 状态 |
|---|---|---:|---:|---:|---:|---|
| MF-r2 | Mean-field residual r2 | **83.5%** | **95.7%** | **86.4%** | **0.442** | ✅ |
| DS-r2 | DeepSet V2d_r2 | 88.8% | 93.8% | 94.5% | 0.998 | ✅ |
| GCBF+ | Official GCBF+ @1000 | **91.3%** | **95.3%** | **95.3%** | **0.652** | ✅ |
| GCBF | Official GCBF mid@270 | **56.7%** | **62.5%** | **83.8%** | **7.504** | ✅ |
| HandCBF | DecShareCBF | **83.4%** | **99.8%** | **83.5%** | **0.035** | ✅ |
| MAPPO-Lag | MAPPO + Lagrangian | **67.6%** | **88.4%** | **74.7%** | **1.875** | ✅ |
| InforMARL | 局部图 GNN + PPO | **62.8%** | **64.3%** | **96.3%** | **5.783** | ✅ |

≈2048 agent-trials；success SE 约 0.6–1.1pp。

---

## 4. 控制计算耗时（辅轴）

同硬件 GPU；JIT warm-up 后；**ms / control call**（一次算出全部 8 agents；**不含** `env.step`）。

| ID | 方法 | 计算内容 | median↓ | mean↓ | p95↓ | 状态 |
|---|---|---|---:|---:|---:|---|
| MF-r2 | Mean-field residual r2 | 特征 + 残差前向 | **1.37 ms** | 1.54 ms | 1.63 ms | ✅ |
| GCBF+ | Official GCBF+ | GNN actor 前向（部署无 QP） | **1.39 ms** | 1.41 ms | 1.57 ms | ✅ |
| HandCBF | DecShareCBF | **在线 ProxQP** | **4.72 ms** | 5.20 ms | 8.55 ms | ✅ |

来源：`baselines/compare/output/latency/control_latency.json`（200 repeats · 32 graphs · seed=1234）。

读数：HandCBF ≈ **3.4×** MF/GCBF+ 中位耗时；MF 与 GCBF+ 同量级。DS / GCBF mid **未测耗时**（按约定）。

---

## 5. 可验证性 / 证书

| ID | 方法 | L0 smoke | L1 near | **L2 near_plus（主验收）** | L3 full（附录） | 经验 only |
|---|---|---|---|---|---|---|
| MF-r2 | Mean-field r2 | PROVEN_FULL | PROVEN_FULL | **PROVEN_FULL** | VIOLATED（11260 CE，均在 L2 外） | — |
| DS-r2 | DeepSet V2d_r2 | — | — | — | — | ✅ |
| GCBF+ | GCBF+ | — | — | — | — | ✅ |
| GCBF | GCBF mid@270 | — | — | — | — | ✅ |
| HandCBF | DecShareCBF | — | — | — | — | ✅（在线 QP，无本仓库形式化） |
| MAPPO-Lag | MAPPO-Lag | — | — | — | — | ✅（CMDP 软约束，无形式化） |
| InforMARL | InforMARL | — | — | — | — | ✅（GNN+PPO 软 cost，无形式化） |

MF L2：`min_d∈[0.14,0.40]`，`|close|≤0.30`，`inv_ttc≤8`，Q0 occupied；IBP `d2_lo≈0.113≥R`。

---

## 6. 一览（32 / 64 / 256 + 耗时）

| 方法 | @32 suc/safe/fin/cost | @64 | @256 | ctrl median | 形式化 |
|---|---|---|---|---:|---|
| **MF-r2** | **82.0 / 96.5 / 84.0 / 0.31** | 82.6 / 94.9 / 85.0 / 0.40 | 83.5 / 95.7 / 86.4 / 0.44 | **1.37 ms** | **L2 PROVEN_FULL** |
| DS-r2 | 87.9 / 93.4 / 94.1 / 1.04 | 88.7 / 93.8 / 94.7 / 0.98 | 88.8 / 93.8 / 94.5 / 1.00 | — | 经验 |
| GCBF+ | 91.4 / 95.7 / 95.3 / 0.44 | 92.2 / 95.9 / 96.1 / 0.45 | 91.3 / 95.3 / 95.3 / 0.65 | **1.39 ms** | 经验 |
| GCBF | 55.5 / 62.1 / 81.2 / 9.25 | 56.6 / 61.7 / 82.8 / 8.53 | 56.7 / 62.5 / 83.8 / 7.50 | — | 经验 |
| HandCBF | 84.8 / **100** / 84.8 / **0.00** | 83.4 / 99.6 / 83.8 / 0.06 | 83.4 / 99.8 / 83.5 / 0.04 | **4.72 ms** | 经验 |
| MAPPO-Lag | 68.4 / 90.6 / 72.7 / 1.38 | 68.8 / 90.2 / 73.6 / 1.19 | 67.6 / 88.4 / 74.7 / 1.88 | — | 经验 |
| InforMARL | 60.9 / 62.5 / 95.7 / 6.93 | 61.3 / 62.7 / 96.3 / 6.11 | 62.8 / 64.3 / 96.3 / 5.78 | — | 经验 |

---

## 7. 待测 / 预留

| 优先级 | 项目 | 填入 | 备注 |
|---|---|---|---|
| P2 | DeepSet 形式化（若做） | 表5 DS-r2 | 非本验收主轨；主验收锁定 MF-r2 |
| P2 | near_plus 盒内采样对照 | `baselines/compare/output/near_plus_sample/` | 经验 only；非形式化 |
| P2 | 障碍密度迁移（obs↑） | `TRANSFER_COMPARE` 扩节 | 人数轴已齐；可选 |
| P3 | DS / GCBF / MAPPO-Lag / InforMARL 耗时 | 表4 | 非必须；主耗时轴已有 MF/GCBF+/HandCBF |
| P3 | 512-epi | 可选附录 | 非必须；256 已够稳 |
| P3 | MAPPO-Lag / InforMARL 加长训 | 对应行 | 当前 400-iter 已够对照；非必须 |
| P3 | 其它外源 | Other | 协议不一致勿写入主表 |

---

## 8. 来源指针

| ID | 权重 / run | @32 | @64 | @256 | 耗时 |
|---|---|---|---|---|---|
| MF-r2 | `acceptance/mf_best_20261006/1_primary/01_models/residual_r2/` | `03_task_eval/residual_r2_validate_32epi.json` | `..._64epi.json` | `..._256epi.json` | `baselines/compare/output/latency/control_latency.json` |
| DS-r2 | `method/deepset/output/phase2/.../V2d_r2/repair_r2_best.pkl` | `.../V2d_r2/validate_32epi.json` | `baselines/compare/output/epi64/ds_r2_epi64.json` | `.../epi256/ds_r2_epi256.json` | — |
| GCBF+ | `logs/.../gcbf+/seed0_20261001232611/models/1000/` | `method/deepset/output/stage1/compare/.../gcbfplus_teacher_epi32_*.json` | `...epi64_*.json` | `.../epi256/gcbfplus_epi256.json` | 同上 latency |
| GCBF | `logs/.../gcbf/seed0_20261006111544/models/270/` · `baselines/gcbf/output/gcbf_mid/` | `validate_32epi.json` | `validate_64epi.json` | `validate_256epi.json` | — |
| HandCBF | `algo=dec_share_cbf` · `baselines/handcbf/output/handcbf/` | `validate_32epi.json` | `validate_64epi.json` | `validate_256epi.json` | 同上 latency |
| MAPPO-Lag | `baselines/mappo_lag/output/mappo_lag/seed0_train/best.pkl` @350 | `validate_32epi.json` | `validate_64epi.json` | `validate_256epi.json` | — |
| InforMARL | `baselines/informarl/output/informarl/seed0_train/best.pkl` @350 | `validate_32epi.json` | `validate_64epi.json` | `validate_256epi.json` | — |

证书：`acceptance/mf_best_20261006/1_primary/00_acceptance/ACCEPTANCE_CERTIFICATE.md` · L2：`05_formal_verify/L2_near_plus_PRIMARY/`。

脚本：`scripts/validate_baseline_handcbf.sh` · `scripts/validate_baseline_mappo_lag.sh` · `scripts/validate_baseline_informarl.sh` · `scripts/bench_control_latency.py` · `scripts/eval_transfer_nagents.sh`。

迁移辅表：`TRANSFER_COMPARE.md`（零样本 N=16/32；产物 `baselines/compare/output/transfer_n/`）。

---

## 9. 读表注意

1. **主验收 / 主对比口径 = 32-epi**；64 / 256 只减方差。  
2. 主验收看 **MF-r2**：任务 + **L2 PROVEN_FULL**。  
3. **HandCBF**：safe/cost 极强，但 finish≈success、偏保守；耗时约为前馈网络的 **~3–4×**。不是「只比速度」。  
4. **GCBF** = 同族消融，不是任务上界。  
5. **MAPPO-Lag**：CMDP 软约束（PPO-clip + λ）；safe/cost 优于 GCBF mid，但 success 明显低于 MF/DS/GCBF+/HandCBF。经验 only。  
6. **InforMARL**：局部图 GNN+PPO + 软 cost；finish 很高（~96%），但 safe≈success、cost 高（~6–7）——偏「敢冲、常撞」。经验 only。  
7. **加人数迁移**见 `TRANSFER_COMPARE.md`（辅轴；不改主验收）。GCBF+ 扩展最稳；MF safe 仍高但 finish↓；InforMARL finish 高、safe 崩。  
8. 耗时是 **控制计算**，不是整局墙钟；HandCBF=QP，MF/GCBF+=网络前向。  
9. 新基线建议交 **32 + 64 + cost**；256 / 耗时可选。

---

*Last filled: 2026-10-06 — InforMARL 主表 + 加人数迁移 N=16/32 已齐。*
