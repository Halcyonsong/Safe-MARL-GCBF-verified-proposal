# 完整数据记录（主验收 + 附录）

> **一份查全表。** 路径均相对本包根目录 `acceptance/mf_best_20261006/`。  
> **修复轨迹 + @32/64/256 + 延展 + 基线全表** → [`EVAL_AND_REPAIR.md`](EVAL_AND_REPAIR.md)  
> 叙事边界见 [`NOTES.md`](NOTES.md)；入口见 [`README.md`](README.md)。  
> 日期：2026-10-06

---

## A. 主验收（锁定）

### A.1 对象与声明

| 项 | 值 | 包内路径 |
|---|---|---|
| 策略 | residual **r2**（physics-gated δ + 冻结 v5 base） | `1_primary/01_models/residual_r2/repair_best.pkl` |
| 冻结 base | MLP512 · step9000 | `1_primary/01_models/base_v5/meanfield_mlp512_student_step9000.pkl` |
| 性质 | TwoStepRadialL2Safe，R=**0.10** | `1_primary/07_source_snapshot/property_l2.py` |
| 声明域 \(X_{\mathrm{near\_plus}}\) | min_d∈**[0.14, 0.40]**，\|close\|≤**0.30**，inv_ttc≤8，Q0 occupied | `1_primary/05_formal_verify/L2_near_plus_PRIMARY/mf_near_plus_manifest.json` |
| 物理包络 | \(d2\_lo=0.14-0.06\cdot0.30-0.009=\)**0.113 ≥ R** | — |
| 形式化结果 | **PROVEN_FULL** · 0 CE · splits=0 · Marabou=0 | `1_primary/05_formal_verify/L2_near_plus_PRIMARY/REPORT.md` |
| 证书 | PASS | `1_primary/00_acceptance/ACCEPTANCE_CERTIFICATE.md` |
| **修复 / 全面测评** | CE 5.95%→0.47%；@32/64/256；N 延展；基线表 | **`EVAL_AND_REPAIR.md`** |
| **论文出图** | 筛出的图数据 + 训练曲线 + PDF/PNG | **`4_paper_figures/`**（含 Fig.8 distill / Fig.9 repair） |

### A.2 分层形式化

| 层 | 盒 | 状态 | 路径 |
|---|---|---|---|
| L0 smoke | 点邻域 | PROVEN_FULL | `1_primary/05_formal_verify/L0_smoke/` |
| L1 near | d∈[0.15,0.35]，\|c\|≤0.25 | PROVEN_FULL | `1_primary/05_formal_verify/L1_near/` |
| **L2 near_plus** | **d∈[0.14,0.40]，\|c\|≤0.30** | **PROVEN_FULL ← 主** | `1_primary/05_formal_verify/L2_near_plus_PRIMARY/` |
| L3 full | d∈[0.12,0.50]，\|c\|≤0.50 | VIOLATED（附录，CE 全在 L2 外） | `1_primary/05_formal_verify/L3_full_appendix/` |

### A.3 反例修复轨迹（base → r1 → r2）

详情与基线全表：[`EVAL_AND_REPAIR.md`](EVAL_AND_REPAIR.md)

| 阶段 | hit_on_danger | @32 suc | @32 safe | @32 fin | @32 cost |
|---|---:|---:|---:|---:|---:|
| base v5 | **5.95%** | 77.7% | 80.9% | 94.1% | 4.86 |
| residual r1 | **1.31%** | 83.2% | 93.0% | 87.9% | 0.72 |
| **residual r2 ★** | **0.47%** | **82.0%** | **96.5%** | **84.0%** | **0.30** |

→ danger hit ≈**12.7×**↓；safe **+15.6pp**；cost ≈**16×**↓。finish 略降（更保守）。  
日志 / CE JSON：`1_primary/02_train_logs/repair_r{1,2}/` · `1_primary/04_ce_repair/r{1,2}/`

### A.4 任务评测（主策略 r2 · 压力）

协议：DoubleIntegrator · N=8 · area4 · obs4 · rays32 · seed=1234

| epi | success | safe | finish | cost | 路径 |
|---:|---:|---:|---:|---:|---|
| 32 | **82.03%** | **96.48%** | **83.98%** | 0.305 | `1_primary/03_task_eval/residual_r2_validate_32epi.json` |
| 64 | 82.62% | 94.92% | 84.96% | 0.400 | `…/residual_r2_validate_64epi.json` |
| 256 | 83.54% | 95.65% | 86.38% | 0.442 | `…/residual_r2_validate_256epi.json` |

近危步落在 L2 带 ≈79.6%；带内两步 hit=0。速查：`1_primary/00_acceptance/METRICS_SUMMARY.md`

### A.5 延展 / 基线 / 延迟（包内已收录）

| 轴 | 包内 |
|---|---|
| 基线 @32/@64/@256 | `2_baselines/BASELINE_COMPARE.md` · `epi_pressure/` |
| 人数延展 N=16/32 | `2_baselines/TRANSFER_COMPARE.md` · `transfer_n/` |
| 控制延迟 | `2_baselines/latency/control_latency.json` |
| **汇总解读** | **`EVAL_AND_REPAIR.md`** |

### A.6 指针快照（打包时）

| 指针 | 指向 | 包内副本 |
|---|---|---|
| ACTIVE_MF_RESIDUAL | `method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl` | `pointers/ACTIVE_MF_RESIDUAL.txt` |
| ACTIVE_MF_VERIFY | `method/meanfield/output/verify/mf_feat44_near_plus_r2` | `pointers/ACTIVE_MF_VERIFY.txt` |
| ACTIVE_MF_BASE | v5 step9000 | `pointers/ACTIVE_MF_BASE.txt` |

本包**未改写**仓库 ACTIVE。

---

## B. 基线对照（主文支撑）

> 全表 + 修复前后解读：**[`EVAL_AND_REPAIR.md`](EVAL_AND_REPAIR.md)** §2–§5

| 文件 | 内容 |
|---|---|
| `2_baselines/BASELINE_COMPARE.md` | @32 / @64 / @256 主对照表 |
| `2_baselines/TRANSFER_COMPARE.md` | N=16/32 零样本延展 |
| `2_baselines/epi_pressure/` | 压力测评 JSON |
| `2_baselines/transfer_n/` | 延展 JSON |
| `2_baselines/latency/` | 控制延迟 |
| `2_baselines/BASELINE_FAIL_EVIDENCE.md` | mid-F 经验 fail 汇总（附录） |
| `2_baselines/near_plus_box_mc/` | 主验收盒产品盒 MC |

论文基线行：**GCBF+ · MAPPO-Lag · InforMARL**（coast / mf_base = 消融）。  
主主张：**证书非对称**（MF ∀-proven；基线无同级证书）——**不是**「主盒上基线 binary L2 必 fail」。

---

## C. 附录数据（非主验收）

### C.1 mid-F 旁支修复轨迹

目标盒：d≥**0.125**，\|c\|≤**0.30**（\(d2\_lo≈0.098 < R\)，网络依赖）。  
证据：`3_appendix/evidence/midF_repair/` · 权重：`3_appendix/sidebranch_midF_r3/midF_r3_repair_best.pkl`

| 轮次 | mid-F CE | mid-F 形式化 | near_plus | @32 suc | @32 safe | @32 fin |
|---|---:|---|---|---:|---:|---:|
| r2 probe | 204 | VIOLATED | PROVEN | 82.03% | 96.48% | 83.98% |
| midF_r1 | 194 | VIOLATED | PROVEN | 83.20% | 98.05% | 84.38% |
| midF_r2 | 183 | VIOLATED | PROVEN | 83.98% | 97.66% | 85.16% |
| **midF_r3** | **177** | VIOLATED | PROVEN | 83.20% | 97.66% | 84.38% |
| midF_r4 | 173 | VIOLATED | PROVEN | 82.81% | 97.27% | 84.38% |

旁支折中：**midF_r3**（任务比 r4 稳；CE 略高于 r4）。

### C.2 压力评测 @64 / @256

协议同上。完整表：`3_appendix/evidence/pressure/SUMMARY.md`

| tag | @64 suc/safe/fin | @256 suc/safe/fin |
|---|---|---|
| r2（主） | 82.6 / 94.9 / 85.0 | 83.5 / 95.7 / 86.4 |
| midF_r2 | 83.8 / 96.5 / 85.7 | 84.1 / 96.5 / 86.5 |
| midF_r3 | 83.2 / 96.5 / 85.2 | 83.4 / 96.6 / 85.7 |
| midF_r4 | 82.6 / 96.1 / 84.8 | 83.3 / 96.6 / 85.6 |

相对 midF_r2：r3/r4 掉幅 \<2pp（未触 soft 退化线）；near_plus 均仍 PROVEN。

### C.3 更宽盒形式化探测（候选界）

表：`3_appendix/evidence/formal/candidate_boxes.json` · 报告：`3_appendix/evidence/formal/*/`

| ckpt | d_min | c_hi | d2_lo | 网络依赖 | 状态 | n_ce |
|---|---:|---:|---:|---|---|---:|
| midF_r3 / r2 | 0.126 | 0.30 | 0.099 | **是** | VIOLATED | 47 / 55 |
| midF_r3 | 0.1268–0.1269 | 0.30 | ≈0.0998 | **是** | VIOLATED | 1 |
| midF_r3 | 0.12695 | 0.30 | ≈0.0999 | **是** | PARTIAL | 0 |
| midF_r3 / r2 | **0.126** | **0.28** | 0.1002 | 否 | **PROVEN** | 0 |
| midF_r3 | **0.127** | **0.30** | 0.1000 | 否 | **PROVEN** | 0 |
| midF_r3 / r2 | ≥0.128 | 0.30 | ≥0.101 | 否 | **PROVEN** | 0 |
| r2（早先） | 0.130 | 0.32 | 0.102 | 否 | PROVEN（mid E） | 0 |

规律：\(d2\_lo ≥ R\) → 包络秒级 PROVEN；网络依赖带未证满。

### C.4 包络界上基线产品盒 MC（n=50k）

| 盒 | coast / mf / MAPPO binary fail |
|---|---|
| d≥0.126，\|c\|≤0.28 | **全 0**（含 hard fringe n=10k） |
| d≥0.127，\|c\|≤0.30 | **全 0** |

路径：`3_appendix/evidence/baselines_mc/`  
→ PROVEN 包络界**不能**支撑「基线 binary L2 反例」主 claim。

### C.5 mid-F CE-fringe 环境评分（n=200）

带：d0∈[0.125,0.132]，c0∈[0.25,0.30] · seed=1234  
路径：`3_appendix/evidence/ce_fringe/`

| policy | n_fail | fail% |
|---|---:|---:|
| coast / mf_base / mf_r2 | 0 | 0 |
| **mappo_lag** | **7** | **3.5%** |
| **gcbfplus** | **4** | **2.0%** |
| informarl | 0 | 0 |

→ 经验 fail 出现在**网络依赖语义带**；该带上 MF 形式化仍 VIOLATED。

---

## D. 主主张 vs 开放问题

| | 状态 |
|---|---|
| 主验收 r2 + near_plus PROVEN | **已达成 · 本包锁定** |
| 证书非对称（基线无同级 ∀ 证明） | **可写** |
| 同盒「MF PROVEN + 基线 binary L2 fail」 | **未达成**（能证过的界基线也过；基线 fail 的界 MF 未证过） |
| mid-F 证满 | **开放**；旁支 CE↓ 但未清零 |

---

## E. 包内路径速查

| 需求 | 路径 |
|---|---|
| 入口 | `README.md` |
| **修复 + 全面测评 + 基线** | **`EVAL_AND_REPAIR.md`** |
| **论文图** | **`4_paper_figures/README.md`** · `figs/` |
| 叙事 / 边界 | `NOTES.md` |
| **本完整数据记录** | **`RECORD.md`**（本文件） |
| 主模型 | `1_primary/01_models/residual_r2/repair_best.pkl` |
| r1/r2 修复产物 | `1_primary/02_train_logs/repair_r{1,2}/` · `04_ce_repair/` |
| 主形式化报告 | `1_primary/05_formal_verify/L2_near_plus_PRIMARY/REPORT.md` |
| 证书 | `1_primary/00_acceptance/ACCEPTANCE_CERTIFICATE.md` |
| 任务 JSON | `1_primary/03_task_eval/` |
| 基线 / 延展 / 延迟 | `2_baselines/` |
| 附录索引 | `3_appendix/README.md` |
| 旁支权重 | `3_appendix/sidebranch_midF_r3/midF_r3_repair_best.pkl` |
| 校验 | `SHA256SUMS.txt` · `MANIFEST.json` |

复现主验证：

```bash
bash method/meanfield/scripts/run_mf_verify_full.sh \
  acceptance/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl \
  near_plus
```
