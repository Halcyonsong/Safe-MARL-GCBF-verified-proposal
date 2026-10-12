# 全面测评 + 反例修复记录（planar netdep · 推荐 Deploy = mf_best r2）
> **最终包路径：** `acceptance/mf_final_20261012/`。主 claim = 平面 netdep；完整基线见 `2_baselines/`。


> 对齐旧包 `acceptance/mf_final_20261012/5_frozen_sources/mf_best_20261006/EVAL_AND_REPAIR.md` 的测评轴：  
> **CE 修复轨迹 · @32/@64/@256 · 人数延展 · 控制延迟 · 基线对照**。  
> 形式化主声明见 `1_primary/00_acceptance/`；叙事边界见 `NOTES.md`。  
> **本文件为新增包专属**；不修改旧 `mf_best` 树。

---

## 0. 一页结论

| 轴 | **推荐：mf_best r2** | 侧枝：netdep r3 |
|---|---|---|
| **形式化 M1–M3（同 X）** | **PROVEN_FULL** | PROVEN_FULL |
| **任务 @32** | **82.0 / 96.5 / 84.0 / 0.31** | 82.0 / 95.7 / 85.2 / 0.69 |
| **压力 @64 / @256** | **82.6/94.9/85.0 · 83.5/95.7/86.4** | 81.6/93.2/87.1 · 82.9/94.7/86.8 |
| **延展 N=16 / 32 safe** | **94.3% / 90.7%** | 93.2% / 82.7% |
| **延迟** | ~1.37 ms（旧测） | **1.35 ms**（本包新测） |
| **相对 GCBF+ safe@32** | **96.5% ≳ 95.7%** | **= 95.7%** |
| **证据** | `mf_best_r2_same_X/` + `mf_best_r2_reference/` | `M{1,2,3}_*` + `pressure/` |

**卖点：** netdep 平面 ∀ 证书（非包络）+ **更强经验 r2 可挂主方法**。  
侧枝 r3：CE 修复链故事（§1）+ 同盒亦可证。详见 `0_docs/DUAL_WEIGHT_FORMAL.md`。

---

## 1. 反例修复轨迹（base → netdep r1 / r2 / r3）

### 1.1 方法

- **冻结 base**：v5 MLP512 distill（`1_primary/01_models/base_v5/`）  
- **残差 δ**：physics-gated；netdep 侧枝 CE 池（迎面 fringe / 两步 L2 危险）  
- **r1**：base CE 上训残差  
- **r2**：合并 CE + warm-start；**manual pick = last**（auto best safe 更差，勿信 auto）  
- **r3**：再修；**manual pick ≈ repair_best** → **侧枝形式化对照**（主推荐为 mf_best r2）

元数据：`1_primary/04_repair_meta/chain/SELECTION_{r1,r2,r3}.md`

### 1.2 经验两步 L2 · hit_on_danger

| 阶段 | 策略 | hit_on_danger | n_hit / n_danger | 来源 |
|---|---|---:|---:|---|
| train pool (r1) | base 危险池 | **6.32%** | 1773 / 28063 | SELECTION_r1 |
| r1 | residual r1 | **0.68%** | 64 / 9401 | SELECTION_r1 |
| r2 | residual r2 | **0.40%** | 37 / 9346 | SELECTION_r2 |
| **r3** | **residual r3** | **0.13%** | **12 / 8968** | SELECTION_r3 |

累计：池内 6.32% → r3 **0.13%**（约 **48×** 下降，相对池；相对 r1 后再压约 5×）。

### 1.3 任务 @32：修复前后对照

协议：DI · **N=8** · area4 · obs4 · rays32 · **seed=1234** · max_step=256

| 策略 | success↑ | safe↑ | finish↑ | cost↓ | 路径 |
|---|---:|---:|---:|---:|---|
| base v5 | 77.7% | 80.9% | 94.1% | 4.86 | `03_task_eval/chain_32epi/base_v5_validate_32epi.json` |
| netdep r1 best | 80.9% | 91.8% | 86.7% | 0.82 | `…/netdep_r1_best_validate_32epi.json` |
| netdep r2 **manual last** | 81.6% | **94.9%** | 85.5% | 0.47 | `…/netdep_r2_manual_pick_…` |
| netdep r2 auto best | 80.9% | 92.6% | 85.9% | 0.90 | （对照，非主） |
| **netdep r3 pick ★** | **82.0%** | **95.7%** | **85.2%** | **0.69** | `…/netdep_r3_pick_…` |

**修复作用（经验层）**

1. **碰撞相关**：danger hit 大幅下降；safe **+14.8pp**（80.9→95.7）；cost 4.86→0.69。  
2. **权衡**：finish 94→85（更保守）；success 相对 base **+4.3pp**。  
3. **形式化入口**：CE 压到 0.13% 后跑 planar SafeSet+Marabou → **M1–M3 PROVEN_FULL**。  
4. **不是**：suc 全面超过 GCBF+；也不是无限时域 / 大 N 形式化。

---

## 2. 基线对照 · 主表（@32）

论文基线行仍为 GCBF+ · MAPPO-Lag · InforMARL（数字来自旧包**同协议快照**；本轮**未**重跑全部外源基线）。  
**本包新测**仅 **netdep r3** 一行。

| ID | 方法 | 角色 | suc | safe | fin | cost | 形式化 |
|---|---|---|---:|---:|---:|---:|---|
| **netdep-r3** | MF residual netdep r3 | **本包主方法** | **82.0** | **95.7** | **85.2** | **0.69** | **planar M1–M3 PROVEN** |
| mf_best-r2 | MF residual r2（旧包） | 历史包络证书 | 82.0 | 96.5 | 84.0 | 0.31 | near_plus 包络 |
| DS-r2 | DeepSet V2d_r2 | 任务并行 | 87.9 | 93.4 | 94.1 | 1.04 | — |
| GCBF+ | Official @1000 | 教师上界 | 91.4 | 95.7 | 95.3 | 0.44 | — |
| GCBF mid | mid@270 | 同族消融 | 55.5 | 62.1 | 81.2 | 9.25 | — |
| HandCBF | DecShareCBF | 解析 QP | 84.8 | 100 | 84.8 | 0 | — |
| MAPPO-Lag | MAPPO+Lagrangian | 外源 | 68.4 | 90.6 | 72.7 | 1.38 | — |
| InforMARL | 局部图 PPO | 外源 | 60.9 | 62.5 | 95.7 | 6.93 | — |

**读法：** netdep-r3 **safe 与 GCBF+ 持平（95.7%）**；suc 不追教师/DS；**唯一**具备本协议 **平面两步 netdep ∀ 证书** 的 MF 行。旧 mf_best-r2 的证是径向包络，贡献叙事不同。

旧包全表：`acceptance/mf_final_20261012/5_frozen_sources/mf_best_20261006/EVAL_AND_REPAIR.md` §2。

---

## 3. 压力测评 · @64 / @256（本包新跑）

协议同 §1.3；权重 = netdep r3 `repair_manual_pick`。

| ID | @32 suc/safe/fin/cost | @64 | @256 |
|---|---|---|---|
| **netdep-r3** | **82.0 / 95.7 / 85.2 / 0.69** | **81.6 / 93.2 / 87.1 / 0.78** | **82.9 / 94.7 / 86.8 / 0.56** |
| mf_best-r2（参考） | 82.0 / 96.5 / 84.0 / 0.31 | 82.6 / 94.9 / 85.0 / 0.40 | 83.5 / 95.7 / 86.4 / 0.44 |
| GCBF+（旧包） | 91.4 / 95.7 / 95.3 / 0.44 | 92.2 / 95.9 / 96.1 / 0.45 | 91.3 / 95.3 / 95.3 / 0.65 |

JSON：

- `1_primary/03_task_eval/pressure/netdep_r3_validate_64epi.json`  
- `1_primary/03_task_eval/pressure/netdep_r3_validate_256epi.json`  
- @32：`…/chain_32epi/netdep_r3_pick_validate_32epi.json`

**读法：** 加 epi 后 suc 稳定在 ~82–83%；safe 在 93–96% 带；**非 32-epi 过拟合。** @64 safe 略低于 @32（93.2 vs 95.7）属采样波动，@256 回升至 94.7%。

---

## 4. 人数延展（零样本 N=16 / 32 · 本包新跑）

训练仍 N=8；评测加人数、不微调；**32-epi · seed=1234**。

格内：`suc% / safe% / fin% / cost`

| 方法 | N=8 | N=16 | N=32 | Δsuc 8→32 |
|---|---|---|---|---:|
| **netdep-r3** | **82.0 / 95.7 / 85.2 / 0.69** | **68.4 / 93.2 / 71.5 / 0.75** | **46.1 / 82.7 / 51.7 / 2.75** | **−35.9pp** |
| mf_best-r2（旧） | 82.0 / 96.5 / 84.0 / 0.31 | 72.9 / 94.3 / 75.6 / 0.73 | 48.4 / 90.7 / 50.8 / 1.23 | −33.6pp |
| GCBF+（旧） | 91.4 / 95.7 / 95.3 / 0.44 | 88.7 / 96.1 / 91.8 / 0.45 | 81.9 / 93.8 / 86.4 / 0.87 | −9.5pp |

JSON：`1_primary/03_task_eval/transfer_n/netdep_r3_N{16,32}_epi32.json`

**读法：** 与旧 MF 同类——**大 N 掉点主要来自 finish/suc**；N=16 时 safe 仍 **93%**；N=32 safe **82.7%**（低于旧 mf_best-r2 的 90.7%，更拥挤下更脆）。**形式化证书不随 N 外推。**

---

## 5. 控制延迟（本包新跑）

同硬件协议：`baselines/compare/scripts/bench_control_latency.py`  
warmup=40 · repeats=200 · 32 graphs · seed=1234 · **ms / 一次 8-agent 控制**（不含 `env.step`）。

| 方法 | median↓ | mean↓ | p95↓ |
|---|---:|---:|---:|
| **netdep-r3** | **1.35** | 1.36 | 1.45 |
| GCBF+ | 1.35 | 1.38 | 1.57 |
| HandCBF（在线 QP） | 5.95 | 6.54 | 10.53 |

产物：`2_baselines/latency_mf_best_era/control_latency.json`（含 `methods.netdep_r3` 与对照 gcbfplus/handcbf）。

**读法：** 与 GCBF+ 同量级；远快于在线 QP。

---

## 6. 形式化（本包主贡献 · 摘要）

### 6.1 原 netdep r3 权重（侧枝）

| 证书 | vg | 状态 | 有效时间 |
|---|---:|---|---|
| M1 | ±0.01 | PROVEN_FULL | Marabou ~25 s |
| M2 | ±0.05 | PROVEN_FULL | ~464 s |
| **M3** | ±0.10 | PROVEN_FULL | vg 二分 ~372+145 s；wall ~25 min |

### 6.2 mf_best r2 权重 · **同一 X**（2026-10-11 新跑 · 推荐论文配对）

| 证书 | vg | 状态 | 方式 | wall |
|---|---:|---|---|---|
| M1 | ±0.01 | **PROVEN_FULL** | root ~28 s | ~65 s |
| M2 | ±0.05 | **PROVEN_FULL** | root TO → vg 二分 ~19+44 s | ~713 s |
| **M3** | ±0.10 | **PROVEN_FULL** | root TO → vg 二分 ~150+563 s | ~1704 s |

**结论：** 更强经验的 **mf_best r2 也能过当前 planar 验收链路**。  
论文叙事可 **任务表用 r2（safe 96.5% / cost 0.31）+ 形式化用同一 r2 的 M1–M3**，不必绑弱一点的 netdep_r3。

详情：`05_formal_verify/mf_best_r2_same_X/` · 原 r3：`05_formal_verify/M{1,2,3}_*` · 证书文：`00_acceptance/`。

---

## 7. 修复 → 测评 → 形式化：怎么串

| 层次 | 证据 | 主张 |
|---|---|---|
| 经验修复 | §1 hit↓ · safe↑ | netdep CE 修复有效 |
| 任务可用 | §2–§3 @32/64/256 | 部署指标稳定 |
| 延展辅轴 | §4 | safe 在 N=16 仍高；大 N finish 瓶颈 |
| 延迟辅轴 | §5 | 推理与 GCBF+ 同量级 |
| **形式化主** | M1–M3 planar | 有界盒 netdep ∀；替代包络叙事 |
| 勿写 | — | 「suc 全面超 GCBF+」；「大 N 已形式化」；「已改 ACTIVE」 |

---

## 8. 产物索引

| 内容 | 路径 |
|---|---|
| 本文件 | `0_docs/EVAL_AND_REPAIR.md` |
| @32 链 | `1_primary/03_task_eval/chain_32epi/` |
| @64/@256 | `1_primary/03_task_eval/pressure/` |
| N=16/32 | `1_primary/03_task_eval/transfer_n/` |
| 延迟 | `2_baselines/latency_mf_best_era/control_latency.json` |
| 台账 | `1_primary/03_task_eval/TASK_EVAL_LEDGER.md` |
| 形式化 | `1_primary/05_formal_verify/` |
