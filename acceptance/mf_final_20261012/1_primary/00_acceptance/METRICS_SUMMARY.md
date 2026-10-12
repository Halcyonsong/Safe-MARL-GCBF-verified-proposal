# 指标速查（最终验收包）

## 0. 推荐配对（一页）

| 轴 | **mf_best r2（推荐）** | netdep r3（侧枝） |
|---|---|---|
| 形式化 M1–M3 | **PROVEN_FULL** 同 X | PROVEN_FULL |
| @32 suc/safe/fin/cost | **0.820 / 0.965 / 0.840 / 0.31** | 0.820 / 0.957 / 0.852 / 0.69 |
| 形式化目录 | `05_formal_verify/mf_best_r2_same_X/` | `05_formal_verify/M{1,2,3}_*` |
| 任务目录 | `03_task_eval/mf_best_r2_reference/` | `chain_32epi/` · `pressure/` · `transfer_n/` |

---

## 1. 形式化主验收

| 项 | 值 |
|---|---|
| 性质 | 平面两步 \(\|p_2\|_2 \ge 0.10\)（签名 a） |
| 路径 | SafeSet(\(a_x\ge a_{\min}\approx 0\)) → Marabou Deploy∈SafeSet |
| 物理 | netdep=**True** · envelope_any_a=**False** · p2_brake≈0.091 · p2_gas≈0.109 |
| 切片 | pose **px=py=2** · threat **中点** · goal_dv=−v |
| 阶梯 | M1→M2→M3 全部 **PROVEN_FULL** · 真 CE=**0**（**两权重**） |

### 1.1 mf_best r2（推荐引用）

| 证书 | vg | 状态 | 方式 | wall_s | 采样 n/CE | ax_min 采样 | ax≥0.99 |
|---|---:|---|---|---:|---|---:|---|
| **M1** | ±0.01 | PROVEN_FULL | root **27.7 s** | **65.0** | 512 / **0** | 0.874 | sat_candidate |
| **M2** | ±0.05 | PROVEN_FULL | root TO → 二分 **18.8+43.9 s** | **713** | 512 / **0** | 0.850 | sat_candidate |
| **M3** | ±0.10 | PROVEN_FULL | root TO → 二分 **150+563 s** | **1704** | 512 / **0** | 0.819 | sat_candidate |

### 1.2 netdep r3（侧枝对照）

| 证书 | vg | 状态 | Marabou 有效 | wall_s | 采样 n/CE | ax 约 | ax≥0.99 |
|---|---:|---|---:|---:|---|---|---|
| M1 | ±0.01 | PROVEN_FULL | **25.0 s** | 52.6 | 256 / 0 | [0.716,0.723] | sat_candidate |
| M2 | ±0.05 | PROVEN_FULL | **463.6 s** | 515.5 | 512 / 0 | [0.688,0.734] | sat_candidate |
| M3 | ±0.10 | PROVEN_FULL | 根 TO；叶 **372+145 s** | 1524 | 512 / 0 | [0.650,0.747] | sat_candidate |

### 可扩展性（时间随 vg；同链路）

| vg 半宽 | 策略 | 量级（r2 / r3） |
|---:|---|---|
| 0.01 | 单盒 Marabou | ~**28 s** / ~**25 s** |
| 0.05 | 单盒或 vg 二分 | r2 二分叶 ~1 min；r3 单盒 ~8 min |
| 0.10 | 根探测 + **vg 二分** | 两叶合计约 **12–15 min** 量级 |

---

## 2. 任务测评（经验）

主文：`0_docs/EVAL_AND_REPAIR.md` · 台账：`03_task_eval/TASK_EVAL_LEDGER.md`

### 2.1 推荐权重 mf_best r2

| | suc | safe | fin | cost | JSON |
|---|---:|---:|---:|---:|---|
| @32 | **82.0%** | **96.5%** | **84.0%** | **0.31** | `mf_best_r2_reference/residual_r2_validate_32epi.json` |
| @64 | 82.6% | 94.9% | 85.0% | 0.40 | `…_64epi.json` |
| @256 | 83.5% | 95.7% | 86.4% | 0.44 | `…_256epi.json` |
| N=16 epi32 | 72.9% | 94.3% | 75.6% | 0.73 | `…/transfer_n/mf_r2_N16_…` |
| N=32 epi32 | 48.4% | 90.7% | 50.8% | 1.23 | `…/transfer_n/mf_r2_N32_…` |

### 2.2 侧枝 netdep 链 @32

| policy | success | safe | finish | cost |
|---|---:|---:|---:|---:|
| base v5 | 0.777 | 0.809 | 0.941 | 4.86 |
| netdep r1 | 0.809 | 0.918 | 0.867 | 0.82 |
| netdep r2 pick | 0.816 | 0.949 | 0.855 | 0.47 |
| netdep r3 | 0.820 | 0.957 | 0.852 | 0.69 |
| GCBF+（旧包表） | 0.914 | 0.957 | 0.953 | 0.44 |

### 2.3 netdep r3 压力 / 延展 / 延迟（本包新跑）

| | 值 |
|---|---|
| @64 | 0.816 / 0.932 / 0.871 / 0.775 |
| @256 | 0.829 / 0.947 / 0.868 / 0.557 |
| N=16 | 0.684 / 0.932 / 0.715 / 0.748 |
| N=32 | 0.461 / 0.827 / 0.517 / 2.749 |
| latency median (r3) | **1.35 ms**（GCBF+ 1.35 · HandCBF 5.95） |

### 读法

- **r2 经验全面优于 r3**（safe/cost/大 N）；形式化 **同等** PROVEN。  
- safe@32：r2 **96.5%** ≳ GCBF+ 95.7%；r3 **= GCBF+**。  
- 主 claim 仍是 planar 形式化，不是 suc 刷分。

---

## 3. 修复元数据（netdep 侧枝 CE）

| 轮 | CE hit_on_danger |
|---|---:|
| r1 | 6.32% → 0.68% |
| r2 | 0.68% → 0.40% |
| r3 | 0.40% → **0.13%** |

详见 `04_repair_meta/chain/SELECTION_*.md`。旧 mf_best r1→r2 CE 轨迹见旧包 `0_docs/EVAL_AND_REPAIR.md`。

---

## 4. 形式化双权重（同 X · 2026-10-11）

| 权重 | M1 | M2 | M3 | 任务 @32 safe/cost |
|---|---|---|---|---|
| netdep r3（侧枝） | PROVEN ~25s | PROVEN ~464s | PROVEN vg-bisect | 95.7% / 0.69 |
| **mf_best r2（推荐）** | **PROVEN ~28s** | **PROVEN vg-bisect** | **PROVEN vg-bisect** | **96.5% / 0.31** |

推荐论文配对：**Deploy = mf_best r2** + planar M1–M3（`05_formal_verify/mf_best_r2_same_X/`）。
