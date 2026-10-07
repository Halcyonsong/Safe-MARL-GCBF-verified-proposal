# Mean-field 有界形式化验收证书

| 字段 | 值 |
|---|---|
| 轨道 | **Mean-field**（并行验证友好路径） |
| 策略 | residual **r2** + 冻结 **v5** base（MLP512，输入 dim=44） |
| 证书状态 | **通过（PASS）** |
| 主层 | **L2 `near_plus` → PROVEN_FULL** |
| 日期 | 2026-10-05 |
| 包 | `acceptance/mf_best_20261006/1_primary/` |

---

## 1. 验收声明（唯一主 claim）

```text
∀ x ∈ X_near_plus ⊂ R^{44}:
    TwoStepRadialL2Safe( Deploy_r2(x); R = 0.10 )
```

### 声明域 \(X_{\mathrm{near\_plus}}\)

| 维族 | 范围 |
|---|---|
| min_d（agent/obs 各象限） | **[0.14, 0.40]** |
| close_max / close_wmean | **[-0.30, 0.30]** |
| inv_ttc_max | **[0, 8]** |
| agent Q0 n | **[1, 4]**（强制占用） |
| 其它象限 n | [0, 4] |
| ego \(v\) | [-0.5, 0.5] |
| goal \(dx,dy\) | [-2, 2] |
| R | **0.10**（COLLISION_AA） |

物理包络（\|a\|_∞≤1）：

\[
d_2 \ge 0.14 - 0.06\cdot 0.30 - 0.009 = 0.113 > 0.10
\]

### 性质（两步径向 L2）

```text
d0, c0 = 占用象限 (n≥0.5) 的 min_d / max(close_max, 0)
d2     = d0 − α c0 − β |a|_∞
α=0.06, β=0.009
Safe  ⇔  d2 ≥ R
```

### 验证结果

| 项 | 值 |
|---|---|
| 状态 | **PROVEN_FULL** |
| 方法 | 整盒 IBP 一次（`d2_lo=0.113`） |
| Marabou | 0 次 |
| splits | 0 |
| 真实 CE | **0** |
| wall | ~2 ms |
| 报告 | `05_formal_verify/L2_near_plus_PRIMARY/REPORT.md` |
| 44 维清单 | `.../mf_near_plus_manifest.json` |

---

## 2. 部署对象

| 项 | 包内路径 |
|---|---|
| **验收网络** | `01_models/residual_r2/repair_best.pkl` |
| 冻结 base | `01_models/base_v5/meanfield_mlp512_student_step9000.pkl` |
| 结构 | `a = clip( π_base(x) + g_phys(x) · δ(x) )` |

指针快照（打包时）：

- `ACTIVE_MF_RESIDUAL` → r2 best  
- `ACTIVE_MF_VERIFY` → near_plus PROVEN_FULL  
- `ACTIVE_MF_BASE` → v5 step9000  

---

## 3. 支持性证据（非主 claim，但纳入交付）

### 3.1 任务测评（32-epi，`eval_seed=1234`）

| 策略 | success | safe | finish |
|---|---:|---:|---:|
| base v5 | 77.7% | 80.9% | 94.1% |
| residual r1 | 83.2% | 93.0% | 87.9% |
| **residual r2** | **82.0%** | **96.5%** | **84.0%** |

来源：`03_task_eval/residual_r2_validate_32epi.json`

### 3.2 经验两步 L2 CE（rollout danger）

| 阶段 | hit_on_danger |
|---|---:|
| base | 5.95% |
| r1 | 1.31% |
| **r2** | **0.47%**（33/7089） |

来源：`04_ce_repair/r2/ce_post_summary.json` · `SELECTION.md`

### 3.3 分层形式化

| 层 | 状态 | 角色 |
|---|---|---|
| L0 smoke | PROVEN_FULL | 工具链 |
| L1 near | PROVEN_FULL | 更紧可避免带 |
| **L2 near_plus** | **PROVEN_FULL** | **本证书主验收** |
| L3 full | VIOLATED_WITH_CES（11260 CE） | 附录；CE 的 d0 **全在 [0.12,0.14)**，**100% 在 L2 外** |

### 3.4 部署近危与 L2 带重合（经验）

24 epi，r2，danger 定义为物理 d≤0.35：

| 指标 | 值 |
|---|---:|
| danger 步落在 L2 (d0,c0) 带 | **≈79.6%** |
| 两步 hit 且在 L2 内 | **0** |
| 两步 hit 且在 L2 外 | 全部残余 |

来源：`05_formal_verify/L2_near_plus_PRIMARY/EMPIRICAL_COVERAGE.json`

---

## 4. 明确不在本证书范围

1. **无限时域 / 整局轨迹** 形式化安全  
2. **默认 full 宽盒**（min_d≥0.12，\|close\|≤0.5）的 PROVEN_FULL  
3. DeepSet (k,m) 分桶 proven 表  
4. 将 MF task success 对齐 DeepSet ~89% 作为 pass 条件  
5. 平面几何（非径向）的必要碰撞条件（径向为**充分**条件）

---

## 5. Soundness 摘要

- 径向 head-on 过近似：径向 safe ⇒ 同 d0/c0 下平面两步 safe（充分）。  
- 特征盒为轴对齐积：可含不可实现组合（过近似）；证明仍 sound。  
- 占用象限 d0/c0 与特征提取器一致。  
- PROVEN_FULL = 声明 \(X\) 全覆盖 + 0 真实策略 CE + 0 unknown。

---

## 6. 签署栏（项目内）

| 项 | 内容 |
|---|---|
| 验收对象 | Mean-field residual r2 + L2 near_plus 形式化 |
| 结果 | **PASS** |
| 交付包 | `acceptance/mf_best_20261006/1_primary/` |
| 完整性 | 见 `MANIFEST.json` / `SHA256SUMS.txt` |

```text
验收结论：Mean-field 并行链路完成；
以 L2 near_plus 有界特征盒两步径向 L2 的 PROVEN_FULL 作为形式化主验收。
```
