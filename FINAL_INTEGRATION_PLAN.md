# 最终验收包整理方案（基于 mf_best_r2 + M1-M3）

**日期**：2026-10-11  
**目标**：整合 mf_best_r2 的完整数据 + M1-M3 planar 验证，清理冗余，准备论文重构

---

## 一、核心决策

### ✅ **选择 mf_best_r2 作为最终验收 checkpoint**

**理由**：
- M1-M3 PROVEN_FULL（与 netdep_r3 同样通过）
- 任务数据完整（32/64/256 全有）
- 性能相当或更好（safe 96.5% > 95.7%）
- baseline 全表齐全
- **零补测工作量**

---

## 二、数据整合清单

### 2.1 必须保留的数据 ✅

#### A. 形式化验证（核心）

**来源**：`method/meanfield/output/verify/planar_safeset_v1/mf_best_r2_ladder/`

```
1_primary/05_formal_verify/
├── M1_vg001/
│   ├── REPORT.json     ← mf_best_r2 M1 结果
│   └── REPORT.md
├── M2_vg005/
│   ├── REPORT.json     ← mf_best_r2 M2 结果
│   └── REPORT.md
├── M3_vg010/
│   ├── REPORT.json     ← mf_best_r2 M3 结果（主引用）
│   └── REPORT.md
└── LADDER_SUMMARY.json ← 三档汇总
```

**关键指标**：
- M1: 65s, ax≥0.874, PROVEN_FULL
- M2: 713s, ax≥0.850, PROVEN_FULL  
- M3: 1704s, ax≥0.819, PROVEN_FULL

#### B. Checkpoint 模型

**来源**：`acceptance/mf_best_20261006/1_primary/01_models/`

```
1_primary/01_models/
├── base_v5/
│   └── meanfield_mlp512_student_step9000.pkl  ← 保留
└── residual_r2/
    ├── repair_best.pkl    ← **主验收 checkpoint**
    └── repair_last.pkl    ← 可选保留（对照）
```

#### C. 任务评估（完整）

**来源**：`acceptance/mf_best_20261006/1_primary/03_task_eval/`

```
1_primary/03_task_eval/
├── base_v5_validate_32epi.json
├── residual_r1_validate_32epi.json
├── residual_r2_validate_32epi.json   ← ✅ 82.0/96.5/84.0
├── residual_r2_validate_64epi.json   ← ✅ 82.6/94.9/85.0
└── residual_r2_validate_256epi.json  ← ✅ 83.5/95.7/86.4
```

#### D. Baseline 全表

**来源**：`acceptance/mf_best_20261006/2_baselines/`

```
2_baselines/
├── BASELINE_COMPARE.md          ← 7方法 × 3规模完整表
├── BASELINE_FAIL_EVIDENCE.md    ← mid-F 经验 fail
├── epi_pressure/
│   ├── epi64/                   ← @64 所有方法
│   └── epi256/                  ← @256 所有方法
├── latency/
│   └── control_latency.json     ← 控制延迟（r2/GCBF+/HandCBF）
├── transfer_n/
│   ├── N8/N16/N32/              ← 迁移能力
│   └── TRANSFER_COMPARE.md
└── near_plus_box_mc/            ← 包络盒 MC（可选）
```

#### E. Paper 图表

**来源**：`acceptance/mf_best_20261006/4_paper_figures/`

```
4_paper_figures/
├── figs/
│   ├── fig_arch_matrix_shapes.pdf
│   ├── fig3_baseline_main_bar.pdf
│   ├── fig1_repair_ce_hit.pdf
│   └── fig2_repair_task_metrics.pdf
├── data/                         ← 图表原始数据
├── make_figures.py               ← 绘图脚本
└── paper_figure_data.json        ← 数据汇总
```

#### F. 修复链路元数据

**来源**：`acceptance/mf_best_20261006/1_primary/04_ce_repair/`

```
1_primary/04_ce_repair/
├── chain/
│   ├── SELECTION_r1.md
│   └── SELECTION_r2.md           ← r2 选择依据
├── repair.csv                    ← base→r1→r2 修复链
└── ce_post_summary.json          ← CE 统计
```

#### G. 源码快照

**来源**：`method/meanfield/stage_mf/` + `scripts/`

```
1_primary/07_source_snapshot/
├── property_planar_l2.py         ← SafeSet + netdep claim
├── marabou_mf.py                 ← Deploy 编码 + McCormick fix
├── verify_planar.py              ← DFS verifier
├── run_mf_verify_planar.sh       ← CLI entry
├── ibp_mf.py
└── encode.py
```

---

### 2.2 需要新增的文档 📝

#### A. 主验收证书（更新）

```
ACCEPTANCE_CERTIFICATE.md         ← 更新为 M1-M3 planar 主叙事
```

**内容要点**：
- 主 claim：平面两步 SafeSet + Marabou（M1-M3 PROVEN_FULL）
- 附录：near_plus 包络（IBP）作为额外证据
- Checkpoint：mf_best_r2 `repair_best.pkl`
- 任务支持：82.0/96.5/84.0 (@32) + 完整 64/256

#### B. 方法对比说明

```
METHOD_EVOLUTION.md               ← 径向包络 → 平面网络依赖
```

**内容要点**：
- 旧路径：near_plus IBP 径向包络（envelope_any_a_safe）
- **新路径**：planar SafeSet + Marabou（network_dependent）
- 为什么更强：网络依赖 vs 几何包络
- 两条路径都 valid，新路径是主叙事

#### C. 实验汇总（简化）

```
METRICS_SUMMARY.md                ← 一页速查
```

**内容**：
- 形式化：M1-M3 三档 + 时间
- 任务：32/64/256 三规模
- Baseline：7 方法对照
- 延迟：控制计算耗时

---

### 2.3 可以删除/归档的数据 🗑️

#### A. netdep_r3 相关（不完整）

**理由**：mf_best_r2 更好且数据完整

```
❌ acceptance/mf_planar_netdep_20261011/1_primary/01_models/residual_netdep_r3/
❌ acceptance/mf_planar_netdep_20261011/1_primary/03_task_eval/chain_32epi/
   （保留一份作为附录：验证链路稳健性证据）
```

**建议**：
- 移动到 `3_appendix/netdep_r3_robustness/`
- 标注"验证方法稳健性证据，但最终验收用 r2"

#### B. 冗余验证结果

```
❌ method/meanfield/output/verify/planar_safeset_v1/min_M1_vg001/
❌ method/meanfield/output/verify/planar_safeset_v1/enlarge_M2_vg005/
❌ method/meanfield/output/verify/planar_safeset_v1/enlarge_M3_vg010/
   （这些是 netdep_r3 的结果，已被 mf_best_r2_ladder 替代）
```

**建议**：
- 归档到 `3_appendix/alternate_checkpoints/`
- 或直接删除（git 历史中可找回）

#### C. 作废的验证（已知）

```
✅ 已标注作废：
3_appendix/_invalidated_pre_mccormick_fix/
3_appendix/bnb_INCOMPLETE_M2/
```

#### D. 旧包的包络验证（可降级为附录）

**来源**：`acceptance/mf_best_20261006/1_primary/05_formal_verify/`

```
旧内容（IBP 径向包络）：
1_primary/05_formal_verify/
├── near_plus_r2_PROVEN/          ← 旧主证书
└── ...
```

**建议**：
- 移动到 `3_appendix/envelope_path/`
- 标注"额外的包络路径，主验收用 planar netdep"

---

## 三、最终验收包结构（整理后）

```
acceptance/mf_unified_planar_20261011/    ← 新统一包名
├── README.md
├── ACCEPTANCE_CERTIFICATE.md             ← 更新：M1-M3 planar 主叙事
├── METRICS_SUMMARY.md                    ← 新增：一页速查
├── METHOD_EVOLUTION.md                   ← 新增：径向→平面演进
├── NOTES.md                              ← 能写/不能写
├── RECORD.md                             ← 实验账本
├── MANIFEST.json
├── SHA256SUMS.txt
│
├── 1_primary/
│   ├── 00_acceptance/
│   │   ├── ACCEPTANCE_CERTIFICATE.md
│   │   ├── METRICS_SUMMARY.md
│   │   └── PACK_*.txt
│   │
│   ├── 01_models/
│   │   ├── base_v5/
│   │   │   └── meanfield_mlp512_student_step9000.pkl
│   │   └── residual_r2/
│   │       ├── repair_best.pkl           ← 主验收
│   │       └── repair_last.pkl
│   │
│   ├── 03_task_eval/
│   │   ├── TASK_EVAL_LEDGER.md
│   │   ├── residual_r2_validate_32epi.json
│   │   ├── residual_r2_validate_64epi.json
│   │   └── residual_r2_validate_256epi.json
│   │
│   ├── 04_ce_repair/
│   │   ├── SELECTION_r2.md
│   │   ├── repair.csv
│   │   └── ce_post_summary.json
│   │
│   ├── 05_formal_verify/
│   │   ├── M1_vg001/                     ← mf_best_r2 M1
│   │   ├── M2_vg005/                     ← mf_best_r2 M2
│   │   ├── M3_vg010/                     ← mf_best_r2 M3 (主)
│   │   └── LADDER_SUMMARY.json
│   │
│   └── 07_source_snapshot/
│       ├── property_planar_l2.py
│       ├── marabou_mf.py
│       ├── verify_planar.py
│       └── run_mf_verify_planar.sh
│
├── 2_baselines/                          ← 完整保留
│   ├── BASELINE_COMPARE.md
│   ├── epi_pressure/
│   ├── latency/
│   ├── transfer_n/
│   └── near_plus_box_mc/
│
├── 3_appendix/
│   ├── netdep_r3_robustness/             ← 新增：netdep_r3 验证结果（稳健性证据）
│   ├── envelope_path/                    ← 降级：near_plus IBP 包络
│   ├── evidence/                         ← 保留
│   └── INVALIDATED_pre_mccormick.md
│
├── 4_paper_figures/                      ← 完整保留
│   ├── figs/
│   ├── data/
│   ├── make_figures.py
│   └── paper_figure_data.json
│
├── aamas_2027_template/
│   ├── AAMAS_2027_sample.tex             ← 需要重写
│   ├── references.bib
│   └── figs/
│
└── part/                                 ← 中文部分（需重写）
    ├── part5_experiments.md
    └── part6_limitations.md
```

---

## 四、整理动作清单（按优先级）

### 🔴 P0：必须做（数据整合）

- [ ] 复制 `mf_best_r2_ladder/M1-M3/` → 新包 `05_formal_verify/`
- [ ] 更新 `ACCEPTANCE_CERTIFICATE.md`（M1-M3 planar 主叙事）
- [ ] 新增 `METRICS_SUMMARY.md`（一页速查）
- [ ] 新增 `METHOD_EVOLUTION.md`（径向→平面演进）
- [ ] 复制完整 baseline 数据（从旧包 `2_baselines/`）
- [ ] 复制完整图表数据（从旧包 `4_paper_figures/`）
- [ ] 生成新的 `MANIFEST.json` + `SHA256SUMS.txt`

### 🟡 P1：强烈建议（清理冗余）

- [ ] 归档 netdep_r3 验证结果 → `3_appendix/netdep_r3_robustness/`
- [ ] 降级 near_plus 包络验证 → `3_appendix/envelope_path/`
- [ ] 删除 `min_M1_vg001/`, `enlarge_M2_vg005/`, `enlarge_M3_vg010/`（netdep_r3 的）
- [ ] 删除或归档 `mf_planar_netdep_20261011/` 整个目录（已被 r2 替代）

### 🟢 P2：可选（优化）

- [ ] 统一所有 REPORT.md 格式
- [ ] 添加 quick-start 复现脚本
- [ ] 生成可视化对比图（M1-M3 时间 scaling）

---

## 五、论文重构清单

### 5.1 LaTeX 需要修改的部分

**文件**：`aamas_2027_template/AAMAS_2027_sample.tex`

#### A. Abstract

**修改要点**：
- 主 claim：平面两步 **网络依赖域** 形式化验证（SafeSet + Marabou）
- 证书：M1→M2→M3 PROVEN_FULL，vg ±0.01 → ±0.10
- 可扩展性：验证时间次线性增长
- 任务支持：82.0/96.5/84.0 + 完整 32/64/256

#### B. §3 Method

**新增 §3.3**：Network-Dependent Formal Certification
- SafeSet 几何反推（从碰撞 → 安全动作集）
- Marabou 编码 Deploy + SafeSet 约束
- M1-M3 梯度扩展策略

**修改 §3.2**：去掉 CEGAR 过度声称
- 改为"Geometric-Guided Residual Repair"
- 诚实说明：经验采样 + planar QP 标签
- 验证是事后证书（不是闭环驱动修复）

#### C. §4.2 Experiments

**更新主表**：
- 形式化：M1-M3 三档 + 时间（表格）
- 任务：residual_r2 完整 32/64/256（表格）
- Baseline：7 方法对照（bar 图）

**新增图表**：
- M1-M3 scaling（时间 vs vg）
- 安全边际（采样 ax 分布）

#### D. §5 Limitations

**诚实披露**：
- pose 固定（px=py=2）
- threat 中点（特征单点）
- vg ±0.10（20% 物理范围）
- 修复与验证弱耦合（不是紧 CEGAR）

**Future work**：
- 自由 pose、threat 全宽
- 前向不变性（闭环重规划）
- 更紧的 CEGAR 集成

---

### 5.2 中文部分需要重写

**文件**：`part/part5_experiments.md`, `part6_limitations.md`

**要点**：
- part5：更新为 M1-M3 数据 + r2 任务表
- part6：诚实披露 pose/threat/vg 限制 + 修复-验证解耦

---

## 六、最终检查清单（发布前）

### 数据完整性

- [ ] M1-M3 验证结果齐全（REPORT.json + md）
- [ ] residual_r2 checkpoint 存在且 SHA256 正确
- [ ] 任务评估 32/64/256 齐全
- [ ] Baseline 7 方法 × 3 规模全表
- [ ] Paper 图表 4 张 + 数据源

### 文档一致性

- [ ] ACCEPTANCE_CERTIFICATE 指向 mf_best_r2
- [ ] MANIFEST.json 包含所有文件
- [ ] SHA256SUMS.txt 校验通过
- [ ] NOTES.md 边界清晰（能写/不能写）

### 论文对齐

- [ ] Abstract 与 M1-M3 数据一致
- [ ] §4.2 表格数据与验收包一致
- [ ] Limitations 诚实披露限制
- [ ] 不声称：全 pose、全 threat、CEGAR 闭环

---

## 七、时间估算

| 任务 | 工作量 | 备注 |
|---|---|---|
| 数据整合（P0） | 2-3h | 复制文件 + 更新文档 |
| 清理冗余（P1） | 1-2h | 归档/删除 |
| LaTeX 重写（P0） | 4-6h | Abstract + Method + Experiments + Limitations |
| 中文重写（P1） | 2-3h | part5 + part6 |
| 生成新图表（P1） | 2-3h | M1-M3 scaling + 安全边际 |
| 校验测试（P0） | 1h | SHA256 + 复现验证 |

**总计**：约 **12-18 小时**（1.5-2 天）

---

## 八、推荐工作流

### Day 1 上午：数据整合（P0）
1. 创建新包 `acceptance/mf_unified_planar_20261011/`
2. 复制 mf_best_r2_ladder 验证结果
3. 复制旧包完整数据（baseline/图表/任务）
4. 更新核心文档（ACCEPTANCE_CERTIFICATE, METRICS_SUMMARY, METHOD_EVOLUTION）

### Day 1 下午：清理冗余（P1）
5. 归档 netdep_r3 验证结果
6. 降级 near_plus 包络路径
7. 删除重复/作废数据
8. 生成 MANIFEST + SHA256

### Day 2 上午：LaTeX 重写（P0）
9. Abstract 改为 M1-M3 planar 主叙事
10. §3 Method 调整（去 CEGAR，加 SafeSet）
11. §4.2 Experiments 更新表格和图表
12. §5 Limitations 诚实披露

### Day 2 下午：图表+中文+测试（P1）
13. 生成 M1-M3 scaling 图
14. 重写中文 part5/part6
15. 复现验证（SHA256 + 一个 M1 验证）
16. 最终检查

---

## 九、一句话总结

**用 mf_best_r2（数据完整，性能好）+ M1-M3 planar 验证（网络依赖强）= 最强验收包，零补测，1.5-2 天可完成整理+论文重构。**
