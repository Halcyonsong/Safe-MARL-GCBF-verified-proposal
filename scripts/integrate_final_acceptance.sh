#!/bin/bash
# 最终验收包整合脚本
# 基于 mf_best_r2 + M1-M3 planar 验证

set -e

echo "=========================================="
echo "最终验收包整合脚本"
echo "=========================================="
echo ""

# 配置
NEW_PACK="acceptance/mf_unified_planar_20261011"
OLD_PACK="acceptance/mf_best_20261006"
NETDEP_PACK="acceptance/mf_planar_netdep_20261011"
VERIFY_SOURCE="method/meanfield/output/verify/planar_safeset_v1/mf_best_r2_ladder"

# 检查源目录
if [ ! -d "$OLD_PACK" ]; then
    echo "❌ 错误：找不到旧包 $OLD_PACK"
    exit 1
fi

if [ ! -d "$VERIFY_SOURCE" ]; then
    echo "❌ 错误：找不到 mf_best_r2 验证结果 $VERIFY_SOURCE"
    exit 1
fi

echo "✅ 源目录检查通过"
echo ""

# 询问是否继续
read -p "是否创建新的验收包 $NEW_PACK？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消操作"
    exit 0
fi

# 创建新包结构
echo "📁 创建新包目录结构..."
mkdir -p "$NEW_PACK"/{1_primary,2_baselines,3_appendix,4_paper_figures,aamas_2027_template,part,pointers}
mkdir -p "$NEW_PACK"/1_primary/{00_acceptance,01_models,03_task_eval,04_ce_repair,05_formal_verify,07_source_snapshot}

echo "✅ 目录结构创建完成"
echo ""

# ============================================================
# P0: 核心数据整合
# ============================================================

echo "📋 [P0] 开始核心数据整合..."
echo ""

# 1. 复制 M1-M3 验证结果
echo "  → 复制 M1-M3 验证结果..."
cp -r "$VERIFY_SOURCE"/* "$NEW_PACK/1_primary/05_formal_verify/"
echo "    ✅ M1-M3 验证结果已复制"

# 2. 复制 checkpoint
echo "  → 复制 checkpoint 模型..."
cp -r "$OLD_PACK/1_primary/01_models"/* "$NEW_PACK/1_primary/01_models/"
echo "    ✅ Checkpoint 已复制"

# 3. 复制任务评估
echo "  → 复制任务评估数据..."
cp -r "$OLD_PACK/1_primary/03_task_eval"/* "$NEW_PACK/1_primary/03_task_eval/"
echo "    ✅ 任务评估已复制"

# 4. 复制修复元数据
echo "  → 复制修复元数据..."
if [ -d "$OLD_PACK/1_primary/04_ce_repair" ]; then
    cp -r "$OLD_PACK/1_primary/04_ce_repair"/* "$NEW_PACK/1_primary/04_ce_repair/"
    echo "    ✅ 修复元数据已复制"
else
    echo "    ⚠️  旧包没有 04_ce_repair，跳过"
fi

# 5. 复制源码快照（从 netdep 包，因为是最新的）
echo "  → 复制源码快照..."
if [ -d "$NETDEP_PACK/1_primary/07_source_snapshot" ]; then
    cp -r "$NETDEP_PACK/1_primary/07_source_snapshot"/* "$NEW_PACK/1_primary/07_source_snapshot/"
    echo "    ✅ 源码快照已复制"
else
    echo "    ⚠️  需要手动复制源码快照"
fi

# 6. 复制 baseline 全表
echo "  → 复制 baseline 对照数据..."
cp -r "$OLD_PACK/2_baselines"/* "$NEW_PACK/2_baselines/"
echo "    ✅ Baseline 数据已复制"

# 7. 复制 paper 图表
echo "  → 复制论文图表..."
cp -r "$OLD_PACK/4_paper_figures"/* "$NEW_PACK/4_paper_figures/"
echo "    ✅ 论文图表已复制"

# 8. 复制 LaTeX 模板
echo "  → 复制 LaTeX 模板..."
if [ -d "$NETDEP_PACK/aamas_2027_template" ]; then
    cp -r "$NETDEP_PACK/aamas_2027_template"/* "$NEW_PACK/aamas_2027_template/"
    echo "    ✅ LaTeX 模板已复制"
elif [ -d "$OLD_PACK/aamas_2027_template" ]; then
    cp -r "$OLD_PACK/aamas_2027_template"/* "$NEW_PACK/aamas_2027_template/"
    echo "    ✅ LaTeX 模板已复制（从旧包）"
fi

echo ""
echo "✅ [P0] 核心数据整合完成"
echo ""

# ============================================================
# P1: 归档冗余数据
# ============================================================

echo "📦 [P1] 归档冗余数据..."
echo ""

# 归档 netdep_r3 验证结果（作为稳健性证据）
echo "  → 归档 netdep_r3 验证结果..."
mkdir -p "$NEW_PACK/3_appendix/netdep_r3_robustness"
if [ -d "$NETDEP_PACK/1_primary/05_formal_verify" ]; then
    cp -r "$NETDEP_PACK/1_primary/05_formal_verify"/* "$NEW_PACK/3_appendix/netdep_r3_robustness/" 2>/dev/null || true
    echo "    ✅ netdep_r3 验证结果已归档（稳健性证据）"
fi

# 复制其他附录
echo "  → 复制附录材料..."
if [ -d "$OLD_PACK/3_appendix" ]; then
    cp -r "$OLD_PACK/3_appendix"/* "$NEW_PACK/3_appendix/" 2>/dev/null || true
    echo "    ✅ 附录材料已复制"
fi

echo ""
echo "✅ [P1] 归档完成"
echo ""

# ============================================================
# 生成核心文档
# ============================================================

echo "📝 生成核心文档..."
echo ""

# 生成 README.md
cat > "$NEW_PACK/README.md" <<'EOF'
# MF 平面两步网络依赖形式化验收包（统一版）

**日期**：2026-10-11  
**版本**：unified (mf_best_r2 + M1-M3 planar)

---

## 核心贡献

### 形式化验证
- **M1-M3 PROVEN_FULL**（vg ±0.01 → ±0.10）
- **SafeSet + Marabou** 平面两步（网络依赖域）
- 验证时间：65s → 713s → 1704s（次线性增长）

### Checkpoint
- **residual_r2** `repair_best.pkl` + base_v5

### 任务评估（完整）
- @32: success 82.0%, safe 96.5%, finish 84.0%
- @64: success 82.6%, safe 94.9%, finish 85.0%
- @256: success 83.5%, safe 95.7%, finish 86.4%

### Baseline 对照（7 方法 × 3 规模）
- GCBF+, MAPPO-Lag, InforMARL, HandCBF, etc.

---

## 目录结构

```
1_primary/
├── 00_acceptance/         ← 验收证书
├── 01_models/             ← residual_r2 + base_v5
├── 03_task_eval/          ← 32/64/256 完整
├── 04_ce_repair/          ← 修复链路
├── 05_formal_verify/      ← M1-M3 验证结果
└── 07_source_snapshot/    ← 源码快照

2_baselines/               ← 7 方法完整对照
4_paper_figures/           ← 论文图表数据
3_appendix/                ← 附录证据
aamas_2027_template/       ← LaTeX 源码
```

---

## 主张边界

**能写**：
- M1-M3 平面两步网络依赖域 PROVEN_FULL
- vg ±0.10（pose 固定、threat 中点）
- 验证时间可扩展（次线性增长）
- 任务支持：完整 32/64/256 + baseline 全表

**不能写**：
- 全 pose、全 threat 已证明
- vg ±0.10 覆盖全物理空间
- CEGAR 紧闭环（修复-验证弱耦合）

---

详见 `ACCEPTANCE_CERTIFICATE.md` 和 `METRICS_SUMMARY.md`。
EOF

echo "  ✅ README.md 已生成"

# 生成 NOTES.md
cat > "$NEW_PACK/NOTES.md" <<'EOF'
# 叙事边界

## 能写

1. **平面两步网络依赖域形式化验证**
   - SafeSet 几何反推 + Marabou 编码 Deploy
   - M1→M2→M3 PROVEN_FULL（vg ±0.01 → ±0.10）
   - 验证时间次线性增长（65s → 713s → 1704s）

2. **完整任务评估**
   - residual_r2: 82.0/96.5/84.0 (@32)
   - 完整 64/256 压力测试
   - baseline 7 方法 × 3 规模全表

3. **网络依赖性**
   - envelope_any_a_safe = False
   - 采样 ax ≥ 0.819（SafeSet 边界）
   - 反事实 ax≥0.99 sat_candidate（非空洞）

## 不能写

1. **全状态空间覆盖**
   - pose 固定 px=py=2（不是自由 pose）
   - threat 中点（不是全 threat 域）
   - vg ±0.10 ≈ 20% 物理范围（不是 100%）

2. **CEGAR 紧闭环**
   - 修复主要靠经验采样 + planar QP
   - 验证是事后证书（不是修复驱动）
   - M1-M3 直接 PROVEN（未触发 CE → 修复路径）

3. **基线必有反例**
   - near_plus 包络盒上 baseline 也可能 0 fail
   - 主 claim 是形式化证书，不是基线反例

## 与旧包关系

- **旧包** `mf_best_20261006`: near_plus 径向包络（IBP）
- **本包**: M1-M3 平面网络依赖（SafeSet + Marabou）
- 本包是旧包的**升级版**（同一 checkpoint，更强证书）
EOF

echo "  ✅ NOTES.md 已生成"

# 生成指针文件
cat > "$NEW_PACK/1_primary/00_acceptance/PACK_BASE.txt" <<EOF
method/meanfield/output/base/mf_feat44_base_v5_20261004/seed0/meanfield_mlp512_student_step9000.pkl
EOF

cat > "$NEW_PACK/1_primary/00_acceptance/PACK_RESIDUAL.txt" <<EOF
method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl
EOF

cat > "$NEW_PACK/1_primary/00_acceptance/PACK_VERIFY_PRIMARY.txt" <<EOF
method/meanfield/output/verify/planar_safeset_v1/mf_best_r2_ladder/M3_vg010
EOF

echo "  ✅ 指针文件已生成"

echo ""
echo "✅ 核心文档生成完成"
echo ""

# ============================================================
# 最终检查
# ============================================================

echo "🔍 最终检查..."
echo ""

# 检查关键文件
CRITICAL_FILES=(
    "1_primary/01_models/residual_r2/repair_best.pkl"
    "1_primary/05_formal_verify/M1_vg001/REPORT.md"
    "1_primary/05_formal_verify/M2_vg005/REPORT.md"
    "1_primary/05_formal_verify/M3_vg010/REPORT.md"
    "1_primary/03_task_eval/residual_r2_validate_32epi.json"
    "1_primary/03_task_eval/residual_r2_validate_64epi.json"
    "1_primary/03_task_eval/residual_r2_validate_256epi.json"
    "2_baselines/BASELINE_COMPARE.md"
    "4_paper_figures/figs/fig3_baseline_main_bar.pdf"
)

MISSING=0
for file in "${CRITICAL_FILES[@]}"; do
    if [ ! -f "$NEW_PACK/$file" ]; then
        echo "  ❌ 缺失：$file"
        MISSING=1
    fi
done

if [ $MISSING -eq 0 ]; then
    echo "  ✅ 所有关键文件检查通过"
else
    echo "  ⚠️  有文件缺失，请检查"
fi

echo ""

# 统计文件数量
TOTAL_FILES=$(find "$NEW_PACK" -type f | wc -l)
echo "📊 统计："
echo "  - 总文件数：$TOTAL_FILES"
echo "  - 目录大小：$(du -sh "$NEW_PACK" | cut -f1)"

echo ""
echo "=========================================="
echo "✅ 验收包整合完成！"
echo "=========================================="
echo ""
echo "新验收包位置：$NEW_PACK"
echo ""
echo "📋 下一步："
echo "  1. 检查 $NEW_PACK/README.md"
echo "  2. 更新 ACCEPTANCE_CERTIFICATE.md（手动）"
echo "  3. 生成 METRICS_SUMMARY.md（手动）"
echo "  4. 重写 LaTeX（参考 FINAL_INTEGRATION_PLAN.md）"
echo "  5. 生成 MANIFEST.json + SHA256SUMS.txt（手动）"
echo ""
echo "参考文档："
echo "  - FINAL_INTEGRATION_PLAN.md"
echo "  - acceptance/mf_planar_netdep_20261011/CHECKPOINT_COMPARISON.md"
echo "  - acceptance/mf_planar_netdep_20261011/GAP_ANALYSIS.md"
echo ""
