# 新增包完整性检查与补测建议

**日期**：2026-10-11  
**包**：`acceptance/mf_final_20261012/5_frozen_sources/mf_planar_netdep_20261011/`  
**目标**：评估是否可覆盖旧形式化验证链路，识别缺失数据

---

## 一、完整性对比

### 1.1 核心形式化证据 ✅ **齐全**

| 项目 | 状态 | 说明 |
|---|---|---|
| **M1-M3 证书** | ✅ 齐全 | PROVEN_FULL，0 真 CE，反事实非空 |
| **验证源码** | ✅ 齐全 | `1_primary/07_source_snapshot/` 含完整链路 |
| **checkpoint** | ✅ 齐全 | r3 repair_manual_pick.pkl + base v5 |
| **修复链路** | ✅ 齐全 | base→r1→r2→r3 SELECTION + CE 摘要 |
| **SHA256 校验** | ✅ 通过 | 所有文件完整性已验证 |

### 1.2 任务评估数据 ⚠️ **部分缺失**

| 评估类型 | 旧包 `mf_best_20261006` | 新包 `mf_planar_netdep_20261011` | 缺失 |
|---|---|---|---|
| **@32 epi 主评估** | ✅ 完整 | ✅ **完整**（base→r3 链） | — |
| **@64 epi 压力** | ✅ 完整 | ❌ **无** | netdep r3 @64 |
| **@256 epi 大压力** | ✅ 完整 | ❌ **无** | netdep r3 @256 |
| **baseline 完整表** | ✅ 齐全（7 方法 × 3 规模） | ⚠️ **引用旧包** | — |

**关键发现**：
- 新包 **只有** netdep r3 的 @32 评估
- @64/@256 压力测试 **完全缺失**
- baseline 对照表 **未针对 r3 重跑**，仍引用旧包 mf_best r2 的基线数据

### 1.3 基线对照 ⚠️ **未重新运行**

| 对照实验 | 旧包 | 新包 | 缺失 |
|---|---|---|---|
| **GCBF+/MAPPO/InforMARL** | ✅ @32/64/256 全表 | ⚠️ **引用旧包** | 无针对 r3 的重跑 |
| **控制延迟** | ✅ 有数据 | ❌ **无** | r3 延迟测试 |
| **Transfer N** | ✅ N=8/16/32 | ❌ **无** | r3 迁移能力 |
| **near_plus MC 采样** | ✅ 有 | ❌ **无**（性质不同） | netdep 盒 MC |

### 1.4 论文素材 ⚠️ **部分缺失**

| 论文支持 | 旧包 | 新包 | 缺失 |
|---|---|---|---|
| **LaTeX 源码** | ✅ （AAMAS 不入库） | ✅ **有**（复制旧版） | — |
| **图表数据** | ✅ `4_paper_figures/` 完整 | ⚠️ **只有 1 张图** | 无 r3 专属图表数据 |
| **paper 图片** | ✅ 4 张主图 + 附录 | ⚠️ **只有 baseline bar** | 缺 r3 repair curves、hit 下降等 |
| **中文部分稿** | ✅ `part/part5_*.md` | ❌ **无** | 需要根据 M1-M3 重写 |

---

## 二、差距分析

### 2.1 形式化验证链路 ✅ **可完全覆盖**

**结论**：新包的形式化验证证据 **完整且优于旧包**

| 维度 | 旧包 `near_plus` | 新包 `netdep M1-M3` | 优势 |
|---|---|---|---|
| **性质** | 径向包络（任意 a 安全） | **平面网络依赖**（Deploy∈SafeSet） | ✅ 网络依赖更强 |
| **方法** | 主要 IBP | **SafeSet + Marabou** | ✅ 更精确 |
| **证书** | PROVEN_FULL | **PROVEN_FULL** | ✅ 同级 |
| **可扩展性** | 单盒 | **M1→M2→M3 梯度** | ✅ 展示扩展性 |
| **时间** | 秒级（包络简单） | **25s → 8min → 25min** | ✅ 非平凡证明 |

**推荐**：形式化验证链路 **完全可以**用新包覆盖旧包。

### 2.2 任务评估 ⚠️ **需要补测**

**问题**：
1. **r3 只有 @32，缺 @64/@256** → 无法与旧包 r2 的完整压力测试对比
2. **baseline 未重跑** → 无法证明 r3 相对 baseline 的性能
3. **无控制延迟** → 无法证明 r3 部署可行性

**影响**：
- 如果论文要写 **netdep r3 作为主方法**，需要补全 @64/@256
- 如果只写 **形式化证书**（M1-M3），可以引用 r2 的任务数据作为"经验支持"

### 2.3 论文图表 ⚠️ **需要重新生成**

**问题**：
- 新包只有 1 张从旧包复制的 baseline bar 图
- 缺少 **r3 专属的 repair curves**（hit 下降、任务指标变化）
- 缺少 **M1-M3 验证时间 scaling 图**

---

## 三、补测建议（按优先级）

### 🔴 **P0：必须补（如果要完全覆盖）**

1. **netdep r3 @64/@256 任务评估**
   ```bash
   # 复现命令（示例）
   python method/meanfield/scripts/eval_mf.py \
     --ckpt acceptance/mf_final_20261012/5_frozen_sources/mf_planar_netdep_20261011/1_primary/01_models/residual_netdep_r3/repair_manual_pick.pkl \
     --n_epi 64 --seed 1234
   
   python method/meanfield/scripts/eval_mf.py \
     --ckpt ... --n_epi 256 --seed 1234
   ```
   **理由**：论文主方法必须有完整 32/64/256 压力测试（与旧包对齐）

2. **论文 LaTeX 重写**
   - 修改 `AAMAS_2027_sample.tex`：
     - Abstract：改为 planar netdep 主叙事
     - §4.2 Experiments：更新为 M1-M3 数据
     - §5 Limitations：诚实披露 pose/threat/vg 限制
   - 中文部分 `part/part5_experiments.md` + `part6_limitations.md` 需要重写

### 🟡 **P1：强烈建议补（论文完整性）**

3. **生成 r3 专属图表**
   - repair curves（base→r1→r2→r3 hit 下降）
   - task metrics（suc/safe/finish 变化）
   - M1-M3 scaling（验证时间 vs vg 半宽）
   
   ```bash
   # 使用旧包的 make_figures.py 作为模板
   cd acceptance/mf_planar_netdep_20261011
   mkdir -p paper_figures/{data,figs}
   # 需要写新的绘图脚本
   ```

4. **控制延迟测试**
   ```bash
   # 测试 r3 前向推理时间
   python method/meanfield/scripts/measure_latency.py \
     --ckpt ... --n_repeats 200 --seed 1234
   ```

5. **netdep 盒 MC 采样验证**
   ```bash
   # 在 netdep fringe 上采样，确认 Deploy.ax ≥ 0
   # 类似旧包 2_baselines/near_plus_box_mc/
   ```

### 🟢 **P2：可选补（增强说服力）**

6. **Transfer N 测试**（N=8/16/32 迁移能力）
7. **baseline 针对 r3 重跑**（确认 r3 vs GCBF+/MAPPO 对比）
8. **Ablation**（residual vs base vs coast）

---

## 四、覆盖策略建议

### 策略 A：**完全覆盖**（推荐，但需补测）

**动作**：
1. ✅ 用新包 `5_frozen_sources/mf_planar_netdep_20261011/` 替换旧验证链路
2. ⚠️ 补测 r3 @64/@256（P0）
3. ⚠️ 重写论文 LaTeX（P0）
4. ⚠️ 生成 r3 图表（P1）

**优势**：
- 形式化叙事更强（网络依赖 vs 包络）
- M1-M3 梯度展示可扩展性
- 诚实披露限制（pose/threat scoped）

**风险**：
- 需要 1-2 天补测
- LaTeX 需要大改（Abstract、Experiments、Limitations）

### 策略 B：**混合引用**（快速投稿）

**动作**：
1. ✅ 形式化验证部分：引用 **新包 M1-M3**
2. ✅ 任务评估部分：引用 **旧包 r2 @32/64/256**
3. ⚠️ LaTeX 写成"两层叙事"：
   - 主叙事：near_plus 包络（旧包）作为主验收
   - 进阶叙事：M1-M3 netdep（新包）作为"网络依赖域的首次证明"

**优势**：
- **无需补测**，可立即投稿
- 保留旧包的完整任务数据
- 新包作为"额外贡献"

**劣势**：
- 叙事分裂（包络 vs netdep 两条线）
- 审稿人可能质疑"哪个是主贡献"

### 策略 C：**新包作为独立工作**（workshop/poster）

**动作**：
1. 新包 **不覆盖** 旧包
2. 新包作为 **独立的 short paper/workshop** 投稿
3. 主会议论文仍用旧包 `mf_best_20261006`

**适用场景**：
- 时间紧张，不想大改论文
- 新包作为"网络依赖域验证"的独立贡献

---

## 五、最终建议

### ✅ **形式化验证链路覆盖** → **合适**

**结论**：新包的形式化证据 **完整且优于旧包**，可以覆盖。

### ⚠️ **需要补测的数据**（按策略 A）

| 项目 | 优先级 | 工作量 | 截止前可完成？ |
|---|---|---|---|
| **r3 @64 评估** | P0 | 1-2 小时 | ✅ 可 |
| **r3 @256 评估** | P0 | 2-3 小时 | ✅ 可 |
| **LaTeX 重写** | P0 | 4-6 小时 | ✅ 可 |
| **生成 r3 图表** | P1 | 2-3 小时 | ✅ 可 |
| **控制延迟** | P1 | 1 小时 | ✅ 可 |
| **baseline 重跑** | P2 | 4-8 小时 | ⚠️ 视情况 |

**总工作量（P0+P1）**：约 **1-2 天**。

### 推荐方案

如果 **1-2 天内可以补测** → **策略 A**（完全覆盖，P0+P1 必做）  
如果 **立即投稿** → **策略 B**（混合引用，无需补测，但叙事需调整）

---

## 六、检查清单

**覆盖前必须确认**：

- [ ] r3 @64 评估完成（P0）
- [ ] r3 @256 评估完成（P0）
- [ ] LaTeX Abstract 改为 planar netdep（P0）
- [ ] LaTeX §4.2 更新为 M1-M3 数据（P0）
- [ ] LaTeX §5 Limitations 诚实披露（P0）
- [ ] 生成 repair curves 图（P1）
- [ ] 生成 M1-M3 scaling 图（P1）
- [ ] 控制延迟测试（P1）
- [ ] 更新 ACTIVE 指针（如需要）
- [ ] git commit + tag 新验收包

**如果选策略 B（混合引用）**：

- [ ] LaTeX 写成"两层叙事"（包络主 + netdep 进阶）
- [ ] 明确告知审稿人：r2（旧包）是主验收，M1-M3（新包）是网络依赖域首次证明
- [ ] 不需要改 ACTIVE 指针

---

## 附录：文件完整性校验

```bash
cd acceptance/mf_planar_netdep_20261011
sha256sum -c SHA256SUMS.txt
```

**结果**：✅ 所有 59 个文件校验通过。
