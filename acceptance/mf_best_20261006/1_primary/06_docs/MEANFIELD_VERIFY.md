# MF 形式化验证约定（已定稿）

> 与 DeepSet 分桶验证 **不同**。DeepSet `method/deepset/output/phase2/ACTIVE_*` 不改。

## 1. 一句定义

```text
∀ x ∈ X ⊂ R^{44}:   TwoStepRadialL2Safe( Deploy(x); R )
```

| 部分 | 含义 |
|---|---|
| **查询域 X** | 网络输入 **每一个自由维** 的区间积（ego 速度、目标、各象限 min_d / close_wmean / inv_ttc …） |
| **全覆盖** | 在 X 上 IBP + Marabou 分治，目标 **PROVEN_FULL**；剩余真实 CE **全部收集** |
| **性质** | **两步** L2（地平线简化）；径向过近似 \(d_2=d_0-\alpha c_0-\beta\|a\|_\infty \ge R\) |
| **不做** | DeepSet (k,m) 桶；主验收 AA/AO 分表 |

## 2. 与 DeepSet 对照

| | DeepSet | Mean-field |
|---|---|---|
| 输入 | 变长实体槽 | 固定 44 维特征 |
| 域简化 | 必须 (k,m) 分桶 | **不按桶**；按 44 维切盒 |
| 性质简化 | 两步 L2 | **同样两步**（不是取消两步） |
| 主指标 | (1,0)/(0,1) proven% | **PROVEN_FULL / PARTIAL / CE 列表** |

## 3. 性质细节

```text
d0, c0  = 仅对占用象限 (n≥0.5) 取 min_d / max(close_max,0)
α       = n_steps · dt = 0.06
β       = (dt²/mass)·n(n-1)/2 = 0.009
d2      = d0 − α c0 − β |a|_∞
Safe    ⇔ d2 ≥ R   (默认 R=COLLISION_AA=0.10)
```

- 空象限与特征提取器一致：不参与 d0/c0。
- 径向头对头过近似：**证明安全是充分的**；SAT 可能虚警 → 真策略再检。
- Marabou 侧可用「全体象限 min/max」作更松 d0/c0（UNSAT 仍 sound）。

## 4. 分层声明盒（可避免带）

避免把「a=0 已撞」点塞进声明域（否则会淹没 CE）。

| 层 | mode | min_d | \|close\| | inv_ttc | Q0 | 角色 | 最坏 \(d_2\)（\|a\|_∞=1） |
|---|---|---|---|---|---|---|---|
| L0 | `smoke` | ≈0.25 点邻域 | — | — | 占用 | 工具链自检 | — |
| L1 | `near` | [0.15, 0.35] | ≤0.25 | ≤8 | n∈[1,4] | 基线可避免带 | **0.126** |
| **L2** | **`near_plus`** | **[0.14, 0.40]** | **≤0.30** | **≤8** | **n∈[1,4]** | **有界主验收** | **0.113** |
| L3 | `full` | [0.12, 0.50] | ≤0.50 | ≤20 | n∈[1,8] | 宽盒附录（不冲 PROVEN） | **0.081 \< R** |

共用（near / near_plus）：ego \(v\in[\pm0.5]\)，goal \(\in[\pm2]\)，R=**0.10**，agent/obs 其它 n∈[0,4]。

校验：

```text
near:      d0−α c0 − β ≥ 0.15 − 0.06·0.25 − 0.009 = 0.126 > 0.10
near_plus: d0−α c0 − β ≥ 0.14 − 0.06·0.30 − 0.009 = 0.113 > 0.10
full lo:   d0−α c0 − β ≥ 0.12 − 0.06·0.50 − 0.009 = 0.081 < 0.10  → 不主验 PROVEN
```

| 项 | 默认 | 说明 |
|---|---|---|
| R | **0.10** | 更强可 `--radius 0.05` |
| 分治 | max_splits≈600–800，depth≈14，marabou_timeout≈25s | 到限 → PARTIAL + 采 CE |
| Marabou 触发 | 性质相关维宽度 ≤ `marabou_max_width`（默认 0.08） | 大盒先 IBP+切分 |
| wall | 900–1200s / claim | 硬上限 |

## 5. 代码入口

```bash
bash scripts/run_mf_verify_full.sh $(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt) smoke
bash scripts/run_mf_verify_full.sh $(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt) near
bash scripts/run_mf_verify_full.sh $(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt) near_plus
# 可选覆盖：D_MIN=0.18 D_MAX=0.40 CLOSE_HI=0.2 RADIUS=0.10 bash scripts/...
```

| 文件 | 作用 |
|---|---|
| `stage_mf/domain.py` | 44 维盒 + claim manifest（含 near_plus） |
| `stage_mf/property_l2.py` | 两步径向性质 + IBP |
| `stage_mf/marabou_mf.py` | Marabou：π + d2\<R 查询 |
| `stage_mf/verify_full.py` | 分治 + 收集全部 CE |
| `method/meanfield/output/verify/` | 产物；`ACTIVE_MF_VERIFY.txt` 指向最新 |

## 6. 状态语义

| 状态 | 含义 |
|---|---|
| **PROVEN_FULL** | 声明 X 每个叶子 proven，0 真实 CE，0 unknown |
| **PARTIAL** | 资源内未盖满；unknown 上已采样 CE |
| **VIOLATED** | ≥1 条真实策略 CE → `counterexamples_all.json` |

## 7. 当前结果（r2 residual，2026-10-05）

对象：`ACTIVE_MF_RESIDUAL` = r2 `repair_best.pkl`。

| mode | 声明盒 | 结果 | 方法 / 产物 |
|---|---|---|---|
| smoke | 点邻域 | **PROVEN_FULL** | IBP；`verify/mf_feat44_smoke` |
| near | min_d∈[0.15,0.35], \|c\|≤0.25 | **PROVEN_FULL** | 整盒 IBP d2_lo=0.126；`verify/mf_feat44_near_r2` |
| **near_plus** | **min_d∈[0.14,0.40], \|c\|≤0.30** | **PROVEN_FULL** | **整盒 IBP d2_lo=0.113**；`verify/mf_feat44_near_plus_r2`（**ACTIVE 主验收**） |
| full (L3 附录) | min_d∈[0.12,0.50], \|c\|≤0.50 | **VIOLATED_WITH_CES** | 物理 d2_lo=0.081\<R；11260 CE；`verify/mf_feat44_full_r2` + `APPENDIX_L3.md` |

**L3 CE 要点（支持 L2 切界）：**

| 项 | 值 |
|---|---|
| CE 数 | 11260（max_splits 用尽后 unknown 采样） |
| d0 分布 | **全部 ∈ [0.12, 0.14)**（med≈0.126） |
| c0 分布 | med≈0.44；**≈95.5% 有 c0>0.30** |
| 落在 near_plus 外 (d 或 c) | **100%** |
| 物理 coast-doomed \(d_0-\alpha c_0<R\) | ≈41% |

**正确读法：**

```text
near_plus PROVEN_FULL  =  有界主验收通过（声明可避免 44 维盒全覆盖）
L3 full VIOLATED       =  预期：宽盒含 coast-risk 下缘；CE 全在 L2 切界外
                       ≠  L2 主验收失败
                       ≠  无限时域部署零间隙
全量验证能力           =  架构上对任意声明盒可 IBP+B&B+Marabou 完备分治（附录）
```

- `ACTIVE_MF_VERIFY.txt` → `method/meanfield/output/verify/mf_feat44_near_plus_r2`（仅 PROVEN_FULL 提升）  
- `LATEST_MF_VERIFY.txt` / `LATEST_MF_VERIFY_FULL.txt` → 最近一次 / L3 附录

## 8. 仍可选

- R 用 0.10 还是 0.05
- full 宽盒 CE 收获（附录，不冲 PROVEN）
- 径向 → 以后加平面几何辅助

**主设计已定：44 维自由维全覆盖 + 两步性质 + 收齐 CE；有界主验收 = near_plus。**
