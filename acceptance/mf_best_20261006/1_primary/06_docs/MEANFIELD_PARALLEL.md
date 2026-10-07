# Mean-field 并行探究约定

> 与 DeepSet 主路径 **并行**，不覆盖 Phase-1/Phase-2 DeepSet 产物与 active 指针。  
> DeepSet 完整地图：[`DEEPSET_PATH_CATALOG.md`](DEEPSET_PATH_CATALOG.md)。

---

## 1. 目标

验证友好架构对照实验：

```text
input(~22) → Dense(256)×3 → ReLU → Dense(2)
ego(8) + nearest_agent(4) + nearest_obs(4) + sum_agents(2) + sum_obs(2) + counts(2)
```

- **不分 (k,m) 桶**；一个输入盒 + CROWN / Marabou 分治  
- ReLU 数约 768（vs DeepSet (1,1) ≈4608）  
- 任务性能待测；success 目标 ≥85% 可继续；&lt;80% 启用扩展精确槽预案

---

## 2. 目录布局（已预留）

```text
stage_mf/                         # mean-field 专用代码（新建，勿改 stage1/student.py）
  __init__.py
  student.py                      # MeanFieldStudent + extract_meanfield_sets
  train.py                        # 蒸馏（可 thin-wrap stage1/train 逻辑）
  validate.py
  README.md

method/meanfield/config/
  default.yaml
  mlp256.yaml                     # hidden=256 主配置

scripts/
  run_mf_distill.sh               # 从冻结 GCBF+ 蒸馏 mean-field
  run_mf_eval.sh                  # 32-epi + 可选 D-hit
  # 后续: run_mf_crown.sh / run_mf_marabou.sh

method/meanfield/output/                   # ★ 全部 MF 产物（与 phase2 隔离）
  README.md
  distill/<run_id>/seed*/
  verify/<run_id>/
  compare/                        # 与 DeepSet 对照表

docs/
  MEANFIELD_PARALLEL.md           # 本文件
  DEEPSET_PATH_CATALOG.md         # DeepSet 冻结索引
```

**可复用（只读 import，不复制改坏）：**

| 模块 | 用途 |
|---|---|
| `stage2/dynamics.py` | 两步动力学、碰撞半径 |
| `stage2/l2_domain.py` | preventable-D |
| `stage2/collision.py` | AA/AO 评估 |
| `stage3/metrics_io.py` | JSON/CSV |
| `core/paths.py` / `core/gcbfplus` | 环境与教师 |
| `scripts/cuda_env.sh` | 环境 |
| 教师权重 | `logs/.../seed0_20261001232611` |

**不要复用/改写：**

- `stage1/student.py`（保持 DeepSet）  
- `stage2/encode.py` / `marabou_build.py`（DeepSet 分桶专用；MF 另写 `stage_mf/encode.py` 等）  
- `method/deepset/output/phase2/ACTIVE_*` 指针  
- `method/deepset/config/stage1/actor512.yaml`

---

## 3. 与 DeepSet 对照协议

同一评估协议（便于公平对比）：

| 项 | 值 |
|---|---|
| 环境 | DoubleIntegrator 8 agents，与 Stage0 相同 |
| 任务 | 32-epi primary，`eval_seed=1234`；64 optional |
| D-hit | `seed=123`，2000/桶（若 MF 定义了 AA/AO/(1,1) 近域） |
| 教师 | 同一冻结 GCBF+ seed0 |
| DeepSet 对照 base | `method/deepset/output/phase2/ACTIVE_BASE.txt` → redistill seed0 |
| DeepSet 对照 residual | `method/deepset/output/phase2/ACTIVE_RESIDUAL.txt` → V2d_r2 |

对比报告输出到：

```text
method/meanfield/output/compare/<date>_mf_vs_deepset.md
```

建议表头：

```text
| 策略 | success@32 | D-hit mean | ReLU | 分桶 | CROWN proven% | Marabou 单查询 | 备注 |
| DeepSet seed0 | 89.5% | 39% | ~4608 | (k,m) | 低 | 难 | catalog §5 |
| DeepSet V2d_r2 | 87.9% | 19.3% | base+gate+δ | 同左 | 局部 PoC | 局部 | residual |
| Mean-field mlp256 | ? | ? | 768 | 无 | ? | ? | 本 track |
```

---

## 4. 落地顺序（建议）

1. **实现** `stage_mf/student.py`：`MeanFieldStudent` + `extract_meanfield_sets`  
2. **蒸馏** `run_mf_distill.sh` → `method/meanfield/output/distill/<id>/`  
3. **任务验** 32-epi；若 success &lt; 82% → 最近 2 邻 + 2 障预案  
4. **验证** CROWN（`auto_LiRPA` 或自写）→ 未知子盒 Marabou  
5. **写对照报告**；**不**自动替换 DeepSet active 指针

---

## 5. 禁止事项

- 用 mean-field 结果覆盖 `method/deepset/output/stage1/frozen/` 或 `method/deepset/output/phase2/LATEST` 下 DeepSet ckpt  
- 修改 `ACTIVE_BASE.txt` / `ACTIVE_RESIDUAL.txt` 指向 MF（除非明确切换主方法并更新 INDEX）  
- 在 `stage2/verify_l2.py` 里混入 MF 分支导致 DeepSet 复现失败  
- 声称 MF「全域 proven」而未写明盒定义与松弛（sum 盒过近似）

---

## 6. 状态（2026-10-05 更新）

| 项 | 状态 |
|---|---|
| 目录骨架 | ✅ `stage_mf/` `method/meanfield/config/` `method/meanfield/output/` |
| 特征 | ✅ **v5** 4-quad + wmean + inv_ttc，**dim=44** |
| Base 蒸馏 | ✅ v5 step9000：success@32 **77.7%** / @64 **78.7%** |
| Residual | ✅ r2 physics-gate：hit_on_danger **0.47%**，safe@32 **96.5%** |
| **形式化** | ✅ 44-D 特征盒 + 两步径向 L2；**near IBP PROVEN_FULL**（r2） |
| DeepSet 隔离 | ✅ `phase2/ACTIVE_*` 未改 |

### 6.1 验证哲学（与 DeepSet 的关键区别）

定稿见 [`MEANFIELD_VERIFY.md`](MEANFIELD_VERIFY.md)。

```text
DeepSet:  变长图 → 必须按 (k,m) 桶分别验证；主看 (1,0)/(0,1) 指标
Mean-field:
  ∀ x ∈ X ⊂ R^{44}:  TwoStepRadialL2Safe(Deploy(x); R)
  域 = 每个自由维区间积（不按桶、不主分 AA/AO）
  性质 = 两步径向 L2（地平线简化）
  IBP + Marabou 分治 → PROVEN_FULL；CE 全部收集
```

| 模块 | 路径 |
|---|---|
| 域 / claim 盒 | `stage_mf/domain.py` |
| 两步性质 + IBP | `stage_mf/property_l2.py` |
| 权重编码 | `stage_mf/encode.py` |
| 动作 IBP | `stage_mf/ibp_mf.py` |
| Marabou（base+gate+δ + d2\<R） | `stage_mf/marabou_mf.py` |
| 全量分治入口 | `stage_mf/verify_full.py` |
| 启动 | `scripts/run_mf_verify_full.sh [ckpt] [smoke\|near\|full]` |

产出：`method/meanfield/output/verify/mf_feat44_*/`（`REPORT.md`、`counterexamples_all.json`、`*_manifest.json`）。

**PROVEN_FULL** = 声明盒每个叶子 proven、0 真实 CE、0 unknown。  
**PARTIAL** = 有 unknown 子盒。  
**VIOLATED** = 已收集真实策略 CE。

Soundness：径向过近似对平面碰撞是**充分**的；特征盒是可实现场景的过近似；UNSAT 对真实对成立。  
当前 **near** 默认盒为可避免带（min_d∈[0.15,0.35]，|close|≤0.25，Q0 占用）——排除 a=0 已撞点。
详见 `method/meanfield/output/ACTIVE_MF_*.txt` 与 `method/meanfield/output/verify/`。
