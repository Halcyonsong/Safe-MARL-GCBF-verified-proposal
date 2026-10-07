# 说明（叙事 · 边界 · 附录结论）

> 叙事与边界在此。  
> **修复轨迹 + @32/64/256 + 延展 + 基线全表** → [`EVAL_AND_REPAIR.md`](EVAL_AND_REPAIR.md)  
> **全部关键数据 / 附录** → [`RECORD.md`](RECORD.md)

## 1. 主验收定义

```text
BEST = residual r2
     + L2 near_plus PROVEN_FULL
     + task @32 ≈ 82.0% / 96.5% / 84.0%
```

指针（打包时）：`ACTIVE_MF_RESIDUAL` → r2 best；`ACTIVE_MF_VERIFY` → near_plus PROVEN。

## 2. 主文可写 / 不可写

**可写**

1. r2 在 \(X_{\mathrm{near\_plus}}\)（d≥0.14，\|c\|≤0.30）上两步径向 L2（R=0.10）**PROVEN_FULL**。  
2. 这是有界盒上的 **∀ 证书**；MAPPO / GCBF+ / InforMARL **无同级证书**。  
3. 任务评测支持部署可用性（见 `1_primary/03_task_eval/`）。  
4. （附录）更宽**物理包络**盒上亦可 PROVEN；属包络性质，**不**构成相对基线的 binary-L2 分离。

**不可写**

1. 「mid-F / 网络依赖带已形式化证满」——**未证满**。  
2. 「主验收盒上基线必有两步 L2 反例」——包络下 binary fail≈0。  
3. 「coast / mf_base 是论文基线」——它们是消融。

## 3. 附录结论（Oct-6）

| 主题 | 结论 |
|---|---|
| mid-F 修复 | CE 204→177（r3）/173（r4）；仍 VIOLATED；near_plus 未破 |
| 旁支折中 ckpt | **midF_r3**（任务比 r4 稳）→ `3_appendix/sidebranch_midF_r3/` |
| 包络加宽 | `d≥0.127/0.30`、`0.126/0.28` 等 PROVEN（r2 与 midF_r3 皆可） |
| 包络上基线 | 产品盒 MC **全员 0 fail** → 不宜作「基线反例」主 claim |
| mid-F CE-fringe | MAPPO 7 / GCBF+ 4 / InforMARL 0（n=200）→ 仅经验附录 |

物理：\|a\|∞≤1 时 \(d2\_lo = d_{\min}-0.06\,c_{\mathrm{hi}}-0.009\)。  
\(d2\_lo ≥ R\) → 任意 clip 策略径向安全（包络）；\(d2\_lo < R\) → 才可能「网络依赖 + 基线 fail」。

## 4. 后续（本包外）

目标「同盒 MF PROVEN + 基线反例」**尚未达成**。若继续：

1. 专修网络依赖发丝 / mid-F  
2. **同一界**重测三基线  
3. 显式 lock 后建**新**包（勿覆盖本目录与 `acceptance/mf_best_20261006/1_primary/`）

## 5. 提案备忘：Ego 输入维冗余（未改主验收）

当前 44-D 中 ego 为 8 维：`[px, py, vx, vy, goal_dx, goal_dy, goal_dvx, goal_dvy]`。

| 维 | 现状判断 | 备注 |
|---|---|---|
| `goal_dx, goal_dy` | **必要** | 相对目标二维向量，**有方向**（不是标量距离） |
| `vx, vy` | **必要** | 自身动力学 |
| `px, py` | **冗余（候选删除）** | 威胁/L2 全由相对几何；绝对坐标对避障非必要 |
| `goal_dvx, goal_dvy` | **冗余（候选删除）** | 本任务目标静止 ⇒ \(\Delta v_{\mathrm{goal}}\approx -v\)，与 `vx,vy` 重复 |

**决议（本包）**

- 只写入提案 / 备忘，**不**改 r2 特征、不重训、不动 `ACTIVE_*` / `acceptance/mf_best_20261006/1_primary/`。
- 完整梳理特征与验证边界后，再考虑**新开对照版**（更干净 ego，如 `[vx,vy,goal_dx,goal_dy]` + 同象限威胁），单独蒸馏 / 修复 / 验证；与主验收并行对照，勿覆盖本目录。
- 主文可诚实写：现 8 维 ego 含状态拷贝冗余；L2 主性质不依赖 `px,py` 与 \(\Delta v_{\mathrm{goal}}\)。
