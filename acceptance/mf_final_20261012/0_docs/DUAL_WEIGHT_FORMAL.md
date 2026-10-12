# 双权重形式化（r2 推荐 · r3 侧枝）

**结论：两权重在同一 planar X 上 M1–M3 均为 PROVEN_FULL；论文 Deploy+形式化挂 mf_best r2。**

| | **mf_best r2（推荐）** | netdep r3（侧枝） |
|--|------------------------|-------------------|
| 形式化 M1–M3 | **PROVEN_FULL** | PROVEN_FULL |
| wall M1/M2/M3 | 65s / 713s / **1704s** | 53s / 516s / 1524s |
| @32 suc/safe/fin/cost | **82.0 / 96.5 / 84.0 / 0.31** | 82.0 / 95.7 / 85.2 / 0.69 |
| 权重 | `1_primary/01_models/residual_mf_best_r2_ref/` | `…/residual_netdep_r3/repair_manual_pick.pkl` |
| 证书 | `05_formal_verify/mf_best_r2_same_X/` | `05_formal_verify/M{1,2,3}_*` |

## Claim 盒（公共）

- 平面两步 \(\|p_2\|_2\ge R=0.10\)，**网络依赖**（envelope_any_a_safe=False）
- \(X_{\mathrm{rel}}\)：dx∈[0.118,0.122], |dy|≤0.04, dvx∈[−0.30,−0.25], |dvy|≤0.05
- pose px=py=2 · threat 中点 · vg 阶梯 ±0.01/0.05/0.10 · goal_dv≡−v
- McCormick 已修；pre-fix 作废见 `3_appendix/INVALIDATED_pre_mccormick.md`

## 选用 r2 的理由

1. 同盒形式化已过（与 r3 同性质）
2. 经验 safe 更高、cost 更低
3. 避免「任务挂 r2、证书挂 r3」双权重叙事

侧枝 r3 仅作 CE 修复链对照，**不**继承 r2 证书除非重验。

## 证据路径

- 摘要：`1_primary/00_acceptance/METRICS_SUMMARY.md`
- 任务：`0_docs/EVAL_AND_REPAIR.md` · `1_primary/03_task_eval/`
- 证书细节：`1_primary/05_formal_verify/INDEX.md`
