# 主验收证据树（`1_primary/`）

> 包入口见上级 [`../README.md`](../README.md)。  
> 本目录 = residual **r2** · L2 `near_plus` · **PROVEN_FULL** 的自洽证据副本。

## 一页结论

| 项 | 内容 |
|---|---|
| **部署策略** | residual r2（physics-gated δ + 冻结 v5 base MLP512） |
| **形式化** | \(\forall x \in X_{\mathrm{near\_plus}}:\ \mathrm{TwoStepRadialL2Safe}(\mathrm{Deploy}(x); R=0.10)\) → **PROVEN_FULL** |
| **声明盒** | min_d∈[0.14, 0.40]，\|close\|≤0.30 |
| **任务 @32** | success 82.0% / safe 96.5% / finish 84.0% |
| **L3 full** | 附录 VIOLATED — **不**作主验收 |

证书：[`00_acceptance/ACCEPTANCE_CERTIFICATE.md`](00_acceptance/ACCEPTANCE_CERTIFICATE.md)

## 树结构

```text
00_acceptance/     证书 · ACTIVE 快照 · 指标摘要
01_models/         base_v5 · residual_r1 · residual_r2★
02_train_logs/     distill / repair_r1 / repair_r2
03_task_eval/      @32/@64/@256
04_ce_repair/      r1 · r2
05_formal_verify/  L0 · L1 · L2★ · L3（附录）
06_docs/           打包时文档快照（过时；以仓库 docs/ 为准）
07_source_snapshot/
```

## 复现

```bash
bash method/meanfield/scripts/run_mf_verify_full.sh \
  acceptance/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl \
  near_plus
```

校验请在包根执行：`cd .. && sha256sum -c SHA256SUMS.txt`。
