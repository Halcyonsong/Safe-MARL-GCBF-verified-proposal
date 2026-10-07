# method/meanfield/output/

Mean-field parallel track (isolated from DeepSet `method/deepset/output/phase2/`).

## Active pointers

| 指针 | 含义 |
|---|---|
| `ACTIVE_MF_BASE.txt` | v5 **step9000** |
| `ACTIVE_MF_RESIDUAL.txt` | **r2** physics-gated residual best |
| `ACTIVE_MF_VERIFY.txt` | **near_plus** PROVEN_FULL（有界主验收） |

DeepSet `method/deepset/output/phase2/ACTIVE_*` **未改**.

## Distill (base)

| Run | Feature | success@32 / @64 |
|---|---|---|
| `distill/mf_distill_20261005_v5` step9000 | v5, dim=44 | **77.7% / 78.7%** |

## Repair progression

| stage | hit_on_danger | success@32 | safe@32 | finish@32 |
|---|---:|---:|---:|---:|
| base | 5.95% | ~77.7% | ~80.9% | ~94% |
| r1 residual | 1.31% | 83.2% | 93.0% | 87.9% |
| **r2 residual** | **0.47%** | **82.0%** | **96.5%** | **84.0%** |

```text
method/meanfield/output/repair/mf_residual_v5_20261005/     # r1
method/meanfield/output/repair/mf_residual_v5_r2_20261005/  # r2 (ACTIVE)
```

Policy: `a = clip(base + g_phys * δ)` — base frozen MLP512; only δ (128-h) trained.

## Formal verification (定稿)

详见 [`docs/MEANFIELD_VERIFY.md`](../docs/MEANFIELD_VERIFY.md)。

```text
∀ x ∈ X ⊂ R^{44}:  TwoStepRadialL2Safe(Deploy(x); R)

域  = 每个自由维（速度/目标/象限 min_d·wmean·inv_ttc…）
性质 = 两步 L2（简化地平线）
不做 = DeepSet (k,m) 桶、主表 AA/AO
```

```bash
bash method/meanfield/scripts/run_mf_verify_full.sh "$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt)" smoke
bash method/meanfield/scripts/run_mf_verify_full.sh "$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt)" near
bash method/meanfield/scripts/run_mf_verify_full.sh "$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt)" near_plus
```

| 产物 | 含义 |
|---|---|
| `verify/mf_feat44_*/REPORT.md` | PROVEN_FULL / PARTIAL / VIOLATED |
| `verify/.../counterexamples_all.json` | **全部**真实 CE |
| `verify/.../*_manifest.json` | 44 维 lo/hi 清单 |
| `ACTIVE_MF_VERIFY.txt` | 最新 verify 目录 |

### 当前结果（r2 residual）

| mode | 声明盒 | 结果 | 方法 |
|---|---|---|---|
| smoke | 点邻域 | **PROVEN_FULL** | IBP |
| near | min_d∈[0.15,0.35], \|close\|≤0.25, Q0 occupied | **PROVEN_FULL** | IBP（d2_lo=0.126≥R） |
| **near_plus** | **min_d∈[0.14,0.40], \|close\|≤0.30, Q0 occupied** | **PROVEN_FULL** | **IBP（d2_lo=0.113≥R）** ← **有界主验收** |
| full (L3 附录) | min_d∈[0.12,0.50], \|close\|≤0.50 | **VIOLATED**（11260 CE） | d2_lo=0.081\<R；CE 全在 near_plus 外 |

- `ACTIVE_MF_VERIFY.txt` → `verify/mf_feat44_near_plus_r2`  
- L3 附录：`verify/mf_feat44_full_r2/APPENDIX_L3.md`

## 验收交付包（根目录）

链路以 **Mean-field L2 near_plus PROVEN_FULL** 作为形式化主验收。  
归档副本（模型 / 日志 / 测评 / 形式化 / 证书）：

```text
acceptance/mf_best_20261006/1_primary/
  README.md
  00_acceptance/ACCEPTANCE_CERTIFICATE.md   ← PASS 证书
  01_models/residual_r2/repair_best.pkl     ← 部署网络
  05_formal_verify/L2_near_plus_PRIMARY/    ← 主 PROVEN_FULL
```

## Launch

```bash
bash method/meanfield/scripts/run_mf_residual_repair.sh
bash method/meanfield/scripts/run_mf_verify_full.sh \
  "$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt)" near_plus
```
