# CEGAR R2 Snapshot — V2d residual post-repair

- generated: 2026-10-04T23:47:39
- residual_in (V2d r1 best): `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d/repair_best.pkl`
- residual_out (V2d r2 best): `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/repair_r2_best.pkl`
- base: seed0 redistill DeepSet
- protocol: declared L2 two-step preventable-D × {AA,AO,(1,1)}
- formal note: residual path = **dense sampling diagnosis only** (not Marabou proven of base+g·δ)

## Day0 — post-V2d sampling verify (4k/bucket on D)

| bucket | verdict | hit_rate_on_D |
|---|---|---:|
| l2_aa_near | **violated** | 18.40% |
| l2_ao_near | **violated** | 31.47% |
| l2_11_near | **violated** | 28.75% |

→ all 3 buckets **violated** → open CEGAR round-2.

## Day1 — residual CE + warm-start short repair

- CE: 3075 usable (AA765 + AO775 + dual1535), min_intervention
- steps=600, dual_ce_weight=2.0, lr=6e-5, floors success≥0.85 safe≥0.92 max_aa≤0.18 max_ao≤0.32
- best eligible @ step **500**: short-eval success **90.62%**, mean_D_hit **19.75%** (AA10.25 / AO25.37 / 11 23.62)
- n_eligible evals: 9/12

## Day2 — re-verify + 32-epi + D-hit (seed123)

### Sampling verify (4k) r1 → r2

| bucket | V2d r1 | V2d r2 | Δ |
|---|---:|---:|---:|
| l2_aa_near | 18.40% (violated) | 10.45% (violated) | -7.95pt |
| l2_ao_near | 31.47% (violated) | 24.60% (violated) | -6.87pt |
| l2_11_near | 28.75% (violated) | 22.65% (violated) | -6.10pt |

### Standard D-hit (2k, seed123) vs prior V2d

| | AA | AO | (1,1) | mean |
|---|---:|---:|---:|---:|
| V2d r1 (prior) | 18.65% | 32.85% | 28.25% | 26.58% |
| V2d r2 (CEGAR) | 10.90% | 24.90% | 22.05% | 19.28% |
| GCBF+ GNN (ref) | ~22.7% | ~33.6% | ~26.8% | ~27.7% |
| seed0 base / frozen (ref) | ~36% | ~43% | ~38% | ~39% |

### Task 32-epi (eval_seed=1234)

- V2d r1 (prior): see phase2_seed0base validate (~88.3% success)
- V2d r2: `checkpoint=method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/repair_r2_best.pkl architecture=deepset_residual_gate base=method/deepset/output/stage1/redistill/stage1_redistill_20261004_183322/seed0/deepset_actor512_student_best.pkl gate_mode=hybrid episodes=32 reward=-12.929 cost=1.039 safe_rate=93.359% finish_rate=94.141% success_rate=87.891%`

## Interpretation

1. **Safety (claim slice):** CEGAR r2 further cut D-hit — AA ~18.7→**10.9%**, AO ~32.9→**24.9%**, (1,1) ~28.3→**22.1%** (seed123 2k). Mean ~26.6→**~19.3%**. All three now **clearly below GNN** on this protocol.
2. **Formal claim:** still **violated** on dense 4k sampling (concrete CEs remain). `sampling_clear` / Marabou **proven** not achieved. Residual graph still not in Marabou ONNX path.
3. **Task:** 32-epi success **87.9%** (safe 93.4%) — holds ≥0.85 floor; ~0.4pt below prior V2d ~88.3%; still below GNN ~91%.
4. **Next if chasing proven:** another CEGAR, residual Marabou encoding, or per-bucket stronger repair — not guaranteed in finite rounds.

## Artifacts

- `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/verify_r1_residual_sampling.json`
- `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/ce_r2.json`
- `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/repair_r2_best.pkl`
- `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/repair_r2.json`
- `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/verify_r2_residual_sampling.json`
- `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/validate_32epi.txt`
- `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/d_hit_seed123.json`
- `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/MANIFEST.json`
