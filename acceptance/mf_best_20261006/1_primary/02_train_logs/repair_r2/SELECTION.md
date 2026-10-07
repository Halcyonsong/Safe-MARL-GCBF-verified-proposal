# MF residual repair — round 2

- base: `method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl`
- init residual (r1): `method/meanfield/output/repair/mf_residual_v5_20261005/seed0/repair_best.pkl`
- residual best (r2): `method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl`
- gate: physics (open=3.0, close=1.0, mix=0.5)
- warm-start δ from r1; lr=5e-5, idle_w=2.5, delta_l2=2e-3, steps=500
- CE pool: merge(r1 post + fresh r1-policy CEs + base baseline), 262 labeled-safe / 736 raw

## CE hit_rate on danger (two-step L2)

| phase | policy | hit_on_danger | n_hit | n_danger |
|---|---|---:|---:|---:|
| base (v5 step9000) | base | **5.95%** | 542 | 9108 |
| residual r1 | r1 | **1.31%** | 128 | 9783 |
| r1 recheck (seed456) | r1 | **0.77%** | 66 | 8535 |
| **residual r2** | **r2** | **0.47%** | **33** | **7089** |

Cumulative vs base: **5.95% → 0.47% ≈ 12.7× drop**.

## Task 32-epi (seed=1234)

| policy | success | safe | finish |
|---|---:|---:|---:|
| base step9000 (distill ref) | ~77.7% | ~80.9% | ~94.1% |
| residual r1 | 83.20% | 92.97% | 87.89% |
| **residual r2 best** | **82.03%** | **96.48%** | **83.98%** |

Notes:
- r2 further cuts sampling collisions and raises safe (~96.5%).
- finish slipped further (~84%); success roughly flat vs r1.
- Short-eval best during train: step200 success 89.1% / safe 96.9% / finish 92.2% (8-epi, noisier).

Empirical repair only. Formal verification: CE now **sub-1%** on danger — **eligible to start MF box PoC**, not a full-pass claim yet.
