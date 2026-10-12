# MF residual repair

- base: `method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl`
- init residual: `(random delta)`
- residual best: `method/meanfield/output/repair/mf_residual_v5_netdep_r1_20261010/seed0/repair_best.pkl`
- gate: physics (open=3.0, close=1.0, mix=0.5)
- ce sources: `method/meanfield/output/repair/mf_residual_v5_netdep_r1_20261010/seed0/ce_baseline.json`
- lr=0.0001 idle_w=1.5 delta_l2=0.001 steps=600

## CE hit_rate on danger (two-step L2)

| phase | hit_on_danger | n_hit | n_danger | n_kept |
|---|---:|---:|---:|---:|
| train_ce_pool | 6.32% | 1773 | 28063 | 1773 |
| this_residual | 0.68% | 64 | 9401 | 64 |

## Task 32-epi (seed=1234)

| policy | success | safe | finish |
|---|---:|---:|---:|
| residual best | 80.86% | 91.80% | 86.72% |

Empirical repair only. Formal verification deferred until CE rate is low.
