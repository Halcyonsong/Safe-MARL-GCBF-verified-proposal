# MF residual repair

- base: `method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl`
- residual best: `method/meanfield/output/repair/mf_residual_v5_20261005/seed0/repair_best.pkl`
- gate: physics (open=3.0, close=1.0, mix=0.5)

## CE hit_rate on danger (two-step L2)

| phase | hit_on_danger | n_hit | n_danger | n_kept |
|---|---:|---:|---:|---:|
| base | 5.95% | 542 | 9108 | 542 |
| residual | 1.31% | 128 | 9783 | 128 |

## Task 32-epi (seed=1234)

| policy | success | safe | finish |
|---|---:|---:|---:|
| residual best | 83.20% | 92.97% | 87.89% |

Empirical repair only. Formal verification deferred until CE rate is low.
