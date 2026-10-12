# MF residual repair

- base: `method/meanfield/output/distill/mf_distill_20261005_v5/seed0/meanfield_mlp512_student_step9000.pkl`
- init residual: `method/meanfield/output/repair/mf_residual_v5_netdep_r2_20261010/seed0/repair_last.pkl`
- residual best: `method/meanfield/output/repair/mf_residual_v5_netdep_r3_20261010/seed0/repair_best.pkl`
- gate: physics (open=3.0, close=1.0, mix=0.5)
- ce sources: `method/meanfield/output/repair/mf_residual_v5_netdep_r1_20261010/seed0/ce_baseline.json,method/meanfield/output/repair/mf_residual_v5_netdep_r1_20261010/seed0/ce_post_residual.json,method/meanfield/output/repair/mf_residual_v5_netdep_r2_20261010/seed0/ce_post_residual.json`
- lr=5e-05 idle_w=2.5 delta_l2=0.002 steps=800

## CE hit_rate on danger (two-step L2)

| phase | hit_on_danger | n_hit | n_danger | n_kept |
|---|---:|---:|---:|---:|
| prior_residual | 0.40% | 37 | 9346 | 37 |
| this_residual | 0.13% | 12 | 8968 | 12 |

## Task 32-epi (seed=1234)

| policy | success | safe | finish |
|---|---:|---:|---:|
| residual best | 82.03% | 95.70% | 85.16% |

Empirical repair only. Formal verification deferred until CE rate is low.
