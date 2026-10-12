# Planar reachable_v1 · box→env · all baselines

- N=256, seed=20261010
- feat_contract: goal_xy≤comm, goal_dv=-v
- rel_err mean=8.31e-08, feat_l2 mean=2.083

| policy | n | planar fail% | ax_mean | repairable |
|---|---:|---:|---:|---:|
| `coast_a0` | 256 | 0.00% | 0.000 | — |
| `const_a=-1` | 256 | 85.55% | -1.000 | 100% |
| `const_a=+1` | 256 | 0.00% | 1.000 | — |
| `mf_base` | 256 | 0.00% | 0.253 | — |
| `mf_r2` | 256 | 0.00% | 0.557 | — |
| `netdep_r3` | 256 | 0.00% | 0.476 | — |
| `mappo_lag` | 256 | 2.34% | 0.454 | 100% |
| `gcbfplus` | 256 | 0.00% | 0.678 | — |
| `informarl` | 256 | 13.67% | 0.045 | 100% |
| `handcbf` | 128 | 0.00% | 0.255 | — |
| `oracle_11x11` | 256 | 0.00% | 1.000 | — |
