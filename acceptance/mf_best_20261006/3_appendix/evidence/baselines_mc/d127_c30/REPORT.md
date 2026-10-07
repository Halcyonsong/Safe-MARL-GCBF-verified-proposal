# Candidate box MC · d≥0.127, |c|≤0.3

> d2_lo=0.1000 · net_dep=False · n=50000 · seed=1234
> mf_ckpt=`method/meanfield/output/repair/mf_residual_v5_midF_r3_20261006/seed0/repair_best.pkl`

| policy | n | fail% | n_fail | L2-safe% | margin min | ‖a‖∞ |
|---|---:|---:|---:|---:|---:|---:|
| coast_a0 | 50000 | 0.000 | 0 | 100.00 | 0.0091 | 0.000 |
| mf_cand | 50000 | 0.000 | 0 | 100.00 | 0.0001 | 0.890 |
| mf_r2 | 50000 | 0.000 | 0 | 100.00 | 0.0001 | 0.898 |
| mappo_lag | 50000 | 0.000 | 0 | 100.00 | 0.0001 | 0.977 |
| gcbfplus | — | N/A | — | — | — | — |
| informarl | — | N/A | — | — | — | — |
