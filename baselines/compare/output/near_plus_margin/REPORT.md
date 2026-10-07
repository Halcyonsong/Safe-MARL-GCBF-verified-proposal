# near_plus · margin + hard-corner (discriminative)

> Binary L2-safe% ≈100% for any clipped policy (physics envelope).  
> **Use margin \(d_2-R\)** and **hard corner** (low \(d_0\), high \(c_0\)) to compare.

- uniform n=50000 · hard n=20000 (d0≤0.18, c0≥0.22)
- seed=1234 · wall=6.8s · R=0.1

| policy | slice | n | L2% | margin mean | p10 | min | % margin<0.02 | ‖a‖∞ mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coast_a0 | uniform | 50000 | 100.0 | 0.0589 | 0.0296 | 0.0221 | 0.0 | 0.000 |
| mf_r2 | uniform | 50000 | 100.0 | 0.0525 | 0.0230 | 0.0131 | 5.0 | 0.709 |
| mappo_lag | uniform | 50000 | 100.0 | 0.0511 | 0.0218 | 0.0131 | 6.7 | 0.863 |
| coast_a0 | hard_corner | 20000 | 100.0 | 0.0404 | 0.0266 | 0.0221 | 0.0 | 0.000 |
| mf_r2 | hard_corner | 20000 | 100.0 | 0.0340 | 0.0197 | 0.0131 | 10.7 | 0.714 |
| mappo_lag | hard_corner | 20000 | 100.0 | 0.0326 | 0.0187 | 0.0132 | 15.0 | 0.866 |

## How to read

1. **L2%** near 100% on both slices → expected; not a ranking signal.
2. **Higher margin** = more clearance under the same radial two-step model.
3. **Hard corner** stresses the preventable band edge (still inside X).
4. **coast_a0** is the physics reference (often largest margin).
5. MF formal claim remains PROVEN_FULL on the whole box; this table is empirical contrast only.
6. DS / GCBF+ / HandCBF need graphs → not in this table.
