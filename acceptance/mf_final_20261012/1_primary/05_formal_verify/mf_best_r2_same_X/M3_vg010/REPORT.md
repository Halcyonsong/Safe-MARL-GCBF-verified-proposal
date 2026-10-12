# M3 mf_best r2 · vg±0.1 (pose fixed, threat mid)

- **status: PROVEN_FULL**
- wall: 1704.002s
- checkpoint: `acceptance/mf_final_20261012/5_frozen_sources/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl`

## Results
| check | result |
|-------|--------|
| netdep | True envelope=False |
| encode | sat |
| samples CE | 0 ax[0.819,0.915] |
| root Marabou | timeout (900.619s) |
| cover | PROVEN_FULL |
| ax≥0.99 cf | sat_candidate |

### vg bisect leaves
- planar_netdep_reachable_pose_fixed_vg0.1_thr_mid_lo: **proven** t=150.102s
- planar_netdep_reachable_pose_fixed_vg0.1_thr_mid_hi: **proven** t=562.685s

ACTIVE untouched.
