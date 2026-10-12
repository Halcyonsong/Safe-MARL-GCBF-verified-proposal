# M2 mf_best r2 · vg±0.05 (pose fixed, threat mid)

- **status: PROVEN_FULL**
- wall: 713.053s
- checkpoint: `acceptance/mf_final_20261012/5_frozen_sources/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl`

## Results
| check | result |
|-------|--------|
| netdep | True envelope=False |
| encode | sat |
| samples CE | 0 ax[0.850,0.897] |
| root Marabou | timeout (600.63s) |
| cover | PROVEN_FULL |
| ax≥0.99 cf | sat_candidate |

### vg bisect leaves
- planar_netdep_reachable_pose_fixed_vg0.05_thr_mid_lo: **proven** t=18.79s
- planar_netdep_reachable_pose_fixed_vg0.05_thr_mid_hi: **proven** t=43.909s

ACTIVE untouched.
