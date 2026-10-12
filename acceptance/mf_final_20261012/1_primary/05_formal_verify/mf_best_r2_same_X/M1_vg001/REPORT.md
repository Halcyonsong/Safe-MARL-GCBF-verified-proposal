# M1 mf_best r2 · vg±0.01 (pose fixed, threat mid)

- **status: PROVEN_FULL**
- wall: 64.954s
- checkpoint: `acceptance/mf_final_20261012/5_frozen_sources/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl`
- parent ladder: same X as netdep_r3 M1–M3

## Scope
pose=2 fixed; vx,vy,goal ∈[-0.01,0.01]; threat mid; X_rel netdep fringe; R=0.10

## Results
| check | result |
|-------|--------|
| netdep | True envelope=False |
| encode unconstrained | sat |
| samples 512 CE | 0 ax[0.874,0.882] |
| root Marabou | proven (27.712s) |
| cover | PROVEN_FULL |
| ax≥0.99 counterfactual | sat_candidate |

ACTIVE untouched.
