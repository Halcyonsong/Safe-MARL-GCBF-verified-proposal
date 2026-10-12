# M3 enlarge vg±0.10 (pose fixed, threat mid)

- **status: PROVEN_FULL**
- wall: 1523.951s
- parent ladder: M1 vg±0.01 → M2 vg±0.05 → **M3 vg±0.10**

## Scope
pose=2 fixed; vx,vy,goal ∈[-0.10,0.10]; threat mid; X_rel unchanged; R=0.10

## Results
| check | result |
|-------|--------|
| netdep | True envelope=False |
| encode unconstrained | sat |
| samples 512 CE | 0 ax[0.650,0.747] |
| root Marabou | timeout (900.332s) |
| cover | PROVEN_FULL |
| ax≥0.99 counterfactual | sat_candidate |
