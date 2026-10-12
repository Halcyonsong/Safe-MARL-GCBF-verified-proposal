# M2 enlarge vg±0.05 (pose fixed, threat mid)

- **status: PROVEN_FULL**
- wall: 515.471s
- encoding: McCormick **fixed**
- parent: `min_M1_vg001` (vg±0.01 PROVEN_FULL)

## Scope

| block | bounds |
|-------|--------|
| X_rel | same netdep fringe as M1 |
| px, py | **2.0 fixed** |
| vx,vy,goal_xy | **[-0.05, 0.05]** |
| threat | midpoint |
| R | 0.10 |

## Results

| check | result |
|-------|--------|
| netdep / not envelope | **True** / envelope=False |
| unconstrained encode | **sat** |
| samples 512 CE | **0** ax∈[0.688,0.734] |
| root Marabou ax≥≈0 | **proven** (463.635s) |
| cover | "proven" |
| counterfactual ax≥0.99 | **sat_candidate** |

## Note

Generic `verify_planar` DFS on this already-Marabou-ready box hit depth_cap after root timeouts
(see `bnb_INCOMPLETE/`). This certificate uses **direct SafeSet Marabou** ± vg bisection cover.

ACTIVE untouched.
