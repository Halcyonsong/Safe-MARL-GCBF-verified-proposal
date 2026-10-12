# Valid minimal planar certificate (post McCormick fix)

- **status: PROVEN_FULL**
- wall: 52.575s (Marabou 24.986s)
- encoding: **fixed** (not the false ~5s certificate)

## X (scoped)

| block | bounds |
|-------|--------|
| dx | [0.118, 0.122] |
| dy | [-0.04, 0.04] |
| dvx | [-0.30, -0.25] |
| dvy | [-0.05, 0.05] |
| px, py | 2.0 fixed |
| vx,vy,goal_xy | **[-0.01, 0.01]** |
| threat feats | midpoint |
| R | 0.10 |

## Network-dependence checks (not envelope)

| check | result |
|-------|--------|
| envelope_any_a_safe | **False** |
| p2 brake (a=-1) | 0.09100 < R → would collide |
| p2 gas (a=+1) | 0.10900 ≥ R |
| true Deploy ax (256) | min=0.716 mean=0.719 (all ≥ 0) |
| Marabou ax≥≈0 | **proven** |
| Marabou ax≥0.99 (counterfactual) | **sat_candidate** (must not be vacuous proven) |

## Claim

> Inside this X only: formal full coverage via SafeSet+Marabou on **encoded Deploy**.
> Depends on network weights. Geometry alone does **not** make all actions safe.

## Not claimed

- vg ±0.05 box (timed out at 180s after fix)
- free pose / full threat width / ACTIVE promotion
