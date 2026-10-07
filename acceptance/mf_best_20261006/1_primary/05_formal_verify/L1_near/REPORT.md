# MF verification — 44-D feature box + two-step L2

- checkpoint: `method/meanfield/output/repair/mf_residual_v5_r2_20261005/seed0/repair_best.pkl`
- mode: **near**
- aggregate: **PROVEN_FULL**
- real CEs: **0**

## Design (agreed)

| | |
|---|---|
| Query domain | **Each free dim of 44-D MF features** (speed, goal, quad min_d / wmean / inv_ttc, …) |
| Coverage | Branch-and-bound on feature box → aim **PROVEN_FULL** |
| Property | **Two-step** radial L2 (horizon simplification) |
| Not used | DeepSet (k,m) buckets; primary AA/AO claim split |

## Per-claim

| claim | status | proven | CE | unknown | marabou | splits | time |
|---|---|---:|---:|---:|---:|---:|---:|
| mf_near | PROVEN_FULL | 1 | 0 | 0 | 0 | 0 | 0.002s |

PROVEN_FULL = full coverage of declared X, 0 real CE, 0 unknown.
PARTIAL = unfinished leaves under resource limits (CEs still harvested).
VIOLATED = ≥1 true-policy CE in `counterexamples_all.json`.

DeepSet `method/deepset/output/phase2/ACTIVE_*` untouched.
