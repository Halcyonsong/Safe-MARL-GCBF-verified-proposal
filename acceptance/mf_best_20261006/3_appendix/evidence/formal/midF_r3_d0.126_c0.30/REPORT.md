# MF verification — 44-D feature box + two-step L2

- checkpoint: `method/meanfield/output/repair/mf_residual_v5_midF_r3_20261006/seed0/repair_best.pkl`
- mode: **near_plus**
- aggregate: **VIOLATED_WITH_CES**
- real CEs: **47**

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
| mf_near_plus | VIOLATED | 2 | 47 | 1199 | 0 | 1200 | 7.06s |

PROVEN_FULL = full coverage of declared X, 0 real CE, 0 unknown.
PARTIAL = unfinished leaves under resource limits (CEs still harvested).
VIOLATED = ≥1 true-policy CE in `counterexamples_all.json`.

DeepSet `method/deepset/output/phase2/ACTIVE_*` untouched.
