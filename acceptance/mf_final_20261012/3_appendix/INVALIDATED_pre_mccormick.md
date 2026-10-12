# Invalidated runs (do not cite)

Before fixing `marabou_mf._add_mccormick` (last inequality sign), any
**positive-width** feature box made the Deploy encoding globally UNSAT.
Marabou SafeSet queries therefore returned UNSAT → mislabeled **proven**.

Discarded artifacts (removed from tree):
- min_M1 early ~5s PROVEN_FULL (false)
- netdep_r3_pose_fixed* marabou_proven leaves (false / untrustworthy)
- netdep_r3_smoke* / free-pose incomplete runs under broken encode

Valid certificate after fix: `../min_M1_vg001/`
