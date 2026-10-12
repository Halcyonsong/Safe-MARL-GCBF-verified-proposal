# Planar SafeSet side-branch — status log

**Updated:** 2026-10-11  
**ACTIVE / acceptance:** untouched.  
**Encoding:** McCormick fixed.

## Certificates (cite these)

| ID | Path | Status | Scope | Formal |
|----|------|--------|-------|--------|
| **M1** | `min_M1_vg001/` | **PROVEN_FULL** | pose=2; vg±**0.01**; threat mid | Marabou ~25s |
| **M2** | `enlarge_M2_vg005/` | **PROVEN_FULL** | vg±**0.05** | root ~464s |
| **M3** | `enlarge_M3_vg010/` | **PROVEN_FULL** | vg±**0.10** | root 900s timeout → **2 vg-halves** proven (~372s + ~145s); wall ~25min |

All three: netdep (not envelope); encode SAT; sample CE=0; ax≥0.99 counterfactual not vacuous.

### Difficulty (honest)

These are **scoped slices**, not “full workspace easy wins”:

| fixed / easy | still real |
|--------------|------------|
| pose pinned (px=py=2) | network must keep ax≥0 on netdep fringe |
| threat features = midpoint | X_rel full interval; vg full interval in claim |
| single / few Marabou boxes | McCormick outer approx; UNSAT is sound |

Harder next: threat width, free pose, wider X_rel — not “same difficulty forever”.

## Ladder

| Step | vg | Status |
|------|-----|--------|
| M1–M3 | ±0.01 → ±0.10 | **done** |
| M4 candidate | ±0.25 or threat subboxes | next if desired |
| free pose / full threat | expensive | later |

## Do not cite

- `_invalidated_pre_mccormick_fix/`
- `enlarge_M2_vg005/bnb_INCOMPLETE/`
