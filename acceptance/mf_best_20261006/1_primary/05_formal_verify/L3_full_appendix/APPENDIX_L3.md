# L3 full appendix — NOT primary acceptance

> Primary bounded acceptance remains **near_plus PROVEN_FULL**  
> (`ACTIVE_MF_VERIFY` → `mf_feat44_near_plus_r2`). This run is an **appendix**.

## Claim

```text
∀ x ∈ X_full ⊂ R^{44}: TwoStepRadialL2Safe(Deploy(x); R=0.10)
X_full: min_d∈[0.12, 0.50], |close|≤0.50, inv_ttc≤20, Q0 occupied
```

Physics whole-box envelope: \(d_2 \ge 0.12 - 0.06\cdot0.50 - 0.009 = **0.081 < 0.10**\) → cannot expect easy PROVEN_FULL.

## Result

| item | value |
|---|---|
| aggregate | **VIOLATED_WITH_CES** |
| proven leaves | 1 |
| unknown leaves | 1200 (hit max_splits) |
| Marabou calls | 0 |
| real CEs | **11260** |
| wall | ~7.0s |

## CE analysis (all 11260 / d0–c0)

| metric | value |
|---|---|
| d0 range | [0.120, 0.139] (med 0.126) |
| c0 range | [0.190, 0.500] (med 0.442) |
| d2 range | [0.081, 0.100] (med 0.093) |
| **d0 < 0.14** | **100.0%** |
| **c0 > 0.30** | **95.5%** |
| **outside near_plus (d or c)** | **100.0%** |
| physics coast-doomed \(d_0-\alpha c_0 < R\) | 41.2% |

**Conclusion:** L3 CEs concentrate in the **coast-risk / high-close fringe below the near_plus cut** (`d0∈[0.12,0.14)`).  
This **supports** keeping L2 `near_plus` as the bounded primary claim — L3 is honest documentation of the wider envelope, not a failed main acceptance.

## Correct reading

```text
L2 near_plus PROVEN_FULL  =  primary bounded acceptance (unchanged)
L3 full VIOLATED          =  expected on wide box; CEs outside L2 cut
≠  L2 is wrong
≠  infinite-horizon deploy certified
```

DeepSet `method/deepset/output/phase2/ACTIVE_*` untouched.
