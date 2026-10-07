# Baseline vs ablation — reading guide

## Paper / compare roles

| name | role | use in “MF certified vs baselines fail” story |
|---|---|---|
| **GCBF+** | baseline | **must** show empirical L2 fail on target domain \(X^\star\) |
| **MAPPO-Lag** | baseline | **must** show empirical L2 fail on \(X^\star\) |
| **InforMARL** | baseline | **must** show empirical L2 fail on \(X^\star\) |
| HandCBF (optional) | baseline | nice-to-have |
| **MF-r2 → MF-repaired** | **ours** | formal **PROVEN_FULL** on same \(X^\star\) |
| coast \(a=0\) | **ablation / physics ref** | shows band is preventable; **not** a competitor |
| mf_base | **ablation (pre-repair)** | shows repair changed Deploy; **not** a competitor |
| DS-r2 | parallel mainline (DeepSet) | optional internal; not the MF-vs-baseline table |

## Desired table shape (after successful mid-F repair)

| method | on \(X^\star\)=mid-F | evidence type |
|---|---|---|
| **MF (repaired)** | **PROVEN_FULL** (0 CE) | formal |
| GCBF+ | fail% > 0 | empirical (shared env / box if applicable) |
| MAPPO-Lag | fail% > 0 | empirical (product-box MC OK — feature policy) |
| InforMARL | fail% > 0 | empirical (shared env GraphsTuple) |
| coast / mf_base | footnote ablation only | not in main baseline columns |

## Gate before starting repair

Do **not** start CE repair until:

1. **MAPPO** fails on mid-F product box (✅ already: uniform 0.46%, hard fringe 3.24%).
2. **GCBF+** and **InforMARL** show **nonzero** two-step L2 fails on **shared env states** whose \((d_0,c_0)\) sit in the mid-F / CE-fringe band  
   — previous pack (warmer band) was insufficient (InforMARL 0 fail; GCBF+ 1/400).  
   — **CE-fringe env collect running:** `baselines/compare/output/midF_ce_fringe_baselines/`  
     band `d0∈[0.125,0.132]`, `c0∈[0.25,0.30]`.

If graph baselines **still** never fail on reachable fringe states, then either:

- widen / retarget \(X^\star\) until they do, or  
- narrow the claim: product-box story for MAPPO + “certificate asymmetry” for graph methods (weaker).
