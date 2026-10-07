# Target narrative (locked)

## What we want

A domain \(X^\star\) such that:

| role | claim |
|---|---|
| **MF (after repair)** | formal **PROVEN_FULL** on \(X^\star\) (two-step radial L2) |
| **Baselines** | empirical failures of the **same** property on \(X^\star\) (or shared env states in \(X^\star\) semantics) |
| Baselines in scope | **GCBF+ · MAPPO-Lag · InforMARL** (optional: HandCBF) |
| **Ablations only** (not baseline rows) | coast \(a=0\), **mf_base** (pre-repair), maybe DS if needed for internal compare |

## What we do **not** claim with coast / mf_base

- coast = physics reference / preventable-band sanity  
- mf_base = pre-repair ablation (“repair changed something”)  
- Neither is a competitor method in the paper table

## Current gap

| domain | MF formal | MAPPO fail? | GCBF+ / InforMARL fail? |
|---|---|---|---|
| L2 `near_plus` (acceptance) | PROVEN_FULL | binary L2 ≈ no (envelope) | binary L2 ≈ no on env |
| mid-F product box | **VIOLATED** (r2) | **Yes** (box MC) | **N/A on abstract box** (need graphs) |
| mid-F env band (prev pack) | — | binary ≈ no | binary ≈ no (1 GCBF+) |

**Before repairing mid-F:** confirm the three baselines show **nonzero failures** on the *same* mid-F target we will certify — especially graph baselines via shared env / harder fringe.  
If only MAPPO fails on the product box and graph baselines never fail on reachable mid-F states, the story weakens for GCBF+/InforMARL.
