# near_plus box · direct Monte-Carlo (empirical)

> **Not formal.** Samples \(x\) uniformly from the **same 44-D product box** as MF-r2 L2 acceptance.
> No env rollout. Graph policies (DS / GCBF+ / HandCBF) = **N/A** here.

- n=50000 · seed=1234 · wall=4.4s · R=0.1

| policy | n | L2-safe% | mean d2 | mean ‖a‖∞ | note |
|---|---:|---:|---:|---:|---|
| coast_a0 | 50000 | 100.00 | 0.1589 | 0.000 | a=0 reference; envelope expects mostly safe in this box |
| mf_r2 | 50000 | 100.00 | 0.1525 | 0.709 | acceptance policy; formal PROVEN_FULL on this box |
| mappo_lag | 50000 | 100.00 | 0.1511 | 0.863 | a=tanh(mean) only; uref omitted (needs graph). Approximate vs deploy. |
| ds_r2 | 0 | — | — | — | N/A under direct box sampling — DeepSet needs ego/entity local sets (graph) |
| gcbfplus | 0 | — | — | — | N/A under direct box sampling — GNN actor needs GraphsTuple |
| handcbf | 0 | — | — | — | N/A under direct box sampling — CBF-QP needs GraphsTuple |

## Why this matches acceptance better than rollout filtering

Certificate quantifies ∀x∈ product box. Empty-quad `min_d≈0.5` makes real states miss the box;
direct sampling evaluates policies **on the declared X**.

MF-r2 already has **PROVEN_FULL** on this box; high empirical L2-safe% here is a sanity check,
not a new proof.
