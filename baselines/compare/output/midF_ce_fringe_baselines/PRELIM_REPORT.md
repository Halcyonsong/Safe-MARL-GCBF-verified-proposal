# mid-F CE-fringe · shared env graphs (FINAL n=200)

> Same GraphsTuple batch for all methods. Filter: `d0∈[0.125,0.132]`, `c0∈[0.25,0.30]`.
> **Not** the formal 44-D product box. Formal mid-F on MF-r2 = VIOLATED (204→177 CE after midF_r3 repair; still open).

- in-band agents=**200** · graphs=176 · seed=1234 · R=0.1
- d0 realized ~[0.125,0.132] · c0 ~[0.25,0.30]

| role | policy | n | fail% | n_fail | L2-safe% | margin mean | margin min | ‖a‖∞ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ablation | coast_a0 | 200 | 0.00 | 0 | 100.00 | 0.0124 | +0.0080 | 0.000 |
| ablation | mf_base | 200 | 0.00 | 0 | 100.00 | 0.0096 | +0.0032 | 0.305 |
| ours | mf_r2 | 200 | 0.00 | 0 | 100.00 | 0.0081 | +0.0008 | 0.478 |
| baseline | mappo_lag | 200 | **3.50** | **7** | 96.50 | 0.0067 | **−0.0010** | 0.627 |
| baseline | gcbfplus | 200 | **2.00** | **4** | 98.00 | 0.0079 | **−0.0013** | 0.497 |
| baseline | informarl | 200 | 0.00 | 0 | 100.00 | 0.0076 | +0.0006 | 0.523 |

**Baseline fails (env-reachable fringe):** `mappo=7`, `gcbf+=4`, `informarl=0` → all-three gate? **False**  
**MF / ablation fails:** all 0 on this reachable pack (formal mid-F still VIOLATED — abstract corners undersampled by env).

## Gate note

User-locked go gate for *narrative* was: all three baselines fail>0 on CE-fringe evidence.  
InforMARL remains 0 fail on env-reachable fringe even at n=200.  
Product-box / hard-fringe MC still show MAPPO (and feature policies) fail; GCBF+ fails here.  
If later \(X^\star\) is **tightened**, re-score this protocol on the new box — do not reuse these counts blindly.
