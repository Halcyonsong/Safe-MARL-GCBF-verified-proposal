# mid-F semantic band · shared env graphs (empirical)

> Same GraphsTuple batch for all methods. Filter: `d0∈[0.125,0.15]`, `c0∈[0.12,0.3]` (mid-F semantics).
> **Not** the formal 44-D product box. Formal mid-F on MF-r2 = VIOLATED (204 CE).

- in-band agents=400 · graphs=329 · seed=1234 · R=0.1

| policy | n | L2-safe% | n_fail | fail% | margin mean | p10 | min | ‖a‖∞ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coast_a0 | 400 | 100.00 | 0 | 0.00 | 0.0268 | 0.0157 | 0.0091 | 0.000 |
| mf_base | 400 | 100.00 | 0 | 0.00 | 0.0246 | 0.0129 | 0.0052 | 0.250 |
| mf_r2 | 400 | 100.00 | 0 | 0.00 | 0.0236 | 0.0113 | 0.0027 | 0.357 |
| mappo_lag | 400 | 100.00 | 0 | 0.00 | 0.0213 | 0.0093 | 0.0013 | 0.611 |
| ds_r2 | 400 | 100.00 | 0 | 0.00 | 0.0219 | 0.0091 | 0.0013 | 0.549 |
| gcbfplus | 400 | 99.75 | 1 | 0.25 | 0.0220 | 0.0104 | -0.0024 | 0.536 |
| informarl | 400 | 100.00 | 0 | 0.00 | 0.0222 | 0.0103 | 0.0021 | 0.507 |

## How to read

1. Failures here mean two-step **radial** L2 unsafe under that policy's action.
2. Expect **coast** often safer (small ‖a‖); aggressive policies fail more in this band.
3. MF-r2 may still fail empirically here (matches formal mid-F VIOLATED) — repair target, not a claim of zero fail yet.
4. After CE repair + PROVEN on formal mid-F, re-score this pack to show separation.
