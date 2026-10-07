# mid-F semantic band · shared env graphs (empirical)

> Same GraphsTuple batch for all methods. Filter: `d0∈[0.125,0.132]`, `c0∈[0.25,0.3]` (mid-F semantics).
> **Not** the formal 44-D product box. Formal mid-F on MF-r2 = VIOLATED (204 CE).

- in-band agents=200 · graphs=176 · seed=1234 · R=0.1

| policy | n | L2-safe% | n_fail | fail% | margin mean | p10 | min | ‖a‖∞ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coast_a0 | 200 | 100.00 | 0 | 0.00 | 0.0124 | 0.0091 | 0.0080 | 0.000 |
| mf_base | 200 | 100.00 | 0 | 0.00 | 0.0096 | 0.0059 | 0.0032 | 0.305 |
| mf_r2 | 200 | 100.00 | 0 | 0.00 | 0.0081 | 0.0039 | 0.0008 | 0.478 |
| mappo_lag | 200 | 96.50 | 7 | 3.50 | 0.0067 | 0.0021 | -0.0010 | 0.627 |
| gcbfplus | 200 | 98.00 | 4 | 2.00 | 0.0079 | 0.0041 | -0.0013 | 0.497 |
| informarl | 200 | 100.00 | 0 | 0.00 | 0.0076 | 0.0041 | 0.0006 | 0.523 |

## How to read

1. Failures here mean two-step **radial** L2 unsafe under that policy's action.
2. Expect **coast** often safer (small ‖a‖); aggressive policies fail more in this band.
3. MF-r2 may still fail empirically here (matches formal mid-F VIOLATED) — repair target, not a claim of zero fail yet.
4. After CE repair + PROVEN on formal mid-F, re-score this pack to show separation.
