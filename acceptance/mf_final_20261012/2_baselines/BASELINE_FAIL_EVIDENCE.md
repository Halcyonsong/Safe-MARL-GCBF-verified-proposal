# mid-F · baseline failure evidence (before repair)

> Acceptance pack **untouched**. This is a go/no-go check: do baselines (and r2) fail inside the wider mid-F regime?

## Setup

| item | value |
|---|---|
| mid F box | `min_d∈[0.125,0.40]`, `|close|≤0.30`, `R=0.10` |
| physics | \(d2\_lo≈0.098 < R\) → **network-dependent** |
| formal r2 probe | **VIOLATED** · 204 CEs · `mf_r2_midF_probe_20261006` |

## A. Formal CE set (n=204) — feature policies on the *same* CE vectors

These are the abstract product-box CEs found against **mf_r2**. Scoring other nets on them is **informative but asymmetric** (set was mined for r2).

| policy | fail% | margin mean | ‖a‖∞ |
|---|---:|---:|---:|
| coast_a0 | **0.0** | +0.0082 | 0.00 |
| mf_base | **36.8** | +0.0019 | 0.71 |
| **mf_r2** | **100.0** | −0.0007 | 0.99 |
| mappo_lag (tanh mean) | **60.3** | +0.0003 | 0.88 |
| DS / GCBF+ / HandCBF / InforMARL | N/A | — | need GraphsTuple |

## B. Fair MC — uniform mid-F product box (n=30k)

Same declared X for everyone (feature-input policies).

| policy | fail% | L2-safe% | margin mean | ‖a‖∞ |
|---|---:|---:|---:|---:|
| coast_a0 | **0.00** | 100.00 | 0.0457 | 0.00 |
| mf_base | **0.18** | 99.82 | 0.0403 | 0.60 |
| mf_r2 | **0.38** | 99.62 | 0.0392 | 0.72 |
| mappo_lag | **0.46** | 99.54 | 0.0379 | 0.86 |

→ **All learned policies have nonzero failures** on mid-F; coast has none (preventable band). Rates are small because most of the box is still easy; failures concentrate on the fringe.

## C. Hard fringe of mid-F (d0∈[0.125,0.135], c0≥0.22) · n≈7.5k from 50k draw

| policy | fail% | margin mean | ‖a‖∞ |
|---|---:|---:|---:|
| coast_a0 | **0.00** | 0.0138 | 0.00 |
| mf_base | **1.29** | 0.0084 | 0.60 |
| mf_r2 | **2.55** | 0.0073 | 0.73 |
| mappo_lag | **3.24** | 0.0061 | 0.86 |

→ Clear ranking: **coast ≫ mf_base > mf_r2 > mappo** on fail% / margin (radial model).  
r2 is **not** best here — consistent with formal VIOLATED and saturated ‖a‖ on CEs.

## D. Shared env GraphsTuple (same pack · all methods)

Band: `d0∈[0.125,0.15]`, `c0∈[0.12,0.30]` · **n=400** in-band agents · **329** graphs · explorer=`mf_base`  
Pack: `env_pack.pkl` · report: `REPORT.md` / `midF_env_l2.json`

| policy | L2-safe% | n_fail | margin mean | margin min | ‖a‖∞ |
|---|---:|---:|---:|---:|---:|
| coast_a0 | **100** | 0 | **0.0268** | 0.0091 | 0.00 |
| mf_base | 100 | 0 | 0.0246 | 0.0052 | 0.25 |
| **mf_r2** | 100 | 0 | 0.0236 | 0.0027 | 0.36 |
| informarl | 100 | 0 | 0.0222 | 0.0021 | 0.51 |
| gcbfplus | **99.75** | **1** | 0.0220 | **−0.0024** | 0.54 |
| ds_r2 | 100 | 0 | 0.0219 | 0.0013 | 0.55 |
| mappo_lag | 100 | 0 | 0.0213 | 0.0013 | 0.61 |

**Reading**

- On *reachable* env states in this band, binary L2 fails are rare (only GCBF+ had 1/400).
- **Margin / ‖a‖** still rank: coast ≫ mf_base > mf_r2 > others; mappo most aggressive.
- This does **not** contradict formal mid-F VIOLATED: formal CEs live in the abstract product box (incl. unrealizable combos); env rollouts undersample the worst abstract corners.
- For “baselines also fail on declared X”: use **§B/§C box MC** (and formal CE set §A). Env pack is the fair graph-policy contrast on shared states.

## E. CE-fringe shared env (tighter band) · n=200 FINAL

Band: `d0∈[0.125,0.132]`, `c0∈[0.25,0.30]` · graphs=176 · seed=1234  
Pack: `../midF_ce_fringe_baselines/` · `REPORT.md`

| policy | L2-safe% | n_fail | fail% | margin min | ‖a‖∞ |
|---|---:|---:|---:|---:|---:|
| coast / mf_base / **mf_r2** | 100 | 0 | 0 | ≥+0.0008 | ≤0.48 |
| **mappo_lag** | 96.50 | **7** | **3.50** | −0.0010 | 0.63 |
| **gcbfplus** | 98.00 | **4** | **2.00** | −0.0013 | 0.50 |
| informarl | 100 | 0 | 0 | +0.0006 | 0.52 |

→ On **env-reachable** CE-fringe: MAPPO + GCBF+ fail; **InforMARL still 0**.  
→ Formal mid-F VIOLATED coexists with mf_r2 env-fringe 0 fail (abstract corners).

## Go / no-go for mid-F CE repair

| question | answer |
|---|---|
| Do baselines fail on mid-F **product box**? | **Yes** (uniform 0.2–0.5%; hard fringe 1–3%) |
| Do graph policies fail on **CE-fringe env** (n=200)? | **MAPPO 7 + GCBF+ 4**; InforMARL **0** |
| All-three env-fringe gate? | **Not met** (InforMARL=0) — user may relax or dig harder / tighten+retest |
| Is r2 clean on mid-F formal? | **No** (probe 204 CE; after midF_r3 still ~177 CE) |
| Is repair plausible? | **Yes** — CEs 100% throttle-preventable; CE drop slow (~10/round) |
| Touch acceptance now? | **No** — only after side-branch PROVEN |
| If \(X^\star\) **tightened** later? | **Must re-score** MAPPO/GCBF+/InforMARL on the new box |

## Files

- `midF_box_uniform.json` / `BOX_UNIFORM_REPORT.md`
- `midF_box_hard_fringe.json` / `BOX_HARD_FRINGE_REPORT.md`
- `env_pack.pkl` / `REPORT.md` / `midF_env_l2.json` — shared env graphs
- formal probe: `method/meanfield/output/verify/mf_r2_midF_probe_20261006/`
- env script: `scripts/sample_midF_env_baselines.py`
