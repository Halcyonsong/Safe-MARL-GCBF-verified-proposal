# Expected metrics (before run completes)

## Protocol (unchanged — same as GCBF+ / DeepSet / MF)
- Env: DoubleIntegrator · **8 agents · area 4 · obs 4 · rays 32**
- Eval: **32 episodes · `eval_seed=1234`** · `scripts/validate_baseline_gcbf.sh`
- Not paper’s large-N / many-obstacle stress test; same small task as our other baselines.

## Train (official-ish)
| | Prev weak | This run |
|---|---|---|
| n_env_train | 2 | **16** |
| batch / buffer | 32 / 256 | **256 / 512** |
| inner_epoch | 4 | **8** |
| lr_actor / lr_cbf | 3e-5 | **1e-5** (README/paper) |
| loss_action_coef | 1e-4 | 1e-4 (README) |
| loss_h_dot_coef | 0.01 | 0.01 (CLI default) |
| steps / seed | 1000 / 0 | 1000 / 0 |

## What we expect (ranges, not guarantees)

Paper (GCBF+ T-RO): in crowded 2D obstacle settings **GCBFv0 success ≈20%**, **GCBF+ >95%**; nonlinear hard cases GCBFv0 **<10%**.  
Our task (8 agents, **4** obs) is easier than paper’s crowded figures, and our GCBF+ teacher already hits **~91% success / ~96% safe** even at 2-env scale.

| Outcome | success@32 | safe@32 | Notes |
|---|---:|---:|---|
| Prev weak GCBF (2-env) | 51.6% | 53.9% | baseline floor |
| **Likely (official-scale)** | **~65–80%** | **~70–85%** | more data + paper lr; still no QP |
| Optimistic | ~80–88% | ~85–92% | if GCBF learns well on this easier task |
| Ceiling (not expected) | ≈ GCBF+ 91% | ≈ 96% | QP teacher gap remains |

**Bottom line:** expect a clear lift over 52%/54%, and still a **visible gap under GCBF+**.  
If after 1000 steps safe stays <60%, treat as “GCBF needs class-default `loss_h_dot=0.2`” follow-up, not eval-protocol failure.
