# r3 manual re-rank @32 seed=1234

| rank | ckpt | suc | safe | fin | cost |
|---:|---|---:|---:|---:|---:|
| 1 | `r3_train_report` | 82.0% | 95.7% | 85.2% | 0.688 | <- pick
| 2 | `r3_best` | 82.0% | 95.7% | 85.2% | 0.688 |
| 3 | `r3_last` | 81.6% | 95.3% | 85.5% | 0.797 |
| 4 | `r2_last` | 81.6% | 94.9% | 85.5% | 0.473 |

## Reference (acceptance pack @32)
| method | suc | safe | fin |
|---|---:|---:|---:|
| GCBF+ | 91.4% | 95.7% | 95.3% |
| MF main r2 | 82.0% | 96.5% | 84.0% |
| HandCBF | 84.8% | 100% | 84.8% |

**Goal:** safe ≳ GCBF+ (~95.7%), success need not match GCBF+.

