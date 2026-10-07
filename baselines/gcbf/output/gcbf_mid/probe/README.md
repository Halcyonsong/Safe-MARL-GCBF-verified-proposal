# Mid-scale GCBF probe (paused @ ~270/1000)
Eval protocol: 32-epi, seed=1234, 8 agents, obs4, area4.
| ckpt | refine | success | safe | finish |
|---|---|---:|---:|---:|
| mid step270 | no | 55.5% | 62.1% | 81.2% |
| mid step250 | no | 46.9% | 53.9% | 74.6% |
| mid step200 | no | 44.5% | 57.4% | 67.2% |
| mid step270 | **yes** | 59.4% | 60.2% | 95.7% |
| weak step1000 | no | 51.6% | 53.9% | 91.4% |
| GCBF+ teacher | — | 91.4% | 95.7% | 95.3% |

**Best default: step270** — suc 55.5% / safe 62.1% / fin 81.2%.
vs weak: +3.9pp suc, +8.2pp safe, but finish lower (81 vs 91).
Still far below GCBF+ (~91/96).

Train paused; can resume from same run if desired.
