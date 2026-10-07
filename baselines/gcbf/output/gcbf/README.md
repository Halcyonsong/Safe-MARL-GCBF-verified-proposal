# Official GCBF baseline (comparison / ablation)

Non-plus **GCBF** (no QP teacher during training). Same DoubleIntegrator protocol and memory-limited hparams as `scripts/run_baseline_gcbfplus.sh`.

## Run

```text
baselines/gcbf/log/gcbf/seed0_20261006100552/models/1000/
```

Mirrored weights: `actor.pkl`, `cbf.pkl` in this directory.

## Eval (32-epi, `eval_seed=1234`)

| 策略 | success↑ | safe↑ | finish↑ |
|---|---:|---:|---:|
| **GCBF** (本目录) | **51.6%** | **53.9%** | **91.4%** |
| GCBF+ 教师 (对照) | 91.4% | 95.7% | 95.3% |

→ QP 教师（GCBF+）相对无 QP 的 GCBF 带来大幅安全/成功率提升；本行用于同族消融，不是上界。

```bash
bash scripts/run_baseline_gcbf.sh
bash scripts/validate_baseline_gcbf.sh baselines/gcbf/log/gcbf/seed0_20261006100552 1000 32
```

Empirical only. Not formally verified.
