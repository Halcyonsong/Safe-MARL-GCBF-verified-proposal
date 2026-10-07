# HandCBF baseline（DecShareCBF）

官方 **手工 CBF + 在线 QP**（`algo=dec_share_cbf`），无训练、无神经网络证书。

## 协议

DoubleIntegrator · 8 agents · area 4.0 · obs 4 · rays 32 · `eval_seed=1234` · `alpha=1.0`

## 任务结果

| 规模 | success↑ | safe↑ | finish↑ | cost↓ | 产物 |
|---|---:|---:|---:|---:|---|
| 32-epi | 84.8% | **100%** | 84.8% | **0.000** | `validate_32epi.json` |
| 64-epi | 83.4% | 99.6% | 83.8% | 0.059 | `validate_64epi.json` |
| 256-epi | 83.4% | 99.8% | 83.5% | 0.035 | `validate_256epi.json` |

读数要点：safe 极高、cost 近 0，但 finish≈success，偏保守（卡死/绕不开会伤 success）。

## 评测

```bash
bash scripts/validate_baseline_handcbf.sh 32
bash scripts/validate_baseline_handcbf.sh 64
bash scripts/validate_baseline_handcbf.sh 256
```

实现：`baselines/handcbf/code/validate.py` → 调用 `baselines/gcbfplus/code/test_official.py --algo dec_share_cbf`。

## 控制耗时

见 `baselines/compare/output/latency/control_latency.json`（与 MF / GCBF+ 同脚本）。
