# MAPPO-Lag baseline

CMDP soft constraint：共享参数 PPO-clip + 对偶 λ 压 env cost。  
观测 = MF 同款 44-D 特征；动作 = `clip(u_ref + tanh(μ(x)))`。  
**经验 only**，无 CBF、无形式化。

## 协议

DoubleIntegrator · 8 · area4 · obs4 · rays32 · `eval_seed=1234` · `cost_limit=0.5`

## 任务结果（best@350 · seed=1234）

| 规模 | success↑ | safe↑ | finish↑ | cost↓ | 产物 |
|---|---:|---:|---:|---:|---|
| 32-epi | **68.4%** | 90.6% | 72.7% | 1.383 | `validate_32epi.json` |
| 64-epi | 68.8% | 90.2% | 73.6% | 1.193 | `validate_64epi.json` |
| 256-epi | 67.6% | 88.4% | 74.7% | 1.875 | `validate_256epi.json` |

读数：safe/cost 明显好于 GCBF mid；success 低于 MF / DS / GCBF+ / HandCBF（软约束 + 短训对照，非任务上界）。

## 训练 / 评测

```bash
bash scripts/run_baseline_mappo_lag.sh 400 0 0.5
bash scripts/validate_baseline_mappo_lag.sh 32
bash scripts/validate_baseline_mappo_lag.sh 64
bash scripts/validate_baseline_mappo_lag.sh 256
```

权重：`seed0_train/best.pkl`（`ACTIVE.txt` 指向）。
