# InforMARL baseline

局部 agent–entity 图 + 注意力 GNN + PPO-clip。  
安全仅靠训练 reward 里的软 cost（`r - cost_weight·cost`），**无 CBF / 无 Lagrangian / 无形式化**。

动作：`clip(u_ref + tanh(μ(GNN(graph))))`。  
图结构复用 DoubleIntegrator 的 `GraphsTuple`（agent / goal / lidar）。

## 协议

DoubleIntegrator · 8 · area4 · obs4 · rays32 · `eval_seed=1234` · `cost_weight=1.0`

## 任务结果（best@350 · seed=1234）

| 规模 | success↑ | safe↑ | finish↑ | cost↓ | 产物 |
|---|---:|---:|---:|---:|---|
| 32-epi | **60.9%** | 62.5% | **95.7%** | 6.934 | `validate_32epi.json` |
| 64-epi | 61.3% | 62.7% | 96.3% | 6.109 | `validate_64epi.json` |
| 256-epi | 62.8% | 64.3% | 96.3% | 5.783 | `validate_256epi.json` |

读数：finish 很高，但 safe≈success、cost 明显高于 MAPPO-Lag / MF / GCBF+ —— 典型「无证书导航 MARL：敢冲、常撞」。

## 训练 / 评测

```bash
bash scripts/run_baseline_informarl.sh 400 0 1.0
bash scripts/validate_baseline_informarl.sh 32
bash scripts/validate_baseline_informarl.sh 64
bash scripts/validate_baseline_informarl.sh 256
```

权重：`seed0_train/best.pkl`（`ACTIVE.txt` 指向）。
