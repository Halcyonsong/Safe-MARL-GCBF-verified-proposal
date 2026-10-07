# GCBF mid-scale baseline (ablation)

**Frozen probe ckpt: step 270** (train paused; not full 1000).

Eval protocol = same as GCBF+ / DeepSet / MF: 8 agents · obs4 · area4 · **32-epi · seed=1234**.

| 策略 | success↑ | safe↑ | finish↑ | 角色 |
|---|---:|---:|---:|---|
| GCBF weak (2-env @1000) | 51.6% | 53.9% | 91.4% | 更弱下限 |
| **GCBF mid @270（本目录）** | **55.5%** | **62.1%** | **81.2%** | **推荐消融行** |
| mid @270 + online refine | 59.4% | 60.2% | 95.7% | 附录 |
| GCBF+ 教师 | 91.4% | 95.7% | 95.3% | 同族上界 / 教师 |

## 够不够当基线？

**够用（作同族消融）**：在同一测评协议下证明「无 QP 的 GCBF ≪ GCBF+」。  
**不够（若想当任务竞品上界）**：不要拿它和 DeepSet/MF 比谁 success 更高；那不是它的定位。

Run: `baselines/gcbf/log/gcbf/seed0_20261006111544/models/270`
