# 指标速查（验收包）

## 形式化主验收

| 项 | 值 |
|---|---|
| claim | L2 near_plus，44-D，两步径向 L2，R=0.10 |
| box | min_d∈[0.14,0.40]，\|close\|≤0.30，Q0 occupied |
| status | **PROVEN_FULL** |
| d2_lo (IBP) | **0.113** |
| real CE | 0 |

## 任务 32-epi

| | success | safe | finish |
|---|---:|---:|---:|
| base v5 | 0.777 | 0.809 | 0.941 |
| r1 | 0.832 | 0.930 | 0.879 |
| **r2** | **0.820** | **0.965** | **0.840** |

## 经验 CE hit_on_danger

| | rate |
|---|---:|
| base | 0.0595 |
| r1 | 0.0131 |
| **r2** | **0.0047** |

## 近危覆盖（经验）

| | |
|---|---:|
| danger 步 ∈ L2 (d,c) | 79.6% |
| L2 内两步 hit | 0 |

## L3 附录

| | |
|---|---|
| status | VIOLATED_WITH_CES |
| n_CE | 11260 |
| CE 在 L2 外 | 100% |
