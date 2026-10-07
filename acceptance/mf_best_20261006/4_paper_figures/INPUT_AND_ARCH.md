# 输入盒 + 网络结构（矩阵/向量）

## 部署式

```text
a = clip( base(x) + g_phys(x) * delta(x) ) ∈ [-1,1]^2
```

| 模块 | 形状（矩阵乘） | 激活 |
|---|---|---|
| **x** | 44×1 | — |
| Base W0 | 512×44 | ReLU |
| Base W1 | 512×512 | ReLU |
| Base W2 | 512×512 | ReLU |
| Base W3 | 2×512 | 线性 → a_base |
| δ D0 | 128×44 | ReLU |
| δ D1 | 128×128 | ReLU |
| δ Dout | 2×128 | 线性 |
| g_phys | 标量 ∈[0,1]（由 min_d / inv_ttc 拼成，非学习 sigmoid） | — |

图：`figs/fig_arch_matrix_shapes.pdf`

## 输入 \(x\in\mathbb{R}^{44}\)（near_plus）

```text
[0:8)   ego:   px,py, vx,vy, goal_dx,dy,dvx,dvy
[8:28)  agent: 4 quads × (n, min_d, close_max, close_wmean, inv_ttc_max)
[28:44) obs:   4 quads × (n, min_d, close_max, inv_ttc_max)
```

象限：Q0(+x,+y), Q1(−x,+y), Q2(−x,−y), Q3(+x,−y)

### 向量（numpy）

```python
lo, hi = ...  # shape (44,)
# 见 data/input_box_near_plus.npz  /  input_box_lo_hi.csv
```

关键族范围：

| 族 | 范围 |
|---|---|
| min_d | **[0.14, 0.40]** |
| close_* | **[−0.30, 0.30]** |
| inv_ttc_max | **[0, 8]** |
| n（一般） | [0, 4]；**agent Q0 n ∈[1,4]**（强制占用） |
| ego v | [−0.5, 0.5] |
| goal dx,dy | [−2, 2] |

> **提案备忘（未改主验收）**：ego 8 维中 `px,py` 对相对避障非必要；目标静止时 `goal_dvx,dvy ≈ -v`。`goal_dx,dy` 为有方向的相对向量（非标量）。完整梳理后可新开精简-ego 对照版；当前证书仍基于完整 44-D。

### 矩阵块

- `agent_lo/hi`: shape **(4,5)**
- `obs_lo/hi`: shape **(4,4)**

图：`figs/fig_input_numpy_matrix.pdf`
