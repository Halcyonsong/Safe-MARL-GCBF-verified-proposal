# 基线对照

> 解读与「修复作用」说明见上级 **[`EVAL_AND_REPAIR.md`](../EVAL_AND_REPAIR.md)**。

| 文件 / 目录 | 用途 |
|---|---|
| `BASELINE_COMPARE.md` | @32 / @64 / @256 主对照表 + 延迟 |
| `TRANSFER_COMPARE.md` | N=16/32 零样本延展 |
| `epi_pressure/epi{64,256}/` | 压力测评原始 JSON |
| `transfer_n/N{8,16,32}/` | 延展原始 JSON |
| `latency/` | 控制延迟 |
| `BASELINE_FAIL_EVIDENCE.md` | mid-F 经验 fail（附录） |
| `near_plus_box_mc/` | 主验收盒产品盒 MC |

论文基线行：GCBF+ · MAPPO-Lag · InforMARL。coast / mf_base 为消融。
