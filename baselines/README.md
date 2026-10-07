# baselines/ — 官方对照基线（非本方法主路径）

> 对照 / 教师 / 消融用。DeepSet / Mean-field 主方法产物不在此目录。

## 布局（每方法）

```text
baselines/<algo>/{code,config,log,output,scripts}
```

| 路径 | 含义 |
|---|---|
| `baselines/gcbfplus/` | 官方 **GCBF+** |
| `baselines/gcbf/` | 官方 **GCBF**（含 mid / official 产物） |
| `baselines/handcbf/` | **HandCBF** = `DecShareCBF` |
| `baselines/mappo_lag/` | **MAPPO-Lag** |
| `baselines/informarl/` | **InforMARL** |
| `baselines/compare/` | 对照表、transfer、box MC、latency |
| `core/gcbfplus/` | 官方源码树（`import gcbfplus`） |

`import baseline.<algo>` 由 `core.paths.ensure_import_path()` 映射到 `baselines/<algo>/code`。

## 已有权重（勿随意覆盖）

| 方法 | log / 入口 | output 镜像 | 32-epi |
|---|---|---|---|
| **GCBF+** | `baselines/gcbfplus/log/gcbf+/seed0_20261001232611` | `.../output/gcbfplus/` | success **91.4%** |
| **GCBF mid@270** | `baselines/gcbf/log/gcbf/seed0_20261006111544` | `.../output/gcbf_mid/` | success **55.5%** |
| **HandCBF** | `algo=dec_share_cbf` | `.../output/handcbf/` | success **84.8%** / safe **100%** |
| **MAPPO-Lag** | `.../output/mappo_lag/seed0_train/best.pkl` | 同左 | success **68.4%** |
| **InforMARL** | `.../output/informarl/seed0_train/best.pkl` | 同左 | success **60.9%** |

根对照表：`baselines/compare/BASELINE_COMPARE.md`。

## 启动

```bash
bash baselines/gcbfplus/scripts/validate_baseline_gcbfplus.sh \
  baselines/gcbfplus/log/gcbf+/seed0_20261001232611 1000 32

bash baselines/gcbf/scripts/validate_baseline_gcbf.sh \
  baselines/gcbf/log/gcbf/<run_dir> 1000 32

bash baselines/handcbf/scripts/validate_baseline_handcbf.sh 32
bash baselines/mappo_lag/scripts/run_baseline_mappo_lag.sh 400 0 0.5
bash baselines/informarl/scripts/run_baseline_informarl.sh 400 0 1.0
python baselines/compare/scripts/bench_control_latency.py
```
