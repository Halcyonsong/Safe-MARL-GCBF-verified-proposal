# MF 最终验收包（2026-10-12）

| | |
|---|---|
| **主贡献** | 平面两步 SafeSet **网络依赖**证书 |
| **推荐 Deploy** | residual **r2** |
| **主证书** | r2 · **M3** vg±0.10 · **PROVEN_FULL** |
| **任务 @32** | suc 82.0% · safe **96.5%** · fin 84.0% · cost **0.31** |
| **证书定级** | 局部两步 pairwise 动作安全（声明 feat–rel） |
| **ACTIVE** | **未改写** |
| **入库范围** | 主树证据 + part/figs；`5_frozen_sources/` 与 `aamas_2027_template/` **本地保留、不入库** |

---

## 先读

1. **本页**
2. **`1_primary/00_acceptance/ACCEPTANCE_CERTIFICATE.md`**
3. **`0_docs/EVAL_AND_REPAIR.md`**
4. **`0_docs/DUAL_WEIGHT_FORMAL.md`**
5. **`1_primary/00_acceptance/METRICS_SUMMARY.md`**
6. **`0_docs/NOTES.md`**
7. 论文草稿 → **`4_paper/part/`** · 图 → **`4_paper/figs/final/`**

---

## 目录

```text
mf_final_20261012/
├── README.md
├── 0_docs/                 ← 验收主文
├── pointers/
├── 1_primary/              ← ★ Deploy · 任务 · 形式化
├── 2_baselines/
├── 3_appendix/             ← 历史切片 · 作废 · 过程笔记
├── 4_paper/                ← part + figs/final；aamas_ 本地不入库
├── MANIFEST.json
└── SHA256SUMS.txt
```

本地可选（**git 忽略**）：`5_frozen_sources/` 完整历史归档。

---

## 推荐配对

| 角色 | 路径 |
|------|------|
| Deploy | `1_primary/01_models/residual_mf_best_r2_ref/repair_best.pkl` |
| 主证 M3 | `1_primary/05_formal_verify/mf_best_r2_same_X/M3_vg010/` |
| 任务 | `1_primary/03_task_eval/mf_best_r2_reference/` |
| 基线 | `2_baselines/BASELINE_COMPARE.md` |
| 图 | `4_paper/figs/final/` |
| 正文草稿 | `4_paper/part/` |

---

## 复现

```bash
export PYTHONPATH="$PWD:$PWD/method/meanfield:$PWD/method/deepset"
export JAX_PLATFORMS=cpu
CKPT=acceptance/mf_final_20261012/1_primary/01_models/residual_mf_best_r2_ref/repair_best.pkl
FREEZE_POSE=1 VG_ABS=0.10 THREAT_MID=1 MARABOU_TIMEOUT=900 \
  bash method/meanfield/scripts/run_mf_verify_planar.sh "$CKPT" /tmp/planar_r2_m3_repro
```

```bash
cd acceptance/mf_final_20261012 && sha256sum -c SHA256SUMS.txt
```
