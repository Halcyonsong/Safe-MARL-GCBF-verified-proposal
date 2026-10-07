# MF 验收交付包（当前最佳 · 2026-10-06）

| | |
|---|---|
| **主验收** | residual **r2** · L2 `near_plus` · **PROVEN_FULL** |
| **模型** | `1_primary/01_models/residual_r2/repair_best.pkl` |
| **声明域** | min_d∈[0.14, 0.40]，\|close\|≤0.30，R=0.10 |
| **任务 @32** | success 82.0% · safe 96.5% · finish 84.0% |
| **主主张** | MF 有 ∀ 证书；基线无同级证书（非对称） |
| **非主验收** | 更宽包络 / mid-F → `3_appendix/` |

本目录为仓库内**唯一**验收交付入口。活权重指针仍在 `method/meanfield/output/ACTIVE_MF_*`。

---

## 先读顺序

1. **本页** — 结构与复现  
2. **`EVAL_AND_REPAIR.md`** — 修复轨迹 · @32/64/256 · 基线主表  
3. **`NOTES.md`** — 能写 / 不能写  
4. **`1_primary/00_acceptance/ACCEPTANCE_CERTIFICATE.md`** — 形式化证书  
5. 需要全表数字 → `RECORD.md`；写论文 → `docs/PAPER_AAMAS_CN.md`

---

## 目录

```text
acceptance/mf_best_20261006/
├── README.md                 ← 本页
├── EVAL_AND_REPAIR.md        ← ★ 日常主数据文
├── RECORD.md                 ← 全量数字账本（含附录）
├── NOTES.md                  ← 叙事边界
├── MANIFEST.json / SHA256SUMS.txt
├── pointers/                 ← 打包时 ACTIVE_*（与 1_primary/00_acceptance/ 同义快捷）
├── docs/                     ← 论文稿 / 口述草稿（非证据）
│   ├── PAPER_AAMAS_CN.md     ← ★ 主论文稿
│   ├── PAPER_DRAFT_CN.md     ← 长草稿
│   └── INTRODUCTION.md       ← 口述提纲（有已知不准，勿当终稿）
├── 1_primary/                ← ★ 主验收证据树
├── 2_baselines/              ← 基线对照 JSON + 表
├── 3_appendix/               ← 非主验收（更宽盒 / mid-F / 旁支权重）
│   └── sidebranch_midF_r3/   ← 优选旁支 ckpt（原根目录 4_sidebranch/）
└── 4_paper_figures/          ← 出图数据 + PDF/PNG（原 5_；编号已顺延）
```

| 想查… | 打开 |
|---|---|
| 修复作用 · 测评 · 基线 | **`EVAL_AND_REPAIR.md`** |
| 证书 / 模型 / CE / L2 报告 | **`1_primary/`** |
| 论文图 | **`4_paper_figures/`** |
| mid-F / 更宽盒 | **`3_appendix/`** |
| 能写 / 不能写 | **`NOTES.md`** |

---

## 复现主验证

```bash
source scripts/cuda_env.sh
bash method/meanfield/scripts/run_mf_verify_full.sh \
  acceptance/mf_best_20261006/1_primary/01_models/residual_r2/repair_best.pkl \
  near_plus
```

```bash
cd acceptance/mf_best_20261006 && sha256sum -c SHA256SUMS.txt
```
