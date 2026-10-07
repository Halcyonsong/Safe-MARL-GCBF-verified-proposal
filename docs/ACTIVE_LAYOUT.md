# Active workspace layout

Restructured 2026-10-07 for GitHub-friendly top-level grouping.  
Hard cut: old roots `stage*`, `stage_mf`, `baseline`, `runs`, `logs`, `config` no longer exist at repo root.

Backup (pre-move): `/tmp/opencode/Safe-MARL-GCBF-verified-proposal.backup-20261007/`

## Top level

```text
method/          core method code + configs + scripts + outputs (+ deepset/tests)
baselines/       per-baseline code/config/log/output/scripts + compare/
acceptance/      delivery packs (internal files not fine-tuned this round)
docs/            design / catalogs
core/            official GCBF+ (import gcbfplus)
scripts/         ★ shared only (see below) — track launchers live under method/*/scripts
refer/
```

## Shared `scripts/` vs track scripts

| Path | Role | Keep at root? |
|---|---|---|
| `scripts/cuda_env.sh` | JAX/CUDA env；几乎所有 launcher `source "$ROOT/scripts/cuda_env.sh"` | **是**（跨 method + baselines 共用） |
| `scripts/plot_train_curves.py` | 通用 eval/repair CSV 画曲线 | **是**（DeepSet / MF 都能用） |
| `method/*/scripts/` · `baselines/*/scripts/` | 各轨训练 / 验证 / 评测入口 | 已按轨拆分 |

根 `scripts/` **不是**旧的大脚本仓；训练入口已迁走。不要把 `run_mf_*` / `run_stage*` 再塞回这里。

## Tests（按轨归类）

| Path | Covers |
|---|---|
| `method/deepset/tests/test_residual_gate.py` | Stage3 residual gate（`stage3.residual_policy`） |
| `method/deepset/tests/test_dual_verification.py` | Stage2 dual-threat IBP / Marabou 语义 |
| `method/deepset/tests/conftest.py` | pytest 启动时 `ensure_import_path()` |

- 根目录 **不再有** `tests/`（原 DeepSet 单测已迁入上表）。
- MF 若补测 → `method/meanfield/tests/`。
- 跑：`python -m pytest method/deepset/tests/ -q`

## Two method tracks

| Track | Code | Config | Scripts | Tests | Output |
|---|---|---|---|---|---|
| **DeepSet** (frozen reference) | `method/deepset/stage{1,2,3}/` | `method/deepset/config/` | `method/deepset/scripts/` | `method/deepset/tests/` | `method/deepset/output/` |
| **Mean-field** (main acceptance) | `method/meanfield/stage_mf/` | `method/meanfield/config/` | `method/meanfield/scripts/` | （暂无） | `method/meanfield/output/` |

Import names unchanged: `stage1` / `stage2` / `stage3` / `stage_mf`.  
`core.paths.ensure_import_path()` adds `method/deepset` and `method/meanfield` to `sys.path`.

## Baselines

Each algorithm:

```text
baselines/<algo>/{code,config,log,output,scripts}
```

| Algo | Notes |
|---|---|
| `gcbfplus` | Official teacher; log under `log/gcbf+/` |
| `gcbf` | mid / weak / official outputs under `output/gcbf*` |
| `handcbf` | DecShareCBF; no train weights |
| `mappo_lag` | CMDP + Lagrangian |
| `informarl` | local-graph GNN + PPO |

Compare / transfer / box MC:

```text
baselines/compare/{scripts,output,BASELINE_COMPARE.md,TRANSFER_COMPARE.md}
```

`import baseline.<algo>` is wired by `core.paths` to `baselines/<algo>/code` (no root `baseline/` directory).

## Acceptance

```text
acceptance/mf_best_20261006/           # ★ sole delivery pack (MF-r2 near_plus)
  1_primary/  2_baselines/  3_appendix/  4_paper_figures/  docs/
  # midF_r3 sidebranch weights live under 3_appendix/sidebranch_midF_r3/
```

Former duplicate `acceptance/meanfield_r2/` removed 2026-10-07 (was byte-identical to `1_primary/`).  
Live training pointers remain `method/meanfield/output/ACTIVE_MF_*`.

GitHub tracking rules: root [`.gitignore`](../.gitignore) (acceptance full; method/baseline dumps mostly ignored; `refer/` local-only).

## Path migration cheat sheet

| Old | New |
|---|---|
| `stage_mf/` | `method/meanfield/stage_mf/` |
| `stage{1,2,3}/` | `method/deepset/stage{1,2,3}/` |
| `runs/meanfield/` | `method/meanfield/output/` |
| `runs/phase2/` `runs/stage*/` | `method/deepset/output/{phase2,stage*}/` |
| `baseline/<algo>/` | `baselines/<algo>/code/` |
| `runs/baseline/<algo>/` | `baselines/<algo>/output/<algo>/` |
| `logs/DoubleIntegrator/gcbf+/` | `baselines/gcbfplus/log/gcbf+/` |
| `config/meanfield/` | `method/meanfield/config/` |
| `config/stage*/` | `method/deepset/config/stage*/` |
| `runs/compare_baselines/` | `baselines/compare/output/` |
| `acceptance_mf_best_20261006/` | `acceptance/mf_best_20261006/` |
| `acceptance/meanfield_r2/` | `acceptance/mf_best_20261006/1_primary/`（重复包已删） |
| `scripts/run_mf_*.sh` | `method/meanfield/scripts/` |
| `scripts/run_stage*.sh` `run_phase2*.sh` | `method/deepset/scripts/` |
| `scripts/run_baseline_*.sh` | `baselines/<algo>/scripts/` |

## Active pointers

```text
method/meanfield/output/ACTIVE_MF_BASE.txt
method/meanfield/output/ACTIVE_MF_RESIDUAL.txt
method/meanfield/output/ACTIVE_MF_VERIFY.txt
method/deepset/output/phase2/ACTIVE_BASE.txt
method/deepset/output/phase2/ACTIVE_RESIDUAL.txt
```

Do **not** point DeepSet `ACTIVE_*` at mean-field. Do **not** promote MF side-branches without an explicit lock.

## Archives

`archive/` **已于 2026-10-07 整夹清除**（含 Phase-1 重 dump / 废弃脚本 / 过程文档）。  
若需找回：`/tmp/opencode/Safe-MARL-GCBF-verified-proposal.backup-20261007/archive/`。

Active docs only: `ACTIVE_LAYOUT.md` · `MEANFIELD_VERIFY.md` · `DESIGN_PROPOSAL.md`（历史）· `README.md`。
