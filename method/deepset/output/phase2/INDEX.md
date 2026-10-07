# Phase-2 run index (active) — DeepSet track

Single entry point for current **DeepSet** Phase-2 products. Phase-1 frozen / pilot slots are untouched.

Layout: [`docs/ACTIVE_LAYOUT.md`](../../../../docs/ACTIVE_LAYOUT.md).  
Mean-field runs live under `method/meanfield/output/` (not here).

## Active lineage (use this)

```text
method/deepset/output/phase2/phase2_seed0base_20261004/          ← ACTIVE Phase-2 root
├── INDEX.md / SUMMARY.md                       (this folder)
├── SMOKE_MANIFEST.json                         Stage2 smoke + CE path
├── ce/l2_counterexamples_v2d_seed0.json        joint CE on seed0 base
├── stage2/l2_collision_report.json             base L2 diagnose
└── seed0/
    ├── V2d/                                    round-1 residual repair
    │   ├── repair_best.pkl                     ★ V2d r1 best
    │   ├── repair.pkl / repair.json / repair.csv
    │   ├── ce.json, d_hit_seed123.json, validate_32epi.txt
    │   ├── MANIFEST.json
    │   ├── compare_task/                       32/64 epi task dumps
    │   └── steps/                              intermediate step*.pkl
    └── V2d_r2/                                 CEGAR round-2 (sibling)
        ├── repair_r2_best.pkl                  ★ current safest residual
        ├── repair_r2.pkl / repair_r2.json / .csv
        ├── ce_r2.json, verify_r{1,2}_*.json
        ├── d_hit_seed123.json, validate_32epi.*
        ├── SNAPSHOT_CEGAR_R2.md
        ├── MANIFEST.json
        └── steps/                              intermediate step*.pkl
```

## Quick pointers (copy-paste)

| Role | Path |
|---|---|
| **Stage1 base (recommended)** | `method/deepset/output/stage1/redistill/stage1_redistill_20261004_183322/seed0/deepset_actor512_student_best.pkl` |
| **Phase-1 frozen base (canon)** | `method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl` |
| **V2d r1 residual** | `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d/repair_best.pkl` |
| **V2d r2 residual (CEGAR, current)** | `method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d_r2/repair_r2_best.pkl` |
| **Joint CE (base)** | `method/deepset/output/phase2/phase2_seed0base_20261004/ce/l2_counterexamples_v2d_seed0.json` |
| **Compare table** | `method/deepset/output/stage1/compare/gcbf_vs_seed0_20261004/` |

Symlinks (also under `method/deepset/output/phase2/`):

- `LATEST` → `phase2_seed0base_20261004`
- `ACTIVE_BASE.txt` — Stage1 base path
- `ACTIVE_RESIDUAL.txt` — current residual best (r2)

## Relocated / do not use as primary

| Path | Note |
|---|---|
| `method/deepset/output/phase2/phase2_seed0base_cegar_r2_20261004/` | **Moved** → see `MOVED.md`; artifacts live under `.../seed0/V2d_r2/` |
| `method/deepset/output/stage3/repair/*` | Phase-1 pilot only; do not overwrite |
| `method/deepset/output/stage1/redistill/.../seed2/` | **Aborted** mid-train; not ranked |

## How new runs should write

| Script | Default out |
|---|---|
| `run_phase2_seed.sh V2d 0 <RUN_ID>` | `method/deepset/output/phase2/<RUN_ID>/seed0/V2d/` |
| `run_cegar_v2d_r2.sh [parent_V2d]` | **`<parent>/../V2d_r2/`** (same seed lineage) |
| `run_stage1_redistill.sh` | `method/deepset/output/stage1/redistill/<id>/seed*/` |

Prefer **one** `RUN_ID` folder per base lineage; put CEGAR as sibling `V2d_r2`, not a second top-level run id.
