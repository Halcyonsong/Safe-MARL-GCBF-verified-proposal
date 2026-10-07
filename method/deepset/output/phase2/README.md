# Phase-2 full-pipeline runs

**Start here:** [`INDEX.md`](INDEX.md)

## Active tree

```text
method/deepset/output/phase2/
  INDEX.md                         ← pointers + layout
  LATEST → phase2_seed0base_20261004/
  ACTIVE_BASE.txt / ACTIVE_RESIDUAL.txt
  phase2_seed0base_20261004/       ← current lineage
    SUMMARY.md
    ce/  stage2/  seed0/V2d/  seed0/V2d_r2/
```

## Conventions

| Item | Rule |
|---|---|
| One lineage per base | `method/deepset/output/phase2/<run_id>/seed{k}/` |
| V2d main repair | `.../V2d/` (`repair_best.pkl`) |
| CEGAR follow-up | `.../V2d_r2/` (sibling, not a new top-level run id) |
| Step checkpoints | `.../steps/` only; keep `*_best.pkl` at top |
| Phase-1 pilot | stay in `method/deepset/output/stage3/repair/` — do not clobber |

## Launch

```bash
# V2d on seed0 base
BASE_CHECKPOINT=method/deepset/output/stage1/redistill/.../seed0/..._best.pkl \
  bash method/deepset/scripts/run_phase2_seed.sh V2d 0 phase2_seed0base_YYYYMMDD

# CEGAR r2 (writes sibling V2d_r2 by default)
bash method/deepset/scripts/run_cegar_v2d_r2.sh \
  method/deepset/output/phase2/phase2_seed0base_20261004/seed0/V2d
```

Layout: `docs/ACTIVE_LAYOUT.md`.
