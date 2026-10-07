# Stage1 — DeepSet distillation

Distill official GCBF+ Actor → fixed DeepSet student.

## Phase-1 canon (do not overwrite)

```text
method/deepset/output/stage1/frozen/deepset_actor512_stage1_baseline.pkl
```

## Re-distill stronger base (recommended)

From **frozen GCBF+ teacher** (no GNN retrain by default), multi-seed, longer steps, CSV curves:

```bash
# default: seeds 0,1,2 · steps 12000 · teacher seed0_20261001232611
bash method/deepset/scripts/run_stage1_redistill.sh stage1_redistill_$(date +%Y%m%d)

# custom
bash method/deepset/scripts/run_stage1_redistill.sh my_run 0,1,2 12000
```

Outputs:

```text
method/deepset/output/stage1/redistill/<run_id>/
  seed0/ deepset_actor512_student{,_best}.pkl .json .csv
         validate_32epi.txt  d_hit_seed123.json  MANIFEST.json
  seed1/ ...
  ranking.json  SELECTION.md  figures/
```

Promote only after reviewing `SELECTION.md` (copy to a **new** frozen name, or pass `--base-checkpoint` into Stage3).

## Single-seed legacy launcher

```bash
bash method/deepset/scripts/run_stage1.sh baselines/gcbfplus/log/gcbf+/seed0_... 12000 16 1000 512 0
bash method/deepset/scripts/validate_stage1.sh method/deepset/output/stage1/actor/deepset_actor512_student_best.pkl 32
```

## Logging

Each run writes `*.json` (eval_history) + `*.csv` (loss/mae/success curves).

```bash
python scripts/plot_train_curves.py \
  --glob 'method/deepset/output/stage1/redistill/*/seed*/deepset_actor512_student.csv' \
  --out-dir method/deepset/output/stage1/redistill/figures
```

QP distill path was retired with `archive/` (2026-10-07); backup under `/tmp/opencode/...backup-20261007/archive/`.
