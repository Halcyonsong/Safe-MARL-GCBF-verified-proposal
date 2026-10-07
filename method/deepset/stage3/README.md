# Stage3 — Counterexample-guided repair

Repairs the **frozen Stage1 DeepSet**. Formal verification stays in `stage2/`.

## Main path (keep these)

| Script | Role |
|---|---|
| `collect_counterexamples.py` | Sample CEs on preventable D |
| `retrain_l2_safe.py` | Full-net repair (**R1**) |
| `retrain_residual_gate.py` | Residual `a=clip(base+g·δ)` (**V2 / V2d**) |
| `validate_residual.py` | 32-epi rollout for residual ckpts |
| `metrics_io.py` | Shared JSON + **CSV curve** writers |

## Preferred launch (Phase-2)

```bash
# Joint main (AA+AO+11) — recommended DeepSet path
bash method/deepset/scripts/run_phase2_seed.sh V2d 0 phase2_$(date +%Y%m%d)

# Ablation: AA+AO residual only
bash method/deepset/scripts/run_phase2_seed.sh V2 0 phase2_$(date +%Y%m%d)

# Strong-safety full-net reference
bash method/deepset/scripts/run_phase2_seed.sh R1 0 phase2_$(date +%Y%m%d)
```
Outputs only under `method/deepset/output/phase2/<run_id>/seed*/METHOD/`:

```text
ce.json
repair.pkl / repair_best.pkl
repair.json          # summary + eval_history
repair.csv           # plot-ready curves
d_hit_seed123.json
validate_32epi.txt
MANIFEST.json
run.log
```

## Curves

```bash
python scripts/plot_train_curves.py \
  --csv method/deepset/output/phase2/phase2_YYYYMMDD/seed0/V2d/repair.csv \
  --out-dir method/deepset/output/phase2/phase2_YYYYMMDD/figures
```

## Label modes

| mode | meaning |
|---|---|
| `min_intervention` | min ‖a−π‖ among two-step-safe actions (V2/V2d default) |
| `max_clearance` | max clearance (R1 strong safety; higher task tax) |

## Phase-1 canon (do not overwrite)

```text
method/deepset/output/stage3/repair/deepset_l2_safe_repair*.pkl
method/deepset/output/stage3/repair/deepset_residual_gate_v2*.pkl
method/deepset/output/stage3/ce/l2_counterexamples.json
```

Legacy pilot trainers (QP, hybrid, repulsive) were removed with `archive/` (2026-10-07).
