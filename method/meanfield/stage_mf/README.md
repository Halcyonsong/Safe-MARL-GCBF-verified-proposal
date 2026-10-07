# stage_mf — Mean-field MLP (parallel track)

`architecture=meanfield_mlp`, **`feature_version=5`**.

## Feature v5 (current)

```text
ego(8)
+ agent ×4 × (n, min_d, close_max, close_wmean, inv_ttc_max)  # 20
+ obs   ×4 × (n, min_d, close_max, inv_ttc_max)               # 16
= 44 dims
```

- **Max threat:** `min_d`, `close_max`, **`inv_ttc_max`** (nearest-in-quad only)
- **Mean threat:** `n` + **distance-weighted** `close_wmean`, `w=exp(-d/(0.5·R_comm))`
- **4 quadrants** (8-sector not used yet)
- No global nearest

Empty: `n=0`, `min_d=comm_radius`, `close_*=0`, `inv_ttc=0`.

MLP: `44 → 512 → ReLU ×3 → 2`, deploy `clip ±1`.

## Prior

| ver | note | success@32 |
|---|---|---:|
| v4 | pure 4quad equal-mean | 72.3% |
| v3 | 1a+1o + quad rest | ~65% short (aborted) |

## Launch

```bash
# distill
bash method/meanfield/scripts/run_mf_distill.sh mf_distill_$(date +%Y%m%d)_v5 0 10000

# residual repair (physics gate; freeze ACTIVE_MF_BASE)
bash method/meanfield/scripts/run_mf_residual_repair.sh
# or:
python method/meanfield/stage_mf/collect_ce.py \
  --checkpoint "$(cat method/meanfield/output/ACTIVE_MF_BASE.txt)" \
  --episodes 96 --out-json method/meanfield/output/repair/.../ce_baseline.json
python method/meanfield/stage_mf/retrain_residual.py \
  --base-checkpoint ... --ce-json ... --checkpoint ...

# formal verify (main acceptance box)
bash method/meanfield/scripts/run_mf_verify_full.sh \
  "$(cat method/meanfield/output/ACTIVE_MF_RESIDUAL.txt)" near_plus
```

## Residual

```text
a = clip( base(x) + g_phys(x) * δ(x) )
g_phys from min_d + inv_ttc (no learned sigmoid)
```

Outputs under `method/meanfield/output/` only. DeepSet `ACTIVE_*` untouched.
