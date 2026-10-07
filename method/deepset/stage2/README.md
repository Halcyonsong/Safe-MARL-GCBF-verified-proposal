# Stage2 — L2 domain & verification

| Module | Role |
|---|---|
| `l2_domain.py` / `l2_query.py` / `bounds.py` | Preventable D, dynamics, boxes |
| `verify_l2.py` / `verify_l1.py` | Formal / sampling verify |
| `eval_l2_gcbfplus.py` | Same-standard D-hit vs GCBF+ |
| `export_onnx.py` / `marabou_*` | Toolchain |

```bash
bash method/deepset/scripts/run_stage2_l2.sh
python method/deepset/stage2/eval_l2_gcbfplus.py --samples 2000
```

Repair code lives in **stage3/** (not here). Symlinks: `ce` → `../stage3/ce`, `repair` → `../stage3/repair`.
