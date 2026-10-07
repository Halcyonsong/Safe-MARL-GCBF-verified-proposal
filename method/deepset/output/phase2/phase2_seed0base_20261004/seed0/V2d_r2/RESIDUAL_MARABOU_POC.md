# Residual Marabou feasibility PoC

Date: 2026-10-05  
Checkpoint: `repair_r2_best.pkl`

## Purpose

Test whether the **deployed residual path** can enter the existing Marabou
pipeline:

```text
base DeepSet + gate * correction + clip + two-step held-action dynamics
```

The learned gate is not encoded as if it were a ReLU network. It contains
sigmoid, norms, minimums, and a physics gate. The PoC therefore uses the
following sound over-approximation:

```text
g ∈ [0, 1]
z = g * delta  (McCormick envelope using interval bounds on delta)
```

The relaxation contains every action produced by the actual deployed gate.
Therefore:

- `UNSAT` proves the actual residual safe for the queried square collision
  proxy and input box;
- `SAT` is only a witness for the relaxed gate/product model, not necessarily
  for the learned residual;
- `TIMEOUT` / `UNKNOWN` is not a proof.

## Results

| Box | AA | AO | Network | Note |
|---|---:|---:|---:|---|
| center, `dist=0.15`, `radius=0.01` | `UNSAT`, 0.53 s | `UNSAT`, 0.47 s | 7,841 vars; 4,246 eq; 8 inequalities | conservative local proof |
| center, `dist=0.15`, `radius=0.025` | `UNSAT`, 0.54 s | `UNSAT`, 0.47 s | same | conservative local proof |
| center, `dist=0.15`, `radius=0.05` | `UNSAT`, 0.54 s | `UNSAT`, 0.48 s | same | conservative local proof |
| center, `dist=0.08`, `radius=0.01` | `TIMEOUT` | `UNSAT`, 0.50 s | same | AA reaches the hard boundary |

Raw JSON outputs:

- `residual_marabou_poc_final.json`
- `residual_marabou_poc_r0025.json`
- `residual_marabou_poc_r005.json`
- `residual_marabou_poc_d008_r001.json`

## Reproduce

From the repository root:

```bash
python stage2/verify_residual_marabou.py \
  --checkpoint method/deepset/output/phase2/LATEST/seed0/V2d_r2/repair_r2_best.pkl \
  --buckets aa,ao \
  --radius 0.01 \
  --dist 0.15 \
  --timeout 10 \
  --out method/deepset/output/phase2/LATEST/seed0/V2d_r2/residual_marabou_poc.json
```

Implementation: `stage2/verify_residual_marabou.py`.

## Interpretation

This is a successful **feasibility result**, not a full L2 claim proof. The
current residual remains unproven on the full `l2_aa_near`, `l2_ao_near`, and
`l2_11_near` domains. The AA boundary timeout also shows that the unrestricted
`g ∈ [0,1]` relaxation is too loose for efficient full-domain solving.

The next verification-oriented experiment should use a gate-aware partition:

1. prove the gate-off region with the existing base verifier;
2. prove small gate-on AA/AO/(1,1) boxes with the residual encoding;
3. only then consider tightening the gate encoding or redesigning the residual
   to a smaller ReLU/linear certificate-friendly module.
