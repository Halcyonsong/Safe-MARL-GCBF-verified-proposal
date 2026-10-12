#!/usr/bin/env python3
"""Re-rank repair checkpoints with full N-epi eval. Does not write ACTIVE_*."""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[3]
    run = pathlib.Path(args.run_dir).resolve()
    out = run / f"manual_rerank_{args.episodes}epi_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    names = ["repair_best.pkl", "repair.pkl", "repair_last.pkl"]
    ckpts = [run / n for n in names if (run / n).is_file()]
    ckpts += sorted(run.glob("repair_step*.pkl"))
    seen, uniq = set(), []
    for p in ckpts:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    ckpts = uniq
    if not ckpts:
        raise SystemExit(f"no repair*.pkl in {run}")
    active = root / "method/meanfield/output/ACTIVE_MF_RESIDUAL.txt"
    before = active.read_text() if active.exists() else ""
    rows = []
    for ck in ckpts:
        jpath = out / f"{ck.stem}_validate_{args.episodes}epi.json"
        cmd = [
            sys.executable,
            str(root / "method/meanfield/stage_mf/validate_residual.py"),
            "--checkpoint", str(ck),
            "--episodes", str(args.episodes),
            "--seed", str(args.seed),
            "--out-json", str(jpath),
        ]
        print("[rerank]", ck.name, flush=True)
        subprocess.check_call(cmd, cwd=str(root))
        m = json.loads(jpath.read_text())
        suc = float(m["success_rate"])
        safe = float(m["safe_rate"])
        fin = float(m["finish_rate"])
        cost = float(m["cost"])
        rows.append({
            "name": ck.name,
            "path": str(ck),
            "success": suc,
            "safe": safe,
            "finish": fin,
            "cost": cost,
            "score_safe_success": 2.0 * safe + suc,
        })
    rows.sort(key=lambda r: (-r["safe"], -r["success"], r["cost"]))
    (out / "RERANK.json").write_text(json.dumps({"pick": rows[0] if rows else None, "candidates": rows}, indent=2) + "\n")
    lines = [
        f"# Manual re-rank @ {args.episodes}-epi seed={args.seed}",
        "",
        "| rank | ckpt | suc | safe | fin | cost |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        mark = " <- pick" if i == 1 else ""
        lines.append(
            f"| {i} | `{r["name"]}` | {100*r["success"]:.1f}% | {100*r["safe"]:.1f}% | "
            f"{100*r["finish"]:.1f}% | {r["cost"]:.3f} |{mark}"
        )
    pick = rows[0]["path"] if rows else ""
    lines += ["", f"**Pick:** `{pick}`", "", "cp pick -> repair_manual_pick.pkl for next INIT_RESIDUAL"]
    (out / "RERANK.md").write_text("\n".join(lines) + "\n")
    print((out / "RERANK.md").read_text())
    after = active.read_text() if active.exists() else ""
    if before != after:
        raise SystemExit("ERROR: ACTIVE_MF_RESIDUAL changed")
    print("[rerank] ACTIVE_* unchanged OK")

if __name__ == "__main__":
    main()
