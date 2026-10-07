"""Shared helpers for Stage3 training logs (JSON summary + CSV curves).

CSV is the preferred input for plotting. JSON keeps full nested D-hit maps.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Flat columns written every eval step (stable for pandas / matplotlib).
EVAL_CSV_FIELDS: Sequence[str] = (
    "step",
    "wall_time_s",
    "loss",
    "mean_hit_D",
    "hit_aa",
    "hit_ao",
    "hit_11",
    "gate_aa",
    "gate_ao",
    "gate_11",
    "reward",
    "cost",
    "safe_rate",
    "finish_rate",
    "success_rate",
    "eligible",
    "is_best",
)


def _hit(d: Mapping[str, Any] | None, name: str) -> float | None:
    if not d:
        return None
    v = d.get(name)
    if isinstance(v, Mapping):
        return float(v.get("hit_rate_on_D", v.get("hit", 0.0)))
    if v is None:
        return None
    return float(v)


def _gate(d: Mapping[str, Any] | None, name: str) -> float | None:
    if not d:
        return None
    v = d.get(name)
    if v is None:
        return None
    return float(v)


def flatten_eval_row(
    row: Mapping[str, Any],
    *,
    wall_time_s: float | None = None,
    is_best: bool = False,
) -> dict[str, Any]:
    """Normalize one eval_history entry into a flat CSV row."""
    hit = row.get("hit_D") or {}
    gates = row.get("mean_gate_on_D") or {}
    return {
        "step": int(row.get("step", -1)),
        "wall_time_s": None if wall_time_s is None else round(float(wall_time_s), 3),
        "loss": row.get("loss"),
        "mean_hit_D": row.get("mean_hit_D"),
        "hit_aa": _hit(hit, "l2_aa_near"),
        "hit_ao": _hit(hit, "l2_ao_near"),
        "hit_11": _hit(hit, "l2_11_near"),
        "gate_aa": _gate(gates, "l2_aa_near"),
        "gate_ao": _gate(gates, "l2_ao_near"),
        "gate_11": _gate(gates, "l2_11_near"),
        "reward": row.get("reward"),
        "cost": row.get("cost"),
        "safe_rate": row.get("safe_rate"),
        "finish_rate": row.get("finish_rate"),
        "success_rate": row.get("success_rate"),
        "eligible": bool(row.get("eligible", False)),
        "is_best": bool(is_best),
    }


def write_eval_csv(
    path: Path | str,
    history: Iterable[Mapping[str, Any]],
    *,
    best_steps: set[int] | None = None,
    wall_times: Mapping[int, float] | None = None,
) -> Path:
    """Write eval curve CSV next to a checkpoint / summary."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_steps = best_steps or set()
    wall_times = wall_times or {}
    rows = []
    for h in history:
        step = int(h.get("step", -1))
        rows.append(
            flatten_eval_row(
                h,
                wall_time_s=wall_times.get(step),
                is_best=step in best_steps,
            )
        )
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(EVAL_CSV_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def write_summary_json(path: Path | str, summary: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    return out


def attach_run_meta(
    summary: dict[str, Any],
    *,
    method: str | None = None,
    train_seed: int | None = None,
    phase2_run_id: str | None = None,
    protocol_note: str | None = None,
) -> dict[str, Any]:
    """Optional Phase-2 identity fields for aggregation / plots."""
    meta = dict(summary.get("run_meta") or {})
    if method is not None:
        meta["method"] = method
    if train_seed is not None:
        meta["train_seed"] = int(train_seed)
    if phase2_run_id is not None:
        meta["phase2_run_id"] = phase2_run_id
    if protocol_note is not None:
        meta["protocol_note"] = protocol_note
    if meta:
        summary = dict(summary)
        summary["run_meta"] = meta
    return summary
