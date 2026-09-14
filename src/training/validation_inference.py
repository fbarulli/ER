"""Deterministic helpers for post-training held-out SKU inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def resolve_best_checkpoint(source: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve exactly the checkpoint selected by load_best_model_at_end."""
    selections: list[tuple[float, int, str, Path, dict[str, Any]]] = []
    for state_path in sorted(source.glob("_checkpoints/**/checkpoint-*/trainer_state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        recorded = state.get("best_model_checkpoint")
        if not recorded:
            continue
        selected = state_path.parent.parent / Path(str(recorded)).name
        if not selected.is_dir():
            continue
        selections.append((
            float(state.get("best_metric", float("-inf"))),
            int(state.get("global_step", 0)),
            selected.as_posix(), selected, state,
        ))
    if not selections:
        raise FileNotFoundError(
            f"no materialized trainer-recorded best checkpoint under {source / '_checkpoints'}"
        )
    _, _, _, checkpoint, state = max(selections, key=lambda item: item[:3])
    return checkpoint, state


def threshold_assignment_metrics(scores: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    """Summarize SKU assignment coverage without inventing absent labels."""
    scores = np.asarray(scores, dtype=float)
    rows = []
    for threshold in thresholds:
        matched = int(np.sum(scores >= threshold))
        rows.append({
            "threshold": float(threshold),
            "matched": matched,
            "unmatched": int(len(scores) - matched),
            "matched_rate": float(matched / len(scores)) if len(scores) else 0.0,
        })
    return pd.DataFrame(rows)
