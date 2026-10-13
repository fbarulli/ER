"""Deterministic helpers for post-training held-out SKU inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.bundle import resolve_best_checkpoint as _resolve_best_checkpoint


def resolve_best_checkpoint(source: Path) -> tuple[Path, dict[str, Any]]:
    """Resolve exactly the checkpoint selected by load_best_model_at_end.

    Delegates to :func:`core.bundle.resolve_best_checkpoint` (the ONE
    trainer-selected best-checkpoint resolver); this wrapper keeps the
    ``training.validation_inference.resolve_best_checkpoint`` import surface
    the staged-model/inference tests monkeypatch.
    """
    resolved = _resolve_best_checkpoint(source)
    assert resolved is not None  # raise_on_missing=True by default
    return resolved


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
