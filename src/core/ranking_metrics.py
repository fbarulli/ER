"""Deterministic ranking metrics for binary candidate lists.

The input is one scored evaluation population: positives are relevant
candidates and negatives are non-relevant candidates.  Metrics describe the
top of that score ranking, not a thresholded classifier decision.
"""

from __future__ import annotations

import numpy as np


def ranking_at_k(
    labels: np.ndarray, scores: np.ndarray, ks: tuple[int, ...]
) -> dict[str, float]:
    """Return precision/recall at each K plus Hits@1.

    Ties use stable input order, making every reported rank reproducible.
    A short candidate list uses its available prefix for Precision@K while
    Recall@K still divides by all relevant candidates in the population.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if len(labels) == 0:
        raise ValueError("ranking metrics require at least one candidate")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("ranking metric labels must be binary")
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    n_relevant = int(ranked.sum())
    result: dict[str, float] = {"hits_at_1": float(ranked[0] == 1)}
    for k in ks:
        if k < 1:
            raise ValueError(f"K must be >= 1, got {k}")
        top = ranked[:k]
        hits = int(top.sum())
        result[f"precision_at_{k}"] = hits / len(top)
        result[f"recall_at_{k}"] = hits / n_relevant if n_relevant else 0.0
    return result
