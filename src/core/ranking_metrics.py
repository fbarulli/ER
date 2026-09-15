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
    if any(k < 1 for k in ks):
        raise ValueError(f"K must be >= 1, got {min(ks)}")
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


def ranking_at_k_by_query(
    labels: np.ndarray,
    scores: np.ndarray,
    query_ids: np.ndarray,
    ks: tuple[int, ...],
) -> dict[str, float]:
    """Aggregate retrieval metrics over independent source queries.

    Each query must have at least one relevant candidate.  Recall@K is the
    fraction of queries whose relevant candidate appears in the top K; it is
    intentionally not divided by the number of positive pair rows globally.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    query_ids = np.asarray(query_ids)
    if labels.ndim != 1 or scores.ndim != 1 or query_ids.ndim != 1:
        raise ValueError("ranking inputs must be aligned one-dimensional arrays")
    if not (len(labels) == len(scores) == len(query_ids)) or len(labels) == 0:
        raise ValueError("ranking inputs must be non-empty and aligned")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("ranking metric labels must be binary")

    result: dict[str, float] = {}
    hits_by_k: dict[int, list[int]] = {k: [] for k in ks}
    precision_by_k: dict[int, list[float]] = {k: [] for k in ks}
    for query in pd_unique(query_ids):
        mask = query_ids == query
        group = labels[mask][np.argsort(-scores[mask], kind="stable")]
        if not group.any():
            continue
        for k in ks:
            top = group[:k]
            hits_by_k[k].append(int(top.sum() > 0))
            precision_by_k[k].append(float(top.mean()) if len(top) else 0.0)
    result["hits_at_1"] = (
        float(np.mean(hits_by_k[1])) if 1 in hits_by_k and hits_by_k[1] else 0.0
    )
    for k in ks:
        hits = hits_by_k[k]
        result[f"precision_at_{k}"] = float(np.mean(precision_by_k[k])) if hits else 0.0
        result[f"recall_at_{k}"] = float(np.mean(hits)) if hits else 0.0
    return result


def pd_unique(values: np.ndarray) -> np.ndarray:
    """Stable unique values without adding a pandas dependency to this module."""
    return np.asarray(list(dict.fromkeys(values.tolist())))
