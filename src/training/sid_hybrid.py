"""Hybrid SID+cosine fusion (Phase 1, analysis-only).

The bi-encoder keeps its job (encode, cosine, retrieve). The SID path is a
training-free second opinion: two items sharing a coarse code prefix are
likely the same family; a pair diverging at ``c0`` is likely a mismatch even
when cosine is high (the collapse regime). Fusion is additive so the lane
runs exactly as today when ``beta=0``:

    hybrid = alpha * cosine + beta * prefix_overlap(sid_a, sid_b)

Pure numpy (no ``core`` imports, so importing this never touches config).
``alpha``/``beta`` live in the caller's config/CLI layer — this module only
enforces they are non-negative weights.
"""

from __future__ import annotations

import numpy as np

from training.semantic_ids import prefix_overlap

DEFAULT_ALPHA = 0.8
DEFAULT_BETA = 0.2

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "hybrid_score",
    "hybrid_matrix",
    "leading_overlap_matrix",
]


def _check_weights(alpha: float, beta: float) -> tuple[float, float]:
    alpha, beta = float(alpha), float(beta)
    if not (alpha >= 0.0 and beta >= 0.0):
        raise ValueError(f"alpha/beta must be non-negative, got {alpha}/{beta}")
    if alpha == 0.0 and beta == 0.0:
        raise ValueError("alpha and beta cannot both be zero")
    return alpha, beta


def hybrid_score(
    cosine: float,
    sid_a: np.ndarray,
    sid_b: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> float:
    """Fuse one pair: ``alpha*cosine + beta*prefix_overlap`` (length-checked)."""
    alpha, beta = _check_weights(alpha, beta)
    cosine = float(cosine)
    if not -1.0 <= cosine <= 1.0:
        raise ValueError(f"cosine must be in [-1, 1], got {cosine}")
    return alpha * cosine + beta * float(prefix_overlap(sid_a, sid_b))


def leading_overlap_matrix(
    sku_sids: np.ndarray, canon_sids: np.ndarray
) -> np.ndarray:
    """Full (n_sku, n_canon) prefix-overlap matrix, vectorized.

    Entry [i, j] = fraction of leading SID levels on which row i and row j
    agree (1.0 identical … 0.0 diverge at c0). Computed as the row-mean of
    the cumulative-AND of per-level equality — no Python pair loop, so
    1k x 13k x 3 stays a ~40MB boolean temp.
    """
    sku = np.asarray(sku_sids, dtype=np.int64)
    canon = np.asarray(canon_sids, dtype=np.int64)
    if sku.ndim != 2 or canon.ndim != 2 or sku.shape[1] != canon.shape[1]:
        raise ValueError(
            f"SID shape mismatch: {sku.shape} vs {canon.shape} "
            "(both must be 2D with equal level counts)"
        )
    if sku.shape[0] == 0 or canon.shape[0] == 0:
        raise ValueError("SID matrices must be non-empty")
    equal = sku[:, None, :] == canon[None, :, :]
    leading = np.logical_and.accumulate(equal, axis=2)
    return leading.mean(axis=2)


def hybrid_matrix(
    cosine_matrix: np.ndarray,
    sku_sids: np.ndarray,
    canon_sids: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> np.ndarray:
    """Full (n_sku, n_canon) hybrid matrix from a cosine matrix + SID tables."""
    alpha, beta = _check_weights(alpha, beta)
    cos = np.asarray(cosine_matrix, dtype=np.float64)
    sku = np.asarray(sku_sids)
    canon = np.asarray(canon_sids)
    if cos.shape != (sku.shape[0], canon.shape[0]):
        raise ValueError(
            f"cosine matrix {cos.shape} does not match "
            f"({sku.shape[0]}, {canon.shape[0]}) SKU/canonical counts"
        )
    return alpha * cos + beta * leading_overlap_matrix(sku, canon)
