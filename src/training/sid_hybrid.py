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
DEFAULT_GAMMA = 0.1

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEFAULT_GAMMA",
    "hybrid_score",
    "hybrid_matrix",
    "leading_overlap_matrix",
    "veto_score",
    "veto_matrix",
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


def veto_score(
    cosine: float,
    sid_a: np.ndarray,
    sid_b: np.ndarray,
    *,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """One-sided alarm: subtract ``gamma`` when the coarse code diverges.

    A shared ``c0`` leaves the cosine untouched (no reward — the Phase 1
    lesson: true pairs share L0 only ~26% of the time, so a symmetric bonus
    punishes correct answers). A diverged ``c0`` subtracts ``gamma``
    (conflicts diverge ~100% of the time, so the alarm is all signal).
    Same veto philosophy as the lane's targeted_veto_gates: block, never
    promote.
    """
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    cosine = float(cosine)
    if not -1.0 <= cosine <= 1.0:
        raise ValueError(f"cosine must be in [-1, 1], got {cosine}")
    a = np.asarray(sid_a).ravel()
    b = np.asarray(sid_b).ravel()
    if a.shape != b.shape or a.shape[0] == 0:
        raise ValueError(f"SID length mismatch/empty: {a.shape} vs {b.shape}")
    return cosine - (gamma if int(a[0]) != int(b[0]) else 0.0)


def veto_matrix(
    cosine_matrix: np.ndarray,
    sku_sids: np.ndarray,
    canon_sids: np.ndarray,
    *,
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    """Full (n_sku, n_canon) veto matrix, vectorized on the c0 column only."""
    gamma = float(gamma)
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    cos = np.asarray(cosine_matrix, dtype=np.float64)
    sku = np.asarray(sku_sids, dtype=np.int64)
    canon = np.asarray(canon_sids, dtype=np.int64)
    if sku.ndim != 2 or canon.ndim != 2 or sku.shape[1] != canon.shape[1]:
        raise ValueError(f"SID shape mismatch: {sku.shape} vs {canon.shape}")
    if cos.shape != (sku.shape[0], canon.shape[0]):
        raise ValueError(
            f"cosine matrix {cos.shape} does not match "
            f"({sku.shape[0]}, {canon.shape[0]}) SKU/canonical counts"
        )
    mismatch = sku[:, None, 0] != canon[None, :, 0]
    return cos - gamma * mismatch.astype(np.float64)
