"""Hierarchical Semantic IDs via residual-quantized KMeans (Phase 0, analysis-only).

Phase 0 asks whether discrete SID prefix overlap separates true pairs from
volume/pack-conflict pairs where bi-encoder cosine collapses.  Nothing here
trains an encoder: codebooks are fit on FROZEN zero-shot embeddings and the
module is pure numpy+sklearn (no ``core`` imports, so importing it never
touches config).

Method choice (TIGER / RQ-KMeans): TIGER (Rajput et al., 2023,
"Recommender Systems with Generative Retrieval") builds Semantic IDs with a
trained RQ-VAE over residuals.  Phase 0 substitutes RQ-KMeans — greedy
k-means on the residual at each level — because the embeddings are frozen
(no gradient training exists to fit an RQ-VAE against) and sklearn
``KMeans(random_state=seed, n_init=10)`` is exactly reproducible.  The
residual structure is identical (level ``l`` quantizes what levels ``< l``
failed to reconstruct), so prefix-overlap semantics carry over: items that
share a coarse cluster share ``c0``, and deeper levels only refine.

SID layout: ``(c0, c1, c2)`` with ``n_clusters`` codes per level, plus a
collision ``tidbit`` (occurrence index among identical full codes, in row
order) so the persisted catalog table is unique per GTIN even when two
canonicals quantize identically.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

DEFAULT_N_LEVELS = 3
DEFAULT_N_CLUSTERS = 256
DEFAULT_SEED = 42

__all__ = [
    "DEFAULT_N_LEVELS",
    "DEFAULT_N_CLUSTERS",
    "DEFAULT_SEED",
    "fit_rq_kmeans",
    "assign_sids",
    "add_collision_tidbits",
    "prefix_overlap",
    "unique_ids_proportion",
    "codebook_usage",
    "save_codebooks",
    "load_codebooks",
    "save_sid_table",
    "load_sid_table",
]


def fit_rq_kmeans(
    embeddings: np.ndarray,
    n_levels: int = DEFAULT_N_LEVELS,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Fit residual-quantization codebooks (RQ-KMeans, TIGER-style levels).

    Level 0 runs k-means on ``embeddings``; each deeper level runs k-means
    on the residual left by the levels above it.  Deterministic: every level
    uses ``KMeans(random_state=seed, n_init=10)``.

    Returns ``(n_levels, k_effective, dim)`` centroids.  ``k_effective`` is
    ``min(n_clusters, n_rows)`` — k-means cannot mint more clusters than
    points, so smoke-sized fits degrade loudly (via the returned shape, and
    callers should print it) instead of crashing.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"embeddings must be a non-empty 2D array, got {X.shape}")
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    n, dim = X.shape
    k_effective = min(int(n_clusters), n)
    codebooks = np.zeros((int(n_levels), k_effective, dim), dtype=np.float64)
    residual = X.copy()
    for level in range(int(n_levels)):
        kmeans = KMeans(
            n_clusters=k_effective, random_state=int(seed), n_init=10
        )
        labels = kmeans.fit_predict(residual)
        codebooks[level] = kmeans.cluster_centers_
        residual = residual - codebooks[level][labels]
    return codebooks


def assign_sids(embeddings: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Greedily assign residual-quantization codes; returns ``(n, n_levels)`` ints.

    Mirrors the fit path exactly: at each level pick the nearest centroid of
    the current residual, then subtract it before the next level.  Distance
    is chunked squared-Euclidean (``||r||^2 - 2 r.C + ||C||^2``) so a
    13k x 256 x 384 assignment never materializes the full broadcast.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    codebooks = np.asarray(codebooks, dtype=np.float64)
    if codebooks.ndim != 3:
        raise ValueError(f"codebooks must be 3D (L, K, d), got {codebooks.shape}")
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {X.shape}")
    if X.shape[1] != codebooks.shape[2]:
        raise ValueError(
            f"dim mismatch: embeddings {X.shape[1]} vs codebooks {codebooks.shape[2]}"
        )
    n, n_levels = X.shape[0], codebooks.shape[0]
    sids = np.zeros((n, n_levels), dtype=np.int64)
    residual = X.copy()
    chunk = 2048
    for level in range(n_levels):
        centers = codebooks[level]
        center_norm = np.sum(centers**2, axis=1)
        for start in range(0, n, chunk):
            block = residual[start : start + chunk]
            dist2 = (
                np.sum(block**2, axis=1, keepdims=True)
                - 2.0 * block @ centers.T
                + center_norm[None, :]
            )
            best = np.argmin(dist2, axis=1)
            sids[start : start + chunk, level] = best
            residual[start : start + chunk] -= centers[best]
    return sids


def add_collision_tidbits(sids: np.ndarray) -> np.ndarray:
    """Append the collision tidbit: occurrence index within identical full codes.

    The first row carrying a code gets tidbit 0, the next identical row
    tidbit 1, and so on, in row order — deterministic with no RNG.  Returns
    ``(n, n_levels + 1)`` ints, so catalog tables are unique per row even
    when two canonicals quantize identically.
    """
    codes = np.asarray(sids)
    if codes.ndim != 2 or codes.shape[0] == 0:
        raise ValueError(f"sids must be a non-empty 2D array, got {codes.shape}")
    seen: dict[tuple[int, ...], int] = {}
    tidbits = np.zeros(codes.shape[0], dtype=np.int64)
    for i, row in enumerate(codes.tolist()):
        key = tuple(int(v) for v in row)
        tidbits[i] = seen.get(key, 0)
        seen[key] = seen.get(key, 0) + 1
    return np.column_stack([codes.astype(np.int64), tidbits])


def prefix_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Longest common SID prefix as a fraction of levels (TIGER prefix semantics).

    Identical codes score 1.0; a pair diverging at ``c0`` scores 0.0 — the
    coarse level carries the split decision, deeper levels only refine it.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        raise ValueError(f"SID length mismatch: {a.shape} vs {b.shape}")
    if a.shape[0] == 0:
        return 0.0
    depth = int(np.argmax(a != b)) if bool(np.any(a != b)) else int(a.shape[0])
    return depth / float(a.shape[0])


def unique_ids_proportion(sids: np.ndarray) -> float:
    """Fraction of rows with a distinct full SID (1.0 = no collisions)."""
    codes = np.asarray(sids)
    if codes.ndim != 2 or codes.shape[0] == 0:
        return 0.0
    return float(len(np.unique(codes, axis=0))) / float(codes.shape[0])


def codebook_usage(sids: np.ndarray, n_clusters: int) -> list[float]:
    """Per-level fraction of the ``n_clusters`` codes actually used.

    Values are capped at 1.0 so SIDs assigned under a larger codebook still
    report sanely.  A collapsed level (usage near 0) means that depth carries
    no signal and its agreement rate must not drive the GO/STOP call.
    """
    codes = np.asarray(sids)
    if codes.ndim != 2:
        raise ValueError(f"sids must be 2D, got {codes.shape}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    return [
        min(1.0, float(len(np.unique(codes[:, level]))) / float(n_clusters))
        for level in range(codes.shape[1])
    ]


def save_codebooks(path: str | Path, codebooks: np.ndarray) -> Path:
    """Persist codebooks (+ shapes) to ``.npz``; returns the written path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    codebooks = np.asarray(codebooks, dtype=np.float64)
    np.savez(
        out,
        codebooks=codebooks,
        n_levels=np.int64(codebooks.shape[0]),
        n_clusters=np.int64(codebooks.shape[1]),
        dim=np.int64(codebooks.shape[2]),
    )
    return out


def load_codebooks(path: str | Path) -> np.ndarray:
    """Load codebooks saved by :func:`save_codebooks` (validates 3D shape)."""
    data = np.load(str(path), allow_pickle=False)
    codebooks = np.asarray(data["codebooks"], dtype=np.float64)
    if codebooks.ndim != 3:
        raise ValueError(f"codebooks must be 3D (L, K, d), got {codebooks.shape}")
    return codebooks


_SID_TABLE_COLUMNS = ("gtin", "c0", "c1", "c2", "tidbit")


def save_sid_table(
    path: str | Path, gtins: list[str] | np.ndarray, sids: np.ndarray
) -> Path:
    """Write the catalog SID table (``gtin,c0,c1,c2,tidbit`` CSV).

    ``sids`` must already carry the tidbit column — see
    :func:`add_collision_tidbits` — so what lands on disk is exactly the
    unique-per-GTIN code the analysis evaluated (no silent re-derivation).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gtins = [str(g) for g in list(gtins)]
    codes = np.asarray(sids, dtype=np.int64)
    if codes.ndim != 2 or codes.shape[1] != len(_SID_TABLE_COLUMNS) - 1:
        raise ValueError(
            "sids must be (n, 4) [c0, c1, c2, tidbit] — "
            f"got {codes.shape}; run add_collision_tidbits first"
        )
    if len(gtins) != codes.shape[0]:
        raise ValueError(
            f"gtin/sid count mismatch: {len(gtins)} vs {codes.shape[0]}"
        )
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_SID_TABLE_COLUMNS)
        for gtin, row in zip(gtins, codes.tolist()):
            writer.writerow([gtin, *[int(v) for v in row]])
    return out


def load_sid_table(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read a table written by :func:`save_sid_table`; validates the header."""
    gtins: list[str] = []
    rows: list[list[int]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _SID_TABLE_COLUMNS:
            raise ValueError(
                f"bad SID table header {reader.fieldnames}; "
                f"expected {list(_SID_TABLE_COLUMNS)}"
            )
        for record in reader:
            gtins.append(str(record["gtin"]))
            rows.append([int(record[c]) for c in _SID_TABLE_COLUMNS[1:]])
    return gtins, np.asarray(rows, dtype=np.int64)
