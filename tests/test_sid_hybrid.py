"""SID Phase 0 oracles — deterministic, no network, no model download."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans

from training.semantic_ids import (
    add_collision_tidbits,
    assign_sids,
    codebook_usage,
    fit_rq_kmeans,
    load_codebooks,
    load_sid_table,
    prefix_overlap,
    save_codebooks,
    save_sid_table,
    unique_ids_proportion,
)
from training.sid_hybrid import (
    hybrid_matrix,
    hybrid_score,
    leading_overlap_matrix,
)


def _blobs() -> np.ndarray:
    rng = np.random.RandomState(7)
    return np.vstack(
        [
            rng.normal(loc=center, scale=0.5, size=(120, 4))
            for center in ([0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0])
        ]
    )


def test_rq_kmeans_blob_shapes_usage_and_determinism() -> None:
    X = _blobs()
    codebooks = fit_rq_kmeans(X, n_levels=2, n_clusters=3, seed=42)
    assert codebooks.shape == (2, 3, 4)
    sids = assign_sids(X, codebooks)
    assert sids.shape == (360, 2)
    # three well-separated blobs → L0 must use all three codes, none starved
    counts = np.bincount(sids[:, 0], minlength=3)
    assert set(sids[:, 0].tolist()) == {0, 1, 2}
    assert bool((counts >= 36).all())
    assert codebook_usage(sids, 3)[0] == pytest.approx(1.0)
    # deterministic: refit is bit-identical, and matches raw sklearn KMeans at L0
    assert np.array_equal(codebooks, fit_rq_kmeans(X, n_levels=2, n_clusters=3, seed=42))
    ref = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X).cluster_centers_
    assert np.array_equal(
        np.sort(codebooks[0], axis=0), np.sort(ref, axis=0)
    )


def test_prefix_overlap_identity_and_early_divergence() -> None:
    X = _blobs()
    codebooks = fit_rq_kmeans(X, n_levels=3, n_clusters=3, seed=42)
    sids = assign_sids(X, codebooks)
    assert prefix_overlap(sids[0], sids[0]) == pytest.approx(1.0)
    # blobs 0 and 1 split at the coarse level → zero shared prefix
    far = assign_sids(
        np.array([[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]), codebooks
    )
    assert far[0][0] != far[1][0]
    assert prefix_overlap(far[0], far[1]) == pytest.approx(0.0)
    # a one-level-deep partial prefix scores 1/3
    assert prefix_overlap(
        np.array([5, 1, 2]), np.array([5, 9, 9])
    ) == pytest.approx(1.0 / 3.0)


def test_collision_tidbit_makes_duplicate_codes_unique() -> None:
    sids = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 0], [1, 2, 3]])
    full = add_collision_tidbits(sids)
    assert full.shape == (4, 4)
    assert full[:, -1].tolist() == [0, 1, 0, 2]
    assert unique_ids_proportion(sids) == pytest.approx(0.5)
    assert unique_ids_proportion(full) == pytest.approx(1.0)


def test_save_load_roundtrip(tmp_path) -> None:
    X = _blobs()[:90]
    codebooks = fit_rq_kmeans(X, n_levels=3, n_clusters=8, seed=42)
    path = save_codebooks(tmp_path / "codebooks.npz", codebooks)
    assert np.array_equal(load_codebooks(path), codebooks)
    gtins = [f"gtin-{i}" for i in range(len(X))]
    full = add_collision_tidbits(assign_sids(X, codebooks))
    table = save_sid_table(tmp_path / "canonical_sids.csv", gtins, full)
    back_gtins, back_sids = load_sid_table(table)
    assert back_gtins == gtins
    assert np.array_equal(back_sids, full)


def test_hybrid_score_weights_cosine_and_prefix_overlap() -> None:
    a = np.array([5, 1, 2])
    assert hybrid_score(0.9, a, a) == pytest.approx(0.8 * 0.9 + 0.2 * 1.0)
    assert hybrid_score(0.9, a, a, alpha=1.0, beta=0.0) == pytest.approx(0.9)
    # beta-only arm is pure prefix overlap: identical 1.0, L0-diverged 0.0
    assert hybrid_score(0.9, a, a, alpha=0.0, beta=1.0) == pytest.approx(1.0)
    assert hybrid_score(0.9, a, np.array([6, 1, 2]), alpha=0.0, beta=1.0) == pytest.approx(0.0)
    # a high-cosine L0-diverged conflict is downweighted vs a true pair
    true = hybrid_score(0.85, a, a)
    conflict = hybrid_score(0.93, a, np.array([6, 1, 2]))
    assert conflict < true


def test_hybrid_score_rejects_bad_inputs() -> None:
    a = np.array([5, 1, 2])
    with pytest.raises(ValueError):
        hybrid_score(0.9, a, a, alpha=-0.1, beta=0.2)
    with pytest.raises(ValueError):
        hybrid_score(0.9, a, a, alpha=0.0, beta=0.0)
    with pytest.raises(ValueError):
        hybrid_score(1.5, a, a)
    with pytest.raises(ValueError):
        hybrid_score(0.9, a, np.array([5, 1]))


def test_hybrid_matrix_matches_pair_loop() -> None:
    rng = np.random.RandomState(3)
    sku = rng.randint(0, 8, size=(9, 3))
    canon = rng.randint(0, 8, size=(14, 3))
    cos = rng.uniform(-1.0, 1.0, size=(9, 14))
    mat = hybrid_matrix(cos, sku, canon)
    assert mat.shape == (9, 14)
    assert leading_overlap_matrix(sku, canon).shape == (9, 14)
    for i in range(9):
        for j in range(14):
            assert mat[i, j] == pytest.approx(hybrid_score(cos[i, j], sku[i], canon[j]))
    with pytest.raises(ValueError):
        hybrid_matrix(cos, sku, canon[:, :2])
    with pytest.raises(ValueError):
        hybrid_matrix(cos[:5], sku, canon)
