"""MNRL pair selection keeps augmentation and avoids in-batch false negatives."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers.base.sampler import NoDuplicatesBatchSampler

from training.training import (
    _build_mnrl_training_triples,
    _mnrl_shared_positive_gtin_rows,
)


def test_augmented_negative_keeps_its_copy_as_anchor_and_source_positive() -> None:
    positives = np.array([[1, 2], [10, 11]])
    negatives = np.array([[1, 3], [20, 3], [21, 4], [99, 3], [10, 11]])
    audit = [
        {"anchor_payload_idx": 1, "copy_payload_idx": 20, "pair_payload_idx": 3,
         "target_mode": "random"},
        {"anchor_payload_idx": 1, "copy_payload_idx": 21, "pair_payload_idx": 4,
         "target_mode": "targeted"},
    ]

    triples = _build_mnrl_training_triples(
        positives, negatives, mask_audit=[], hard_negative_mask_audit=audit
    )

    assert triples == [(1, 2, 3), (20, 2, 3), (21, 2, 4)]
    # Both augmentation types survive once; missing source positives and
    # self-negatives stay excluded, rather than acquiring a gtin proxy.


def test_augmented_negative_lineage_is_tied_to_its_target() -> None:
    triples = _build_mnrl_training_triples(
        np.array([[1, 2]]),
        np.array([[20, 4]]),
        mask_audit=[],
        hard_negative_mask_audit=[
            {"anchor_payload_idx": 1, "copy_payload_idx": 20, "pair_payload_idx": 3}
        ],
    )
    assert triples == []


def test_symmetric_swap_uses_generated_positive_and_excludes_source_positive_as_negative() -> None:
    triples = _build_mnrl_training_triples(
        # The source positive and symmetric generated pair are distinct
        # positive edges in the training fold.
        np.array([[1, 2], [20, 21]]),
        # The original positive can appear in the negative list due to a
        # different augmentation path; it must not become this copy's N.
        np.array([[1, 2], [1, 3]]),
        mask_audit=[
            {
                "anchor_payload_idx": 1,
                "copy_payload_idx": 20,
                "pair_payload_idx": 2,
                "copy_pair_payload_idx": 21,
                "target_mode": "swap_values",
            }
        ],
        hard_negative_mask_audit=[],
    )

    assert triples == [(1, 2, 3), (20, 21, 3)]


def test_symmetric_swap_requires_source_and_generated_positive_edges_in_fold() -> None:
    audit = [{
        "anchor_payload_idx": 1,
        "copy_payload_idx": 20,
        "pair_payload_idx": 2,
        "copy_pair_payload_idx": 21,
        "target_mode": "swap_values",
    }]

    source_edge_missing = _build_mnrl_training_triples(
        np.array([[20, 21]]), np.array([[1, 3]]),
        mask_audit=audit, hard_negative_mask_audit=[],
    )
    generated_edge_missing = _build_mnrl_training_triples(
        np.array([[1, 2]]), np.array([[1, 3]]),
        mask_audit=audit, hard_negative_mask_audit=[],
    )

    assert source_edge_missing == []
    assert generated_edge_missing == [(1, 2, 3)]
    assert (20, 21, 3) not in generated_edge_missing


def test_semantic_false_negative_exposure_counts_repeated_positive_gtins() -> None:
    row_bc = np.array(["unused", "A", "A", "B", "A", "C", "C"])
    triples = [(1, 2, 3), (4, 2, 5), (6, 5, 3)]
    assert _mnrl_shared_positive_gtin_rows(triples, row_bc) == 2


def test_no_duplicates_sampler_keeps_same_text_copies_apart() -> None:
    rows = [
        {"anchor": "same anchor", "positive": "canonical A", "negative": "hard X"},
        {"anchor": "same anchor", "positive": "canonical B", "negative": "hard Y"},
        {"anchor": "third anchor", "positive": "canonical A", "negative": "hard Z"},
        {"anchor": "fourth anchor", "positive": "canonical C", "negative": "hard X"},
    ]

    class _Dataset:
        column_names = ("anchor", "positive", "negative")

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            return rows[index]

    dataset = _Dataset()
    sampler = NoDuplicatesBatchSampler(
        dataset,
        batch_size=2,
        drop_last=False,
        generator=torch.Generator(),
        seed=17,
    )

    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(4))
    for batch in batches:
        texts = [
            dataset[index][column]
            for index in batch
            for column in ("anchor", "positive", "negative")
        ]
        assert len(texts) == len(set(texts))
