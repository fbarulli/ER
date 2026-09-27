"""MNRL pair selection keeps augmentation and avoids in-batch false negatives."""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers.base.sampler import NoDuplicatesBatchSampler

from training.training import (
    _build_mnrl_training_triples,
    _mnrl_shared_positive_barcode_rows,
)


def test_augmented_negative_keeps_its_copy_as_anchor_and_source_positive() -> None:
    positives = np.array([[1, 2], [10, 11]])
    negatives = np.array([[1, 3], [20, 3], [21, 4], [99, 3], [10, 11]])
    audit = [
        {"anchor_payload_idx": 1, "copy_payload_idx": 20, "pair_payload_idx": 3,
         "target_mode": "random"},
        {"anchor_payload_idx": 1, "copy_payload_idx": 21, "pair_payload_idx": 4,
         "target_mode": "swap_agreed"},
    ]

    triples = _build_mnrl_training_triples(
        positives, negatives, hard_negative_mask_audit=audit
    )

    assert triples == [(1, 2, 3), (20, 2, 3), (21, 2, 4)]
    # Both augmentation types survive once; missing source positives and
    # self-negatives stay excluded, rather than acquiring a barcode proxy.


def test_augmented_negative_lineage_is_tied_to_its_target() -> None:
    triples = _build_mnrl_training_triples(
        np.array([[1, 2]]),
        np.array([[20, 4]]),
        hard_negative_mask_audit=[
            {"anchor_payload_idx": 1, "copy_payload_idx": 20, "pair_payload_idx": 3}
        ],
    )
    assert triples == []


def test_semantic_false_negative_exposure_counts_repeated_positive_gtins() -> None:
    row_bc = np.array(["unused", "A", "A", "B", "A", "C", "C"])
    triples = [(1, 2, 3), (4, 2, 5), (6, 5, 3)]
    assert _mnrl_shared_positive_barcode_rows(triples, row_bc) == 2


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
