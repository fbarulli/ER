"""G1: MNRL mint/mask provenance survives training (per-pair backprop carry).

The defect this locks down: the MNRL lane kept only POPULATION-level sets
(base/masked/twin) and ``_write_datapoint_usage`` was gated to the contrastive
loss, so a minted/masked row's lineage never reached the per-pair usage artifact
and could not be attributed to backprop. ``_tracking_mnrl_loss`` now carries the
per-pair lineage the orchestrator builds with ``_build_pair_lineage`` and counts
every presented triple as backpropagated (the row term is averaged into the
batch objective).

The coverage population is the explicit negative's registry-safe SOURCE (never
the unregistered monitoring tag base/masked/twin): a coverage census keyed on an
undeclared population raises ``UnregisteredDatapointPopulationError``, so the
mapping itself is the load-bearing contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.training import _mnrl_pair_populations, _tracking_mnrl_loss


def test_mnrl_per_pair_backprop_carries_registry_safe_source_lineage():
    triples = [(1, 2, 3), (1, 2, 30)]
    train_neg = np.array([[1, 3], [1, 30]])
    sources = np.array(["gate", "counterfactual"], dtype=object)
    populations = _mnrl_pair_populations(triples, train_neg, sources)
    # base triple -> its gate negative; twin triple -> the minted counterfactual.
    assert populations == ["gate", "counterfactual"]

    loss = _tracking_mnrl_loss(None, monitoring_enabled=True, warmup_enabled=False,
                               warmup_epochs=2, twin_weight=0.25)
    loss.set_pair_lineage([
        {"population": populations[0], "lineage_id": "gate:1:3",
         "source_anchor_payload_idx": 1, "source_pair_payload_idx": 3,
         "is_masked_copy": 0, "target_mode": ""},
        {"population": populations[1], "lineage_id": "hard_negative:1:9",
         "source_anchor_payload_idx": 1, "source_pair_payload_idx": 9,
         "is_masked_copy": 1, "target_mode": "counterfactual"},
    ])
    loss.set_triple_populations(["base", "twin"])
    loss.set_epoch(1)
    loss.set_batch_pair_ids(torch.tensor([0, 1]))
    embeddings = [torch.randn(2, 8, requires_grad=True) for _ in range(2)]
    loss.compute_loss_from_embeddings(embeddings, torch.zeros(2))

    rows = {row["pair_id"]: row for row in loss.pair_usage_rows()}
    assert rows[0]["backprop_count"] == 1 and rows[1]["backprop_count"] == 1
    assert rows[1]["is_masked_copy"] == 1
    assert rows[1]["target_mode"] == "counterfactual"
    # The writer's presentation keys carry the source label, never base/twin.
    assert {key[2] for key in loss.presentation_counts()} == {"gate", "counterfactual"}
    assert {row["population"] for row in loss.pair_usage_rows_by_epoch()} == {
        "gate", "counterfactual"}


def test_mnrl_sources_follow_exact_edges_and_copy_lineage() -> None:
    positives = np.array([[1, 2], [1, 4], [20, 21]])
    negatives = np.array([[1, 3], [8, 3], [30, 2]])
    sources = np.array(["gate", "attribute", "counterfactual+aug"], dtype=object)
    mask = [{"anchor_payload_idx": 1, "pair_payload_idx": 2,
             "copy_payload_idx": 20, "copy_pair_payload_idx": 21}]
    twin = [{"anchor_payload_idx": 1, "pair_payload_idx": 2,
             "copy_payload_idx": 30, "target_mode": "counterfactual"}]
    from training.training import _build_mnrl_training_triples

    triples = _build_mnrl_training_triples(
        positives, negatives, mask_audit=mask, hard_negative_mask_audit=twin)
    populations = _mnrl_pair_populations(
        triples, negatives, sources, mask_audit=mask,
        hard_negative_mask_audit=twin)
    assert dict(zip(triples, populations, strict=True)) == {
        (1, 2, 3): "gate", (20, 21, 3): "gate",
        (1, 2, 30): "counterfactual", (1, 4, 30): "counterfactual"}
    canonical = [{**twin[0], "copy_source_payload_idx": 2}]
    assert _mnrl_pair_populations(
        [(2, 1, 30)], negatives, sources,
        hard_negative_mask_audit=canonical) == ["counterfactual"]


@pytest.mark.parametrize("sources", [None, np.array(["gate"], dtype=object)])
def test_mnrl_sources_reject_missing_source_rows(sources: np.ndarray | None) -> None:
    with pytest.raises(ValueError, match="source"):
        _mnrl_pair_populations([(1, 2, 3)], np.array([[1, 3], [4, 5]]), sources)


def test_mnrl_sources_reject_an_unresolved_negative() -> None:
    with pytest.raises(ValueError, match="provenance"):
        _mnrl_pair_populations([(1, 2, 99)], np.array([[1, 3]]),
                               np.array(["gate"], dtype=object))


def test_mnrl_positive_copy_keeps_direct_edges_and_inherits_twin_sources() -> None:
    mask = [{"anchor_payload_idx": 1, "pair_payload_idx": 2,
             "copy_payload_idx": 20, "copy_pair_payload_idx": 21}]
    twin = [{"anchor_payload_idx": 1, "pair_payload_idx": 2,
             "copy_payload_idx": 30, "copy_source_payload_idx": 2,
             "target_mode": "counterfactual"}]
    negatives = np.array([[20, 5], [30, 2]])
    sources = np.array(["attribute", "counterfactual"], dtype=object)
    assert _mnrl_pair_populations(
        [(20, 21, 5), (20, 21, 30)], negatives, sources,
        mask_audit=mask, hard_negative_mask_audit=twin) == ["attribute", "counterfactual"]
