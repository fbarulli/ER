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
