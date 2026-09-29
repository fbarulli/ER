"""MNRL subset monitoring + twin-loss warmup (TODO Item 7, EXP-03).

Three guarantees are locked down, all driven directly (no training run
needed):

1. BIT-IDENTICAL DISABLED PATH — with ``training.mnrl_monitoring.enabled``
   and ``training.twin_loss_warmup.enabled`` both false, ``_make_loss``
   returns the installed MultipleNegativesRankingLoss unchanged, and the
   tracking wrapper (if ever constructed) computes the exact installed loss
   value — so existing behavior and loss bytes never change.
2. SUBSET LOSS — when monitoring is enabled, the wrapper attributes each
   triple's row loss to its population (base/masked/twin) and reports the
   per-epoch per-population mean loss that feeds
   ``mnrl_subset_loss_by_epoch_fold{i}.csv``.
3. CONFIG DEFAULTS — the new spec blocks default to DISABLED so every
   existing training.yaml still validates.
"""

from __future__ import annotations

import numpy as np
import torch

from core.schemas import TrainingSpec
from sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking import (
    MultipleNegativesRankingLoss,
)
from training.training import (
    _build_mnrl_training_triples,
    _build_mnrl_triple_populations,
    _make_loss,
    _tracking_mnrl_loss,
)


def _plain_loss():
    return MultipleNegativesRankingLoss(model=None)


def _tracked_loss(
    monitoring_enabled=False,
    warmup_enabled=False,
    warmup_epochs=2,
    twin_weight=0.25,
):
    return _tracking_mnrl_loss(
        None,
        monitoring_enabled=monitoring_enabled,
        warmup_enabled=warmup_enabled,
        warmup_epochs=warmup_epochs,
        twin_weight=twin_weight,
    )


def _embeddings():
    torch.manual_seed(0)
    return [torch.randn(4, 8), torch.randn(4, 8)], torch.zeros(4)


def test_make_loss_returns_installed_mnrl_when_monitoring_and_warmup_disabled():
    loss = _make_loss(
        None,
        "mnrl",
        structured_feature_weight=0.0,
        uniformity_weight=0.0,
        uniformity_temperature=1.0,
        uniformity_min_batch_size=4,
        label_smoothing=0.0,
        mnrl_monitoring_enabled=False,
        twin_warmup_enabled=False,
    )
    assert type(loss) is MultipleNegativesRankingLoss


def test_tracked_loss_disabled_is_bit_identical_to_installed():
    embeddings, labels = _embeddings()
    plain = _plain_loss().compute_loss_from_embeddings(embeddings, labels)
    tracked = _tracked_loss().compute_loss_from_embeddings(embeddings, labels)
    assert torch.equal(tracked, plain)


def test_tracked_loss_monitoring_only_preserves_installed_value():
    embeddings, labels = _embeddings()
    plain = _plain_loss().compute_loss_from_embeddings(embeddings, labels)
    tracked = _tracked_loss(monitoring_enabled=True).compute_loss_from_embeddings(
        embeddings, labels
    )
    assert torch.allclose(tracked, plain)


def test_subset_loss_attribution_by_population():
    loss = _tracked_loss(monitoring_enabled=True)
    loss.set_epoch(1)
    loss.set_triple_populations(["base", "twin", "masked", "twin"])
    loss.set_batch_pair_ids(torch.tensor([0, 1, 2, 3]))
    embeddings, labels = _embeddings()
    loss.compute_loss_from_embeddings(embeddings, labels)

    rows = loss.mnrl_subset_rows_by_epoch()
    by_pop = {row["population"]: row for row in rows}
    assert {row["population"] for row in rows} == {"base", "twin", "masked"}
    assert rows[0]["epoch"] == 1
    assert by_pop["twin"]["triple_count"] == 2
    assert by_pop["base"]["triple_count"] == 1
    assert by_pop["masked"]["triple_count"] == 1


def _two_row_embeddings():
    # Seed chosen so BOTH row losses are clearly nonzero (seed 0 leaves the
    # second row's loss at exactly 0, which would mask the down-weighting).
    torch.manual_seed(19)
    return [torch.randn(2, 8), torch.randn(2, 8)], torch.zeros(2)


def test_twin_warmup_down_weights_twin_rows_in_epoch_one():
    loss = _tracked_loss(warmup_enabled=True, warmup_epochs=2, twin_weight=0.25)
    loss.set_epoch(1)
    loss.set_triple_populations(["base", "twin"])
    loss.set_batch_pair_ids(torch.tensor([0, 1]))
    embeddings, labels = _two_row_embeddings()

    weighted = loss.compute_loss_from_embeddings(embeddings, labels)

    unweighted = _tracked_loss(
        monitoring_enabled=True, warmup_enabled=False
    ).compute_loss_from_embeddings(embeddings, labels)
    # Twin row (index 1) contributes 0.25x at epoch 1 of a 2-epoch ramp, so the
    # weighted batch mean is strictly below the unweighted mean.
    assert weighted < unweighted


def test_twin_warmup_reaches_full_weight_after_warmup_epochs():
    loss = _tracked_loss(warmup_enabled=True, warmup_epochs=2, twin_weight=0.25)
    loss.set_epoch(2)
    loss.set_triple_populations(["base", "twin"])
    loss.set_batch_pair_ids(torch.tensor([0, 1]))
    embeddings, labels = _two_row_embeddings()

    weighted = loss.compute_loss_from_embeddings(embeddings, labels)
    unweighted = _tracked_loss(
        monitoring_enabled=True, warmup_enabled=False
    ).compute_loss_from_embeddings(embeddings, labels)
    assert torch.allclose(weighted, unweighted)


def test_triple_populations_align_with_triples():
    positives = np.array([[1, 2], [10, 11]])
    negatives = np.array([[1, 3], [20, 3], [21, 4], [10, 11]])
    audit = [
        {
            "anchor_payload_idx": 1,
            "copy_payload_idx": 20,
            "pair_payload_idx": 3,
            "target_mode": "random",
        },
        {
            "anchor_payload_idx": 1,
            "copy_payload_idx": 21,
            "pair_payload_idx": 4,
            "target_mode": "targeted",
        },
    ]

    triples = _build_mnrl_training_triples(
        positives, negatives, mask_audit=[], hard_negative_mask_audit=audit
    )
    populations = _build_mnrl_triple_populations(
        positives, negatives, mask_audit=[], hard_negative_mask_audit=audit
    )

    assert len(triples) == len(populations)
    # (1,2,3) is base (organic source negative); the two copy-anchored triples
    # (20,2,3) and (21,2,4) are masked hard-negative copies.
    assert populations == ["base", "masked", "masked"]


def test_twin_triples_are_tagged_twin():
    positives = np.array([[1, 2], [10, 11]])
    # The twin edge (30, 2) must survive negative balancing to be eligible.
    negatives = np.array([[1, 3], [30, 2]])
    audit = [
        {
            "anchor_payload_idx": 1,
            "copy_payload_idx": 30,
            "pair_payload_idx": 2,
            "target_mode": "counterfactual",
        }
    ]

    triples = _build_mnrl_training_triples(
        positives, negatives, mask_audit=[], hard_negative_mask_audit=audit
    )
    populations = _build_mnrl_triple_populations(
        positives, negatives, mask_audit=[], hard_negative_mask_audit=audit
    )

    assert (1, 2, 30) in triples
    twin_index = triples.index((1, 2, 30))
    assert populations[twin_index] == "twin"


def test_config_defaults_are_disabled():
    spec = TrainingSpec.MnrlMonitoringSpec()
    assert spec.enabled is False

    warmup = TrainingSpec.TwinLossWarmupSpec()
    assert warmup.enabled is False
    assert warmup.warmup_epochs == 2
    assert warmup.twin_weight == 0.25
