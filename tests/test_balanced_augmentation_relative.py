"""Public contracts of the relative R1/R2 balanced-augmentation counts.

The carve's class sizes are known only at bundle-build time, so config declares
the counts RELATIVELY: mint negatives up to the positive count (R1), then mask
that share of each (now equal) class (R2). These pins assert the three public
behaviours: relative resolution closes ``produced == requested`` on a synthetic
train split, relative and absolute counts are mutually exclusive, and every
minted/masked audit row carries the measured difficulty it was stratified by.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core.schemas import BalancedAugmentationSpec
from training.balanced_augmentation import RelativeCounts, augment_balanced


def _relative_inputs():
    """Synthetic train split with P=3 positives and N=1 negative, two fields.

    Every positive anchor declares volume + pack + prose, so the measured pair
    difficulty is defined (not ``unknown``); the donor rows carry conflicting
    values for both fields, so minting has headroom.
    """
    df = pd.DataFrame({"retailer": ["amazon", "walmart", "ebay", "target",
                                    "kroger", "aldi", "walmart", "ebay", "amazon"]})
    payload = [
        "Cola drink 500ml volume_ml_500 pack_qty_6",     # 0 anchor
        "Cola Classic 500ml volume_ml_500 pack_qty_6",   # 1 entity A sibling
        "Juice box 330ml volume_ml_330 pack_qty_12",     # 2 donor
        "Milk carton 900ml volume_ml_900 pack_qty_1",    # 3 donor
        "Big soda 1500ml volume_ml_1500 pack_qty_24",    # 4 donor
        "Cola Walmart 500ml volume_ml_500 pack_qty_6",   # 5 entity F
        "Cola drink 500ml volume_ml_500 pack_qty_6",     # 6 canonical c0
        "Cola drink 500ml volume_ml_500 pack_qty_6",     # 7 canonical c1
        "Cola drink 500ml volume_ml_500 pack_qty_6",     # 8 canonical c2
    ]
    row_bc = np.asarray(["75678164126", "75678164126", "33000000000123",
                         "90000000000123", "15000000000123", "75000000000123",
                         "75678164126", "75678164126", "75678164126"], dtype=object)
    features = np.zeros((len(payload), 10), dtype=np.float32)
    pos = np.array([[0, 6], [0, 7], [0, 8]], dtype=int)
    neg = np.array([[2, 3]], dtype=int)
    spec = BalancedAugmentationSpec.model_validate({
        "enabled": True,
        "counts": {"minted_negatives": 0, "masked_minted_negatives": 0,
                   "masked_positives": 0, "vendor_variation_positives": 0},
        "balance_negatives_to_positives": True,
        "masked_share_per_class": 0.25,
        "min_attribute_pairs": 2,
    })
    return {"pos": pos, "neg": neg, "payload": payload, "row_bc": row_bc,
            "features": features, "df": df,
            "train_indices": set(range(len(payload))),
            "canonical_indices": {6, 7, 8}, "spec": spec, "seed": 7}


def test_relative_counts_resolve_to_the_actual_class_sizes() -> None:
    """R1 mints P-N; R2 masks 25% of each (now equal) class, echoed verbatim."""
    args = _relative_inputs()
    (*_rest, coverage) = augment_balanced(**args)
    assert coverage.model_dump(mode="json")["requested_counts"] == {
        "minted_negatives": 2, "masked_minted_negatives": 1,
        "masked_positives": 1, "vendor_variation_positives": 0,
    }
    # The engine's exact-equality coverage validator only accepts a run where
    # produced == requested, so reaching this point IS the closure pin.
    assert coverage.minted_negatives == 2
    assert coverage.masked_minted_negatives == 1
    assert coverage.masked_positives == 1


def test_relative_and_absolute_counts_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        BalancedAugmentationSpec.model_validate({
            "balance_negatives_to_positives": True,
            "counts": {"minted_negatives": 5, "masked_minted_negatives": 0,
                       "masked_positives": 0, "vendor_variation_positives": 0},
        })
    with pytest.raises(ValidationError, match="mutually exclusive"):
        BalancedAugmentationSpec.model_validate({
            "masked_share_per_class": 0.25,
            "counts": {"minted_negatives": 10, "masked_minted_negatives": 0,
                       "masked_positives": 4, "vendor_variation_positives": 0},
        })


def test_infeasible_relative_share_dies_loud() -> None:
    """A share that cannot fit the negative-first invariant is refused."""
    spec = BalancedAugmentationSpec.model_validate({
        "balance_negatives_to_positives": True,
        "masked_share_per_class": 0.9,
        "counts": {"minted_negatives": 0, "masked_minted_negatives": 0,
                   "masked_positives": 0, "vendor_variation_positives": 0},
    })
    with pytest.raises(ValidationError, match="masked <= minted"):
        RelativeCounts.resolve(spec, positives=100, negatives=90)


def test_minted_and_masked_audits_carry_their_measured_difficulty() -> None:
    """Every minted/masked copy records the difficulty stratum it came from."""
    args = _relative_inputs()
    (*_rest, pos_audit, audits, _coverage) = augment_balanced(**args)
    # Two minted negatives plus the one masked_minted copy appended to audits.
    assert len(audits) == 3 and len(pos_audit) == 1
    assert all(row["difficulty_slice"] != "unknown" for row in audits + pos_audit)
    assert all(row["target_stratum"] for row in audits + pos_audit)
