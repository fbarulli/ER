"""The carve's published populations drive the bundle build's R1/R2 augmentation.

The 50/50 carve publishes its gtin->split populations as NORMALIZED 14-digit
gtins, while a bundle's ``row_bc`` carries the raw spelling its payload was
built from. The bundle build must map the carve onto those raw keys (``pairs_in
set`` filters raw row_bc directly, so a normalized holdout matches nothing) and
resolve the relative R1/R2 counts from the CARVE's train class sizes — not a
freshly derived split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.schemas import BalancedAugmentationSpec
from model_tracks.smart_setup import SmartPartitionPopulations
from training.balanced_augmentation import augment_balanced


def _carve_spec() -> BalancedAugmentationSpec:
    return BalancedAugmentationSpec.model_validate({
        "enabled": True,
        "counts": {"minted_negatives": 0, "masked_minted_negatives": 0,
                   "masked_positives": 0, "vendor_variation_positives": 0},
        "balance_negatives_to_positives": True,
        "masked_share_per_class": 0.25,
        "min_attribute_pairs": 2,
    })


def test_the_build_resolves_r1_r2_from_the_carve_train(tmp_path: Path) -> None:
    # Product A appears under TWO raw spellings (both normalize to the carve's
    # one published gtin). B and D are train, C is dev, E is test.
    raw = ["75678164126", "075678164126", "33000000000123", "90000000000123",
           "15000000000123", "75000000000123",
           "75678164126", "75678164126", "75678164126"]
    row_bc = np.asarray(raw, dtype=object)
    populations = {
        "train": ["00075678164126", "33000000000123", "15000000000123"],
        "dev": ["90000000000123"],
        "test": ["75000000000123"],
    }
    path = tmp_path / "smart_partition_populations.json"
    path.write_text(json.dumps(populations), encoding="utf-8")

    splits = SmartPartitionPopulations.read(path).on_row_keys(row_bc)
    # The carve's normalized gtin maps onto BOTH raw spellings of product A.
    assert splits["train"] == sorted({"75678164126", "075678164126",
                                      "33000000000123", "15000000000123"})
    assert splits["dev"] == ["90000000000123"]
    train_rows = {i for i, value in enumerate(row_bc) if str(value) in set(splits["train"])}
    assert train_rows == {0, 1, 2, 4, 6, 7, 8}

    row_text = [
        "Cola drink 500ml volume_ml_500 pack_qty_6",
        "Cola Classic 500ml volume_ml_500 pack_qty_6",
        "Juice box 330ml volume_ml_330 pack_qty_12",
        "Milk carton 900ml volume_ml_900 pack_qty_1",
        "Big soda 1500ml volume_ml_1500 pack_qty_24",
        "Cola Walmart 500ml volume_ml_500 pack_qty_6",
        "Cola drink 500ml volume_ml_500 pack_qty_6",
        "Cola drink 500ml volume_ml_500 pack_qty_6",
        "Cola drink 500ml volume_ml_500 pack_qty_6",
    ]
    df = pd.DataFrame({"retailer": ["amazon", "walmart", "ebay", "target",
                                    "kroger", "aldi", "walmart", "ebay", "amazon"]})
    (*_rest, coverage) = augment_balanced(
        pos=np.array([[0, 6], [0, 7], [0, 8]], dtype=int),
        neg=np.array([[2, 4]], dtype=int),
        payload=row_text, row_bc=row_bc,
        features=np.zeros((len(row_text), 10), dtype=np.float32),
        df=df, train_indices=train_rows, canonical_indices={6, 7, 8},
        spec=_carve_spec(), seed=7,
    )
    # R1 mints P-N from the carve's train; R2 masks 25% of the now-equal class.
    assert coverage.model_dump(mode="json")["requested_counts"] == {
        "minted_negatives": 2, "masked_minted_negatives": 1,
        "masked_positives": 1, "vendor_variation_positives": 0,
    }
