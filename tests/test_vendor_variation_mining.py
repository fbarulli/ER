"""Pin: deepened round-robin mining of cross-vendor same-entity positives.

History: the vendor-variation lane selected ONE pair per entity in a single
pass and exhausted the 10k cohort's supply at 230 < the requested 300, so
``AugmentationCoverage`` failed the exact-equality contract. The fix mines
DEEPER with the same rules: pass 1 replays the historical selection byte for
byte; deeper passes round-robin across entities, each contributing its next
valid pair per pass until the quota is met or the pool is exhausted.

Pinned here on synthetic data (no pipeline, no real data state modified):

1. Pass-1 selection equals the original single-pass loop's selection when the
   eligible-entity population covers the quota (byte-identical list).
2. Round-robin reaches the exact quota across passes (and stops mid-pass),
   in deterministic entity/pass order, with no rng anywhere in the mining.
3. A quota larger than the valid-pair pool exhausts, and the exact-equality
   validator rejects the shortfall (produced != requested dies loudly).
4. End-to-end ``augment_balanced`` closes produced == requested on a full
   synthetic run where the deepened mining supplies the vendor positives.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core.schemas import AugmentationCounts, BalancedAugmentationSpec
from training.balanced_augmentation import (
    AugmentationCoverage,
    AttributeAllocation,
    _entity_vendor_pairs,
    _mine_vendor_pairs,
    augment_balanced,
)
from training.masking import _FIELD_PREFIXES, _field_values_conflict

FIELD_KEYS = {
    field: AttributeAllocation(
        positive_both_observed=0, negative_both_observed=0,
        eligible_anchors=0, donor_values=0, requested=0,
        initial_requested=0, minted=0, masked=0, shortfall=0,
        status="eligible")
    for field in _FIELD_PREFIXES
}


def rich_structures():
    """Entity A: 3 cross-vendor members (3 valid pairs); entity B: 2 members.

    All members declare identical volume evidence, so every cross-retailer
    member pair is conflict-free and valid.
    """
    fields = {index: {"volume": ["volume_ml_500"]} for index in range(5)}
    by_entity = defaultdict(list)
    by_entity["111111111111"] = [0, 1, 2]
    by_entity["222222222222"] = [3, 4]
    retail_of = lambda index: ["amazon", "walmart", "target", "amazon", "kroger"][index]  # noqa: E731
    return fields, by_entity, retail_of


def old_loop_selection(by_entity, fields, retail_of, quota):
    """The ORIGINAL single-pass loop, verbatim, for the byte-equality pin."""
    vendor_pairs = []
    for members in by_entity.values():
        if len(vendor_pairs) >= quota:
            break
        for left in members:
            right = next(
                (i for i in members
                 if retail_of(left) != retail_of(i)
                 and all(not _field_values_conflict(
                     field, fields[left].get(field, []),
                     fields[i].get(field, [])) for field in _FIELD_PREFIXES)),
                None)
            if right is not None:
                vendor_pairs.append((left, right))
                break
    return vendor_pairs


def test_pass_one_matches_the_legacy_single_pass_loop():
    fields, by_entity, retail_of = rich_structures()
    quota = 2  # three eligible entities exceed the quota
    mined = _mine_vendor_pairs(by_entity, fields, retail_of, quota)
    assert mined == old_loop_selection(by_entity, fields, retail_of, quota) \
        == [(0, 1), (3, 4)]


def test_pass_one_is_the_legacy_choice_on_a_quota_starved_population():
    """When the quota is UNREACHABLE in pass 1, pass 1 still reproduces the
    legacy loop's whole output (one pair per entity), and pass 2+ only adds
    further pairs afterward in the same fixed entity order."""
    fields, by_entity, retail_of = rich_structures()
    deep = _mine_vendor_pairs(by_entity, fields, retail_of, 10)
    assert deep == [(0, 1), (3, 4), (0, 2), (1, 2)]


def test_round_robin_reaches_the_quota_and_stops_mid_pass():
    fields, by_entity, retail_of = rich_structures()
    # Pass 1: (0,1), (3,4); pass 2: (0,2); pass 3: (1,2) -> quota 4 met.
    assert _mine_vendor_pairs(by_entity, fields, retail_of, 4) \
        == [(0, 1), (3, 4), (0, 2), (1, 2)]
    # Quota 3 stops mid-pass before entity A's third pair.
    assert _mine_vendor_pairs(by_entity, fields, retail_of, 3) \
        == [(0, 1), (3, 4), (0, 2)]


def test_mirror_pair_licensed_once_and_conflict_vetoed_pairs_exhaust():
    """(left,right) and its mirror stay one license; a pack conflict vetoes."""
    fields = {1: {"volume": ["volume_ml_500"], "pack": ["pack_qty_6"]},
              2: {"volume": ["volume_ml_500"], "pack": ["pack_qty_12"]},
              3: {"volume": ["volume_ml_500"], "pack": ["pack_qty_12"]}}
    member_retailer = {1: "amazon", 2: "walmart", 3: "target"}
    pairs = _entity_vendor_pairs([1, 2, 3], member_retailer, fields)
    # (1,2)/(1,3) vetoed by the pack conflict; (2,3) valid; no mirror repeat.
    assert pairs == [(2, 3)]


def test_zero_quota_and_empty_population_untouched():
    fields, by_entity, retail_of = rich_structures()
    assert _mine_vendor_pairs(by_entity, fields, retail_of, 0) == []
    assert _mine_vendor_pairs(defaultdict(list), fields, retail_of, 5) == []


def test_exhausted_pool_returns_every_valid_pair_and_validator_rejects_the_gap():
    fields, by_entity, retail_of = rich_structures()
    # Entity A holds 3 ordered pairs, entity B 1: the pool is 4 < requested.
    mined = _mine_vendor_pairs(by_entity, fields, retail_of, 10)
    assert mined == [(0, 1), (3, 4), (0, 2), (1, 2)]
    attributes = dict(FIELD_KEYS)
    attributes["volume"] = AttributeAllocation(
        positive_both_observed=0, negative_both_observed=0,
        eligible_anchors=3, donor_values=1, requested=6,
        initial_requested=6, minted=6, masked=0, shortfall=0,
        status="eligible")
    with pytest.raises(ValidationError, match="augmentation output shortfall"):
        AugmentationCoverage.model_validate({
            "source_train_positives": 3, "source_train_negatives": 0,
            "vendor_variation_positives": len(mined), "masked_positives": 0,
            "minted_negatives": 6, "masked_minted_negatives": 0,
            "attributes": attributes, "rejections": {},
            "requested_counts": {"minted_negatives": 6, "masked_minted_negatives": 0,
                                 "masked_positives": 0,
                                 "vendor_variation_positives": 10}})


def _e2e_inputs():
    """Full synthetic run: minting + masking + deep vendor mining together."""
    df = pd.DataFrame({
        "retailer": ["amazon", "walmart", "ebay", "target", "kroger",
                     "aldi", "walmart", "ebay"]})
    payload = [
        "Cola drink 500ml volume_ml_500",      # 0 (entity A)
        "Cola Classic 500ml volume_ml_500",    # 1 (entity A)
        "Cola Fresh 500ml volume_ml_500",      # 2 (entity A)
        "Juice box 330ml volume_ml_330",       # 3 donor (entity C)
        "Milk carton 900ml volume_ml_900",     # 4 donor (entity D)
        "Big soda 1500ml volume_ml_1500",      # 5 donor (entity E)
        "Cola Walmart 500ml volume_ml_500",    # 6 (entity F)
        "Cola Ebay 500ml volume_ml_500",       # 7 (entity F)
        "Cola drink 500ml volume_ml_500",      # 8 canonical c0
        "Cola drink 500ml volume_ml_500",      # 9 canonical c1
        "Cola drink 500ml volume_ml_500",      # 10 canonical c2
    ]
    gtins = ["75678164126", "75678164126", "75678164126",
             "33000000000123", "90000000000123", "15000000000123",
             "75000000000123", "75000000000123",
             "75678164126", "75678164126", "75678164126"]
    row_bc = np.asarray(gtins, dtype=object)
    features = np.zeros((len(payload), 10), dtype=np.float32)
    pos = np.array([[0, 8], [0, 9], [0, 10]], dtype=int)
    neg = np.zeros((0, 2), dtype=int)
    spec = BalancedAugmentationSpec.model_validate({
        "enabled": True,
        "counts": {"minted_negatives": 6, "masked_minted_negatives": 2,
                   "masked_positives": 2, "vendor_variation_positives": 4},
        "min_attribute_pairs": 2,
    })
    return {"pos": pos, "neg": neg, "payload": payload, "row_bc": row_bc,
            "features": features, "df": df,
            "train_indices": set(range(len(payload))),
            "canonical_indices": {8, 9, 10}, "spec": spec, "seed": 7}


def test_augment_balanced_produced_equals_requested():
    """Deepened mining fills the requested vendor quota exactly; inside
    augment_balanced the coverage validator (untouched) would reject any
    other outcome — its acceptance IS the produced==requested pin."""
    args = _e2e_inputs()
    (final_pos, final_neg, new_payload, new_bc, new_features,
     _pos_audit, _audits, coverage) = augment_balanced(**args)
    counts = coverage.model_dump(mode="json")
    assert counts["vendor_variation_positives"] == 4
    assert counts["minted_negatives"] == 6
    assert counts["masked_minted_negatives"] == 2
    assert counts["masked_positives"] == 2
    # Deep-mining order: pass 1 both entities' heads, pass 2 A's second pair
    # (B's single pair is already exhausted), pass 3 fills the remainder.
    vendor_rows = [tuple(row) for row in final_pos[3:7].tolist()]
    assert vendor_rows == [(0, 1), (6, 7), (0, 2), (1, 2)]
    retailers = args["df"]["retailer"].astype(str).to_numpy()
    for left, right in vendor_rows:
        assert retailers[left] != retailers[right], "cross-vendor only"
        assert str(new_bc[left]) == str(new_bc[right]), "same entity only"
    assert len(new_payload) == len(new_bc) == len(new_features)


def test_requested_excess_quotas_are_refused_at_the_count_contract():
    """Requested-count integrity: an infeasible vendor quota can only enter the
    lane when the negative budget covers it — otherwise the spec guard refuses
    the counts before any mining runs (requested counts stay SSOT-declared)."""
    with pytest.raises(ValidationError,
                       match="configured positive augmentation exceeds negative"):
        AugmentationCounts(minted_negatives=6, masked_minted_negatives=2,
                           masked_positives=2, vendor_variation_positives=99)


def test_augment_balanced_shortfall_with_mint_deficit_zero_dies_loud():
    """Requested vendor quota 6, mint side closes at 6/2/2 — only the vendor
    quota exceeds the valid-pair pool: the exact-equality coverage check dies."""
    args = _e2e_inputs()
    args["spec"] = args["spec"].model_copy(update={"counts": AugmentationCounts(
        minted_negatives=6, masked_minted_negatives=2, masked_positives=2,
        vendor_variation_positives=6)})
    with pytest.raises(ValidationError, match="augmentation output shortfall"):
        augment_balanced(**args)
