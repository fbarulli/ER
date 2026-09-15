from __future__ import annotations

import pandas as pd
import pytest

from training.rand_matching import (
    RandMatcher,
    candidate_gate_fields,
    confidence_penalty_mask,
    flavor_overlap_penalty,
)

SETTINGS = {
    "enabled": True,
    "critical_attributes": ["volume", "pack", "flavor"],
    "minimum_joint_missing": 2,
    "penalty_per_joint_missing": 0.01,
    "max_penalty": 0.03,
    "preserve_exact_gtin": True,
}

FLAVOR_SETTINGS = {
    "enabled": True,
    "minimum_overlap": 0.5,
    "max_penalty": 0.05,
    "preserve_exact_gtin": True,
}


def _info(*, volume=(), pack=(), flavor="") -> dict[str, object]:
    return {"volume": set(volume), "pack": set(pack), "flavor": flavor}


def test_mask_penalizes_only_shared_missing_evidence() -> None:
    penalty, reason = confidence_penalty_mask(
        _info(volume={500}),
        _info(volume={500}),
        exact_gtin=False,
        config=SETTINGS,
    )
    assert penalty == pytest.approx(0.02)
    assert reason == "jointly_missing:pack,flavor"


def test_mask_preserves_exact_gtin_even_when_all_attributes_are_missing() -> None:
    penalty, reason = confidence_penalty_mask(
        _info(), _info(), exact_gtin=True, config=SETTINGS
    )
    assert penalty == 0.0
    assert reason == "exact_gtin_preserved"


def test_mask_does_not_penalize_when_joint_missing_minimum_is_not_met() -> None:
    penalty, reason = confidence_penalty_mask(
        _info(volume={500}, pack={12}),
        _info(volume={500}, pack={12}),
        exact_gtin=False,
        config=SETTINGS,
    )
    assert penalty == 0.0
    assert reason == "sufficient_attribute_evidence"


def test_mask_is_capped_and_never_negative() -> None:
    settings = {**SETTINGS, "penalty_per_joint_missing": 0.2, "max_penalty": 0.04}
    penalty, _ = confidence_penalty_mask(
        _info(), _info(), exact_gtin=False, config=settings
    )
    assert penalty == pytest.approx(0.04)


def test_candidate_score_uses_mask_but_keeps_raw_score() -> None:
    row = pd.Series({"SKU_ID": "sku-1", "barcode": "", "title": "water"})
    result = candidate_gate_fields(
        row,
        _info(),
        "candidate-1",
        {},
        0.83,
        sku_id="sku-1",
        source_row_index="0",
    )
    assert result["raw_score"] == pytest.approx(0.83)
    assert result["confidence_penalty"] == pytest.approx(0.03)
    assert result["score"] == pytest.approx(0.80)


def test_flavor_overlap_penalty_triggers_for_mismatch() -> None:
    jaccard, overlap, penalty, reason = flavor_overlap_penalty(
        _info(flavor="vanilla"),
        _info(flavor="lime"),
        exact_gtin=False,
        config=FLAVOR_SETTINGS,
    )
    assert jaccard == 0.0
    assert overlap == 0.0
    assert penalty == pytest.approx(0.05)
    assert reason == "low_flavor_overlap"


def test_flavor_overlap_penalty_accepts_normalized_shared_flavor() -> None:
    jaccard, overlap, penalty, reason = flavor_overlap_penalty(
        _info(flavor="Apple-Lemon flavour"),
        _info(flavor="apple_lemon flavored"),
        exact_gtin=False,
        config=FLAVOR_SETTINGS,
    )
    assert jaccard == 1.0
    assert overlap == 1.0
    assert penalty == 0.0
    assert reason == "sufficient_flavor_overlap"


def test_flavor_penalty_preserves_exact_gtin_score() -> None:
    row = pd.Series(
        {"SKU_ID": "sku-exact", "barcode": "4006381333931", "title": "lime"}
    )
    result = candidate_gate_fields(
        row,
        _info(flavor="lime"),
        "4006381333931",
        {
            "canonical": "vanilla drink",
            "volume_set": "[]",
            "pack_set": "[]",
            "mode_flavor": "vanilla",
        },
        0.81,
        sku_id="sku-exact",
        source_row_index="0",
    )
    assert result["flavor_penalty_reason"] == "exact_gtin_preserved"
    assert result["flavor_penalty"] == 0.0
    assert result["confidence_penalty"] == 0.0
    assert result["score"] == pytest.approx(result["raw_score"])


MISSING_ATTRIBUTE_FAILURE_FIXTURES = (
    ("clear Sparkling", "70118568"),
    (
        "Liquid Labs Rapid Hydration Electrolyte Drink Mix "
        "Tropical Fruit 20 Stick",
        "818594019908",
    ),
    ("sole DECO PLATE italy", "8011087136416"),
    ("Fresh apple- lemon", "4009300016908"),
    ("XYIENCE Frostberry Blast Energy Drink", "842885097146"),
)


def test_all_none_zero_failures_are_retained_by_exact_candidate_rescue() -> None:
    """The five observed missing-evidence cases must reach scoring.

    ANN top-K is intentionally empty in this fixture.  Each truth candidate
    must still be grabbed by the exact-GTIN rescue, after which the shared
    scorer emits explicit missing-evidence fields instead of filtering it.
    """
    matcher = object.__new__(RandMatcher)
    matcher.item_index = {
        gtin: index
        for index, (_, gtin) in enumerate(MISSING_ATTRIBUTE_FAILURE_FIXTURES)
    }
    retained = []
    for source_row, (title, gtin) in enumerate(MISSING_ATTRIBUTE_FAILURE_FIXTURES):
        grabbed = matcher._candidate_indexes([], gtin)
        candidate_index = matcher.item_index[gtin]
        assert grabbed[candidate_index] == (None, "exact_gtin_rescue")
        record = candidate_gate_fields(
            pd.Series(
                {
                    "SKU_ID": f"sku-{source_row}",
                    "barcode": gtin,
                    "title": title,
                }
            ),
            _info(),
            gtin,
            {
                "canonical": title,
                "volume_set": "[]",
                "pack_set": "[]",
                "mode_flavor": "",
            },
            0.25,
            sku_id=f"sku-{source_row}",
            source_row_index=str(source_row),
            retrieval_source="exact_gtin_rescue",
        )
        assert record["exact_gtin"] == 1
        assert record["jointly_missing_attribute_count"] >= 1
        assert record["jointly_missing_attributes"]
        retained.append(record)

    assert len(retained) == len(MISSING_ATTRIBUTE_FAILURE_FIXTURES) == 5
