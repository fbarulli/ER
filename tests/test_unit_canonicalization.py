from __future__ import annotations

import pytest

from core.attribute_conflicts import attribute_conflict_types, canonical_attribute_info
from core.structured_features import append_text, info_from_sets
from core.unit_canonicalization import canonical_pack_count, canonical_volume_ml
from pipeline import extract_all


@pytest.mark.parametrize(
    ("value", "unit", "expected_ml"),
    [
        ("0.237", "l", 237.0),
        ("23.7", "cl", 237.0),
        ("8", "fl oz", 237.0),
        ("1", "qt", 946.0),
    ],
)
def test_volume_units_share_a_whole_ml_representation(
    value: str, unit: str, expected_ml: float
) -> None:
    assert canonical_volume_ml(value, unit) == expected_ml


def test_pack_count_rejects_fractional_values() -> None:
    assert canonical_pack_count("24") == 24
    with pytest.raises(ValueError, match="must be an integer"):
        canonical_pack_count("2.5")


def test_pipeline_canonicalizes_title_and_attribute_units() -> None:
    title = extract_all("Sparkling water 8 fl oz, case of 12", "")
    attributes = extract_all("Sparkling water", "Volume: 0.75 L; Count per Unit: 6")
    assert title["volume_ml"] == 237.0
    assert title["pack_qty"] == 12
    assert attributes["volume_ml"] == 750.0
    assert attributes["pack_qty"] == 6


def test_structured_tokens_are_canonical_before_encoder_input() -> None:
    info = info_from_sets([236.588], ["12"])
    assert append_text("water", info) == "water volume_ml_237 pack_qty_12"


def test_canonicalized_equivalent_volume_does_not_create_conflict() -> None:
    left = {"volume": {canonical_volume_ml(8, "fl oz")}, "pack": set(), "flavor": ""}
    right = canonical_attribute_info(
        {"volume_set": "[237]", "pack_set": "[]", "mode_flavor": ""}
    )
    assert attribute_conflict_types(left, right) == []
