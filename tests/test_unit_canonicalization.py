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


@pytest.mark.parametrize(
    ("title", "expected_volume_ml", "expected_pack", "expected_package_type"),
    [
        ("Cock n Bull Ginger Beer 12 Pack 12oz Soda Cans", 355.0, 12, "can"),
        ("Cock n Bull Ginger Beer 12 Pack 12oz Soda Bottles", 355.0, 12, "bottle"),
        ("Heartsease Elderflower Presse 750ml Pack of 6", 750.0, 6, None),
        ("Perfect Hydration 16.9 oz Pack of 12", 500.0, 12, None),
        ("Hal's Lime Seltzer 16 Oz Cans Pack of 24", 473.0, 24, "can"),
        ("Victor Allen Vanilla Coffee 11 oz cans", 325.0, 1, "can"),
    ],
)
def test_fold_zero_title_attributes_reach_structured_scoring(
    title: str,
    expected_volume_ml: float,
    expected_pack: int,
    expected_package_type: str | None,
) -> None:
    extracted = extract_all(title, "")
    assert extracted["volume_ml"] == expected_volume_ml
    assert extracted["pack_qty"] == expected_pack
    if expected_pack == 1:
        assert extracted["pack_confidence"] == 0.0
    else:
        assert extracted["pack_confidence"] > 0.0
    if expected_package_type is not None:
        assert expected_package_type in extracted["package_types"]


@pytest.mark.parametrize(
    ("phrase", "expected_pack", "expected_type"),
    [
        ("sparkling water 12-pack cans", 12, "can"),
        ("sparkling water 12 pack cans", 12, "can"),
        ("sparkling water 12 cans", 12, "can"),
        ("hydration mix 14 packets", 14, "packet"),
        ("elderflower presse pack of 6", 6, None),
        ("elderflower presse 6-pack", 6, None),
        ("coffee 6 packages", 6, None),
        ("water case of 12 bottles", 12, "bottle"),
        ("water 12 cases", 12, None),
        ("juice 4 boxes", 4, "box"),
        ("juice 4 cartons", 4, "carton"),
        ("soda 8 tins", 8, "tin"),
    ],
)
def test_packaging_spacing_plural_and_hyphen_variants(
    phrase: str, expected_pack: int, expected_type: str | None
) -> None:
    extracted = extract_all(phrase, "")
    assert extracted["pack_qty"] == expected_pack
    assert extracted["pack_confidence"] > 0.0
    if expected_type is not None:
        assert expected_type in extracted["package_types"]
