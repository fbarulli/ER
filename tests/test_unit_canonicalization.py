from __future__ import annotations

import pytest

from core.attribute_conflicts import (
    attribute_conflict_types,
    canonical_attribute_info,
    conflict_columns,
    sku_attribute_info,
)
from core.structured_features import append_text, info_from_sets, sku_info
from core.unit_canonicalization import canonical_pack_count, canonical_volume_ml
from pipeline import canonical_evidence_text, extract_all


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["Made with apples", "No added sugar"], "made with apples no added sugar"),
        ("['Made with apples', 'No added sugar']", "made with apples no added sugar"),
        ('["Made with apples", "No added sugar"]', "made with apples no added sugar"),
        ("[]", ""),
        ("plain evidence", "plain evidence"),
    ],
)
def test_canonical_evidence_is_plain_deterministic_text(
    value: object, expected: str
) -> None:
    result = canonical_evidence_text(value)
    assert result == expected
    assert not any(marker in result for marker in ("[", "]", "'", '"'))


def test_canonical_evidence_sequence_order_is_stable() -> None:
    assert canonical_evidence_text({"zebra", "apple"}) == "apple zebra"


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


def test_source_structured_pack_defaults_to_singleton_when_count_is_unknown() -> None:
    info = sku_info("Plain tea", "")

    assert info["pack"] == {1.0}
    assert "[FIELD_PACK_SIZE] pack_qty_1" in append_text("plain tea", info)
    # The gate's separate confidence-aware parser must not inherit the
    # model-only singleton default.
    assert sku_attribute_info("Plain tea", "")["pack"] == set()


@pytest.mark.parametrize(
    ("title", "attributes", "expected_pack"),
    [
        ("Sparkling water 12-pack", "", 12.0),
        ("Sparkling water", "Count per Unit: 6", 6.0),
    ],
)
def test_source_structured_pack_preserves_explicit_counts(
    title: str, attributes: str, expected_pack: float
) -> None:
    assert sku_info(title, attributes)["pack"] == {expected_pack}


def test_structured_tokens_are_canonical_before_encoder_input() -> None:
    info = info_from_sets([236.588], ["12"])
    # The structured channel now emits a stable per-field marker before each
    # value group, so field boundaries survive the encoder's tokenizer.
    assert (
        append_text("water", info)
        == "water [FIELD_VOLUME] volume_ml_237 [FIELD_PACK_SIZE] pack_qty_12"
    )


def test_canonicalized_equivalent_volume_does_not_create_conflict() -> None:
    left = {"volume": {canonical_volume_ml(8, "fl oz")}, "pack": set(), "flavor": ""}
    right = canonical_attribute_info(
        {"volume_set": "[237]", "pack_set": "[]", "mode_flavor": ""}
    )
    assert attribute_conflict_types(left, right) == []


def test_bottle_and_can_are_first_class_disjoint_attributes() -> None:
    bottle = sku_attribute_info("Acme soda 6 pack 12 oz bottles", "")
    can = canonical_attribute_info(
        {
            "canonical": "Acme soda cans",
            "mode_flavor": "",
            "volume_set": "[355]",
            "pack_set": "[6]",
            "package_type_set": "['can']",
        }
    )
    assert bottle["package_type"] == {"bottle"}
    assert can["package_type"] == {"can"}
    assert attribute_conflict_types(bottle, can) == ["package_type"]
    assert conflict_columns(bottle, can)["package_type_conflict"] == 1


def test_package_type_reaches_the_shared_model_text_representation() -> None:
    info = info_from_sets([355], [6], ["bottle"])
    text = append_text("Acme soda", info)
    assert "[FIELD_PACKAGE_TYPE]" in text
    assert text.endswith("package_type_bottle")


def test_structured_field_markers_are_grouped_and_deterministic() -> None:
    """Markers appear once per populated field, in a stable field order."""
    info = info_from_sets(
        [355, 330],
        [6],
        ["bottle"],
        flavor={"cola", "lemon"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    marker_order = [
        token
        for token in append_text("Acme soda", info).split()
        if token.startswith("[FIELD_")
    ]
    assert marker_order == [
        "[FIELD_VOLUME]",
        "[FIELD_PACK_SIZE]",
        "[FIELD_PACKAGE_TYPE]",
        "[FIELD_FLAVOR]",
        "[FIELD_CARBONATION]",
        "[FIELD_SWEETENER_DIET]",
        "[FIELD_PULP]",
    ]
    # A field with no evidence contributes no marker, and repeated calls (which
    # iterate sets) must produce byte-identical text.
    empty_pulp = info_from_sets([355], [6], ["bottle"], flavor={"cola"})
    assert "[FIELD_PULP]" not in append_text("Acme soda", empty_pulp)
    assert len({append_text("Acme soda", info) for _ in range(25)}) == 1


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
