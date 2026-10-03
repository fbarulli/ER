from types import SimpleNamespace

import pytest

from core.common import data_cfg
from core.schemas import ExtractionPolicySpec
from pipeline import extract_all, extract_pack_evidence, extract_pack_from_title, fuse_confidence


def test_disagreement_uses_every_reader_and_is_order_independent():
    claims = [(330, .95, "attribute"), (330, .90, "sku_name_eng"), (500, .20, "sku_url")]
    assert fuse_confidence(claims) == pytest.approx(.20)
    assert fuse_confidence(list(reversed(claims))) == pytest.approx(.20)


def test_repeated_listing_surfaces_are_not_independent_corroboration():
    claims = [(330, .90, "sku_name_eng"), (330, .90, "sku_url"), (330, .90, "image_url")]
    assert fuse_confidence(claims) == pytest.approx(.90)
    assert fuse_confidence(claims + [(330, .90, "attribute")]) == pytest.approx(.99)


def test_nested_counts_and_outer_inner_quantities_remain_distinct():
    assert extract_pack_from_title("Reed's 6X4 / 12 Oz") == (24, data_cfg().extraction.pack_confidence["nested"])
    assert extract_pack_from_title("84 Cases, 2016 Bottles")[0] == 2016
    evidence = extract_pack_evidence("84 Cases, 2016 Bottles")
    assert {(entry["role"], entry["count"]) for entry in evidence} == {("outer_count", 84), ("unit_count", 2016)}
    assert extract_pack_from_title("84 Cases") == (1, 0.)


def test_negated_ingredient_keeps_both_source_claims_and_conflict():
    parsed = extract_all("No Stevia (12pk)", "Sweetener: stevia; Volume: 330")
    assert parsed["negated_sweetener_type_set"] == ["stevia"]
    assert "stevia" in parsed["sweetener_type_set"]
    assert "sweetener_source_conflict:stevia" in parsed["attribute_consistency_flags"]


def test_multipack_does_not_exempt_implausible_per_unit_volume():
    parsed = extract_all("Soda 40645-Ounce Pack of 6", "Pack Type: Bottle")
    assert "ambiguous_volume" in parsed["attribute_consistency_flags"]


def test_named_bulk_container_preserves_volume_and_flags_source_disagreement():
    parsed = extract_all("Spring Water 18 l", "Pack Type: Bucket; Volume: 100")
    assert parsed["volume_ml"] == 18000
    assert "ambiguous_volume" not in parsed["attribute_consistency_flags"]
    assert "volume_sources_disagree" in parsed["attribute_consistency_flags"]


def test_config_rejects_bad_ranges_and_unknown_source_policy():
    raw = data_cfg().extraction.model_dump()
    with pytest.raises(ValueError, match="ordered"):
        ExtractionPolicySpec.model_validate({**raw, "volume_min_ml": raw["volume_max_ml"]})
    with pytest.raises(ValueError, match="source_groups"):
        ExtractionPolicySpec.model_validate({**raw, "source_groups": {"sku_name_eng": "listing"}})
