from core.attribute_conflicts import (
    CRITICAL_NAME_BY_CENSUS_KEY, canonical_attribute_info,
)
from core.attribute_decision import AttributeDecisionEngine, ComparisonResult
from training.rand_matching import targeted_veto_gate


def test_count_and_format_are_distinct_registry_features():
    assert CRITICAL_NAME_BY_CENSUS_KEY["count per unit"] == "pack"
    assert CRITICAL_NAME_BY_CENSUS_KEY["pack type"] == "package_type"
    a = canonical_attribute_info({"pack_set": {6}, "package_type_set": {"bottle"}})
    b = canonical_attribute_info({"pack_set": {12}, "package_type_set": {"bottle"}})
    evidence = AttributeDecisionEngine(volume_relative_tolerance=.05).evaluate(a, b)
    assert evidence.dimensions["count per unit"].result is ComparisonResult.CONFLICT
    assert evidence.dimensions["pack type"].result is ComparisonResult.MATCH


def test_raw_format_evidence_does_not_invent_curated_numeric_count():
    a = canonical_attribute_info({"volume_set": {355}, "pack_set": set(),
                                 "universe_evidence": {"pack type": ["aerosol"]}})
    b = canonical_attribute_info({"volume_set": {355}, "pack_set": set(),
                                 "package_type_set": {"bottle"}})
    evidence = AttributeDecisionEngine(volume_relative_tolerance=.05).evaluate(a, b)
    assert evidence.dimensions["pack type"].result is ComparisonResult.CONFLICT
    assert evidence.dimensions["count per unit"].result is ComparisonResult.INCONCLUSIVE
    gate = targeted_veto_gate(a, b, sku_brand="acme", candidate_brand="acme", exact_gtin=False)
    assert gate["targeted_gate_route"] == "human_review"
    assert not gate["targeted_vetoed_conflicts"]
