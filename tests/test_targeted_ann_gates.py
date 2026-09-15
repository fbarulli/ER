from __future__ import annotations

import pandas as pd
import pytest

from core.common import rand_matching_cfg
from training.rand_matching import (
    _annotate_candidates,
    _assignments_with_trace,
    candidate_gate_fields,
    targeted_veto_gate,
)


SETTINGS = {
    "enabled": True,
    "pack_mismatch_veto": True,
    "volume_mismatch_veto": True,
    "package_type_mismatch_veto": True,
    "brand_mismatch_veto": True,
    "missing_pack_or_volume_route": "human_review",
    "volume_relative_tolerance": 0.05,
    "volume_absolute_tolerance_ml": 5.0,
    "preserve_exact_gtin": True,
}


def _info(*, pack=(), volume=(), package_type=(), flavor="") -> dict[str, object]:
    return {
        "pack": set(pack),
        "volume": set(volume),
        "package_type": set(package_type),
        "flavor": flavor,
    }


@pytest.mark.parametrize(
    ("left", "right", "left_brand", "right_brand", "reason"),
    [
        (
            _info(pack={12}, volume={750}),
            _info(pack={6}, volume={750}),
            "Acme",
            "Acme",
            "pack_mismatch",
        ),
        (
            _info(pack={6}, volume={750}),
            _info(pack={6}, volume={1000}),
            "Acme",
            "Acme",
            "volume_mismatch",
        ),
        (
            _info(pack={6}, volume={750}),
            _info(pack={6}, volume={750}),
            "Acme",
            "Other",
            "brand_mismatch",
        ),
        (
            _info(pack={6}, volume={355}, package_type={"bottle"}),
            _info(pack={6}, volume={355}, package_type={"can"}),
            "Acme",
            "Acme",
            "package_type_mismatch",
        ),
    ],
)
def test_known_conflicts_are_hard_vetoes(left, right, left_brand, right_brand, reason):
    gate = targeted_veto_gate(
        left,
        right,
        sku_brand=left_brand,
        candidate_brand=right_brand,
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_route"] == "reject"
    assert reason in gate["targeted_gate_reason"]


def test_volume_tolerance_is_canonical_and_configurable():
    gate = targeted_veto_gate(
        _info(pack={6}, volume={750}),
        _info(pack={6}, volume={780}),
        sku_brand="Acme",
        candidate_brand="acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_volume_conflict"] == 0
    assert gate["targeted_gate_route"] == "auto_merge"
    assert gate["targeted_volume_ml_a"] == "[750]"
    assert gate["targeted_volume_ml_b"] == "[780]"


def test_missing_pack_or_volume_is_human_review_not_auto_merge():
    gate = targeted_veto_gate(
        _info(pack={12}, volume={750}),
        _info(pack=(), volume={750}),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_decision"] == "defer"
    assert gate["targeted_gate_route"] == "human_review"
    assert gate["targeted_missing_attributes"] == "pack_b"


def test_exact_gtin_lock_bypasses_conflicts_and_missingness():
    gate = targeted_veto_gate(
        _info(pack={12}, volume={750}),
        _info(),
        sku_brand="Acme",
        candidate_brand="Other",
        exact_gtin=True,
        config=SETTINGS,
    )
    assert gate["targeted_gate_decision"] == "exact_gtin_lock"
    assert gate["targeted_gate_route"] == "auto_merge"


def test_package_type_veto_has_explicit_audit_evidence():
    gate = targeted_veto_gate(
        _info(pack={6}, volume={355}, package_type={"bottle"}),
        _info(pack={6}, volume={355}, package_type={"can"}),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_package_type_conflict"] == 1
    assert gate["targeted_package_type_a"] == '["bottle"]'
    assert gate["targeted_package_type_b"] == '["can"]'
    assert gate["targeted_gate_decision"] == "veto"


def test_bottle_sku_to_can_candidate_is_rejected_end_to_end():
    row = pd.Series(
        {
            "SKU_ID": "bottle-sku",
            "barcode": "",
            "title": "Acme soda 6 pack 12 oz bottles",
            "brand": "Acme",
        }
    )
    candidate = candidate_gate_fields(
        row,
        _info(pack={6}, volume={355}, package_type={"bottle"}),
        "candidate-can",
        {
            "canonical": "Acme soda 6 pack 355 ml cans",
            "mode_brand": "Acme",
            "mode_flavor": "",
            "volume_set": "[355]",
            "pack_set": "[6]",
            "package_type_set": "['can']",
        },
        0.99,
        sku_id="bottle-sku",
        source_row_index="0",
    )
    assert candidate["attribute_conflict_type"] == "package_type"
    assert candidate["rule_ok"] == 0
    assert candidate["targeted_gate_route"] == "reject"

    trace = _annotate_candidates(pd.DataFrame([candidate]), 0.61)
    assert not bool(trace.iloc[0]["accepted"])
    assert trace.iloc[0]["rejection_reason"] == "targeted_attribute_veto"


def test_exact_gtin_package_type_disagreement_uses_configured_lock():
    gtin = "4006381333931"
    row = pd.Series(
        {
            "SKU_ID": "exact-bottle",
            "barcode": gtin,
            "title": "Acme soda 6 pack 12 oz bottles",
            "brand": "Acme",
        }
    )
    candidate = candidate_gate_fields(
        row,
        _info(pack={6}, volume={355}, package_type={"bottle"}),
        gtin,
        {
            "canonical": "Acme soda 6 pack 355 ml cans",
            "mode_brand": "Acme",
            "mode_flavor": "",
            "volume_set": "[355]",
            "pack_set": "[6]",
            "package_type_set": "['can']",
        },
        0.10,
        sku_id="exact-bottle",
        source_row_index="0",
    )
    assert candidate["targeted_gate_decision"] == "exact_gtin_lock"
    assert candidate["rule_ok"] == 0
    trace = _annotate_candidates(pd.DataFrame([candidate]), 0.99)
    assert bool(trace.iloc[0]["accepted"])
    assert trace.iloc[0]["attribute_gate"] == "override_exact_gtin"


def _candidate(
    sku: str,
    score: float,
    *,
    route: str,
    exact: int = 0,
    status: str = "different",
) -> dict[str, object]:
    return {
        "SKU_ID": sku,
        "candidate_gtin": f"candidate-{sku}",
        "score": score,
        "gtin_status": status,
        "exact_gtin": exact,
        "rule_ok": 1,
        "attribute_matches": 3,
        "brand_conflict": 0,
        "targeted_gate_route": route,
    }


def test_weakest_06004_edge_fails_new_threshold():
    candidates = pd.DataFrame([_candidate("weak", 0.6004, route="auto_merge")])
    trace = _annotate_candidates(
        candidates,
        0.61,
    )
    assert float(trace.iloc[0]["effective_threshold"]) == pytest.approx(0.61)
    assert not bool(trace.iloc[0]["accepted"])
    assert trace.iloc[0]["rejection_reason"] == "below_threshold"


def test_weak_missing_edge_is_routed_to_review_and_never_selected():
    candidates = pd.DataFrame([_candidate("weak", 0.6004, route="human_review")])
    predictions, trace = _assignments_with_trace(candidates, 0.60)
    assert trace.iloc[0]["assignment_gate"] == "human_review_candidate"
    assert trace.iloc[0]["rejection_reason"] == "human_review_missing_pack_or_volume"
    assert predictions.iloc[0]["ITEM_ID"].startswith("UNMATCHED_")


def test_061_is_monotonic_and_cannot_increase_overmerge_edges():
    candidates = pd.DataFrame(
        [
            _candidate("weak", 0.6004, route="auto_merge"),
            _candidate("veto", 0.90, route="reject"),
            _candidate("review", 0.95, route="human_review"),
            _candidate("strong", 0.82, route="auto_merge"),
        ]
    )
    at_060 = _annotate_candidates(candidates, 0.60)
    at_061 = _annotate_candidates(candidates, 0.61)
    accepted_060 = set(at_060.loc[at_060["accepted"], "SKU_ID"])
    accepted_061 = set(at_061.loc[at_061["accepted"], "SKU_ID"])
    assert accepted_061 <= accepted_060
    assert accepted_060 - accepted_061 == {"weak"}
    assert accepted_061 == {"strong"}


def test_060_override_is_bumped_without_lowering_other_strata():
    thresholds = rand_matching_cfg()["threshold_by_gtin_status"]
    assert thresholds["both_equal"] == pytest.approx(0.80)
    assert thresholds["different"] == pytest.approx(0.80)
    assert thresholds["one_missing"] == pytest.approx(0.61)
    assert thresholds["both_missing"] == pytest.approx(0.80)
