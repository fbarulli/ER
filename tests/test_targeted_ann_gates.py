from __future__ import annotations

import pandas as pd
import pytest

from core.attribute_conflicts import critical_attribute_evaluation
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


def _info(
    *,
    pack=(),
    volume=(),
    package_type=(),
    flavor=(),
    carbonation=(),
    sweetener=(),
    pulp=(),
) -> dict[str, object]:
    """Build the record shape the gate actually consumes.

    Two readers meet on this mapping: ``critical_attribute_evaluation`` reads
    the SET keys (``flavor_set``), while the candidate diagnostics read the
    scalar ``flavor`` display key. The helper used to declare only a scalar
    ``flavor`` string, so the gate silently reported the flavor dimension as
    missing in every case. It now mirrors the real contract for both readers.
    Optional categorical dimensions default to empty (unknown), so a case only
    asserts on the evidence it declares.
    """
    flavor_values = set(flavor)
    return {
        "pack": set(pack),
        "volume": set(volume),
        "package_type": set(package_type),
        "flavor": " ".join(sorted(flavor_values)),
        "flavor_set": flavor_values,
        "carbonation": set(carbonation),
        "carbonation_set": set(carbonation),
        "sweetener": set(sweetener),
        "sweetener_set": set(sweetener),
        "pulp": set(pulp),
        "pulp_set": set(pulp),
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
    # Every critical dimension is declared on both sides so the ONLY thing this
    # case exercises is the configurable volume tolerance (780 vs 750 is 3.8%,
    # inside the 5% relative tolerance).
    common = dict(
        pack={6},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    gate = targeted_veto_gate(
        _info(volume={750}, **common),
        _info(volume={780}, **common),
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
    # The candidate declares every dimension EXCEPT pack, so pack_b is the only
    # missing evidence and the missingness that routes to review is precisely
    # the one this test names.
    gate = targeted_veto_gate(
        _info(
            pack={12},
            volume={750},
            package_type={"bottle"},
            flavor={"cola"},
            carbonation={"carbonated"},
            sweetener={"no_sugar"},
            pulp={"no_pulp"},
        ),
        _info(
            volume={750},
            package_type={"bottle"},
            flavor={"cola"},
            carbonation={"carbonated"},
            sweetener={"no_sugar"},
            pulp={"no_pulp"},
        ),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_gate_decision"] == "defer"
    assert gate["targeted_gate_route"] == "human_review"
    assert gate["targeted_missing_attributes"] == "pack_b"
    assert gate["targeted_missing_attribute_count"] == 1
    # Unknown evidence is NOT a conflict, and must not be reported as one.
    assert gate["targeted_critical_conflicts"] == ""
    # The audit boolean mirrors the same evaluation: partial evidence cannot
    # pass a full-evidence predicate.
    assert gate["targeted_pack_gate_pass"] == 0


def test_all_dimensions_known_and_agreeing_is_the_only_auto_merge_path():
    """The strict audit predicate passes only with complete, agreeing evidence."""
    full = dict(
        pack={6},
        volume={750},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    gate = targeted_veto_gate(
        _info(**full),
        _info(**full),
        sku_brand="Acme",
        candidate_brand="Acme",
        exact_gtin=False,
        config=SETTINGS,
    )
    assert gate["targeted_missing_attributes"] == ""
    assert gate["targeted_critical_conflicts"] == ""
    assert gate["targeted_pack_gate_pass"] == 1
    assert gate["targeted_gate_route"] == "auto_merge"


def test_extended_critical_dimensions_conflict_like_pack_and_volume():
    """carbonation/sweetener/pulp conflicts veto exactly like pack/volume."""
    base = dict(
        pack={6},
        volume={750},
        package_type={"bottle"},
        flavor={"cola"},
        carbonation={"carbonated"},
        sweetener={"no_sugar"},
        pulp={"no_pulp"},
    )
    for dimension, value, expected in (
        ("carbonation", {"still"}, "carbonation_mismatch"),
        ("sweetener", {"sugar"}, "sweetener_mismatch"),
        ("pulp", {"with_pulp"}, "pulp_mismatch"),
        ("flavor", {"orange"}, "flavor_mismatch"),
    ):
        gate = targeted_veto_gate(
            _info(**base),
            _info(**{**base, dimension: value}),
            sku_brand="Acme",
            candidate_brand="Acme",
            exact_gtin=False,
            config=SETTINGS,
        )
        assert gate["targeted_gate_route"] == "reject", dimension
        assert expected in gate["targeted_gate_reason"], dimension
        assert expected.split("_")[0] in gate["targeted_critical_conflicts"], dimension


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


def test_every_hard_no_gate_reason_maps_to_a_known_family():
    """Every reason the gate can emit on a HARD_NO must be classifiable.

    Only hard_no rows enter the balanced pool, so these are the reasons that
    must resolve. A new gate_reason string without a matching prefix makes
    sample_balanced_pairs raise "Unmatched hard-negative gate_reason values"
    and the whole balanced-pool lane hard-fails on fresh artifacts — which is
    exactly what the "Pack blocker:" reason did.
    """
    from training.sample_balanced_pairs import _reason_type

    hard_no_reasons = {
        # The composite blocker introduced with the shared critical-attribute
        # gate: highest-volume hard_no reason in the committed artifact
        # (87,804 of 135,769 rows) and the one that hard-failed the lane.
        "Pack blocker: pack size, package type, or volume mismatch",
        # Legacy single-dimension reasons, still emitted by the tail checks.
        "No volume overlap",
        "No pack overlap",
        "Package type mismatch",
        "Package material mismatch",
        "Flavor mismatch: lemon vs orange",
        # Categorical conflicts arrive as "Critical attribute mismatch: <dims>".
        "Critical attribute mismatch: flavor",
        "Critical attribute mismatch: carbonation,sweetener",
    }
    for reason in sorted(hard_no_reasons):
        assert _reason_type(reason) is not None, reason

    # Non-hard_no outcomes never reach the pool classifier, so they are
    # deliberately NOT required to resolve.
    for reason in ("Known critical attributes compatible", "Low raw volume confidence"):
        assert _reason_type(reason) is None, reason


def test_targeted_miner_never_emits_a_same_canonical_negative():
    """A same-canonical pair is a TRUE MATCH, never a label-0 row.

    build_training_data drops exactly those pairs for the baseline negatives
    and counts them as n_neg_same_canonical_dropped; the targeted miner must
    apply the same identity rule instead of re-adding them.
    """
    import numpy as np
    import pandas as pd

    from core.attribute_conflicts import canonical_attribute_info
    from core.hard_negatives import mine_targeted_attribute_negatives

    df = pd.DataFrame(
        {"title": ["Acme Cola 12 pack 355ml", "Acme Cola 12 pack 355ml"]}
    )
    canon_text = "acme cola still sugar"
    canonical_records = pd.DataFrame(
        [
            {
                "gtin": "A",
                "canonical": canon_text,
                "mode_brand": "acme",
                "mode_flavor": "",
                "volume_set": "[355]",
                "pack_set": "[12]",
                "package_type_set": "[]",
            },
            {
                "gtin": "B",
                "canonical": canon_text,
                "mode_brand": "acme",
                "mode_flavor": "",
                "volume_set": "[1000]",
                "pack_set": "[12]",
                "package_type_set": "[]",
            },
        ]
    )
    gates = pd.DataFrame(
        [{"gtin1": "A", "gtin2": "B", "similarity": 0.90, "gate_decision": "hard_no"}]
    )
    gtin_to_row = {"A": 0, "B": 1}
    gtin_to_canon_idx = {"A": 100, "B": 101}
    canonical_map = {"A": canon_text, "B": canon_text}

    # Sanity: the conflict evaluator DOES see a volume conflict here, so only
    # the same-canonical guard can prevent the pair from being mined.
    assert "volume" in critical_attribute_evaluation(
        canonical_attribute_info(canonical_records.iloc[0].to_dict()),
        canonical_attribute_info(canonical_records.iloc[1].to_dict()),
    )["conflicts"]

    guarded, _ = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        n_target=10,
        min_similarity=0.5,
        canonical_map=canonical_map,
    )
    assert len(guarded) == 0

    unguarded, _ = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        n_target=10,
        min_similarity=0.5,
    )
    assert len(unguarded) == 2  # both directions — the label inversion


def test_targeted_miner_respects_the_gate_volume_tolerance():
    """A within-tolerance volume gap is agreement, not a mineable conflict."""
    import pandas as pd

    from core.hard_negatives import mine_targeted_attribute_negatives

    df = pd.DataFrame({"title": ["Acme Juice 1L", "Acme Juice 1L"]})
    base = {
        "canonical": "acme juice still",
        "mode_brand": "acme",
        "mode_flavor": "",
        "pack_set": "[6]",
        "package_type_set": "[]",
    }
    canonical_records = pd.DataFrame(
        [
            {**base, "gtin": "A", "volume_set": "[480]"},
            {**base, "gtin": "B", "volume_set": "[500]"},
        ]
    )
    gates = pd.DataFrame(
        [{"gtin1": "A", "gtin2": "B", "similarity": 0.90, "gate_decision": "proceed"}]
    )
    gtin_to_row = {"A": 0, "B": 1}
    gtin_to_canon_idx = {"A": 100, "B": 101}

    strict, _ = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        n_target=10,
        min_similarity=0.5,
        volume_relative_tolerance=0.0,
    )
    assert len(strict) == 2  # 480 vs 500 is a conflict at exact-match tolerance

    tolerant, _ = mine_targeted_attribute_negatives(
        df,
        gates,
        canonical_records,
        gtin_to_row,
        gtin_to_canon_idx,
        n_target=10,
        min_similarity=0.5,
        volume_relative_tolerance=0.05,  # the gate's own tolerance
    )
    assert len(tolerant) == 0
