"""tests/test_full_attribute_decisions.py — the owner-ruling decision layer.

Ruling 2026-10-01: "ALL ATTRIBUTES are used to make ALL DECISIONS".

The full-universe evaluation (core.attribute_conflicts.full_attribute_
evaluation) must, on synthetic frames only:

(a) record a water-type conflict on a pair that agrees on volume/pack/flavor
    AND keep today's exact decision precedence — water type is not in
    config's veto_dimensions and not a critical-7 dimension, so the gate and
    the veto minted-conflicts list are UNCHANGED by it;
(b) keep one-sided absence as UNKNOWN — never minted into a conflict, never
    an agreement;
(c) leave the critical-7 outputs byte-stable against the unchanged
    critical_attribute_evaluation when the full wiring is ON;
(d) emit a census audit column (existing naming convention) for EVERY
    AttributeUniverse-registered field, including the veto-eligibility
    ledger covering the whole registry.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.attribute_conflicts import (
    DIMENSION_STATES,
    attribute_conflict_types,
    critical_attribute_evaluation,
    dimension_census_columns,
    full_attribute_evaluation,
    full_dimension_states,
    parse_universe_cell,
    veto_eligibility_ledger,
)
from core.attribute_universe import attribute_registry
from core.critical_attributes import CRITICAL_ATTRIBUTE_DIMENSIONS
from pipeline import attribute_gate_census_column_names


def _pair_record(**over: object) -> dict:
    """A synthetic SKU-side attribute record in the canonical convention."""
    base = {
        "volume": {1000.0},
        "pack": {6},
        "package_type": {"bottle"},
        "flavor": "",
        "flavor_set": {"lemon"},
        "carbonation": {"still"},
        "sweetener": {"no_sugar"},
        "sweetener_type": {"sucralose"},
        "sweetening": {"unsweetened"},
        "pulp": {"no_pulp"},
        "pack_material": {"glass"},
        "universe_evidence": {"water type": frozenset({"mineral"})},
        "unclassified_keys": (),
    }
    base.update(over)
    return base


# ── (a) water-type conflict is recorded BUT precedence is preserved ─────────
def test_water_type_conflict_is_recorded_and_precedence_is_unchanged():
    # Agree on volume/pack/flavor; DISAGREE on water type (both sides live).
    left = _pair_record(universe_evidence={"water type": frozenset({"mineral"})})
    right = _pair_record(
        volume={1010.0},
        universe_evidence={"water type": frozenset({"spring"})},
    )
    evaluation = full_attribute_evaluation(
        left, right, volume_relative_tolerance=0.05
    )
    # the decision layer RECORDS the water-type conflict with the field's own
    # measured semantics (plain frozenset inequality, no overlap rule)
    assert evaluation["dimension_states"]["water type"] == "conflict"
    assert "water type" in evaluation["dimension_conflicts"]
    # ...while the critical-7 decision outputs say exactly what they said
    # before the ruling: volume compatible inside 5%, pack/flavor agree
    assert evaluation["conflicts"] == []
    assert set(evaluation["agreements"]) >= {"volume", "pack", "flavor", "carbonation", "pulp"}
    # SAME precedence: a config-driven veto consumes ONLY the configured
    # veto_dimensions — water type is in none (rand_matching lane applies
    # exactly this substraction over this same evaluation object).
    vetoed = [d for d in evaluation["dimension_conflicts"] if d in {"volume", "pack", "package_type", "flavor", "carbonation", "pulp", "pack_material"}]
    assert vetoed == []


# ── (b) absence on one side stays unknown, no veto minted ────────────────────
def test_absence_on_one_side_stays_unknown_never_a_veto():
    left = _pair_record(universe_evidence={"water type": frozenset({"mineral"})})
    right = _pair_record(universe_evidence={})
    states = full_dimension_states(left, right)
    assert states["water type"] == "missing_right"
    left = _pair_record(universe_evidence={})
    right = _pair_record(universe_evidence={"water type": frozenset({"mineral"})})
    assert full_dimension_states(left, right)["water type"] == "missing_left"
    both = _pair_record(universe_evidence={})
    assert full_dimension_states(both, both)["water type"] == "missing_both"
    evaluation = full_attribute_evaluation(both, both)
    assert evaluation["dimension_conflicts"] == []
    # and the missingness NEVER reaches the critical-7 confusion either
    assert evaluation["conflicts"] == []
    assert evaluation["unknown"] == []


def test_critical_channel_absence_also_stays_unknown():
    evaluation = full_attribute_evaluation(_pair_record(), _pair_record(sweetener=set(), sweetener_type=set(), sweetening=set()))
    assert "sweetener" in evaluation["unknown"]
    assert "sweetener" not in evaluation["conflicts"]


# ── (c) byte-stability of the critical-7 outputs with all wiring ON ─────────
def _rich_pair(**over):
    base = {}
    critical = {
        "volume_set": {500.0},
        "pack_set": {24},
        "package_type": {"can"},
        "package_type_set": {"can"},
        "flavor_set": {"cola", "lemon"},
        "carbonation_set": {"carbonated"},
        "sweetener_set": {"diet"},
        "sweetener_type_set": {"stevia"},
        "sweetening_set": {"unsweetened"},
        "pulp_set": set(),
        "pack_material": {"metal"},
        "universe_evidence": {
            "pack material type": frozenset({"metal"}),
            "juice content": frozenset({"100%"}),
        },
        "volume_confidence": 0.9,
        "pack_confidence": 0.9,
        "volume_consistency": 0.9,
        "pack_consistency": 0.9,
    }
    base.update(critical)
    base.update(over)
    return base


def test_full_evaluation_critical7_outputs_are_byte_stable():
    left = _rich_pair(universe_evidence={
        "pack material type": frozenset({"metal", "glass"}),
        "juice content": frozenset({"0-2%", "100%"}),
    })
    right = _rich_pair(
        pack_material={"plastic"},
        package_type={"bottle"},
        package_type_set={"bottle"},
        universe_evidence={
            "pack material type": frozenset({"plastic", "glass"}),
            "juice content": frozenset({"100%"}),
        },
    )
    full = full_attribute_evaluation(left, right, volume_relative_tolerance=0.05)
    plain = critical_attribute_evaluation(left, right, volume_relative_tolerance=0.05)
    assert full["conflicts"] == plain["conflicts"] == ["package_type"]
    assert full["unknown"] == plain["unknown"]
    assert full["agreements"] == plain["agreements"]


def test_three_way_gate_decision_is_identical_with_full_wiring_present():
    from pipeline import three_way_gate

    # two pairs that differ ONLY in the captured water-type evidence: the
    # gate's decision is byte-identical, because the veto still votes only
    # where the config permits (water type is audit-only today).
    water_a = _rich_pair(universe_evidence={"water type": frozenset({"mineral"})})
    water_b = _rich_pair(universe_evidence={"water type": frozenset({"spring"})})
    assert three_way_gate(water_a, water_b) == three_way_gate(water_a, water_a)
    assert three_way_gate(water_a, water_b)["decision"] == "proceed"


# ── (d) every registered field appears in the trace census columns ──────────
def test_every_registered_field_has_a_census_state_column():
    columns = attribute_gate_census_column_names()
    for key in attribute_registry():
        assert f"{key.replace(' ', '_')}_state" in columns, key
    assert len(columns) == len(attribute_registry()) + 6


def test_census_columns_states_stay_inside_the_audited_vocabulary():
    left = _pair_record(
        universe_evidence={
            "water type": frozenset({"mineral"}),
            "juice content": frozenset({"0-2%"}),
            "caffeine": frozenset({"0-15mg"}),
        }
    )
    right = _pair_record(
        universe_evidence={
            "juice content": frozenset({"100%"}),
            "naturally derived": frozenset({"natural"}),
        }
    )
    states = full_dimension_states(left, right)
    assert set(states) == set(attribute_registry())
    assert set(states.values()) <= DIMENSION_STATES
    columns = dimension_census_columns(states)
    assert columns["water_type_state"] == "missing_right"
    assert columns["juice_content_state"] == "conflict"
    # a caffeine band pair with both sides POPULATED via canonical bands
    # compares through band inequality; a single-sided one is missing, never
    # a conflict
    assert columns["caffeine_state"] == "missing_right"
    assert "naturally_derived_state" in columns
    # a one-side caffeine value is missing, never a conflict; the unknown-parse
    # state fires only on a genuinely unclassifiable band token
    states = full_dimension_states(
        left,
        _pair_record(universe_evidence={"caffeine": frozenset({"no numeric band text"})}),
    )
    assert states["caffeine"] == "unknown_parse"
    with pytest.raises(ValueError):
        dimension_census_columns({"volume": "does-not-exist"})


# ── parse parity: the decision layer uses the census SSOT parser ─────────────
def test_parse_universe_cell_uses_the_census_device():
    universe = parse_universe_cell(
        "Juice Content: 0-2%; Water Type: Mineral; Caffeine: 15-25 mg"
    )
    from core.attribute_universe import AttributeUniverse
    import pandas as pd

    reference = AttributeUniverse(
        pd.DataFrame(
            {
                "attributes": ["Juice Content: 0-2%; Water Type: Mineral; Caffeine: 15-25 mg"],
                "barcode": ["8715600246377"],
            }
        )
    ).parse("Juice Content: 0-2%; Water Type: Mineral; Caffeine: 15-25 mg")
    assert universe == reference
    assert universe["juice content"] == frozenset({"0-2%"})


# ── veto-eligibility ledger: evidence class + config delta ───────────────────
def _fake_veto_config(vetoed: list[str]) -> SimpleNamespace:
    return SimpleNamespace(veto_dimensions=vetoed)


def test_ledger_reads_config_and_reports_owner_delta():
    # the census parameter is the KEYS mapping (the same shape the ledger's
    # own loader returns from results/attribute_universe_census.json).
    census = {
        "water type": {"conflict_rate": 0.1558},
        "pack material type": {"conflict_rate": 0.0964},
        "volume": {"conflict_rate": 0.0146},
        "concentrate format": {"conflict_rate": 0.4850},
        "special edition": {"conflict_rate": 0.0},
    }
    settings = _fake_veto_config(["volume", "pack", "package_type", "flavor", "carbonation", "pulp"])
    ledger = veto_eligibility_ledger(census=census, config=settings)
    assert set(ledger) == set(attribute_registry()) | {
        "volume", "pack", "package_type", "flavor", "carbonation", "sweetener", "pulp", "pack_material"
    }
    water = ledger["water type"]
    # the census run measures water type at 15.58% — ABOVE the 15% ceiling —
    # so the ledger reports the review-lane class, not the veto band.
    assert water["evidence_class"] == "review_lane_above_band"
    assert water["config_state"] == "audit_only"
    assert "veto_dimensions" in water["owner_delta"]
    assert water["conflict_rate"] == 0.1558
    assert water["conflict_semantics"] == "set_inequality"
    pack_material = ledger["pack_material"]
    assert pack_material["evidence_class"] == "veto_band_candidate"
    assert pack_material["in_schema_allow_list"] is True
    # deltas are empty exactly where the config already permits the veto
    assert {d for d, row in ledger.items() if row["owner_delta"]} == set(ledger) - set(
        settings.veto_dimensions
    )


def test_ledger_class_boundaries_follow_the_measured_veto_band():
    census = {"coffee type": {"conflict_rate": 0.0190}}
    ledger = veto_eligibility_ledger(census=census, config=_fake_veto_config([]))
    assert ledger["coffee type"]["evidence_class"] == "below_veto_floor"
    census = {"concentrate format": {"conflict_rate": 0.4850}}
    ledger = veto_eligibility_ledger(census=census, config=_fake_veto_config([]))
    assert ledger["concentrate format"]["evidence_class"] == "review_lane_above_band"
    census = {"keys": {}}
    ledger = veto_eligibility_ledger(census=census, config=_fake_veto_config(["volume"]))
    assert ledger["volume"]["evidence_class"] == "non_attribute_cell_lane"
    # CONSTANT keys are the no-yield class (never a veto, never a donor)
    census = {"special edition": {"conflict_rate": 0.0}}
    ledger = veto_eligibility_ledger(census=census, config=_fake_veto_config([]))
    assert ledger["special edition"]["evidence_class"] == "no_yield"


def test_pack_material_veto_census_key_maps_through_the_ssot_mapping():
    ledger = veto_eligibility_ledger(
        census={"pack material type": {"conflict_rate": 0.0964}},
        config=_fake_veto_config(["pack_material"]),
    )
    assert ledger["pack_material"]["census_key"] == "pack material type"
    assert ledger["pack_material"]["config_state"] == "hard_veto_permitted"
    assert ledger["pack_material"]["owner_delta"] == ""


# ── census census column rollups: failure modes stay loud ─────────────────────
def test_conflicted_column_rollup_is_deterministic():
    states = full_dimension_states(
        _pair_record(universe_evidence={"water type": frozenset({"mineral"})}),
        _pair_record(universe_evidence={"water type": frozenset({"spring"})}),
    )
    columns = dimension_census_columns(states)
    assert columns["dimension_conflicts"] == "water_type"
    assert columns["dimension_conflict_count"] == 1
    assert columns["dimension_missing_left"] == ""
    # and the audit columns of one pair are the SAME dictionary, rerun after
    rerun = dimension_census_columns(
        full_dimension_states(
            _pair_record(universe_evidence={"water type": frozenset({"mineral"})}),
            _pair_record(universe_evidence={"water type": frozenset({"spring"})}),
        )
    )
    assert columns == rerun


# ── stage-8 provenance must survive to the audit trail ───────────────────────
def test_semantic_family_rescue_records_its_provenance(monkeypatch):
    """Regression: the rescue fired but its attribution was wiped.

    AttributeDecisionEngine.evaluate assigned `fallback_from =
    "semantic_family"` inside the adjudication branch and then reset
    `fallback_from = ""` unconditionally on the NEXT statement (the
    assignment sat after the if/else, at the loop-body indent). The verdict
    was still downgraded CONFLICT -> MATCH, so the decision was right, but
    every DimensionDecision reported fallback_from="" — the audit trail the
    stage-8 comment promises ("recorded with its source so audits trace the
    downgrade") could never fire, and a rescue was indistinguishable from a
    clean lexical match.

    Pinned by stubbing the family index: a pair that shares a family must
    report the rescue AND name it.
    """
    import core.attribute_decision as decision_module
    from core.attribute_decision import AttributeDecisionEngine, ComparisonResult

    monkeypatch.setattr(
        decision_module, "semantic_family_shared", lambda left, right: True
    )
    engine = AttributeDecisionEngine(
        volume_relative_tolerance=0.05, volume_absolute_tolerance_ml=5.0
    )

    def record(roast: str) -> dict:
        return {
            "universe_evidence": {
                "coffee type": frozenset({"arabica"}),
                "roast type": frozenset({roast}),
            },
            "volume_set": {355.0},
            "flavor_set": {"coffee"},
        }

    evidence = engine.evaluate(record("dark"), record("medium"))
    rescued = evidence.dimensions["roast type"]

    assert rescued.result is ComparisonResult.MATCH, "rescue must downgrade"
    assert rescued.fallback_from == "semantic_family", (
        "the rescue's provenance must reach DimensionDecision.fallback_from"
    )


def test_unrescued_conflict_reports_no_provenance(monkeypatch):
    """The control: a genuine conflict must NOT claim a rescue."""
    import core.attribute_decision as decision_module
    from core.attribute_decision import AttributeDecisionEngine, ComparisonResult

    monkeypatch.setattr(
        decision_module, "semantic_family_shared", lambda left, right: False
    )
    engine = AttributeDecisionEngine(
        volume_relative_tolerance=0.05, volume_absolute_tolerance_ml=5.0
    )

    def record(roast: str) -> dict:
        return {
            "universe_evidence": {"roast type": frozenset({roast})},
            "volume_set": {355.0},
            "flavor_set": {"coffee"},
        }

    evidence = engine.evaluate(record("dark"), record("medium"))
    conflicted = evidence.dimensions["roast type"]

    assert conflicted.result is ComparisonResult.CONFLICT
    assert conflicted.fallback_from == ""
