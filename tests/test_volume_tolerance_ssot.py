"""tests/test_volume_tolerance_ssot.py — ONE volume tolerance, every lane.

Defect this pins (owner ruling 2026-10-01). The volume comparison predicate
(core.critical_attributes.volumes_compatible) takes TWO tolerances: a
relative cut and an absolute one in ml, and applies whichever is wider. Only
the relative cut was declared in config/training.yaml `gate:`, so every lane
that omitted `volume_absolute_tolerance_ml` silently fell back to the
parameter default of 0.0 while the veto lane applied it.

The absolute cut is not a rounding detail. At small volumes the relative cut
is the STRICTER of the two — 5% of 14ml is 0.7ml — so a relative-only lane
answers "conflict" where the wired veto lane answers "compatible". Measured on
data/canonical_records.csv: 61 within-brand volume pairs, e.g. 14-vs-16ml,
8-vs-7ml, 19-vs-21ml.

A disagreement between lanes that are supposed to be one SSOT is the exact
failure this repo's audit trail keeps hitting: a pair the gate labels
`proceed` gets mined as a hard negative, or a gate `hard_no` is written into
the census as a conflict.

These tests pin the tolerance as a DECLARED, PROPAGATED value:

(a) the gate block declares BOTH cuts and the absolute one is non-zero
    (the omission itself was the defect);
(b) every lane that consults volumes_compatible receives both cuts from the
    same config SSOT — asserted by patching volumes_compatible and reading
    the arguments each lane actually passed, so a future call site that drops
    the argument fails here instead of drifting silently;
(c) the census and the veto lane now AGREE on the affected small-volume pairs
    (the disagreement is the regression);
(d) genuinely distinct volumes stay a conflict — the fix must not have
    loosened the gate into accepting real differences.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.common import training_cfg
from core.critical_attributes import volumes_compatible


def _gate_tolerances() -> tuple[float, float]:
    gate = training_cfg().gate
    return float(gate.vol_tolerance), float(gate.vol_abs_tolerance)


def _attrs(**over: Any) -> dict:
    """A minimal record in the shape pipeline.three_way_gate consumes."""
    base = {
        "volume_set": {355.0},
        "pack_set": {12},
        "package_type_set": {"can"},
        "package_material_set": {"metal"},
        "packaging_level_set": set(),
        "flavor_set": {"vanilla"},
        "carbonation_set": set(),
        "sweetener_set": set(),
        "pulp_set": set(),
        "volume_confidence": 0.9,
        "pack_confidence": 0.9,
        "volume_consistency": 0.9,
        "pack_consistency": 0.9,
    }
    base.update(over)
    return base


# ── (a) the gate block declares both cuts ────────────────────────────────────


def test_gate_block_declares_both_volume_cuts() -> None:
    """The declaration itself. vol_abs_tolerance was absent entirely."""
    gate = training_cfg().gate
    assert gate.vol_tolerance > 0.0
    assert gate.vol_abs_tolerance > 0.0, (
        "gate.vol_abs_tolerance must be declared and non-zero: the relative "
        "cut alone is the stricter rule at small volumes, so leaving the "
        "absolute one at 0.0 makes this lane disagree with the veto lane"
    )


def test_relative_and_absolute_are_the_veto_lane_values() -> None:
    """The gate block and the veto block must not declare two truths.

    masking.py previously read the absolute cut from the veto block while
    reading the relative cut from the gate block — two declarations of one
    tolerance, which is how the lanes drifted. The veto block is retained for
    its own lane, but the gate block is now the SSOT both cuts are read from.
    """
    relative, absolute = _gate_tolerances()
    veto = training_cfg().rand_matching.targeted_veto_gates
    assert relative == pytest.approx(float(veto.volume_relative_tolerance))
    assert absolute == pytest.approx(float(veto.volume_absolute_tolerance_ml))


# ── (c) the census and the veto lane now agree ───────────────────────────────

# The measured affected population: relative-only said "conflict", the wired
# veto lane said "compatible".
SMALL_VOLUME_PAIRS = [(14.0, 16.0), (8.0, 7.0), (19.0, 21.0), (60.0, 65.0)]


@pytest.mark.parametrize(("left", "right"), SMALL_VOLUME_PAIRS)
def test_census_agrees_with_veto_lane_on_small_volumes(
    left: float, right: float
) -> None:
    """The regression: one SSOT, one answer.

    Before the fix the census was called with the relative cut only and
    reported `conflict` for every pair here, while the veto lane (both cuts)
    reported compatible.
    """
    from core.attribute_conflicts import full_dimension_states

    relative, absolute = _gate_tolerances()

    left_record = {"volume_set": {left}, "universe_evidence": {}}
    right_record = {"volume_set": {right}, "universe_evidence": {}}

    census_state = full_dimension_states(
        left_record,
        right_record,
        volume_relative_tolerance=relative,
        volume_absolute_tolerance_ml=absolute,
    )["volume"]

    veto_verdict = volumes_compatible(
        {left},
        {right},
        volume_relative_tolerance=relative,
        volume_absolute_tolerance_ml=absolute,
    )

    assert census_state == ("agree" if veto_verdict else "conflict"), (
        f"{left}ml vs {right}ml: census said {census_state!r} while the veto "
        f"lane said compatible={veto_verdict} — the lanes must not disagree"
    )


@pytest.mark.parametrize(
    ("left", "right"), [(500.0, 5000.0), (1000.0, 2000.0), (355.0, 1000.0)]
)
def test_genuinely_different_volumes_stay_a_conflict(
    left: float, right: float
) -> None:
    """The fix must not loosen the gate into accepting real differences."""
    relative, absolute = _gate_tolerances()
    assert not volumes_compatible(
        {left},
        {right},
        volume_relative_tolerance=relative,
        volume_absolute_tolerance_ml=absolute,
    )


# ── (b) both cuts reach every lane that consults the predicate ───────────────


def test_three_way_gate_propagates_both_cuts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the arguments the gate actually passed.

    Asserting on the predicate's arguments rather than on a gate decision
    keeps this test from silently passing if the volume branch is never
    reached, and pins the specific defect (a dropped absolute argument).
    """
    import pipeline

    observed: list[tuple[float, float]] = []
    real = volumes_compatible

    def spy(left, right, **kwargs):
        observed.append(
            (
                float(kwargs.get("volume_relative_tolerance", 0.0)),
                float(kwargs.get("volume_absolute_tolerance_ml", 0.0)),
            )
        )
        return real(left, right, **kwargs)

    monkeypatch.setattr("core.critical_attributes.volumes_compatible", spy)
    monkeypatch.setattr(pipeline, "volumes_compatible", spy)

    pipeline.three_way_gate(_attrs(), _attrs())

    relative, absolute = _gate_tolerances()
    assert observed, "the gate never reached a volume comparison"
    assert all(calls[0] == pytest.approx(relative) for calls in observed), (
        f"relative tolerance not propagated: {observed}"
    )
    assert all(calls[1] == pytest.approx(absolute) for calls in observed), (
        f"ABSOLUTE tolerance not propagated: {observed} — every call fell "
        f"back to a stricter relative-only comparison"
    )


def test_decision_engine_loads_absolute_from_config() -> None:
    """AttributeDecisionEngine.load read 0.0, not the configured value."""
    from core.attribute_decision import AttributeDecisionEngine

    engine = AttributeDecisionEngine.load()
    relative, absolute = _gate_tolerances()
    assert engine.volume_relative_tolerance == pytest.approx(relative)
    assert engine.volume_absolute_tolerance_ml == pytest.approx(absolute)


def test_decision_engine_honours_an_explicit_override() -> None:
    """Direct construction still wins (the selftest pins known-good values)."""
    from core.attribute_decision import AttributeDecisionEngine

    engine = AttributeDecisionEngine.load(
        overrides={"volume_absolute_tolerance_ml": 0.0}
    )
    assert engine.volume_absolute_tolerance_ml == 0.0


# ── the gate's decision surface, end to end ─────────────────────────────────


@pytest.mark.parametrize(("left", "right"), SMALL_VOLUME_PAIRS[:3])
def test_gate_accepts_small_volume_pairs_within_the_absolute_cut(
    left: float, right: float
) -> None:
    from pipeline import three_way_gate

    result = three_way_gate(_attrs(volume_set={left}), _attrs(volume_set={right}))
    assert result["decision"] == "proceed", (
        f"{left}ml vs {right}ml should be within the configured absolute cut: "
        f"{result['reason']}"
    )


def test_gate_still_rejects_a_genuine_volume_mismatch() -> None:
    from pipeline import three_way_gate

    result = three_way_gate(
        _attrs(volume_set={500.0}), _attrs(volume_set={5000.0})
    )
    assert result["decision"] == "hard_no"
    assert "volume" in result["reason"].lower()


def test_gate_rejects_on_non_volume_attributes_unchanged() -> None:
    """The absolute cut must not soften any non-volume dimension.

    Configured categorical contradictions remain definite negatives.
    Carbonation is excluded by the current veto policy.
    """
    from pipeline import three_way_gate

    for dimension, left, right in [
        ("pulp_set", {"no_pulp"}, {"with_pulp"}),
        ("sweetener_set", {"sugar"}, {"no_sugar"}),
    ]:
        result = three_way_gate(
            _attrs(**{dimension: left}), _attrs(**{dimension: right})
        )
        assert result["decision"] == "hard_no", (
            f"{dimension} conflict must stay a hard_no, got {result}"
        )


def test_absent_volume_routes_to_review() -> None:
    """Missing measurements stay unknown even with optimistic confidence."""
    from pipeline import three_way_gate

    result = three_way_gate(_attrs(volume_set=set()), _attrs())
    assert result["decision"] == "fallback"
    assert result["reason"] == training_cfg().gate.reasons.low_volume_confidence
