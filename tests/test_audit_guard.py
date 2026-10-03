"""Fail-closed audit-guard contracts: each guard must abort on its artifact."""

import pytest

from core.audit_guard import (
    AuditGuardError,
    assert_metrics_not_degenerate,
    assert_not_degenerate,
    assert_vocabulary_overlap,
    guard_dimension,
    guard_dimensions,
    self_comparison_control,
)


def test_vocabulary_overlap_returns_matched_terms() -> None:
    matched = assert_vocabulary_overlap(
        {"lemon", "lime", "grape"}, ["lemon soda", "lime water"], label="flavor"
    )
    assert matched == {"lemon", "lime"}


def test_vocabulary_overlap_fails_on_zero_overlap() -> None:
    with pytest.raises(AuditGuardError, match="does not overlap"):
        assert_vocabulary_overlap(
            {"lemon", "lime"}, ["orange soda"], label="flavor"
        )


def test_vocabulary_overlap_fails_on_empty_vocabulary() -> None:
    with pytest.raises(AuditGuardError, match="vocabulary is empty"):
        assert_vocabulary_overlap(set(), ["lemon soda"], label="flavor")


def test_vocabulary_overlap_fails_on_empty_corpus() -> None:
    with pytest.raises(AuditGuardError, match="corpus is empty"):
        assert_vocabulary_overlap({"lemon"}, [], label="flavor")


def test_vocabulary_overlap_min_terms_is_enforced() -> None:
    with pytest.raises(AuditGuardError, match="does not overlap"):
        assert_vocabulary_overlap(
            {"lemon", "lime"}, ["lemon soda"], label="flavor", min_terms=2
        )


def test_self_comparison_control_passes_on_identity() -> None:
    checked = self_comparison_control(
        lambda a, b: a == b, ["a", "b", "c"], label="extract"
    )
    assert checked == 3


def test_self_comparison_control_fails_when_comparator_never_matches() -> None:
    with pytest.raises(AuditGuardError, match="self-comparison control failed"):
        self_comparison_control(
            lambda a, b: False, ["a", "b"], label="extract"
        )


def test_self_comparison_control_fails_on_no_samples() -> None:
    with pytest.raises(AuditGuardError, match="no samples"):
        self_comparison_control(lambda a, b: True, [], label="extract")


def test_self_comparison_control_normalizes_result() -> None:
    checked = self_comparison_control(
        lambda a, b: "EQUAL",
        ["x"],
        label="extract",
        expected="equal",
        normalize=str.lower,
    )
    assert checked == 1


def test_not_degenerate_rejects_fraction_extremes() -> None:
    with pytest.raises(AuditGuardError, match="degenerate"):
        assert_not_degenerate("rate", 0.0, label="audit")
    with pytest.raises(AuditGuardError, match="degenerate"):
        assert_not_degenerate("rate", 1.0, label="audit")


def test_not_degenerate_accepts_interior_fraction() -> None:
    assert assert_not_degenerate("rate", 0.5, label="audit") == 0.5


def test_not_degenerate_rejects_count_extremes() -> None:
    with pytest.raises(AuditGuardError, match="degenerate"):
        assert_not_degenerate("hits", 0, total=10, label="audit")
    with pytest.raises(AuditGuardError, match="degenerate"):
        assert_not_degenerate("hits", 10, total=10, label="audit")


def test_not_degenerate_returns_fraction_for_count() -> None:
    assert assert_not_degenerate("hits", 3, total=12, label="audit") == 0.25


def test_not_degenerate_rejects_nonpositive_total() -> None:
    with pytest.raises(AuditGuardError, match="total must be positive"):
        assert_not_degenerate("hits", 0, total=0, label="audit")


def test_metrics_not_degenerate_checks_every_entry() -> None:
    assert_metrics_not_degenerate({"a": 0.5, "b": (3, 10)}, label="audit")
    with pytest.raises(AuditGuardError, match="degenerate"):
        assert_metrics_not_degenerate({"a": 0.5, "b": (10, 10)}, label="audit")


def _spec(**overrides):
    base = dict(
        name="flavour",
        values={"lemon", "lime"},
        source_texts=["Flavour: lemon", "Flavour: lime"],
        self_compare=lambda a, b: a == b,
        self_samples=["x", "y"],
        populated=10,
        total=100,
    )
    base.update(overrides)
    return base


def test_guard_dimension_passes() -> None:
    result = guard_dimension(**_spec())
    assert result.passed and not result.unmeasured


def test_guard_dimension_flags_zero_overlap_as_hard_failure() -> None:
    result = guard_dimension(**_spec(source_texts=["Flavour: orange"]))
    assert not result.passed and not result.unmeasured
    assert "does not overlap" in result.detail


def test_guard_dimension_flags_broken_self_comparison() -> None:
    result = guard_dimension(**_spec(self_compare=lambda a, b: False))
    assert not result.passed
    assert "self-comparison" in result.detail


def test_guard_dimension_marks_degenerate_coverage_unmeasured() -> None:
    result = guard_dimension(**_spec(populated=0, total=100))
    assert result.passed and result.unmeasured
    assert "degenerate" in result.detail


def test_guard_dimensions_raises_on_any_hard_failure() -> None:
    with pytest.raises(AuditGuardError, match="dimension guard"):
        guard_dimensions([_spec(), _spec(name="sweetener", source_texts=["Sweetener: stevia"])], label="audit")


def test_guard_dimensions_returns_unmeasured_without_raising() -> None:
    results = guard_dimensions(
        [_spec(), _spec(name="giftbox", populated=100, total=100)], label="audit"
    )
    assert len(results) == 2
    assert [r.unmeasured for r in results] == [False, True]
