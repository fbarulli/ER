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


@pytest.mark.parametrize("value,total", [(float("nan"), None), (float("inf"), 10), (-1, 10), (11, 10), (1.5, None)])
def test_not_degenerate_rejects_invalid_measurements(value, total):
    with pytest.raises(AuditGuardError):
        assert_not_degenerate("coverage", value, total=total)


def test_absent_dimension_is_unmeasured_not_a_vocabulary_failure():
    result = guard_dimension(**_spec(values=set(), populated=0))
    assert result.passed and result.unmeasured


@pytest.mark.parametrize("populated,total", [(-1, 100), (101, 100), (0, 0)])
def test_invalid_dimension_coverage_is_hard_failure(populated, total):
    result = guard_dimension(**_spec(populated=populated, total=total))
    assert not result.passed and not result.unmeasured


def test_parse_gap_census_includes_rare_rows_after_control_sample():
    from core.audit_guard import self_comparison_parse_gaps
    evidence = ["equal"] * 200 + ["unparsed"]
    assert self_comparison_parse_gaps(
        ["caffeine"], evaluate_status=lambda a, b, name: a,
        evidence_samples=evidence,
    ) == {"caffeine": 1}


def test_dimension_control_sample_size_can_be_configured():
    from core.audit_guard import attribute_dimension_guard_specs
    specs = attribute_dimension_guard_specs(
        ["caffeine"], values={"caffeine": {"200mg"}}, source_texts=["200mg"],
        populated={"caffeine": 1}, total=2, evaluate_status=lambda a, b, name: a,
        evidence_samples=["equal", "different"], sample_size=2,
    )
    with pytest.raises(AuditGuardError, match="self-comparison"):
        guard_dimensions(specs)


def test_reading_sheet_keeps_empty_attribute_keys_as_missed_evidence():
    from scripts.audit_attribute_readings import _key_present
    assert _key_present("Caffeine: ; Flavor: Lemon", "caffeine")


def test_guard_result_rejects_contradictory_outcome():
    from pydantic import ValidationError
    from core.audit_guard import DimensionGuardResult
    with pytest.raises(ValidationError, match="hard failure"):
        DimensionGuardResult(name="caffeine", passed=False, unmeasured=True)


@pytest.mark.parametrize("script", ["audit_attribute_readings", "audit_identity_dimensions"])
def test_sparse_catalog_cli_publishes_missing_dimension_evidence(tmp_path, monkeypatch, script):
    import importlib
    import json
    import sys
    import pandas as pd
    module = importlib.import_module(f"scripts.{script}")
    dataset = tmp_path / "catalog.csv"
    pd.DataFrame([
        {"sku_id": "1", "gtin": "", "retailer": "shop", "sku_name_eng": "Water", "attribute": "Caffeine: "},
        {"sku_id": "2", "gtin": "", "retailer": "shop", "sku_name_eng": "Water", "attribute": ""},
    ]).to_csv(dataset, index=False)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [script, "--dataset", str(dataset), "--output-dir", str(output)])
    module.main()
    if script == "audit_attribute_readings":
        sheet = pd.read_csv(output / "reading_sheet.csv", keep_default_na=False)
        assert ((sheet.dimension == "Caffeine") & (sheet.state == "key_present_not_extracted")).any()
        report = json.loads((output / "reading_manifest.json").read_text())
        assert "Caffeine" in report["unmeasured"]
    else:
        report = json.loads((output / "identity_dimensions.json").read_text())
        assert "Caffeine" in report["unmeasured_dimensions"]
        assert all(row["same_gtin_difference_rate"] is None for row in report["dimension_evaluation"])
