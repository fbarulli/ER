"""Attribute separation metrics: support honesty and per-value ranking.

Synthetic populations for the score CONTRACT: how the
score is formed, and the rule that an under-supported value is reported with
its counts but never flagged as a defect. The one exception is the coverage
contract, which is proved on the committed REAL population
(``data/labeled_pairs.csv`` + ``data/canonical_records.csv``) where it exists,
and skipped otherwise -- the artifact itself is still never a dependency.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

import training.attribute_separation as separation
from core.schemas import (
    SEPARATION_SUMMARY_COLUMNS,
    SEPARATION_VALUE_COLUMNS,
    AttributeSeparationSpec,
    SeparationSummaryRow,
    SeparationValueRow,
)
from training.attribute_separation import (
    ATTRIBUTE_SOURCES,
    ATTRIBUTE_UNAVAILABLE,
    attribute_coverage_contract,
    attribute_separation,
    separation_population,
    separation_spec,
)

REPO = Path(__file__).resolve().parents[1]

SPEC = AttributeSeparationSpec(
    enabled=True, min_pairs_per_class=3, min_value_support=3, flag_below=0.10
)


def _canonicals(rows: dict[str, dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"gtin": gtin, **values} for gtin, values in rows.items()]
    )
    for column in {c for c, _ in ATTRIBUTE_SOURCES.values()}:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _pairs(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["gtin1", "gtin2", "true_label"])


def test_shipped_config_supplies_every_threshold() -> None:
    spec = separation_spec()
    assert spec.enabled is True
    assert spec.min_pairs_per_class >= 1
    assert spec.min_value_support >= 1
    assert -1.0 <= spec.flag_below <= 1.0


def test_attribute_sources_columns_are_the_record_schema_ssot() -> None:
    """The registry's column side is DERIVED from CanonicalRecord, not retyped.

    Freezing the whole table here is deliberate: it is the byte-neutrality
    proof that deriving the columns from ``core.columns`` moved nothing, and it
    fails the day a rename in ``CanonicalRecord`` silently changes which cell
    the lane reads.
    """
    from core.columns import ATTRIBUTE_DIMENSION_COLUMNS

    assert ATTRIBUTE_SOURCES == {
        "brand": ("mode_brand", "scalar"),
        "volume": ("volume_set", "volume_set"),
        "pack": ("pack_set", "pack_set"),
        "package_type": ("package_type_set", "string_set"),
        "flavor": ("flavor_set", "string_set"),
        "carbonation": ("carbonation_set", "string_set"),
        "sweetener": ("sweetener_set", "string_set"),
        "pulp": ("pulp_set", "string_set"),
    }
    assert {column for column, _ in ATTRIBUTE_SOURCES.values()} == {
        ATTRIBUTE_DIMENSION_COLUMNS[attribute] for attribute in ATTRIBUTE_SOURCES
    }


def test_a_perfect_attribute_scores_one_and_a_useless_one_scores_zero() -> None:
    """separation = P(agree | positive) - P(agree | negative)."""
    canon = _canonicals({
        # volume separates perfectly: positives share it, negatives never do.
        "p1": {"volume_set": "[500]", "flavor_set": "['peach']"},
        "p2": {"volume_set": "[500]", "flavor_set": "['peach']"},
        "n1": {"volume_set": "[330]", "flavor_set": "['peach']"},
        "n2": {"volume_set": "[355]", "flavor_set": "['peach']"},
    })
    pairs = _pairs([
        ("p1", "p2", 1), ("p2", "p1", 1), ("p1", "p2", 1),
        ("n1", "n2", 0), ("n2", "n1", 0), ("n1", "n2", 0),
    ])
    summary, _ = attribute_separation(
        separation_population(pairs, canon), spec=SPEC
    )
    by_attribute = summary.set_index("attribute")
    assert by_attribute.loc["volume", "separation"] == pytest.approx(1.0)
    # flavor agrees everywhere, so it cannot separate at all
    assert by_attribute.loc["flavor", "separation"] == pytest.approx(0.0)
    assert bool(by_attribute.loc["flavor", "negative_class_saturated"]) is True


def test_an_under_supported_value_is_reported_but_never_flagged() -> None:
    """A brand seen twice must not be reported as a defect."""
    canon = _canonicals({
        "p1": {"mode_brand": "Rare"}, "p2": {"mode_brand": "Rare"},
        "n1": {"mode_brand": "Other"}, "n2": {"mode_brand": "Other"},
        "p3": {"mode_brand": "Common"}, "p4": {"mode_brand": "Common"},
        "n3": {"mode_brand": "Common"}, "n4": {"mode_brand": "Common"},
    })
    pairs = _pairs([
        ("p1", "p2", 1), ("n1", "n2", 0),
        ("p3", "p4", 1), ("n3", "n4", 0),
    ])
    _, by_value = attribute_separation(
        separation_population(pairs, canon), spec=SPEC
    )
    rare = by_value[by_value["value"].eq("rare")]
    assert not rare.empty
    assert int(rare["n_positive"].iloc[0]) == 1
    assert bool(rare["reportable"].iloc[0]) is False
    assert bool(rare["flagged_weak"].iloc[0]) is False


def test_support_thresholds_come_from_config_not_literals() -> None:
    """Raising the floor withdraws rows from flagging, and nothing else."""
    canon = _canonicals({
        "a1": {"mode_brand": "X"}, "a2": {"mode_brand": "Y"},
        "b1": {"mode_brand": "X"}, "b2": {"mode_brand": "X"},
    })
    pairs = _pairs([("a1", "a2", 1), ("b1", "b2", 0)])
    population = separation_population(pairs, canon)
    low, _ = attribute_separation(
        population,
        spec=AttributeSeparationSpec(
            enabled=True, min_pairs_per_class=1, min_value_support=1, flag_below=0.10
        ),
    )
    high, _ = attribute_separation(
        population,
        spec=AttributeSeparationSpec(
            enabled=True, min_pairs_per_class=9, min_value_support=9, flag_below=0.10
        ),
    )
    assert bool(low.set_index("attribute").loc["brand", "reportable"]) is True
    assert bool(high.set_index("attribute").loc["brand", "reportable"]) is False
    # the scores themselves are identical; only the support verdict moved
    assert low.set_index("attribute").loc["brand", "separation"] == pytest.approx(
        high.set_index("attribute").loc["brand", "separation"]
    )


def test_population_join_refuses_to_silently_shrink() -> None:
    """A pair whose GTIN has no canonical record must fail, not be dropped."""
    canon = _canonicals({"p1": {"mode_brand": "X"}})
    pairs = _pairs([("p1", "missing-gtin", 1)])
    with pytest.raises(ValueError, match="no canonical record"):
        separation_population(pairs, canon)


def test_schemas_reject_a_flagged_row_without_support() -> None:
    """The flagging rule is enforced by the model, not by convention."""
    with pytest.raises(ValidationError, match="under-supported"):
        SeparationValueRow(
            attribute="brand", value="x", n_positive=1, n_negative=0,
            p_match_positive=0.0, p_match_negative=0.0, separation=0.0,
            reportable=False, flagged_weak=True,
        )
    with pytest.raises(ValidationError, match="under-supported"):
        SeparationSummaryRow(
            attribute="brand", n_positive=1, n_negative=0, n_unobservable=0,
            p_agree_positive=0.0, p_agree_negative=0.0, separation=0.0,
            reportable=False, flagged_weak=True, negative_class_saturated=False,
        )


def test_a_saturated_negative_class_cannot_carry_positive_separation() -> None:
    with pytest.raises(ValidationError, match="impossible"):
        SeparationSummaryRow(
            attribute="brand", n_positive=10, n_negative=10, n_unobservable=0,
            p_agree_positive=1.0, p_agree_negative=1.0, separation=0.5,
            reportable=True, flagged_weak=False, negative_class_saturated=True,
        )


def test_output_frames_match_their_declared_column_contracts() -> None:
    canon = _canonicals({"p1": {"mode_brand": "X"}, "n1": {"mode_brand": "Y"}})
    summary, by_value = attribute_separation(
        separation_population(_pairs([("p1", "n1", 1)]), canon), spec=SPEC
    )
    assert tuple(summary.columns) == SEPARATION_SUMMARY_COLUMNS
    assert tuple(by_value.columns) == SEPARATION_VALUE_COLUMNS
    assert set(summary["attribute"]) == set(ATTRIBUTE_SOURCES)


def test_category_is_declared_unavailable_rather_than_silently_omitted() -> None:
    """The user asked for category 'if available'; it must say why it is not."""
    assert "category" not in ATTRIBUTE_SOURCES
    assert "category" in ATTRIBUTE_UNAVAILABLE
    assert "no category column" in ATTRIBUTE_UNAVAILABLE["category"]


# ── the attribute axis as the GENERAL coverage contract (core.coverage_contracts) ──
def _real_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The committed real population, read exactly as the report producer reads it."""
    labeled_path = REPO / "data" / "labeled_pairs.csv"
    canonical_path = REPO / "data" / "canonical_records.csv"
    if not labeled_path.is_file() or not canonical_path.is_file():
        pytest.skip("real labeled_pairs/canonical_records are not built here")
    return (
        pd.read_csv(labeled_path, dtype={"gtin1": str, "gtin2": str}),
        pd.read_csv(canonical_path, dtype=str, keep_default_na=False),
    )


def _synthetic_population(observable: dict[str, list[bool]]) -> pd.DataFrame:
    """A separation-population frame with a controlled observability pattern.

    ``observable[attribute][row]`` says whether that pair carries a value for the
    attribute on at least one side, in the frame's own cell convention
    (``frozenset`` of normalized values).
    """
    rows = len(next(iter(observable.values())))
    return pd.DataFrame({
        **{f"{attribute}__{side}": [
            frozenset({f"{attribute}-value"} if observable[attribute][row] else ())
            for row in range(rows)]
            for attribute in ATTRIBUTE_SOURCES for side in (1, 2)},
        "true_label": [row % 2 for row in range(rows)],
    })


def _summaries_from(population: pd.DataFrame,
                    observable: dict[str, list[bool]]) -> list[SeparationSummaryRow]:
    """One closing summary per registry attribute, computed FROM the frame."""
    rows = len(population)
    labels = list(population["true_label"])
    summaries = []
    for attribute in ATTRIBUTE_SOURCES:
        carries = observable[attribute]
        n_positive = sum(1 for row in range(rows) if carries[row] and labels[row] == 1)
        n_negative = sum(1 for row in range(rows) if carries[row] and labels[row] == 0)
        summaries.append(SeparationSummaryRow(
            attribute=attribute, n_positive=n_positive, n_negative=n_negative,
            n_unobservable=rows - n_positive - n_negative, p_agree_positive=0.5,
            p_agree_negative=0.5, separation=0.0, reportable=True,
            flagged_weak=False, negative_class_saturated=False))
    return summaries


def test_the_real_attribute_population_passes_the_contract_and_moves_no_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labeled, canonicals = _real_inputs()
    population = separation_population(labeled, canonicals)
    summary, by_value = attribute_separation(population, spec=separation_spec())
    summaries = [SeparationSummaryRow.model_validate(row)
                 for row in summary.to_dict("records")]

    contract = attribute_coverage_contract(population, summaries)
    assert contract.records_total == len(population)
    assert contract.dimensions["attribute"].policy == "overlap"
    # the census is the ROWS' own observable support (the fabricated-records
    # version claimed len(population) for every attribute, whatever the data)
    derived = contract.derived_counts()["attribute"]
    for row in summaries:
        assert derived.get(row.attribute, 0) == row.n_positive + row.n_negative
    assert set(derived) - {"unknown"} <= set(ATTRIBUTE_SOURCES)

    adopted = (summary.to_csv(index=False), by_value.to_csv(index=False))
    monkeypatch.setattr(separation, "attribute_coverage_contract", lambda *_, **__: None)
    summary_off, by_value_off = attribute_separation(population, spec=separation_spec())
    assert adopted == (summary_off.to_csv(index=False), by_value_off.to_csv(index=False))


def test_the_census_follows_the_rows_not_a_fabricated_record_list() -> None:
    """Falsified 2026-10-08: the records were fabricated from the census they
    validated, so the derivation yielded the same census for all-observable and
    none-observable pairs. It now follows the REAL rows."""
    none_observable = {attribute: [False] * 4 for attribute in ATTRIBUTE_SOURCES}
    empty = _synthetic_population(none_observable)
    from_none = attribute_coverage_contract(
        empty, _summaries_from(empty, none_observable))
    assert from_none.derived_counts()["attribute"] == {"unknown": 4}

    all_observable = {attribute: [True] * 4 for attribute in ATTRIBUTE_SOURCES}
    full = _synthetic_population(all_observable)
    from_all = attribute_coverage_contract(
        full, _summaries_from(full, all_observable))
    assert from_all.derived_counts()["attribute"] == {
        attribute: 4 for attribute in ATTRIBUTE_SOURCES}

    # a summary that mis-states an attribute's observable support is rejected:
    # the rows, not the summary, decide the census
    first, second = list(ATTRIBUTE_SOURCES)[:2]
    split_axis = {attribute: ([attribute == first, attribute == first,
                               attribute == second, attribute == second])
                  for attribute in ATTRIBUTE_SOURCES}
    partial = _synthetic_population(split_axis)
    assert attribute_coverage_contract(
        partial, _summaries_from(partial, split_axis)
    ).derived_counts()["attribute"] == {first: 2, second: 2}
    claimed = _summaries_from(partial, split_axis)
    claimed[0] = claimed[0].model_copy(update={
        "n_positive": 2, "n_negative": 1, "n_unobservable": 1})
    with pytest.raises(ValidationError, match="declared counts disagree"):
        attribute_coverage_contract(partial, claimed)


def test_the_attribute_contract_rejects_unclosed_unregistered_and_empty_rows() -> None:
    # every pair is observable for exactly one half of the registry, so no pair
    # is unobservable for everything (the census has no unknown bucket here)
    attributes = list(ATTRIBUTE_SOURCES)
    observable = {attribute: [attribute == attributes[0]] * 2 + [attribute != attributes[0]] * 2
                  for attribute in attributes}
    population = _synthetic_population(observable)
    accepted = attribute_coverage_contract(
        population, _summaries_from(population, observable))
    assert accepted.records_total == 4
    assert accepted.derived_counts()["attribute"] == {
        attribute: 2 for attribute in attributes}

    # a summary that does not close over the population is mis-summed
    unclosed = _summaries_from(population, observable)
    unclosed[-1] = unclosed[-1].model_copy(update={"n_unobservable": 5})
    with pytest.raises(ValueError, match="does not close over the population"):
        attribute_coverage_contract(population, unclosed)

    # an attribute the registry declares but no summary accounts for
    with pytest.raises(ValueError, match="attribute census coverage mismatch"):
        attribute_coverage_contract(
            population, _summaries_from(population, observable)[:-1])
    # and a summary for an attribute the registry does not declare
    unregistered = _summaries_from(population, observable)
    unregistered[-1] = unregistered[-1].model_copy(update={"attribute": "category"})
    with pytest.raises(ValueError, match="attribute census coverage mismatch"):
        attribute_coverage_contract(population, unregistered)

    # zero pair rows cannot claim coverage at all
    with pytest.raises(ValidationError):
        attribute_coverage_contract(
            _synthetic_population({attribute: [] for attribute in ATTRIBUTE_SOURCES}),
            _summaries_from(_synthetic_population({attribute: []
                                                  for attribute in ATTRIBUTE_SOURCES}),
                            {attribute: [] for attribute in ATTRIBUTE_SOURCES}))

    # the records must be pairs of the REAL frame: a frame without the
    # separation columns cannot be censused
    with pytest.raises(ValueError, match="needs the separation population columns"):
        attribute_coverage_contract(
            pd.DataFrame({"true_label": [0, 1]}),
            _summaries_from(population, observable))
