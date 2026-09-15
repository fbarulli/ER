"""Attribute separation metrics: support honesty and per-value ranking.

Synthetic populations only — the real one lives in a regenerated artifact and
must not be a test dependency.  What is pinned here is the CONTRACT: how the
score is formed, and the rule that an under-supported value is reported with
its counts but never flagged as a defect.
"""
from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

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
    attribute_separation,
    separation_population,
    separation_spec,
)

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
