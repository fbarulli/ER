"""The compact comparison key closes the fused-vs-spaced normalisation gap.

``normalized_attribute_text`` folds accents but keeps the word boundary; the
brand veto's own normaliser drops the boundary but keeps the accent.  Measured
on the 388 non-exact brand rows, five pairs are the *same* brand written with
and without a boundary and are reachable only by the union of both behaviours.
"""

from __future__ import annotations

import pytest

from core.critical_attributes import (
    compact_attribute_text,
    normalized_attribute_text,
)

# The five real cases from the 388 non-exact brand rows, each verified against
# dataset.csv.  Every one is one brand spelled two ways.
FUSED_VS_SPACED = [
    ("PureThé", "PURE THE"),
    ("Bio Food", "biofood"),
    ("Bolt 24", "Bolt24"),
    ("Folkington's", "Folkingtons"),
    ("A SHOC", "Ashoc"),
]


@pytest.mark.parametrize(("left", "right"), FUSED_VS_SPACED)
def test_compact_matches_a_fused_and_spaced_spelling_of_one_brand(left, right):
    assert compact_attribute_text(left) == compact_attribute_text(right)


@pytest.mark.parametrize(("left", "right"), FUSED_VS_SPACED)
def test_the_existing_normaliser_alone_would_miss_them(left, right):
    # Guards the reason this function exists: if someone "simplifies" the
    # compact key back onto normalized_attribute_text, this fails first.
    assert normalized_attribute_text(left) != normalized_attribute_text(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [("Côteaux", "Coteaux"), ("Rotbäckchen", "ROTBACKCHEN"), ("Mezzo Mix", "MezzoMix")],
)
def test_compact_also_keeps_the_accent_folding(left, right):
    # NFKC-then-alnum (the brand veto's normaliser) fails every one of these,
    # because 'ô'/'ä' survive isalnum().
    assert compact_attribute_text(left) == compact_attribute_text(right)


def test_compact_does_not_conflate_distinct_brands():
    assert compact_attribute_text("Piacelli") != compact_attribute_text("Premier")
    assert compact_attribute_text("Quellbrunn") != compact_attribute_text("Rheinfels Quelle")
    assert compact_attribute_text("") == ""
