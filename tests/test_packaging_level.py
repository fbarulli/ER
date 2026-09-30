"""Packaging LEVEL is an identity dimension, independent of pack COUNT.

Owner ruling: a change in pack count, case quantity, or packaging level
mints a brand-new GS1 trade item with its own unique identifier. A 12-unit
retail pack and the same 12 units shipped as a case are therefore NOT the
same trade item, even though both extract to pack_qty == 12.

These tests pin that behaviour at three levels: the title parser, the gate
decision, and the generated artifacts.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from core.common import F
from pipeline import extract_packaging_level, three_way_gate

# (title, expected) — the positives are real case listings from the corpus;
# the negatives are the traps that a loose \bcase\b match would swallow.
CASE_TITLES = [
    "Lucky Jack Latte Vanilla Nitro Coffee - Case of 12 / 7.5 fl oz Cans",
    "fresh ginger ale pomegranate hibiscus - case of 12",
    "L and A Juice - All Cranberry - Case of 6 - 32 Fl oz.",
    "Adirondack - Seltzer Sparkling Water - Case Of 3-8 / 12 Fz",
    "100% Fruit Juice Blue Raspberry Slushee Mix | Case of 4 x 1 Gallons",
    "4 Set - Jupina Pineapple Soda 12 oz. Case of 6 Cans",
]

NON_CASE_TITLES = [
    "(NOT A CASE) Enhanced Sparkling Water Pineapple",
    "(NOT A CASE) Juice Lemon",
    "Thick & Easy Thickened Beverage 4 oz. Portion Cup Iced Tea",
    "Thick & Easy Clear Thickened Iced Tea 4 Fl Oz ( Pack Of 12)",
    "Use filtered water in this case only",
    "Lucky Jack Cold Brew Coffee Nitro Latte Vanilla 7.5 fl oz Each",
    "L A Juice Cranberry Delight 32 Oz ( Pack of 6)",
    "",
    "   ",
]


@pytest.mark.parametrize("title", CASE_TITLES)
def test_case_listing_is_detected(title: str) -> None:
    assert extract_packaging_level(title) == {"case"}


@pytest.mark.parametrize("title", NON_CASE_TITLES)
def test_absent_marker_is_unknown_not_retail(title: str) -> None:
    """A missing marker means NO CLAIM, never an affirmative "retail".

    Encoding "single" here would make every title that simply omits the word
    "case" conflict against a real case listing, splitting genuine
    duplicates. The empty set is the whole point.
    """
    assert extract_packaging_level(title) == set()


def _attrs(**over: object) -> dict:
    base = {
        "volume_set": {222.0},
        "pack_set": {12},
        "package_type_set": {"can"},
        "package_material_set": {"metal"},
        "packaging_level_set": set(),
        "flavor_set": {"vanilla"},
        "carbonation_set": {"still"},
        "sweetener_set": set(),
        "pulp_set": set(),
        "volume_confidence": 0.9,
        "pack_confidence": 0.9,
        "volume_consistency": 0.9,
        "pack_consistency": 0.9,
    }
    base.update(over)
    return base


def test_case_vs_unstated_never_merges() -> None:
    """The core owner ruling: same count, different level, no merge."""
    case = _attrs(packaging_level_set={"case"})
    retail = _attrs(packaging_level_set=set())
    result = three_way_gate(case, retail)
    assert result["decision"] != "proceed"
    assert "Packaging level" in result["reason"]


def test_packaging_level_never_becomes_a_hard_negative() -> None:
    """One-sided evidence goes to review, never to a fabricated hard_no.

    A hard_no here would manufacture a negative out of silence, which is
    worse than leaving a pair undecided.
    """
    result = three_way_gate(_attrs(packaging_level_set={"case"}), _attrs())
    assert result["decision"] == "fallback"


def test_hard_negative_outranks_the_review_flag() -> None:
    """A flavour conflict must beat the weaker one-sided level claim.

    Measured 2026-09-30: an earlier ordering placed the level check BEFORE
    the categorical conflict check and silently downgraded 79 genuine
    flavour conflicts from hard_no to fallback. This pins the fix.
    """
    result = three_way_gate(
        _attrs(packaging_level_set={"case"}, flavor_set={"vanilla"}),
        _attrs(packaging_level_set=set(), flavor_set={"strawberry"}),
    )
    assert result["decision"] == "hard_no"
    assert "flavor" in result["reason"]


def test_same_level_on_both_sides_is_not_blocked() -> None:
    """Two case listings of the same product must still be mergeable."""
    result = three_way_gate(
        _attrs(packaging_level_set={"case"}), _attrs(packaging_level_set={"case"})
    )
    assert result["decision"] == "proceed"


def test_no_proceed_pair_has_a_one_sided_level_claim() -> None:
    """Artifact-level invariant over the real generated gate census."""
    canon = pd.read_csv(F["canonical_records"], dtype=str, keep_default_na=False)

    def parse(raw: str) -> set:
        try:
            return set(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            return set()

    levels = dict(
        zip(canon["gtin"], canon["packaging_level_set"].map(parse), strict=True)
    )
    gates = pd.read_csv(
        F["gate_results"], dtype={"gtin1": str, "gtin2": str}, keep_default_na=False
    )
    left = gates["gtin1"].map(levels).map(bool)
    right = gates["gtin2"].map(levels).map(bool)
    violations = gates[left ^ right]["gate_decision"].eq("proceed").sum()
    assert violations == 0, (
        f"{violations} proceed pairs assert a packaging level on one side only"
    )
