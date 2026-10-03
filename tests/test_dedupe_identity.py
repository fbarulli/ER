"""Identity invariants for the tiered dedupe (06).

The regression these lock down is a real, measured data-loss bug: T3 keyed its
collapse on (retailer, title) ALONE, so whenever one retailer listed the same
title string under two different checksum-valid gtins, two distinct products
were merged and one lost its only row. Measured 2026-09-30 on the 71.6k-row
corpus: 692 groups, 1,086 product-listings deleted, 264 products erased from the
matching input entirely (present in canonical_records.csv, absent from the
deduped output, represented by a sibling's gtin).

These tests pin the DECISIONS (what may and may not collapse together), not the
whole pipeline — main() writes real corpus outputs and is exercised by the
stage run, while the predicates below are the part a future edit can silently
regress.
"""

import pandas as pd
import pytest

from core.gtin import gtin_validity
from core.sku_identity import (
    completeness,
    identity_conflict,
    row_identity,
)
from training.dedupe import _same_product_by_title


# Two real, checksum-valid, SAME-RETAILER, SAME-TITLE-STRING gtins from the
# corpus group "Sisi Mango no bubbles zero sugar" (Albert Heijn). Using real
# values keeps the test honest: synthetic 13-digit strings mostly fail the GS1
# checksum and would silently exercise the malformed path instead.
_DISTINCT_BARCODES = ("8715600246377", "8715600248098")


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["retailer", "sku_name_eng", "brand", "attribute", "sku_last_price", "gtin"],
        index=[f"row-{i}" for i in range(len(rows))],
    )


# ── T3 identity partition ───────────────────────────────────────────────────

def test_same_title_different_valid_gtins_get_distinct_identity_keys():
    """The T3 partition must separate two products sharing a title string."""
    df = _frame([
        ("Shop A", "Sisi Mango no bubbles zero sugar", "Sisi",
         "Flavour: Mango", 1.19, _DISTINCT_BARCODES[0]),
        ("Shop A", "Sisi Mango no bubbles zero sugar", "Sisi",
         "Flavour: Mango", 1.29, _DISTINCT_BARCODES[1]),
    ])

    bc = df["gtin"].astype(str)
    assert gtin_validity(bc).all(), "fixture gtins must be checksum-valid"

    keys = {row_identity(r).gtin_key for r in df.to_dict("records")}

    assert keys == set(_DISTINCT_BARCODES), (
        "distinct trusted gtins must produce distinct identity keys, "
        "otherwise T3 collapses two different products"
    )


def test_identity_conflict_proves_the_pair_different_even_when_text_agrees():
    """GTIN is the arbiter when the text cannot tell two rows apart."""
    left, right = (
        row_identity(r) for r in _frame([
            ("Shop A", "Sisi Mango no bubbles zero sugar", "Sisi",
             "Flavour: Mango", 1.19, _DISTINCT_BARCODES[0]),
            ("Shop A", "Sisi Mango no bubbles zero sugar", "Sisi",
             "Flavour: Mango", 1.29, _DISTINCT_BARCODES[1]),
        ]).to_dict("records")
    )

    reasons = identity_conflict(left, right)

    assert "gtin" in reasons, (
        "identical descriptor text must NOT license a merge when the trusted "
        "gtins disagree — the text predicate is a veto, not authority"
    )


def test_gtin_less_rows_still_share_one_identity_partition():
    """Price aggregation must still work where there is no gtin to key on."""
    df = _frame([
        ("Shop A", "Acme Cola 6 pack", "Acme", "", 6.00, ""),
        ("Shop A", "Acme Cola 6 pack", "Acme", "", 7.00, ""),
    ])

    keys = {row_identity(r).gtin_key for r in df.to_dict("records")}

    assert keys == {""}, "rows with no trusted gtin share the '' partition"


def test_invalid_gtin_is_never_a_trusted_identity_key():
    """A checksum-failing gtin is export noise, not identity."""
    left, right = (
        row_identity(r) for r in _frame([
            ("Shop A", "Acme Cola", "Acme", "", 6.00, "123456789013"),
            ("Shop A", "Acme Cola", "Acme", "", 7.00, "123456789013"),
        ]).to_dict("records")
    )

    assert left.gtin_key == "" and right.gtin_key == ""
    assert not left.gtin_trusted and not right.gtin_trusted


# ── T1.5 decision ───────────────────────────────────────────────────────────

def test_malformed_gtin_group_with_conflicting_roast_is_kept_apart():
    """The Cool Brew case: one retailer, one malformed gtin, two roasts."""
    sub = _frame([
        ("amazon", "Cold Brew Concentrate", "Cool Brew",
         "Roast Type: French Roast", 12.0, "53721632036"),
        ("amazon", "Cold Brew Concentrate", "Cool Brew",
         "Roast Type: Vanilla", 19.0, "53721632036"),
    ])

    assert not _same_product_by_title(sub, "amazon", "53721632036"), (
        "a roast split is a genuine product difference and must not collapse"
    )


def test_malformed_gtin_group_with_mixed_diet_claim_is_kept_apart():
    sub = _frame([
        ("amazon", "Stewart's Root Beer", "Stewart's", "", 6.0, "98794313048"),
        ("amazon", "Stewart's Diet Root Beer", "Stewart's", "", 6.0, "98794313048"),
    ])

    assert not _same_product_by_title(sub, "amazon", "98794313048")


def test_malformed_gtin_group_with_a_diet_claim_in_every_row_collapses():
    """A qualifier present in EVERY row is a shared trait, not a split."""
    sub = _frame([
        ("amazon", "Orange Crush Sugar Free", "Orange Crush", "", 6.0, "72392329915"),
        ("amazon", "Orange Crush Sugar Free Singles 6pk", "Orange Crush", "",
         6.0, "72392329915"),
    ])

    assert _same_product_by_title(sub, "amazon", "72392329915"), (
        "diet marker on all rows is a product trait, not a contradiction"
    )


def test_owner_adjudicated_keep_verdict_wins_over_the_predicate():
    """_STEP2_KEEP is authoritative even when the text looks compatible."""
    sub = _frame([
        ("amazon", "Zephyrhills Sparkling Water Lemon", "Zephyrhills", "", 1.0, "1"),
        ("amazon", "Zephyrhills Sparkling Water Lime", "Zephyrhills", "", 1.0, "1"),
    ])

    assert not _same_product_by_title(sub, "amazon", "73430910713")


# ── completeness must ignore price / url / image_url ─────────────────────────

def test_completeness_counts_descriptors_not_price_or_urls():
    """Representative choice must not be won by url/image_url/price noise.

    A listing with two extra URL columns used to outrank a fully-described
    product purely on `df.notna().sum()`.
    """
    bare = {
        "sku_name_eng": "Acme Cola", "brand": "Acme", "category": "Soda",
        "breadcrumbs_eng": "Beverages > Soda", "attribute": "Flavour: Cola",
        "description_short_eng": "A cola.", "sku_last_price": 9.99, "sku_url": "https://x/1",
        "image_url": "https://x/1.jpg",
    }
    described = {
        "sku_name_eng": "Acme Cola", "brand": "Acme", "category": "Soda",
        "breadcrumbs_eng": "Beverages > Soda", "attribute": "Flavour: Cola",
        "description_short_eng": "A cola.", "sku_last_price": 1.99, "sku_url": "",
        "image_url": "",
    }

    assert completeness(bare) == completeness(described), (
        "price/url/image_url must not contribute to the completeness score"
    )
    assert completeness(described) > 0


def test_completeness_ignores_price_and_url_variation_but_counts_descriptors():
    rows = pd.DataFrame([
        {"sku_name_eng": "Acme Cola", "brand": "Acme", "category": "Soda",
         "breadcrumbs_eng": "B > S", "attribute": "Flavour: Cola",
         "description_short_eng": "x", "sku_last_price": 1.0, "sku_url": "", "image_url": ""},
        {"sku_name_eng": "Acme Cola", "brand": "Acme", "category": "Soda",
         "breadcrumbs_eng": "B > S", "attribute": "Flavour: Cola",
         "description_short_eng": "x", "sku_last_price": 999.0, "sku_url": "u", "image_url": "u2"},
    ])

    scores = [completeness(r) for r in rows.to_dict("records")]

    assert scores[0] == scores[1]


# ── absence is not agreement ────────────────────────────────────────────────

def test_missing_descriptor_is_not_a_conflict():
    """A truncated listing must be allowed to collapse with the full one."""
    full, truncated = (
        row_identity(r) for r in _frame([
            ("Shop A", "Acme Sparkling Water Lemon", "Acme",
             "Flavour: Lemon, Carbonated: Yes", 1.0, ""),
            ("Shop A", "Acme Sparkling Water Lemon", "Acme", "", 1.0, ""),
        ]).to_dict("records")
    )

    assert identity_conflict(full, truncated) == []


@pytest.mark.parametrize(
    "left_attr,right_attr,expected",
    [
        ("Flavour: Lemon", "Flavour: Lime", True),
        ("Roast Type: French Roast", "Roast Type: Vanilla", True),
        ("Pack Type: Can", "Pack Type: Bottle", True),
        # `Pack Material Type` is the REAL corpus key (35,571 non-null cells).
        # A regex of `pack\s*material` compiles to `pack\s*material\s*:`, which
        # does not match it, so this dimension was silently dead — the kind of
        # bug that reads as "no conflicts found" rather than "no check ran".
        ("Pack Material Type: Glass", "Pack Material Type: Plastic", True),
        ("Pack Material Type: Glass", "Pack Material Type: Glass", False),
    ],
)
def test_positive_splits_are_detected(left_attr, right_attr, expected):
    left, right = (
        row_identity(r) for r in _frame([
            ("Shop A", "Acme Product", "Acme", left_attr, 1.0, ""),
            ("Shop A", "Acme Product", "Acme", right_attr, 1.0, ""),
        ]).to_dict("records")
    )

    assert bool(identity_conflict(left, right)) is expected


def test_package_material_is_actually_extracted():
    """Guard the dead-dimension bug directly: the value must be non-empty."""
    identity = row_identity({
        "sku_name_eng": "Acme Juice", "brand": "Acme", "category": "Juice",
        "breadcrumbs_eng": "B > J", "attribute": "Pack Material Type: Glass",
        "description_short_eng": "Juice.", "sku_last_price": 3.0, "gtin": "",
        "sku_url": "", "image_url": "",
    })

    assert identity.package_material == frozenset({"glass"}), (
        "package_material extracted nothing — the attribute key regex does not "
        "match the corpus's `Pack Material Type`"
    )
