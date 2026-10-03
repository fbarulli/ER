"""Guards against missed dimensions, false authority, and compatibility chaining."""
import pandas as pd
import pytest

from core.product_dimensions import evaluate_rows, row_dimensions, evaluate_dimensions
from core.sku_identity import row_identity, evaluate_sku_identity
from training.dedupe import _same_product_by_title


def row(attributes="", **overrides):
    return {"sku_name_eng": "Example drink", "brand": "Example", "attribute": attributes,
            "gtin": "", **overrides}


@pytest.mark.parametrize("key,a,b", [
    ("Tea Type", "green", "black"), ("Coffee Type", "arabica", "java"),
    ("Water Type", "spring", "mineral"), ("Juice Content", "100%", "0-2%"),
    ("Caffeine", "0-15 mg", "50-100 mg"), ("Count per Unit", "6", "12"),
])
def test_previously_unchecked_dimensions_reach_identity_review(key, a, b):
    result = evaluate_rows(row(f"{key}: {a}"), row(f"{key}: {b}"))
    assert key in result["review_dimensions"]
    # Count per Unit can also reach the established pack-conflict extractor.
    assert result["decision"] in {"review", "different"}


def test_absence_and_matching_attributes_never_manufacture_identity():
    result = evaluate_rows(row("Tea Type: green"), row())
    assert result["attribute"]["Tea Type"]["status"] == "unknown"
    assert result["decision"] == "compatible_unverified"
    assert evaluate_rows(row("Tea Type: green"), row("Tea Type: green"))["decision"] == "compatible_unverified"


def test_feed_differences_do_not_override_trusted_identity():
    result = evaluate_rows(row("Tea Type: green", gtin="8715600246377"),
                           row("Tea Type: black", gtin="8715600246377"))
    assert result["decision"] == "same"
    assert result["review_dimensions"] == ["Tea Type"]


def test_unknown_key_and_unparsed_numeric_stay_visible():
    result = evaluate_rows(row("New Field: x; Volume: nonsense"), row("New Field: x; Volume: nonsense"))
    assert result["unclassified_keys"] == ["new field"]
    assert result["attribute"]["Volume"]["status"] == "unparsed"


def test_offer_metadata_does_not_become_product_conflict():
    result = evaluate_rows(row(sku_last_price="1.0", sku_url="shop-a"), row(sku_last_price="9.0", sku_url="shop-b"))
    assert result["columns"]["sku_last_price"]["role"] == "offer_context"
    assert result["columns"]["sku_url"]["status"] == "different"
    assert result["decision"] == "compatible_unverified"


def test_negation_and_compound_values_are_not_lost():
    a = row_dimensions(row("Health Claims: no added sugar; Pack Material Type: Paper / Carton"))
    b = row_dimensions(row("Health Claims: added sugar; Pack Material Type: Paper / Carton"))
    assert evaluate_dimensions(a, b)["Health Claims"]["status"] == "different"
    assert a.attributes["Pack Material Type"] == frozenset({"paper / carton"})


def test_full_evidence_is_attached_to_the_existing_identity_object():
    a, b = row_identity(row("Tea Type: green")), row_identity(row("Tea Type: black"))
    assert evaluate_sku_identity(a, b)["decision"] == "review"


def test_malformed_gtin_group_cannot_chain_through_broad_anchor():
    frame = pd.DataFrame([row("Tea Type: green, black"), row("Tea Type: green"), row("Tea Type: black")])
    assert not _same_product_by_title(frame, "Shop", "malformed-not-adjudicated")


def test_dedupe_consults_extra_dimensions_before_collapse():
    frame = pd.DataFrame([row("Caffeine: 0-15 mg"), row("Caffeine: 50-100 mg")])
    assert not _same_product_by_title(frame, "Shop", "malformed-not-adjudicated")
