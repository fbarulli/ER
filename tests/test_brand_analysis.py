"""Brand-string analysis: TF-IDF + fuzzy classification, with support honesty.

Synthetic comparisons for the contract, plus the real catalog clusters so the
one place surface variants actually exist stays covered.
"""
from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from core.schemas import BRAND_PAIR_COLUMNS, BrandAnalysisSpec, BrandPairRow
from training.brand_analysis import (
    CLASS_DIFFERENT,
    CLASS_EXACT,
    CLASS_MISSING,
    analyse_brands,
    brand_features,
    brand_spec,
    catalog_brand_variants,
    classify_pair,
)

SPEC = BrandAnalysisSpec(
    enabled=True,
    tfidf_analyzer="char_wb",
    tfidf_ngram_min=2,
    tfidf_ngram_max=3,
    surface_variant_min_ratio=0.85,
    different_brand_max_ratio=0.60,
    indeterminate_class="ambiguous_similarity",
    min_pair_support=20,
    corporate_suffixes=("gmbh", "ltd", "group"),
)


def _comparisons(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "brand_left": [left for left, _ in rows],
            "brand_right": [right for _, right in rows],
            "title": ["" for _ in rows],
            "support_positive": [100 for _ in rows],
            "support_negative": [100 for _ in rows],
        }
    )


def test_shipped_config_supplies_every_threshold() -> None:
    spec = brand_spec()
    assert spec.enabled is True
    assert spec.tfidf_ngram_min <= spec.tfidf_ngram_max
    assert spec.different_brand_max_ratio < spec.surface_variant_min_ratio
    assert spec.min_pair_support >= 1
    assert spec.corporate_suffixes


def test_inverted_thresholds_are_rejected_at_config_load() -> None:
    with pytest.raises(ValidationError, match="must be below"):
        BrandAnalysisSpec(
            enabled=True, tfidf_analyzer="char_wb", tfidf_ngram_min=2,
            tfidf_ngram_max=3, surface_variant_min_ratio=0.5,
            different_brand_max_ratio=0.9, indeterminate_class="x",
            min_pair_support=1, corporate_suffixes=("ltd",),
        )
    with pytest.raises(ValidationError, match="must not exceed"):
        BrandAnalysisSpec(
            enabled=True, tfidf_analyzer="char_wb", tfidf_ngram_min=4,
            tfidf_ngram_max=2, surface_variant_min_ratio=0.9,
            different_brand_max_ratio=0.5, indeterminate_class="x",
            min_pair_support=1, corporate_suffixes=("ltd",),
        )


def test_features_are_one_for_an_identical_brand() -> None:
    features = brand_features("Powerking", "Powerking", spec=SPEC)
    assert features["ratio"] == pytest.approx(1.0)
    assert features["edit_distance"] == 0
    assert features["normalized_equal"] == 1.0


def test_case_accent_punctuation_and_suffix_variants_collapse() -> None:
    """Every surface form of one brand must reach the same normal form."""
    base = "Coteaux Nantais"
    for variant in (
        "Côteaux Nantais",   # accents
        "COTEAUX NANTAIS",   # case
        "Coteaux  Nantais",  # spacing
        "Coteaux-Nantais",   # punctuation
    ):
        assert brand_features(base, variant, spec=SPEC)["normalized_equal"] == 1.0, variant
    # a corporate suffix is a surface difference, not a different brand
    assert brand_features("Quelle", "Quelle GmbH", spec=SPEC)["normalized_equal"] == 1.0


def test_classification_separates_variants_from_different_brands() -> None:
    variant = brand_features("Brämhults", "Bramhults", spec=SPEC)
    assert classify_pair("Brämhults", "Bramhults", variant, spec=SPEC) == CLASS_EXACT
    different = brand_features("Powerking", "Coca Cola", spec=SPEC)
    assert classify_pair("Powerking", "Coca Cola", different, spec=SPEC) == CLASS_DIFFERENT
    missing = brand_features("", "Powerking", spec=SPEC)
    assert classify_pair("", "Powerking", missing, spec=SPEC) == CLASS_MISSING


def test_the_indeterminate_band_is_not_forced_into_a_verdict() -> None:
    """Between the two thresholds the evidence does not decide."""
    left, right = "Cemilefendi", "Cemil"
    features = brand_features(left, right, spec=SPEC)
    assert SPEC.different_brand_max_ratio < features["ratio"] < SPEC.surface_variant_min_ratio
    assert classify_pair(left, right, features, spec=SPEC) == SPEC.indeterminate_class


def test_a_low_support_brand_is_not_reportable() -> None:
    comparisons = _comparisons([("Radnor", "Rainbow")])
    comparisons["support_positive"] = 2
    comparisons["support_negative"] = 2
    frame = analyse_brands(comparisons, spec=SPEC)
    assert bool(frame["reportable"].iloc[0]) is False


def test_output_frame_matches_its_declared_column_contract() -> None:
    frame = analyse_brands(_comparisons([("A", "B")]), spec=SPEC)
    assert tuple(frame.columns) == BRAND_PAIR_COLUMNS
    with pytest.raises(ValidationError):
        BrandPairRow(
            brand_left="a", brand_right="b", classification="x",
            similarity_ratio=2.0, token_sort_ratio=0.0, token_set_ratio=0.0,
            edit_distance=0, tfidf_cosine=0.0, n_positive=0, n_negative=0,
            reportable=False,
        )


def test_catalog_variants_finds_the_real_clusters() -> None:
    """The catalog is the one place surface variants actually exist."""
    catalog = pd.DataFrame(
        {
            "mode_brand": ["REAL", "Reál", "Réal", "Viva", "Viva!", "Solo", "Solo"],
        }
    )
    variants = catalog_brand_variants(catalog, spec=SPEC)
    assert set(variants["normalized_brand"]) == {"real", "viva"}
    real = variants[variants["normalized_brand"].eq("real")].iloc[0]
    assert int(real["n_variants"]) == 3
    assert int(real["n_gtins"]) == 3
    # "Solo"/"Solo" is ONE raw variant, so it is not a variant cluster
    assert "solo" not in set(variants["normalized_brand"])
