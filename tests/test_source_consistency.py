"""Source contradiction / implausibility flags (review-not-guess)."""

from core.critical_attributes import source_consistency_flags


def test_no_caffeine_claim_beside_declared_caffeine():
    flags = source_consistency_flags(
        "Free From: no caffeine; Caffeine: 50-100 mg", "Joyburst Energy Drink", {"sucralose"}
    )
    assert "caffeine_source_conflict" in flags


def test_no_aspartame_beside_aspartame():
    flags = source_consistency_flags(
        "No Artificial Ingredients: no aspartame; Sweetener: aspartame",
        "Frannie's Sparkling Beverage",
        {"aspartame"},
    )
    assert "no_aspartame_with_aspartame" in flags


def test_no_sugar_claim_beside_cane_sugar():
    flags = source_consistency_flags(
        "Health Claims: no sugar; Sweetener: cane sugar", "Brew Dr Kombucha", {"cane_sugar"}
    )
    assert "no_sugar_with_sugar" in flags


def test_caffeine_on_a_product_with_no_caffeine_source():
    flags = source_consistency_flags(
        "Caffeine: 15-25 mg; Volume: 700", "syrup white peach", {"sugar"}
    )
    assert "caffeine_without_source" in flags


def test_trace_caffeine_band_is_not_pollution():
    # "0-15 mg" includes zero, so it is not a positive declaration.
    flags = source_consistency_flags(
        "Caffeine: 0-15 mg", "spring water", set()
    )
    assert "caffeine_without_source" not in flags


def test_named_caffeine_source_is_clean():
    for title in ("Cold Brew Coffee", "Green Tea", "Cola", "Energy Drink", "Yerba Mate"):
        flags = source_consistency_flags(
            "Caffeine: 50-100 mg", title, set()
        )
        assert "caffeine_without_source" not in flags, title


def test_clean_product_has_no_flags():
    flags = source_consistency_flags(
        "Volume: 330; Flavour: apple; Carbonization: still", "apple juice", {"sugar"}
    )
    assert flags == frozenset()
