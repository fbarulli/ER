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


def test_source_review_sheet_supports_verdict_merger_and_counts_gtins(tmp_path, monkeypatch):
    import csv
    import json
    import sys
    import pandas as pd
    from scripts import review_source_consistency as script
    frame = pd.DataFrame([{
        "gtin": "00123", "canonical": "soda", "attribute": "",
        "attribute_consistency_flags": {"no_sugar_with_sugar", "no_aspartame_with_aspartame"},
    }])
    monkeypatch.setattr(script, "canonical_records_frame", lambda: frame)
    monkeypatch.setattr(sys, "argv", ["review", "--output-dir", str(tmp_path)])
    script.main()
    report = json.loads((tmp_path / "review_manifest.json").read_text())
    assert report["flagged_gtins"] == 1
    assert report["sheet_rows"] == 2
    with (tmp_path / "source_defect_sheet.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["dimension"] == row["flag"] and row["sku_id"] == "00123" for row in rows)
