import pandas as pd

from core.record_linkage import link_barcode_less, strip_pack_multiplicity


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["product_id", "title", "brand", "barcode", "retailer"],
        index=[f"row-{i}" for i in range(len(rows))],
    )


def test_links_pack_variants_across_retailers_but_keeps_flavors_separate():
    df = _frame([
        ("a", "Acme Mocha 6 pack", "Acme", "", "Shop A"),
        ("b", "Acme Mocha pack of 12", "Acme", "bad-checksum", "Shop B"),
        ("c", "Acme Latte 6 pack", "Acme", "", "Shop C"),
    ])

    clusters, census = link_barcode_less(df)

    assert clusters["row-0"] == clusters["row-1"]
    assert clusters["row-2"] != clusters["row-0"]
    assert census["barcode_less_rows"] == 3
    assert census["exact_title_pairs_linked"] == 1


def test_only_valid_gtins_are_excluded_and_same_retailer_does_not_link():
    df = _frame([
        ("valid", "Acme Mocha 6 pack", "Acme", "036000291452", "Shop A"),
        ("bad-a", "Acme Mocha", "Acme", "123456789013", "Shop A"),
        ("bad-b", "Acme Mocha", "Acme", "", "Shop A"),
    ])

    clusters, census = link_barcode_less(df)

    assert set(clusters) == {"row-1", "row-2"}
    assert clusters["row-1"] != clusters["row-2"]
    assert census["exact_title_pairs_linked"] == 0
    assert census["barcode_less_rows"] == 2


def test_exact_title_pair_links_across_retailers():
    df = _frame([
        ("a", "Acme Mocha", "Acme", "", "Shop A"),
        ("b", "Acme Mocha", "Acme", "", "Shop B"),
    ])

    clusters, census = link_barcode_less(df)

    assert clusters["row-0"] == clusters["row-1"]
    assert census["exact_title_pairs_linked"] == 1


def test_missing_brand_or_title_stays_singleton():
    df = _frame([
        ("a", "", "Acme", "", "Shop A"),
        ("b", "", "Acme", "", "Shop B"),
        ("c", "Acme Mocha", None, "", "Shop A"),
        ("d", "Acme Mocha", None, "", "Shop B"),
    ])

    clusters, census = link_barcode_less(df)

    assert len(set(clusters.values())) == 4
    assert census["num_multirow_clusters"] == 0


def test_pack_count_cleanup_handles_normalized_apostrophe_and_volume():
    assert strip_pack_multiplicity("acme 12 s mocha") == "acme mocha"
    assert strip_pack_multiplicity("acme pack of 6 250 ml") == "acme 250 ml"
    assert strip_pack_multiplicity("acme one 6-pack") == "acme"
    assert strip_pack_multiplicity("acme 24 count 12 bottles") == "acme"
    assert strip_pack_multiplicity("acme quantity of 12") == "acme"
    assert strip_pack_multiplicity("acme 24 oz") == "acme 24 oz"
    assert strip_pack_multiplicity("acme of 250 ml") == "acme of 250 ml"
