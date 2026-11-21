"""The cascade's candidate source unions its declared weak keys (one block each).

One public behavior: a listing's candidates are the UNION of its declared keys'
blocks without itself — two listings that share the volume+flavour key but a
different brand are candidates of each other, a virtual supervision endpoint is
never a candidate, and a key a row does not carry contributes nothing.
"""
import pandas as pd

from model_tracks.blocked_candidates import BlockedCandidates


def test_candidates_union_the_declared_keys(tmp_path):
    catalog = tmp_path / "eligible_catalog.csv"
    pd.DataFrame([
        {"sku_id": "100", "brand": "ACME", "category": "Energy Drinks",
         "attribute": "Volume: 250 ml; Flavour: Berry"},
        {"sku_id": "200", "brand": "Other", "category": "Energy Drinks",
         "attribute": "Volume: 250 ml; Flavour: Berry"},
        {"sku_id": "300", "brand": "ACME", "category": "Energy Drinks",
         "attribute": "Volume: 500 ml; Flavour: Mango"},
        {"sku_id": "canonical:300", "brand": "ACME", "category": "Energy Drinks",
         "attribute": "Volume: 500 ml; Flavour: Mango"},
        {"sku_id": "400", "brand": "", "category": "", "attribute": ""},
    ]).to_csv(catalog, index=False)
    owner = BlockedCandidates.from_catalog(
        catalog, keys=(("brand_family",), ("volume", "flavour")))

    # Brand block (300) UNION the volume+flavour block (200).
    assert owner.candidates("100") == ("200", "300")
    assert owner.candidates("200") == ("100",)
    # The virtual endpoint repeats 300's brand and attributes and is still absent.
    assert owner.candidates("300") == ("100",)
    # No declared key value on the row: no block, no candidates.
    assert owner.candidates("400") == ()
