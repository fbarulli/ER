"""A missing title cannot establish identity between gtin-less listings."""
import numpy as np
import pandas as pd

from training.dedupe import _protect_missing_titles


def test_blank_title_rows_keep_separate_t2_and_t3_partitions():
    # Repeated IDs and nonconsecutive indices must not defeat protection.
    frame = pd.DataFrame({
        "sku_id": ["same"] * 5,
        "retailer": ["amazon"] * 5,
        "sku_name_eng": [np.nan, "", "  ", "Lemon juice", "Lemon juice"],
        "brand": ["Cal-O-Sicle", "Mautner Markhof", "Culture Pop", "A", "A"],
        "_ident": [""] * 5,
    }, index=[12, 13, 25, 30, 31])
    guarded = _protect_missing_titles(frame)
    assert guarded._ident.iloc[:3].nunique() == 3
    assert guarded._ident.iloc[3:].tolist() == ["", ""]
    assert frame._ident.tolist() == [""] * 5
    # These are the keys T2 and T3 both use after the identity handoff.
    collapsed = guarded.drop_duplicates(["retailer", "sku_name_eng", "_ident"])
    assert len(collapsed) == 4
    assert collapsed.brand.iloc[:3].tolist() == ["Cal-O-Sicle", "Mautner Markhof", "Culture Pop"]


def test_missing_titles_with_same_trusted_gtin_are_not_title_merged():
    # T1 owns gtin merges. Rows reaching the title tiers still cannot use
    # absent title as an identity claim (e.g. identity-review holds).
    frame = pd.DataFrame({"sku_name_eng": [None, None], "_ident": ["123", "123"]})
    guarded = _protect_missing_titles(frame)
    assert guarded._ident.nunique() == 2


def test_populated_titles_preserve_frame_and_partitions():
    frame = pd.DataFrame({"sku_name_eng": ["Lemon", "Lime"], "_ident": ["a", "b"]})
    assert _protect_missing_titles(frame) is frame


def test_vectorized_identity_agreement_preserves_missing_value_semantics():
    frame = pd.DataFrame({
        "retailer": ["a"] * 9,
        "sku_name_eng": ["same", "same", "different", "different", None, None, "unknown", "unknown", "single"],
        "_t2_bc": ["123", "123", "123", "456", "", "", None, np.nan, "789"],
    }, index=[7, 2, 9, 1, 4, 3, 8, 5, 6])
    groups = frame.groupby(["retailer", "sku_name_eng"], sort=False, dropna=False)["_t2_bc"]
    old = groups.transform(lambda s: "1" if s.nunique() <= 1 else "0").eq("1")
    new = groups.transform("nunique").le(1)
    pd.testing.assert_series_equal(old, new)
