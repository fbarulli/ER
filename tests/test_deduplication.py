"""Representative selection preserves ranking, indices and row lineage."""
import numpy as np
import pandas as pd
import pytest

from core.deduplication import collapse_representatives


def test_selection_matches_stable_keep_first_with_missing_keys():
    frame = pd.DataFrame({
        "retailer": ["a", "a", "a", "a", "b", "b"],
        "sku_name_eng": ["juice", "juice", None, np.nan, "water", "water"],
        "rank": [1, 2, 4, 4, np.nan, 0],
    }, index=[9, 3, 17, 25, 41, 6])
    parent = {i: i for i in frame.index}
    kept, dropped = collapse_representatives(
        frame, ["retailer", "sku_name_eng"], ["rank"], [False], parent=parent
    )
    expected = frame.sort_values("rank", ascending=False, na_position="last", kind="stable").drop_duplicates(["retailer", "sku_name_eng"])
    pd.testing.assert_frame_equal(kept, expected)
    assert parent == {9: 3, 3: 3, 17: 17, 25: 17, 41: 6, 6: 6}
    assert set(dropped) == {9, 25, 41}


def test_empty_selection():
    frame = pd.DataFrame({"key": pd.Series(dtype=str), "rank": pd.Series(dtype=float)})
    parent = {}
    kept, dropped = collapse_representatives(frame, ["key"], ["rank"], [False], parent=parent)
    assert kept.empty and dropped.empty and not parent


def test_nonunique_source_indices_rejected():
    frame = pd.DataFrame({"key": ["x", "x"], "rank": [1, 2]}, index=[0, 0])
    with pytest.raises(ValueError, match="unique source row"):
        collapse_representatives(frame, ["key"], ["rank"], [False], parent={})


def test_missing_title_protection_survives_shared_collapse():
    from training.dedupe import _protect_missing_titles
    frame = pd.DataFrame({"retailer": ["a", "a"], "sku_name_eng": [None, None], "_ident": ["", ""], "rank": [1, 2]})
    kept, dropped = collapse_representatives(
        _protect_missing_titles(frame), ["retailer", "sku_name_eng", "_ident"], ["rank"], [False], parent={}
    )
    assert len(kept) == 2 and dropped.empty


@pytest.mark.parametrize('decision,expected', [('collapse', True), ('keep', False)])
def test_reviewed_adjudication_reads_config(monkeypatch, decision, expected):
    from types import SimpleNamespace
    from training import dedupe
    spec = SimpleNamespace(dedupe_adjudications=[SimpleNamespace(
        retailer='retailer', gtin='123', decision=decision
    )])
    monkeypatch.setattr(dedupe, 'data_cfg', lambda: spec)
    # Overrides settle the pair before any descriptor parsing is needed.
    assert dedupe._same_product_by_title(pd.DataFrame(), 'retailer', '123') is expected
