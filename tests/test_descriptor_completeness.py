from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.product_identity import completeness, completeness_frame


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, "", " \t"])
def test_missing_descriptors_do_not_rank_as_populated(missing):
    row = {"title": "Cola", "brand": missing, "description": missing}
    assert completeness(row) == 1
    assert completeness(SimpleNamespace(**row)) == 1


def test_frame_counts_match_scalar_and_preserve_index():
    frame = pd.DataFrame([
        {"title": "Cola", "brand": np.nan, "description": pd.NA, "price": "9.99"},
        {"title": "Cola", "brand": "Acme", "description": " ", "url": "https://x"},
        {"title": None, "brand": "NA", "description": "None", "price": "1.00"},
    ], index=[7, 2, 9])
    expected = pd.Series([1, 2, 2], index=frame.index, dtype="int64")
    pd.testing.assert_series_equal(completeness_frame(frame), expected)
    assert [completeness(row) for row in frame.to_dict("records")] == expected.tolist()
    assert completeness_frame(frame.iloc[:0]).empty


def test_empty_schema_retains_zero_counts():
    frame = pd.DataFrame({"price": ["1", "2"]}, index=[3, 1])
    pd.testing.assert_series_equal(
        completeness_frame(frame), pd.Series([0, 0], index=[3, 1], dtype="int64"),
    )
