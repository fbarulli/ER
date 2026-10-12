"""tests/test_laya_corpus_column_contract.py — the corpus header seam."""
from __future__ import annotations

import pytest

from scripts.laya_corpus_io import (
    LISTING_PAIRS_REQUIRED_COLUMNS,
    LISTING_PAIRS_TRACE_COLUMNS,
    ColumnContract,
    CsvSource,
)

HEADER = ["sku_id1", "sku_id2", "label", "split"]


def test_accepts_the_exact_contract_header() -> None:
    ColumnContract.require(HEADER, LISTING_PAIRS_REQUIRED_COLUMNS,
                           source="listing_pairs")


def test_accepts_the_declared_trace_column() -> None:
    ColumnContract.require(HEADER + ["example_id"],
                           LISTING_PAIRS_REQUIRED_COLUMNS,
                           optional=LISTING_PAIRS_TRACE_COLUMNS,
                           source="listing_pairs")


@pytest.mark.parametrize("header", [
    HEADER[:3],
    HEADER + ["mystery"],
    HEADER + ["mystery", "mystery"],
])
def test_missing_unknown_and_duplicate_headers_drift_loud(header: list[str]) -> None:
    with pytest.raises(RuntimeError, match="listing_pairs header drifted"):
        ColumnContract.require(header, LISTING_PAIRS_REQUIRED_COLUMNS,
                               optional=LISTING_PAIRS_TRACE_COLUMNS,
                               source="listing_pairs")


def test_csv_source_round_trip(tmp_path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text(",".join(HEADER) + "\na,b,1,train\n", encoding="utf-8")
    header, rows = CsvSource.read(path)
    assert header == HEADER
    assert rows == [{"sku_id1": "a", "sku_id2": "b", "label": "1",
                     "split": "train"}]
