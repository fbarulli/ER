#!/usr/bin/env python3
"""Write the two-column item-resolution submission.

The input may contain diagnostics (scores, nearest candidates, etc.).  Only the
SKU and resolved item identifiers are retained in the output.  Output headers
are lowercase by default for the external submission contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_column(frame: pd.DataFrame, name: str) -> str:
    matches = [column for column in frame.columns if str(column).strip().lower() == name]
    if not matches:
        raise ValueError(f"input is missing required column {name!r}")
    if len(matches) > 1:
        raise ValueError(f"input contains duplicate case-insensitive columns for {name!r}")
    return matches[0]


def format_submission(input_path: Path, output_path: Path, *, uppercase_columns: bool = False) -> None:
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    sku_column = _find_column(frame, "sku_id")
    item_column = _find_column(frame, "item_id")
    output = frame[[sku_column, item_column]].copy()
    output.columns = ["SKU_ID", "ITEM_ID"] if uppercase_columns else ["sku_id", "item_id"]

    if output["SKU_ID" if uppercase_columns else "sku_id"].duplicated().any():
        raise ValueError("input contains duplicate SKU_ID values")
    if output.isna().any().any():
        raise ValueError("submission contains missing identifiers")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="prediction CSV, including SKU_ID and ITEM_ID")
    parser.add_argument("output", type=Path, help="two-column submission CSV to write")
    parser.add_argument(
        "--uppercase-columns",
        action="store_true",
        help="write the internal pipeline spelling (SKU_ID, ITEM_ID) instead of lowercase headers",
    )
    args = parser.parse_args()
    format_submission(args.input, args.output, uppercase_columns=args.uppercase_columns)


if __name__ == "__main__":
    main()
