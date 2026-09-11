"""src/core/gtin.py — GTIN/UPC structural validation (moved from src/core/cache.py).

Pure, import-light (pandas only — no torch, no dataset hashing): every
label-forming surface imports from here, never from core.cache, so adding
validation to a script never drags the 53MB dataset sha256.

Why this module exists (owner ruling, this session): 1,747 of 14,997
distinct barcodes (3,715 rows) FAIL the GS1 check digit — retailer-export
noise. Until now raw gtin was treated as ground truth everywhere (canonical
grouping, eval pairs, dedupe T1), silently creating false-positive labels.
The checksum was validated by dead code (lib.cache was never imported by
the pipeline); README claimed the validation existed. These functions are
now the live SSOT for "this barcode may be trusted as identity".

Semantics: validation only decides TRUST; grouping keys stay the RAW gtin
string (no UPC-12→13 rewriting of keys — that would drift every CSV).
Leading-zero canonicalization happens inside the checksum only, where it
is checksum-neutral (weight of a leading 0 is 0, weights anchor right).
"""

from __future__ import annotations

import pandas as pd


def is_valid_gtin_checksum(gtin: str) -> bool:
    """Validate check digit for GTIN-8, GTIN-12, GTIN-13, GTIN-14."""
    if not gtin or not gtin.isdigit():
        return False

    if len(gtin) not in {8, 12, 13, 14}:
        return False

    digits = [int(d) for d in gtin]
    check_digit = digits[-1]
    body = digits[:-1]

    # GS1 spec (all four GTIN lengths use the same rule): from the right-most
    # BODY digit (check digit excluded), weights alternate 3, 1, 3, 1, ...;
    # check digit = (10 - weighted_sum mod 10) mod 10.
    body_reversed = body[::-1]
    total = sum(body_reversed[0::2]) * 3 + sum(body_reversed[1::2])
    expected = (10 - total % 10) % 10

    return check_digit == expected


def _canonicalize_gtin(x: str | float | None) -> str | float | None:
    """Canonicalize UPC-12 to GTIN-13 by adding a leading zero.

    The input is a raw barcode cell (CSV read as dtype=str, so NaN can be a
    float) — None/NaN passes through unchanged; the caller filters it.
    """
    if pd.isna(x):
        return x

    x = str(x)

    if len(x) == 12:
        return "0" + x

    return x


def normalize_and_validate_gtin(series: pd.Series) -> pd.DataFrame:
    """Clean raw barcode strings and validate GTIN structure.

    Returns:
        gtin_clean
        gtin_structurally_valid
    """
    # Keep only digit sequences.
    cleaned = series.astype(str).str.extract(r"(\d+)", expand=False)

    # Empty strings -> missing.
    cleaned = cleaned.where(cleaned.ne(""))

    # Remove placeholder all-zero barcodes.
    cleaned = cleaned.where(~cleaned.str.fullmatch(r"0+", na=False))

    # UPC-12 -> GTIN-13 canonicalization (checksum-only; grouping keys
    # elsewhere stay raw — see module docstring).
    cleaned = cleaned.map(_canonicalize_gtin)

    # Keep plausible GTIN lengths. (An all-placeholder input — every row
    # dropped by the 0+ filter — leaves an all-NaN object column whose
    # .str accessor dies on object dtype; .astype("string") keeps it
    # usable and NaN lengths stay NaN -> invalid, which is the intent.)
    valid_length = cleaned.astype("string").str.len().isin([8, 12, 13, 14])
    cleaned = cleaned.where(valid_length)

    # Validate checksum without using fillna downcasting.
    is_valid = pd.Series(False, index=series.index, dtype=bool)
    notna_mask = cleaned.notna()

    if notna_mask.any():
        is_valid.loc[notna_mask] = (
            cleaned.loc[notna_mask].map(is_valid_gtin_checksum).astype(bool)
        )

    return pd.DataFrame(
        {
            "gtin_clean": cleaned,
            "gtin_structurally_valid": is_valid,
        }
    )


def barcode_validity(barcodes: pd.Series) -> pd.Series:
    """Boolean mask: may this barcode string be trusted as product identity?

    True only when the digit-extracted, length-plausible barcode passes the
    GS1 check digit. Empty/placeholder/malformed -> False. Label surfaces
    (canonical grouping, eval pairs, dedupe T1) treat False exactly like a
    missing barcode: the row survives in the corpus, but no identity is
    asserted from it.
    """
    return normalize_and_validate_gtin(barcodes)["gtin_structurally_valid"]
