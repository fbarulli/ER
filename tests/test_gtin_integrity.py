"""Integrity invariants for core.gtin (multi-run extraction + sibling equality).

Two regression classes are pinned here:

  * the multi-run truncation fix: extraction used to keep only the FIRST
    digit run of a cell, so "1-735143004010" silently became "1" and the
    real barcode was lost. Extraction now keeps the LONGEST digit run
    (earliest on ties). The dataset.csv census is unchanged by this — no
    current cell has first run != longest run (measured) — so the module's
    behavior only widens for cells that actually carry a describer.
  * `gtin_equivalent`: a raw-equality gate mislabels a UPC-12 against its
    zero-prefixed EAN-13 sibling (same product, different spelling, zero
    collisions measured on this corpus yet structurally unprotected). The
    helper folds UPC-12→13 and applies the checksum on the qualifying
    spelling, without re-padding any node key.

Real checksum-valid laboratories values keep the tests honest: synthetic
13-digit strings mostly fail the GS1 checksum and would silently exercise
the malformed path instead.
"""
from __future__ import annotations

import pandas as pd

from core.gtin import (
    gtin_equivalent,
    is_valid_gtin_checksum,
    normalize_and_validate_gtin,
)


# Real checksum-valid values (verified below at import time), so every
# extracted body lands on the plausibility path, not the malformed one.
_UPC12 = "036000291452"          # UPC-12
_EAN13 = "0036000291452"         # the same code, zero-prefixed
_UPC12_ALT = "735143004010"      # 12-digit body from a real corpus cell
_EAN13_ALT = "0735143004010"
_EAN8 = "87123457"


def test_fixture_checksums_are_valid():
    assert is_valid_gtin_checksum(_UPC12)
    assert is_valid_gtin_checksum(_EAN13)
    assert is_valid_gtin_checksum(_UPC12_ALT)
    assert is_valid_gtin_checksum(_EAN13_ALT)


def test_longest_digit_run_survives_describer_prefix():
    """'1-735143004010' resolves to the checksum-valid 12-digit body."""
    facts = normalize_and_validate_gtin(pd.Series(["1-735143004010"]))
    assert facts["gtin_clean"].iat[0] == "0735143004010"
    assert facts["gtin_structurally_valid"].iat[0]


def test_longest_run_wins_over_noise_around_the_barcode():
    facts = normalize_and_validate_gtin(
        pd.Series(["pack 6, volume 500 - 735143004010"])
    )
    assert facts["gtin_clean"].iat[0] == "0735143004010"
    assert facts["gtin_structurally_valid"].iat[0]


def test_equal_length_runs_take_the_first_and_extend_the_body():
    # "12" and "34" tie on length; concatenation must NOT be how the tie is
    # broken ("1234" would be a different key than either run), and the
    # earliest must win deterministically.
    facts = normalize_and_validate_gtin(pd.Series(["id 87123456 / 87123457"]))
    assert facts["gtin_clean"].iat[0] == "87123456"
    assert facts["gtin_structurally_valid"].iat[0]


def test_all_zero_placeholder_stays_invalid():
    facts = normalize_and_validate_gtin(pd.Series(["0000000000"]))
    assert pd.isna(facts["gtin_clean"].iat[0])
    assert not facts["gtin_structurally_valid"].iat[0]


def test_eleven_digit_body_stays_invalid():
    facts = normalize_and_validate_gtin(pd.Series(["1-" + _UPC12[:-1]]))
    assert pd.isna(facts["gtin_clean"].iat[0])
    assert not facts["gtin_structurally_valid"].iat[0]


def test_single_run_cells_unchanged():
    facts = normalize_and_validate_gtin(pd.Series([_EAN13]))
    assert facts["gtin_clean"].iat[0] == _EAN13
    assert facts["gtin_structurally_valid"].iat[0]


def test_gtin_equivalent_upc12_vs_zero_prefixed_ean13():
    assert gtin_equivalent(_UPC12, _EAN13)
    assert gtin_equivalent(_UPC12_ALT, _EAN13_ALT)
    assert gtin_equivalent(f" {_EAN13_ALT} ", _UPC12_ALT)


def test_gtin_equivalent_distinct_codes_false():
    assert not gtin_equivalent(_UPC12, "0036000291453")
    assert not gtin_equivalent(_UPC12_ALT, _EAN13)
    assert not gtin_equivalent(_UPC12, _UPC12_ALT)


def test_gtin_equivalent_malformed_false():
    assert not gtin_equivalent(_UPC12[:-1], _UPC12[:-1])  # 11 digits
    assert not gtin_equivalent("", "")
    assert not gtin_equivalent(None, None)
    assert not gtin_equivalent(float("nan"), _EAN13)
    assert not gtin_equivalent("n/a", _EAN13)
    assert not gtin_equivalent(f"data {_EAN13_ALT}", _UPC12_ALT)
    assert not gtin_equivalent(_UPC12[:-1] + "3", _UPC12)  # bad check digit
    # checksum-invalid even against itself: structural nonsense asserts nothing
    assert not gtin_equivalent("87123458", "87123458")
