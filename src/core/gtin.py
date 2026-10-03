"""src/core/gtin.py — GTIN/UPC structural validation (moved from src/core/cache.py).

Pure, import-light (pandas only — no torch, no dataset hashing): every
label-forming surface imports from here, never from core.cache, so adding
validation to a script never drags the 53MB dataset sha256.

Why this module exists (owner ruling, this session): 1,747 of 14,997
distinct gtins (3,715 rows) FAIL the GS1 check digit — retailer-export
noise. Until now raw gtin was treated as ground truth everywhere (canonical
grouping, eval pairs, dedupe T1), silently creating false-positive labels.
The checksum was validated by dead code (lib.cache was never imported by
the pipeline); README claimed the validation existed. These functions are
now the live SSOT for "this gtin may be trusted as identity".

Structural validation and reviewed eligibility are distinct.
`gtin_validity` additionally rejects reviewed GLN/formulation holds.
Semantics: validation only decides TRUST; grouping keys stay the RAW gtin
string (no UPC-12→13 rewriting of keys — that would drift every CSV).
Leading-zero canonicalization happens inside the checksum only, where it
is checksum-neutral (weight of a leading 0 is 0, weights anchor right).

MULTI-RUN extraction (truncation fix): a raw cell may carry several digit
runs; retail feeds measure things like "1 liter", "6 pack 4006381333931"
or spell "1-735143004010". The old first-run extraction silently truncated
"1-735143004010" to "1". The census promised no change — the corpus holds
no cell where the first run is followed by a LONGER run (the 22 "7-…"
7-digit prefices all precede the same 13-digit number, never a longer
body) — and the printed census is byte-identical. Extraction now prefers
the LONGEST digit run per cell (earliest wins ties), so a describer
prefix is resolved correctly for any future cell that carries one, while
every single-run cell (i.e. all current data) extracts byte-identically.

SIBLING EQUALITY: `gtin_equivalent(a, b)` compares two spellings after
whitespace strip, UPC-12→13 zero-prefix folding, and the GS1 checksum on
the qualifying spelling each side lands on; True iff both resolve to
identical checksum-valid digits at the same length (8/13/14 — a 12 never
survives the fold, and a 12 never equals a 14 spelling because the lengths
differ). This is sibling-tolerance equality for callers that must not
re-pad node keys; raw-key grouping elsewhere stays untouched.
"""

from __future__ import annotations

import re

import numpy as np
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

    The input is a raw gtin cell (CSV read as dtype=str, so NaN can be a
    float) — None/NaN passes through unchanged; the caller filters it.
    """
    if pd.isna(x):
        return x

    x = str(x)

    if len(x) == 12:
        return "0" + x

    return x


def normalize_and_validate_gtin(series: pd.Series) -> pd.DataFrame:
    """Clean raw gtin strings and validate GTIN structure.

    Extraction keeps the LONGEST digit run in a cell (earliest wins ties),
    so a describer prefix ("1-735143004010") no longer truncates to "1";
    the census of dataset.csv is unchanged by this (measured regression
    gate, see module docstring).

    Returns:
        gtin_clean
        gtin_structurally_valid
    """
    # Keep the longest digit run per cell (earliest wins ties — max returns
    # the first maximal candidate). A cell with no digits maps to "", which
    # the filter below sends to missing. Alignment is positional: the raw
    # series may carry an arbitrary or duplicate index. This replaces
    # first-run-only `str.extract(r"(\d+)")`, which truncated e.g.
    # "1-735143004010" to "1"; no current dataset.csv cell has first run
    # != longest run (measured), so outputs are byte-identical there.
    present_loc = series.notna().to_numpy()
    cleaned_vals = np.full(len(series), None, dtype=object)

    def _longest(raw: str) -> str:
        candidates = re.findall(r"\d+", raw)
        return max(candidates, key=len) if candidates else ""

    if present_loc.any():
        winners = series.loc[present_loc].astype(str).map(_longest)
        cleaned_vals[present_loc] = winners.to_numpy()
    cleaned = pd.Series(cleaned_vals, index=series.index, dtype="object")

    # Empty strings -> missing.
    cleaned = cleaned.where(cleaned.ne(""))

    # Remove placeholder all-zero gtins.
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


def _gtin_siblings_key(x: str | float | None) -> str:
    """Internal key for `gtin_equivalent`: whitespace-strip, fold UPC-12 to
    its zero-prefixed GTIN-13 spelling, then require the checksum of the
    qualifying length each side lands on. Any non-digit, missing, or
    off-length form — including a 11-digit body — returns "". The checksum
    is checked on the FOLDED spelling: a UPC-12 is validated as its 13-digit
    sibling, never raw (weights anchor right, so the 12 alone is a
    different number)."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or not s.isdigit():
        return ""
    if len(s) == 12:
        s = "0" + s
    if len(s) not in {8, 13, 14}:
        return ""
    return s if is_valid_gtin_checksum(s) else ""


def gtin_equivalent(a: str | float | None, b: str | float | None) -> bool:
    """Sibling-tolerance GTIN equality: UPC-12 and its zero-prefixed EAN-13
    spelling are the same code.

    Both inputs are whitespace-stripped; a UPC-12 is folded to 13 by adding
    the leading zero and the GS1 checksum is enforced on the QUALIFYING
    spelling each side resolves to (folded, not raw); equal is True iff
    both resolve to identical checksum-valid digits at the same length
    (8/13/14 — a 12 never survives the fold). Any other spelling —
    malformed, missing, off-length (including 11-digit) — is False by
    construction; nothing is padded or repaired here. Grouping keys
    elsewhere stay raw — see the module docstring.
    """
    ka = _gtin_siblings_key(a)
    return ka != "" and ka == _gtin_siblings_key(b)


def gtin_validity(gtins: pd.Series) -> pd.Series:
    """Boolean mask: may this gtin string be trusted as product identity?

    True only when the digit-extracted, length-plausible gtin passes the
    GS1 check digit and is not held by the reviewed identity policy. Empty/placeholder/malformed -> False. Label surfaces
    (canonical grouping, eval pairs, dedupe T1) treat False exactly like a
    missing gtin: the row survives in the corpus, but no identity is
    asserted from it.
    """
    from core.identity_policy import held_keys
    facts = normalize_and_validate_gtin(gtins)
    held = facts.gtin_clean.astype("string").str.zfill(14).isin(held_keys())
    return facts["gtin_structurally_valid"] & ~held
