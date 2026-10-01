"""Canonical numeric representations for product volume and pack size.

This module is the single unit boundary used before structured attributes are
appended to encoder text or fused with embeddings.  Source strings and catalog
values therefore reach the neural network in the same representation.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

def _volume_factors() -> dict[str, "Decimal"]:
    """unit spelling -> ml-per-unit Decimal, from the config units table.

    The literal dict this replaces DUPLICATED core.text._TO_ML and ner's
    VOLUME_TO_ML (a fourth copy); the copies already disagreed ('dl/decilitre'
    missing here means any decilitre title CRASHED canonical_volume_ml;
    'cc' absent from core.text's parse meant 65 '300 cc' titles were
    rejected; factors drifted 29.5735 vs 29.5735295625). Keys are normalized
    from spellings by normalize_unit so a single config table feeds both
    lookup conventions (this module's stripped keys and core.text's spaced
    norm_unit keys) with no interpolation between them.
    """
    from core.common import data_cfg

    factors: dict[str, Decimal] = {}
    for entry in data_cfg().units.volume:
        for spelling in entry.spellings:
            factors[normalize_unit(spelling)] = Decimal(str(entry.ml_per_unit))
    return factors


_VOLUME_TO_ML = None  # derived view, built lazily (config load deferred)


def _ensure_volume_table() -> dict:
    global _VOLUME_TO_ML
    if _VOLUME_TO_ML is None:
        _VOLUME_TO_ML = _volume_factors()
    return _VOLUME_TO_ML


# Persisted ANN indexes include this value in their preprocessing fingerprint.
# Increment it whenever canonical numeric semantics change.
# v2: the ml-per-unit table moved to config/paths.yaml `units` (one table
# serving core.text's parse, the converter and the NER features), so the
# converter now accepts the whole decilitre family and 'cc' where v1 raised
# on / rejected them. Persisted preprocessing fingerprints must rebuild.
UNIT_CANONICALIZATION_VERSION = "unit-canonical-v2"


def _decimal(value: object) -> Decimal:
    try:
        number = Decimal(str(value).strip().replace(",", "."))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid numeric unit value: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"unit value must be finite and positive: {value!r}")
    return number


def normalize_unit(unit: object) -> str:
    """Normalize punctuation/spacing/plurals for unit lookup."""
    return re.sub(r"[^a-z]", "", str(unit).casefold())


def canonical_volume_ml(value: object, unit: object = "ml") -> float:
    """Convert a positive volume to whole milliliters.

    Whole-milliliter rounding deliberately maps retail equivalents such as
    ``8 fl oz`` (236.588 ml) and a catalog's rounded ``237 ml`` to one stable
    encoder token.
    """
    normalized_unit = normalize_unit(unit)
    try:
        multiplier = _ensure_volume_table()[normalized_unit]
    except KeyError as exc:
        raise ValueError(f"unsupported volume unit: {unit!r}") from exc
    milliliters = (_decimal(value) * multiplier).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    return float(milliliters)


def canonical_pack_count(value: object) -> int:
    """Return a positive integral pack count, rejecting lossy coercions."""
    number = _decimal(value)
    integral = number.to_integral_value(rounding=ROUND_HALF_UP)
    if number != integral:
        raise ValueError(f"pack count must be an integer: {value!r}")
    return int(integral)


__all__ = [
    "UNIT_CANONICALIZATION_VERSION",
    "canonical_pack_count",
    "canonical_volume_ml",
    "normalize_unit",
]
