"""Canonical numeric representations for product volume and pack size.

This module is the single unit boundary used before structured attributes are
appended to encoder text or fused with embeddings.  Source strings and catalog
values therefore reach the neural network in the same representation.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

_VOLUME_TO_ML = {
    "ml": Decimal(1),
    "milliliter": Decimal(1),
    "milliliters": Decimal(1),
    "millilitre": Decimal(1),
    "millilitres": Decimal(1),
    "cc": Decimal(1),
    "cl": Decimal(10),
    "centiliter": Decimal(10),
    "centiliters": Decimal(10),
    "centilitre": Decimal(10),
    "centilitres": Decimal(10),
    "l": Decimal(1000),
    "lt": Decimal(1000),
    "ltr": Decimal(1000),
    "liter": Decimal(1000),
    "liters": Decimal(1000),
    "litre": Decimal(1000),
    "litres": Decimal(1000),
    "floz": Decimal("29.5735295625"),
    "fluidounce": Decimal("29.5735295625"),
    "fluidounces": Decimal("29.5735295625"),
    # Product titles use bare oz for beverage volume in this pipeline.
    "oz": Decimal("29.5735295625"),
    "ounce": Decimal("29.5735295625"),
    "ounces": Decimal("29.5735295625"),
    "pt": Decimal("473.176473"),
    "pint": Decimal("473.176473"),
    "pints": Decimal("473.176473"),
    "qt": Decimal("946.352946"),
    "quart": Decimal("946.352946"),
    "quarts": Decimal("946.352946"),
    "gal": Decimal("3785.411784"),
    "gallon": Decimal("3785.411784"),
    "gallons": Decimal("3785.411784"),
}

# Persisted ANN indexes include this value in their preprocessing fingerprint.
# Increment it whenever canonical numeric semantics change.
UNIT_CANONICALIZATION_VERSION = "unit-canonical-v1"


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
        multiplier = _VOLUME_TO_ML[normalized_unit]
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
