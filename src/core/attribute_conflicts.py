"""Shared attribute parsing and conflict classification.

Training-time mining, pair dumps, and post-run reports must use the same
structured canonical attributes.  Keeping this logic here prevents a report
from disagreeing with the population that was actually mined.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping


def _value_set(value: object) -> set[object]:
    """Parse a canonical CSV set field without trusting CSV dtype inference."""
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid canonical attribute set: {value!r}") from exc
    if isinstance(parsed, (list, tuple, set)):
        return set(parsed)
    raise ValueError(f"canonical attribute set is not a sequence: {value!r}")


def canonical_attribute_info(record: Mapping[str, object]) -> dict[str, object]:
    """Return the structured attributes used by the canonical record lane."""
    return {
        "volume": _value_set(record.get("volume_set")),
        "pack": _value_set(record.get("pack_set")),
        "flavor": str(record.get("mode_flavor", "")).strip().lower(),
    }


def sku_attribute_info(title: object, attributes: object) -> dict[str, object]:
    """Extract SKU-side attributes using the same parser as the data lane."""
    from pipeline import extract_all

    extracted = extract_all(str(title), str(attributes))
    volume_ml = extracted.get("volume_ml")
    # Zero is the extractor's sentinel for "no volume mention".  It must
    # remain unknown here; turning it into {0.0} makes every known canonical
    # volume look like a real conflict.
    try:
        volume = (
            {float(volume_ml)}
            if volume_ml is not None and float(volume_ml) > 0
            else set()
        )
    except (TypeError, ValueError):
        volume = set()
    return {
        "volume": volume,
        "pack": {int(extracted.get("pack_qty") or 1)},
        "flavor": str(extracted.get("flavor") or "").strip().lower(),
    }


def attribute_conflict_types(
    left: Mapping[str, object], right: Mapping[str, object]
) -> list[str]:
    """Classify conflicts using the canonical miner's disjoint-set rules."""
    conflicts: list[str] = []
    if left["volume"] and right["volume"] and not (
        set(left["volume"]) & set(right["volume"])
    ):
        conflicts.append("volume")
    if left["pack"] and right["pack"] and not (
        set(left["pack"]) & set(right["pack"])
    ):
        conflicts.append("pack")
    if left["flavor"] and right["flavor"] and left["flavor"] != right["flavor"]:
        conflicts.append("flavor")
    return conflicts


def conflict_columns(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    """Return the report columns for a pair of parsed attribute records."""
    conflicts = attribute_conflict_types(left, right)
    return {
        "volume_conflict": int("volume" in conflicts),
        "pack_conflict": int("pack" in conflicts),
        "flavor_conflict": int("flavor" in conflicts),
        "attribute_conflict_type": "+".join(conflicts) if conflicts else "none",
    }
