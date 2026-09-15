"""Shared attribute parsing and conflict classification.

Training-time mining, pair dumps, and post-run reports must use the same
structured canonical attributes.  Keeping this logic here prevents a report
from disagreeing with the population that was actually mined.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Mapping

from core.unit_canonicalization import canonical_pack_count, canonical_volume_ml


_FLAVOR_NOISE = frozenset(
    {
        "flavor",
        "flavored",
        "flavour",
        "flavoured",
        "profile",
        "taste",
    }
)
_FLAVOR_TOKEN_ALIASES = {
    "berries": "berry",
    "cocoanut": "coconut",
}
_FLAVOR_LEXICON = frozenset(
    {
        "aloe",
        "apple",
        "berry",
        "cherry",
        "chocolate",
        "citrus",
        "coconut",
        "coffee",
        "cola",
        "cranberry",
        "elderflower",
        "ginger",
        "grapefruit",
        "grape",
        "lemon",
        "lime",
        "mango",
        "mint",
        "orange",
        "passionfruit",
        "passion",
        "peach",
        "pear",
        "pineapple",
        "pomegranate",
        "raspberry",
        "rhubarb",
        "rose",
        "strawberry",
        "tonic",
        "tropical",
        "fruit",
        "vanilla",
        "watermelon",
    }
)


def normalized_flavor_tokens(value: object) -> frozenset[str]:
    """Return stable flavor evidence tokens for overlap-based comparison.

    Separators commonly found in catalog exports (commas, underscores,
    slashes, ampersands, and hyphens) are deliberately equivalent.  Generic
    flavor-label words are removed so they cannot create false overlap.
    """
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    tokens = re.findall(r"[a-z0-9]+", text)
    return frozenset(
        _FLAVOR_TOKEN_ALIASES.get(token, token)
        for token in tokens
        if token not in _FLAVOR_NOISE
    )


def flavor_overlap_metrics(left: object, right: object) -> tuple[float, float]:
    """Return ``(Jaccard, overlap coefficient)`` for flavor evidence.

    Empty evidence is explicit rather than treated as disagreement: both
    values are zero and callers decide whether missingness needs a separate
    confidence mask.
    """
    left_tokens = normalized_flavor_tokens(left)
    right_tokens = normalized_flavor_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0, 0.0
    intersection = len(left_tokens & right_tokens)
    return (
        intersection / len(left_tokens | right_tokens),
        intersection / min(len(left_tokens), len(right_tokens)),
    )


def _flavor_evidence(*values: object) -> str:
    """Collect all recognized flavor mentions, not just the first regex hit."""
    tokens: set[str] = set()
    for value in values:
        tokens.update(normalized_flavor_tokens(value) & _FLAVOR_LEXICON)
    return " ".join(sorted(tokens))


def _value_set(value: object, *, kind: str) -> set[object]:
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
        canonicalizer = canonical_volume_ml if kind == "volume" else canonical_pack_count
        normalized = set()
        for item in parsed:
            try:
                normalized.add(canonicalizer(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid canonical {kind} value: {item!r}"
                ) from exc
        return normalized
    raise ValueError(f"canonical attribute set is not a sequence: {value!r}")


def canonical_attribute_info(record: Mapping[str, object]) -> dict[str, object]:
    """Return the structured attributes used by the canonical record lane."""
    return {
        "volume": _value_set(record.get("volume_set"), kind="volume"),
        "pack": _value_set(record.get("pack_set"), kind="pack"),
        "flavor": _flavor_evidence(
            record.get("mode_flavor", ""), record.get("canonical", "")
        ),
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
            {canonical_volume_ml(volume_ml)}
            if volume_ml is not None and float(volume_ml) > 0
            else set()
        )
    except (TypeError, ValueError):
        volume = set()
    pack_qty = extracted.get("pack_qty")
    pack_confidence = extracted.get("pack_confidence")
    try:
        pack = (
            {canonical_pack_count(pack_qty)}
            if pack_qty is not None
            and int(pack_qty) >= 1
            and float(pack_confidence or 0.0) > 0.0
            else set()
        )
    except (TypeError, ValueError):
        pack = set()
    return {
        "volume": volume,
        "pack": pack,
        "flavor": _flavor_evidence(
            extracted.get("flavor") or "", title, attributes
        ),
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
    if left["flavor"] and right["flavor"] and flavor_overlap_metrics(
        left["flavor"], right["flavor"]
    )[1] == 0.0:
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


__all__ = [
    "attribute_conflict_types",
    "canonical_attribute_info",
    "conflict_columns",
    "flavor_overlap_metrics",
    "normalized_flavor_tokens",
    "sku_attribute_info",
]
