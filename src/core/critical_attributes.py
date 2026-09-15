"""Shared critical product-attribute vocabulary and compatibility rules.

The model text lane, canonical records, hard-negative miners, and inference
gates all import this module.  Explicit evidence can agree or conflict;
absence is kept as unknown and is never converted into agreement.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping


CRITICAL_ATTRIBUTE_DIMENSIONS: tuple[str, ...] = (
    "volume",
    "pack",
    "package_type",
    "flavor",
    "carbonation",
    "sweetener",
    "pulp",
)

FLAVOR_ALIASES: dict[str, str] = {
    "berries": "berry",
    "cocoanut": "coconut",
}
FLAVOR_LEXICON: frozenset[str] = frozenset(
    {
        "aloe", "apple", "berry", "cherry", "chocolate", "citrus",
        "coconut", "coffee", "cola", "cranberry", "elderflower", "fruit",
        "ginger", "grape", "grapefruit", "lemon", "lime", "mango", "mint",
        "orange", "passion", "passionfruit", "peach", "pear", "pineapple",
        "pomegranate", "raspberry", "rhubarb", "rose", "strawberry", "tonic",
        "tropical", "vanilla", "watermelon",
    }
)


def normalized_attribute_text(*values: object) -> str:
    """Normalize punctuation without discarding negation-bearing words."""
    text = " ".join(str(value or "") for value in values)
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def extract_flavor_tokens(*values: object) -> frozenset[str]:
    text = normalized_attribute_text(*values)
    return frozenset(
        FLAVOR_ALIASES.get(token, token)
        for token in text.split()
        if FLAVOR_ALIASES.get(token, token) in FLAVOR_LEXICON
    )


def extract_critical_claims(*values: object) -> dict[str, frozenset[str]]:
    """Extract explicit non-numeric critical claims from source text.

    ``no added sugar`` is retained separately: it does not prove that a
    product contains no naturally occurring sugar.  ``diet`` is compatible
    with ``no_sugar`` but conflicts with an explicit ``sugar`` claim.
    """
    text = normalized_attribute_text(*values)

    no_sugar = bool(
        re.search(
            r"\b(?:no sugar|zero sugar|sugar free|sugarfree|sugarless|"
            r"without sugar|free of sugar)\b",
            text,
        )
    )
    no_added_sugar = bool(
        re.search(r"\b(?:no|without) added sugar\b", text)
    )
    sugar = bool(
        re.search(
            r"\b(?:with added sugar|contains sugar|sweetened with sugar|"
            r"sweetener sugar|sugar sweetened)\b",
            text,
        )
    )
    sweetener: set[str] = set()
    if no_sugar:
        sweetener.add("no_sugar")
    if no_added_sugar:
        sweetener.add("no_added_sugar")
    if sugar:
        sweetener.add("sugar")
    if re.search(r"\bdiet\b", text):
        sweetener.add("diet")

    # Remove explicit negative phrases before looking for positive
    # carbonation so "non-carbonated" cannot emit both states.
    non_carbonated = bool(
        re.search(r"\b(?:non carbonated|uncarbonated|not carbonated)\b", text)
    )
    carbonation_text = re.sub(
        r"\b(?:non carbonated|uncarbonated|not carbonated)\b", " ", text
    )
    carbonation: set[str] = set()
    if non_carbonated or re.search(r"\bstill\b", text):
        carbonation.add("still")
    if re.search(r"\b(?:carbonated|sparkling|fizzy)\b", carbonation_text):
        carbonation.add("carbonated")

    pulp: set[str] = set()
    no_pulp = bool(
        re.search(
            r"\b(?:no pulp|without pulp|pulp free|free of pulp|pulp 0)\b",
            text,
        )
        or re.search(r"\bpulp\s+(?:no|none)\b", text)
    )
    with_pulp = bool(
        re.search(r"\b(?:with (?:extra )?pulp|contains pulp|pulp yes)\b", text)
    )
    if no_pulp:
        pulp.add("no_pulp")
    if with_pulp:
        pulp.add("with_pulp")

    return {
        "flavor": extract_flavor_tokens(text),
        "carbonation": frozenset(carbonation),
        "sweetener": frozenset(sweetener),
        "pulp": frozenset(pulp),
    }


def sweetener_conflict(left: set[str], right: set[str]) -> bool:
    """Return true only for an explicit positive-vs-diet/no-sugar clash."""
    negative = {"no_sugar", "diet"}
    return bool(
        ("sugar" in left and right & negative)
        or ("sugar" in right and left & negative)
    )


def categorical_conflict(
    dimension: str, left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    left_values = set(left.get(dimension) or set())
    right_values = set(right.get(dimension) or set())
    if not left_values or not right_values:
        return False
    if dimension == "sweetener":
        return sweetener_conflict(left_values, right_values)
    return not bool(left_values & right_values)


__all__ = [
    "CRITICAL_ATTRIBUTE_DIMENSIONS",
    "FLAVOR_ALIASES",
    "FLAVOR_LEXICON",
    "categorical_conflict",
    "extract_critical_claims",
    "extract_flavor_tokens",
    "normalized_attribute_text",
    "sweetener_conflict",
]
