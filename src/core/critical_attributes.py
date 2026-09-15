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


def compact_attribute_text(*values: object) -> str:
    """Normalize a field to a comparison key with NO word boundaries.

    The two normalisers already in the tree each solve half of this problem.
    ``normalized_attribute_text`` above folds accents but keeps the space, so
    ``PureThé`` becomes ``purethe`` while ``PURE THE`` becomes ``pure the``.
    The brand veto's own normaliser drops the space but uses NFKC, which does
    not decompose, so ``é`` survives ``isalnum()`` and ``PureThé`` stays
    ``purethé``.  Neither matches both spellings; this does.

    Measured on the 388 non-exact brand rows, the fused-vs-spaced class is the
    one normalization misses the token filter cannot reach: ``PureThé`` vs
    ``PURE THE``, ``Bio Food`` vs ``biofood``, ``Bolt 24`` vs ``Bolt24``,
    ``Folkington's`` vs ``Folkingtons`` and ``A SHOC`` vs ``Ashoc`` are the
    same brand written with and without a boundary.

    Use this where a fused or spaced spelling must compare equal.  Use
    ``normalized_attribute_text`` where word boundaries carry meaning, such as
    the phrase matching in ``extract_critical_claims``.
    """
    text = " ".join(str(value or "") for value in values)
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return "".join(char for char in text if char.isalnum())


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
    # Only unambiguous phrasings are accepted here. A "pulp <value>" enum
    # branch used to exist and was REMOVED (audit 2026-09-15) because the
    # normalizer folds punctuation away, making an enum spelling
    # ("pulp_no", "pulp:0") textually identical to prose. Two real
    # inversions proved this: "with Added Pulp, No Sugar Added" was
    # extracted as no_pulp (the exact opposite of its meaning) and the
    # volume fragment "pulp 0.33l" was read as "pulp 0". A census of 61,529
    # live titles shows the token after "pulp" is dominated by sizes
    # (16/1l/100/750) and by "free" (65), with no enum spellings present, so
    # the branch bought no recall and only risked label inversion. Absence of
    # a recognized phrase now stays unknown instead of inventing a claim.
    no_pulp = bool(
        re.search(
            r"\b(?:no pulp|without pulp|pulp free|free of pulp)\b",
            text,
        )
    )
    with_pulp = bool(
        re.search(
            r"\b(?:with (?:extra )?pulp|contains pulp|pulp yes)\b",
            text,
        )
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


def volumes_compatible(
    left_values: object,
    right_values: object,
    *,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
) -> bool:
    """Return whether any left/right volume pair is within the shared tolerance.

    SSOT (audit 2026-09-15): this predicate was re-implemented in several
    lanes with different answers — the training gate used a relative
    tolerance, the conflict miner used exact set intersection, and the
    calibration veto used a separate helper. Two lanes disagreeing about
    whether the same volumes are compatible is exactly how a pair the gate
    labels ``proceed`` gets emitted as a hard negative. Absent evidence on
    either side is not a conflict.
    """
    left = set(left_values or set())
    right = set(right_values or set())
    if not left or not right:
        return True
    return any(
        abs(float(a) - float(b))
        <= max(
            float(volume_absolute_tolerance_ml),
            float(volume_relative_tolerance) * max(abs(float(a)), abs(float(b))),
        )
        for a in left
        for b in right
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
    "volumes_compatible",
]
