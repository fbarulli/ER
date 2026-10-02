"""Shared critical product-attribute vocabulary and compatibility rules.

The model text lane, canonical records, hard-negative miners, and inference
gates all import this module.  Explicit evidence can agree or conflict;
absence is kept as unknown and is never converted into agreement.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from core.text import normalized_attribute_text


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
    # 2026-09-28 low-hanging-fruit normalization (measured on the
    # 108,046-row payload, scripts/flip_validity_audit.py companion mining:
    # 2,300 rows / 71 pairs carried a lexicon-missed near-flavor word).
    # Plurals -> singular:
    "apples": "apple", "oranges": "orange", "grapes": "grape",
    "lemons": "lemon", "pears": "pear", "limes": "lime",
    "grapefruits": "grapefruit", "coconuts": "coconut", "tonics": "tonic",
    "coffees": "coffee", "roses": "rose", "mints": "mint",
    "pineapples": "pineapple", "fruits": "fruit",
    # adjective forms
    "fruity": "fruit", "peachy": "peach", "lemony": "lemon",
    # foreign-language variants
    "tonica": "tonic", "lemoni": "lemon", "tamarindo": "tamarind",
    # truncation artifacts seen in feed titles
    "strawberr": "strawberry", "grapefru": "grapefruit", "chery": "cherry",
    "rhubar": "rhubarb", "fruite": "fruit",
    # spelling variants measured in titles (2026-10-01 lexicon audit):
    "litchi": "lychee",
    "cassis": "blackcurrant", "anis": "anise", "aniseed": "anise",
    # EXCLUDED deliberately: pearl~pear (different word, 28 rows measured),
    # sucralose~sucrose-class attribute confusions (distinct values).
}
FLAVOR_LEXICON: frozenset[str] = frozenset(
    {
        "aloe", "angelica", "anise", "blackcurrant", "exotic", "apple", "berry", "boysenberry", "buckthorn",
        "calamansi", "cherry", "chokeberry", "chocolate", "clementine",
        "cloudberry", "coconut", "coffee", "cola", "cranberry",
        "citrus", "elderberry", "elderflower", "fruit",
        "ginger", "grape", "grapefruit", "juniper", "lemon", "lime",
        "lingonberry", "lychee", "mango", "mint", "nectarine", "orange",
        "passion", "passionfruit", "peach", "pear", "pecan", "pineapple",
        "pomegranate", "quince",
        "raspberry", "rhubarb", "rose", "strawberry", "tamarind", "tonic",
        "tropical", "vanilla", "violet", "watermelon",
    }
)
# Additional values observed in explicit Flavour/Flavor declarations. Keep
# these field-bound: the same words can describe ingredients or product types
# elsewhere in the SKU text.
DECLARED_FLAVOR_LEXICON: frozenset[str] = frozenset({
    "acai", "agave", "almond", "amaretto", "apricot", "aranciata",
    "aronia", "artichoke", "avocado", "banana", "barley", "basil",
    "beet", "bergamot", "bilberry", "birch", "blackberry",
    "blackcurrant", "blueberry", "bubble gum", "burdock", "cabbage",
    "cactus", "camellia", "camomile", "cannabis", "cappuccino",
    "caramel", "cardamom", "carrot", "charcoal", "chestnut", "chilli",
    "cinnamon", "cocoa", "creme brulee", "cucumber", "currant",
    "dandelion", "eucalyptus", "fennel", "fig", "garlic", "ginseng",
    "guarana", "guava", "hazelnut", "hibiscus", "honey", "irish cream",
    "jasmine", "kiwi", "latte", "lavender", "lychee", "magnolia", "mandarin",
    "maple", "marshmallow", "melon", "menthol", "mocha", "mulberry",
    "nettle", "noni", "nut", "oat", "olive", "onion", "papaya",
    "pea", "pepper", "peppermint", "plum", "prune", "pumpkin",
    "raisin", "rosehip", "rosemary", "sea salt", "spearmint",
    "tangerine", "tea", "thistle", "thyme", "toffee", "tomato",
    "walnut",
    # Measured declared misses (2026-10-01 lexicon audit):
    "wild berries",
})
DECLARED_FLAVOR_FIELD_RE = re.compile(r"(?:^|;)\s*flavou?r\s*:\s*([^;]*)", re.IGNORECASE)


def extract_flavor_tokens(*values: object) -> frozenset[str]:
    text = normalized_attribute_text(*values)
    return frozenset(
        FLAVOR_ALIASES.get(token, token)
        for token in text.split()
        if FLAVOR_ALIASES.get(token, token) in FLAVOR_LEXICON
    )


def extract_declared_flavor_tokens(*values: object) -> frozenset[str]:
    """Accept reviewed flavor values only when the catalog declares the field."""
    found: set[str] = set()
    for value in values:
        for field in DECLARED_FLAVOR_FIELD_RE.finditer(str(value or "")):
            for part in re.split(r"[,/;&]", field.group(1)):
                candidate = normalized_attribute_text(part)
                if candidate in DECLARED_FLAVOR_LEXICON:
                    found.add(candidate)
    return frozenset(found)


# Explicit negative-sugar surfaces only.  A typo is accepted only in the
# anchored phrase "no dugar"; arbitrary fuzzy matches are not trusted claims.
# The numeric branch excludes "0 sugar added", which is a different claim.
NO_SUGAR_RE = re.compile(
    r"\b(?:no (?:sugars?|dugar)(?!\s+added\b)|zero sugars?(?!\s+added\b)|"
    r"0\s*(?:g|grams?)?\s*sugars?\b(?!\s+added\b)|"
    r"sugar free|sugarfree|sugarless|without sugar(?!\s+added\b)|free of sugar)\b"
)
NO_ADDED_SUGAR_RE = re.compile(
    r"\b(?:(?:no|without|zero|0) added sugars?|(?:no|without|zero|0) sugars? added)\b"
)
SUGAR_CLAIM_RE = re.compile(
    r"\b(?:with added sugar|contains sugar|sweetened with sugar|"
    r"made with sugar(?!\s+free\b)|sweetener sugar|sugar sweetened|"
    r"real sugar|pure sugar)\b"
)
# NOTE (audit 2026-09-28): "low in sugar", "reduced in sugar" and
# "reduced in calories and sugar" are deliberately NOT sugar claims.
# They have dedicated sweetening states (core.sweetener_values:
# low_sugar / reduced_sugar). Mapping a reduction to the positive
# `sugar` class manufactures a both-states contradiction on genuinely
# low-sugar products (e.g. "low in sugar" + a no_sugar declaration),
# and sweetener_conflict() then flags a false positive-vs-diet clash.
# Bare "soda" sells two different things: carbonated drinks AND syrups,
# concentrates, cordials, drink mixes, and powders (169 soda titles, nearly
# all Liquid/Powder Concentrates) plus still-declared drinks (282 titles).
# A bare soda is a carbonation claim only with neither signal present.
_SODA_DRY_PRODUCT_RE = re.compile(
    r"\b(?:syrup|concentrate|cordial|drink mix|powder)\b"
)


def extract_critical_claims(*values: object) -> dict[str, frozenset[str]]:
    """Extract explicit non-numeric critical claims from source text.

    ``no added sugar`` is retained separately: it does not prove that a
    product contains no naturally occurring sugar.  ``diet`` is compatible
    with ``no_sugar`` but conflicts with an explicit ``sugar`` claim.
    """
    text = normalized_attribute_text(*values)

    no_sugar = bool(NO_SUGAR_RE.search(text))
    no_added_sugar = bool(NO_ADDED_SUGAR_RE.search(text))
    sugar = bool(SUGAR_CLAIM_RE.search(text))
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
        re.search(r"\b(?:non carbonated|uncarbonated|not carbonated|without carbonic(?: acid)?|no bubbles?)\b", text)
    )
    carbonation_text = re.sub(
        r"\b(?:non carbonated|uncarbonated|not carbonated|without carbonic(?: acid)?)\b", " ", text
    )
    carbonation_text = re.sub(r"\b(?:baking|washing) soda\b", " ", carbonation_text)
    carbonation: set[str] = set()
    if non_carbonated or re.search(r"\bstill\b", text):
        carbonation.add("still")
    # NOTE (audit 2026-09-28): only the unambiguous "soda pop" is an
    # unconditional carbonation claim. Bare "soda" fires only for
    # beverage-like products: syrups/concentrates/mixes (169 titles) and
    # still-declared drinks (282 titles, e.g. Sunny Delight) are excluded.
    # "still" in the set already covers the non-carbonated branch, since
    # that branch always records still.
    if re.search(r"\b(?:carbonated|sparkling|fizzy|soda pop)\b", carbonation_text):
        carbonation.add("carbonated")
    if (
        re.search(r"\bsoda\b", carbonation_text)
        and "still" not in carbonation
        and not _SODA_DRY_PRODUCT_RE.search(carbonation_text)
    ):
        carbonation.add("carbonated")
    if re.search(r"\beffervescent\b", carbonation_text) and not re.search(
        r"\beffervescent(?:\s+\w+){0,3}\s+(?:tablets?|tabs?)\b", carbonation_text
    ):
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
            r"\b(?:with (?:(?:extra|added|real|aloe vera|fruit) )?pulp|contains pulp|pulp yes|juice and pulp|juice with pulp|juice w pulp|juice e pulp|"
            r"(?:extra|light) pulp|pulp of|pulp aloe vera|(?:aloe vera|aloe|orange|coconut|fruit) pulp|orange juice pulp|concentrates and pulps?)\b",
            text,
        )
    )
    # Bits denotes juice pulp only in an explicit juice context. Smooth
    # alone can describe a smoothie or mouthfeel and is not a pulp claim.
    if re.search(r"\bjuice\b", text):
        with_pulp |= bool(re.search(r"\bwith bits\b", text))
        no_pulp |= bool(re.search(r"\b(?:no bits|without bits)\b", text))
        no_pulp |= bool(re.search(r"\bsmooth(?:\s+\w+){0,3}\s+juice\b", text))
    if no_pulp:
        pulp.add("no_pulp")
    if with_pulp:
        pulp.add("with_pulp")

    # Organic claim (audit 2026-10-01, valio pair): the certification is a
    # product differentiator within brands (the organic sibling of a fruit
    # juice), spelled "organic" in English feeds and "luomu" in Finnish
    # ones. "bio" is deliberately ABSENT: in this corpus it is both the EU
    # organic badge and unrelated parts of brand names, too ambiguous to
    # carry a certification claim alone.
    organic: frozenset[str] = (
        frozenset({"organic"}) if re.search(r"\b(?:organic|luomu)\b", text) else frozenset()
    )

    return {
        "flavor": extract_flavor_tokens(text) | extract_declared_flavor_tokens(*values),
        "carbonation": frozenset(carbonation),
        "sweetener": frozenset(sweetener),
        "pulp": frozenset(pulp),
        "organic": organic,
    }


def extract_description_claims(description: object) -> dict[str, frozenset[str]]:
    """Extract only explicit match-relevant claims from catalog descriptions.

    Flavor is omitted: a long description can mention ingredients that are
    not the product's declared flavor.
    """
    found = extract_critical_claims(str(description or ""))
    return {key: found[key] for key in ("carbonation", "sweetener", "pulp", "organic")}


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
    "DECLARED_FLAVOR_LEXICON",
    "FLAVOR_ALIASES",
    "FLAVOR_LEXICON",
    "categorical_conflict",
    "extract_critical_claims",
    "extract_description_claims",
    "extract_declared_flavor_tokens",
    "extract_flavor_tokens",
    "normalized_attribute_text",
    "sweetener_conflict",
    "volumes_compatible",
]
