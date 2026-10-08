"""Declared sweetener ingredients and sweetening status, independent of sugar claims."""

from __future__ import annotations

import re
from functools import lru_cache

from core.critical_attributes import normalized_attribute_text


SWEETENER_TYPES = frozenset({
    "sugar", "cane sugar", "sucralose", "acesulfame potassium", "stevia",
    "aspartame", "hfcs", "fructose", "erythritol", "cyclamate", "monk fruit",
    "saccharin", "glucose", "sucrose", "corn syrup", "allulose", "neotame",
    "neohesperidin dc",
})
ATTRIBUTE_ITEM_RE = re.compile(r"(?:^|;)\s*([^:;]+):\s*([^;]*)")
SWEETENER_CLAIMS = frozenset({"diet", "no sugar", "no added sugar", "sugar free"})

# (state, pattern source, required substring). Every pattern is
# re.IGNORECASE, so the gate cannot be a bare `in` test on the raw text; the
# third column is instead a substring that is NECESSARY for that pattern to
# match — it names one of the literal words the pattern demands — and it is
# tested against a lowercased copy. Keeping all three in one table is what
# stops the gate from drifting away from the pattern it guards.
#
# Measured: the seven searches cost one full case-insensitive scan each over
# the joined title+attribute+description, while a single str.lower() plus seven
# substring tests rejects almost all of them.
_SWEETENING_SPECS: tuple[tuple[str, str, str], ...] = (
    ("unsweetened", r"\bunsweetened\b", "unsweetened"),
    ("no_sweeteners", r"\b(?:no|without)\s+sweeteners?\b", "sweetener"),
    ("no_artificial_sweeteners", r"\b(?:no|without)\s+artificial\s+sweeteners?\b", "artificial"),
    ("no_added_sweeteners", r"\b(?:no|without)\s+added\s+sweeteners?\b", "added"),
    ("low_sugar", r"\b(?:low|less|light)\s+(?:in\s+)?sugar\b", "sugar"),
    ("reduced_sugar", r"\breduced\s+(?:in\s+)?sugar\b", "sugar"),
    ("sweetened", r"\b(?:lightly\s+)?sweetened\b", "sweetened"),
)
SWEETENING_PATTERNS: dict[str, re.Pattern] = {
    state: re.compile(source, re.I) for state, source, _hint in _SWEETENING_SPECS
}
_SWEETENING_HINTS: dict[str, str] = {
    state: hint for state, _source, hint in _SWEETENING_SPECS
}

_SUGAR_KIND_RE = re.compile(r"\b(?:cane|brown|raw) sugar\b")
_CANE_SUGAR_RE = re.compile(r"\bcane sugar\b")
_WITH_STEVIA_RE = re.compile(r"\bwith stevia\b(?!\s+free\b)")
_DECLARED_ITEM_SPLIT = re.compile(r"[,/&]")
# Hoisted module-level dispatch (see core.critical_attributes): each of these
# ran on every non-empty text, through re's pattern cache.
_INGREDIENT_ALTERNATION = "|".join(
    re.escape(value) for value in sorted(SWEETENER_TYPES, key=len, reverse=True)
)
# "Sweetened with X" / "made with X" / "<drink> with X", where X is a declared
# sweetener and X is not part of an "X free" claim.
_DECLARED_SWEETENER_PHRASE_RE = re.compile(
    r"\b(?:sweetened with|made with|(?:drink|soda|beverage|cola|juice|tea|coffee)\s+with)\s+("
    + _INGREDIENT_ALTERNATION + r")\b(?!\s+free\b)"
)
_FREE_INGREDIENT_ALTERNATION = "|".join(
    re.escape(value) for value in sorted(
        SWEETENER_TYPES - {"sugar", "cane sugar"}, key=len, reverse=True
    )
)
_INGREDIENT_FREE_RE = re.compile(r"\b(" + _FREE_INGREDIENT_ALTERNATION + r")\s+free\b")


def extract_sweetening_status(*values: object) -> set[str]:
    """Return narrowly phrased sweetening states, separate from sugar claims."""
    text = " ".join(str(value or "") for value in values)
    # The gate is applied only to ASCII text: str.lower() reproduces exactly the
    # folding IGNORECASE performs over the ASCII letters these words are made of,
    # while IGNORECASE additionally folds U+017F, U+0130, U+0131 and U+212A.
    # Non-ASCII text takes the unchanged seven-search path.
    low = text.lower() if text.isascii() else None
    found: set[str] = set()
    for state, pattern in SWEETENING_PATTERNS.items():
        if low is not None and _SWEETENING_HINTS[state] not in low:
            continue
        if pattern.search(text):
            found.add(state)
    return found


@lru_cache(maxsize=65536)
def _title_sweetener_types_cached(title: str) -> frozenset[str]:
    text = normalized_attribute_text(title)
    text = _negated_ingredient_pattern().sub(' ', text)
    found: set[str] = set()
    if _SUGAR_KIND_RE.search(text):
        found.add("cane_sugar" if _CANE_SUGAR_RE.search(text) else "sugar")
    for match in _DECLARED_SWEETENER_PHRASE_RE.finditer(text):
        found.add(match.group(1).replace(" ", "_"))
    if _WITH_STEVIA_RE.search(text):
        found.add("stevia")
    return frozenset(found)


def title_sweetener_types(title: str) -> set[str]:
    """Capture explicit ingredient phrases in titles, not bare ingredient words."""
    try:
        return set(_title_sweetener_types_cached(title))
    except TypeError:
        return _title_sweetener_types_uncached(title)


def _title_sweetener_types_uncached(title: str) -> set[str]:
    return set(_title_sweetener_types_cached.__wrapped__(title))


@lru_cache(maxsize=1)
def _negated_ingredient_pattern():
    ingredients = '|'.join(re.escape(value) for value in sorted(SWEETENER_TYPES, key=len, reverse=True))
    return re.compile(r'\b(?:no|without|not made with|not sweetened with)\s+(' + ingredients + r')\b(?!\s+added\b)', re.I)


@lru_cache(maxsize=65536)
def _negated_sweetener_types_cached(values: tuple) -> frozenset[str]:
    return frozenset(_negated_sweetener_types_impl(*values))


def negated_sweetener_types(*values: object) -> set[str]:
    """Explicit absent ingredient claims; silence never asserts absence."""
    try:
        hash(values)
    except TypeError:
        return _negated_sweetener_types_impl(*values)
    return set(_negated_sweetener_types_cached(values))


def _negated_sweetener_types_impl(*values: object) -> set[str]:
    text = normalized_attribute_text(*values)
    found = {match.group(1).replace(' ', '_') for match in _negated_ingredient_pattern().finditer(text)}
    # General sugar-free wording belongs to sugar-status claims. Named
    # ingredient-free wording is explicit ingredient absence.
    found.update(match.group(1).replace(" ", "_")
                 for match in _INGREDIENT_FREE_RE.finditer(text))
    return found


def declared_sweeteners(attributes: str) -> dict[str, set[str]]:
    """Parse explicit field values without equating an ingredient with a claim.

    Contradictory source declarations are retained and flagged. Missing and
    unrecognized values remain explicit; neither is inferred from a title.
    """
    cell = str(attributes or "")
    types: set[str] = set()
    states: set[str] = set()
    unknown: set[str] = set()
    # The loop only ever reports a non-empty result through a segment whose key
    # normalizes to "sweetener", and normalized_attribute_text folds case and
    # punctuation without deleting letters, so an ASCII cell that does not
    # contain "sweetener" at all cannot contribute anything. Skipping the walk
    # avoids folding the key of every segment of every such cell (the same
    # repeated-fold shape that paid off in critical_attributes._field_tokens).
    if cell.isascii() and "sweetener" not in cell.lower():
        return {
            "sweetener_type": types,
            "sweetening": states,
            "unmapped": unknown,
            "consistency_flags": set(),
        }
    for item in ATTRIBUTE_ITEM_RE.finditer(cell):
        if normalized_attribute_text(item.group(1)) != "sweetener":
            continue
        for part in _DECLARED_ITEM_SPLIT.split(item.group(2)):
            value = normalized_attribute_text(part)
            if value in SWEETENER_TYPES:
                types.add(value.replace(" ", "_"))
            elif value == "unsweetened":
                states.add("unsweetened")
            elif value and value not in SWEETENER_CLAIMS:
                unknown.add(value)
    return {
        "sweetener_type": types,
        "sweetening": states,
        "unmapped": unknown,
        "consistency_flags": (
            {"unsweetened_with_declared_sweetener"} if states and types else set()
        ),
    }
