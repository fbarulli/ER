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

SWEETENING_PATTERNS = {
    "unsweetened": re.compile(r"\bunsweetened\b", re.I),
    "no_sweeteners": re.compile(r"\b(?:no|without)\s+sweeteners?\b", re.I),
    "no_artificial_sweeteners": re.compile(r"\b(?:no|without)\s+artificial\s+sweeteners?\b", re.I),
    "no_added_sweeteners": re.compile(r"\b(?:no|without)\s+added\s+sweeteners?\b", re.I),
    "low_sugar": re.compile(r"\b(?:low|less|light)\s+(?:in\s+)?sugar\b", re.I),
    "reduced_sugar": re.compile(r"\breduced\s+(?:in\s+)?sugar\b", re.I),
    "sweetened": re.compile(r"\b(?:lightly\s+)?sweetened\b", re.I),
}


def extract_sweetening_status(*values: object) -> set[str]:
    """Return narrowly phrased sweetening states, separate from sugar claims."""
    text = " ".join(str(value or "") for value in values)
    return {state for state, pattern in SWEETENING_PATTERNS.items() if pattern.search(text)}


def title_sweetener_types(title: str) -> set[str]:
    """Capture explicit ingredient phrases in titles, not bare ingredient words."""
    text = normalized_attribute_text(title)
    text = _negated_ingredient_pattern().sub(' ', text)
    found: set[str] = set()
    if re.search(r"\b(?:cane|brown|raw) sugar\b", text):
        found.add("cane_sugar" if re.search(r"\bcane sugar\b", text) else "sugar")
    ingredients = "|".join(re.escape(value) for value in sorted(SWEETENER_TYPES, key=len, reverse=True))
    for match in re.finditer(
        r"\b(?:sweetened with|made with|(?:drink|soda|beverage|cola|juice|tea|coffee)\s+with)\s+("
        + ingredients + r")\b(?!\s+free\b)",
        text,
    ):
        found.add(match.group(1).replace(" ", "_"))
    if re.search(r"\bwith stevia\b(?!\s+free\b)", text):
        found.add("stevia")
    return found


@lru_cache(maxsize=1)
def _negated_ingredient_pattern():
    ingredients = '|'.join(re.escape(value) for value in sorted(SWEETENER_TYPES, key=len, reverse=True))
    return re.compile(r'\b(?:no|without|not made with|not sweetened with)\s+(' + ingredients + r')\b(?!\s+added\b)', re.I)


def negated_sweetener_types(*values: object) -> set[str]:
    """Explicit absent ingredient claims; silence never asserts absence."""
    text = normalized_attribute_text(*values)
    found = {match.group(1).replace(' ', '_') for match in _negated_ingredient_pattern().finditer(text)}
    # General sugar-free wording belongs to sugar-status claims. Named
    # ingredient-free wording is explicit ingredient absence.
    ingredients = "|".join(re.escape(value) for value in sorted(
        SWEETENER_TYPES - {"sugar", "cane sugar"}, key=len, reverse=True
    ))
    found.update(match.group(1).replace(" ", "_") for match in re.finditer(
        r"\b(" + ingredients + r")\s+free\b", text
    ))
    return found


def declared_sweeteners(attributes: str) -> dict[str, set[str]]:
    """Parse explicit field values without equating an ingredient with a claim.

    Contradictory source declarations are retained and flagged. Missing and
    unrecognized values remain explicit; neither is inferred from a title.
    """
    types: set[str] = set()
    states: set[str] = set()
    unknown: set[str] = set()
    for item in ATTRIBUTE_ITEM_RE.finditer(str(attributes or "")):
        if normalized_attribute_text(item.group(1)) != "sweetener":
            continue
        for part in re.split(r"[,/&]", item.group(2)):
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
