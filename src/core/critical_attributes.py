"""Shared critical product-attribute vocabulary and compatibility rules.

The model text lane, canonical records, hard-negative miners, and inference
gates all import this module.  Explicit evidence can agree or conflict;
absence is kept as unknown and is never converted into agreement.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

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

# ── vocabulary (config-owned SSOT) ─────────────────────────────────────────
# Every attribute vocabulary below is DATA, not code: one block in
# config/vocabulary.json (`attribute_vocabulary`) owns them, next to the
# STOPWORDS/CONCEPT_FOLDS/brand_aliases that core.common validates. They are
# read here WITHOUT importing core.common — this module is loaded while
# core.common is still importing (common -> schemas -> ... ->
# attribute_conflicts -> here), so a top-level core.common import would be
# circular. Root discovery goes through core.project_root (a leaf with zero
# core imports), so the env override EUROMONITOR_PROJECT_ROOT is honored
# EXACTLY as core.common honors it — one root, one vocabulary, everywhere.
@lru_cache(maxsize=1)
def _attribute_vocabulary() -> dict:
    from core.project_root import find_project_root

    root = find_project_root(Path(__file__).resolve())
    data = json.loads((root / "config" / "vocabulary.json").read_text(encoding="utf-8"))
    from core.attribute_vocabulary import validated_attribute_vocabulary

    return validated_attribute_vocabulary(data)


_VOCAB = _attribute_vocabulary()
# Variant -> canonical flavor. Includes plural/adjective/foreign/truncation
# aliases, and "-ade" drink words carrying their base fruit (a "lemonade" is
# lemon-flavored; whole-token only, so "made"/"trade"/"gatorade"/"bionade"
# never fire). Measured 2026-10-03: 1,988 title rows gain their base flavor,
# 0 of 855 golden records change.
FLAVOR_ALIASES: dict[str, str] = {
    str(key): str(value) for key, value in (_VOCAB.get("flavor_aliases") or {}).items()
}
FLAVOR_LEXICON: frozenset[str] = frozenset(_VOCAB.get("flavor_lexicon") or ())
# Field-bound: only honored inside an explicit Flavour/Flavor declaration.
DECLARED_FLAVOR_LEXICON: frozenset[str] = frozenset(
    _VOCAB.get("declared_flavor_lexicon") or ()
)
# "Made From" base-ingredient vocabulary, measured from the corpus's declared
# `Made From:` field (116 distinct values, top-80 = 99.6%). Explicit closed
# list so title extraction cannot invent ingredients; multi-word values match
# as phrases.
MADE_FROM_LEXICON: frozenset[str] = frozenset(_VOCAB.get("made_from_lexicon") or ())
MADE_FROM_PHRASES: tuple[str, ...] = tuple(_VOCAB.get("made_from_phrases") or ())
# Words/phrases that legitimately carry caffeine. A positive caffeine band
# declared on a product whose TITLE names none of these is source pollution.
CAFFEINE_SOURCES: tuple[str, ...] = tuple(_VOCAB.get("caffeine_sources") or ())
# Sugar-as-ingredient vocabulary (the sweetener_type channel). A "no sugar"
# claim beside one of these is an internal source contradiction.
SUGAR_INGREDIENTS: frozenset[str] = frozenset(_VOCAB.get("sugar_ingredients") or ())
DECLARED_FLAVOR_FIELD_RE = re.compile(r"(?:^|;)\s*flavou?r\s*:\s*([^;]*)", re.IGNORECASE)


def _field_tokens(attribute: object, key: str) -> frozenset[str]:
    """Lowercased comma-split tokens of one `Key:` field in the attribute cell."""
    from core.text import attribute_field_value

    key = normalized_attribute_text(key)
    return frozenset(attribute_field_value(attribute, key))


def _caffeine_positive(values: frozenset[str]) -> bool:
    """True when a caffeine band's lower bound is > 0 ("0-15 mg" is trace)."""
    for value in values:
        match = re.match(r"\s*(\d+)", value)
        if match and int(match.group(1)) > 0:
            return True
    return False


def _has_caffeine_source(*texts: object) -> bool:
    text = normalized_attribute_text(*texts)
    tokens = set(text.split())
    return any(
        (source in text) if " " in source else (source in tokens)
        for source in CAFFEINE_SOURCES
    )


def _without_field(attribute: object, key: str) -> str:
    """Attribute cell minus one `Key:` field (the `Caffeine:` key itself would
    otherwise always satisfy a caffeine-source search).

    Keeps its own split(';') walk because it reconstructs whole segments —
    including colon-less ones, which core.text.attribute_fields skips by
    contract. The KEY normalization is the shared semantics; the segment
    reconstruction is not expressible over (key, value) pairs.
    """
    key = normalized_attribute_text(key)
    return ";".join(
        part
        for part in str(attribute or "").split(";")
        if not (":" in part and normalized_attribute_text(part.split(":", 1)[0]) == key)
    )


def source_consistency_flags(
    attribute: object, title: object, sweetener_type: frozenset[str] | set[str]
) -> frozenset[str]:
    """Internal source contradictions + implausible declarations.

    Measured 2026-10-03: the extractor is faithful, so these are SOURCE defects
    (a "no sugar" claim beside cane sugar; a caffeine band on a juice). Flagged
    for review, never silently dropped or "corrected" (review-not-guess).
    """
    flags: set[str] = set()
    free_from = _field_tokens(attribute, "free from")
    claims = _field_tokens(attribute, "health claims")
    no_artificial = _field_tokens(attribute, "no artificial ingredients")
    caffeine = _field_tokens(attribute, "caffeine")
    caff_pos = _caffeine_positive(caffeine)
    sweeteners = set(sweetener_type)
    if caff_pos and "no caffeine" in free_from:
        flags.add("caffeine_source_conflict")
    if caff_pos and not _has_caffeine_source(title, _without_field(attribute, "caffeine")):
        flags.add("caffeine_without_source")
    if (sweeteners & SUGAR_INGREDIENTS) and "no sugar" in claims:
        flags.add("no_sugar_with_sugar")
    if "aspartame" in sweeteners and "no aspartame" in no_artificial:
        flags.add("no_aspartame_with_aspartame")
    return frozenset(flags)


def extract_made_from_tokens(*values: object) -> frozenset[str]:
    """Base-ingredient evidence from any text columns (title + attribute).

    Whole-token (single words) and phrase (multi-word) matches against the
    measured MADE_FROM_LEXICON. Deliberately title+attribute aware so a
    listing whose title says "turmeric" is captured even when the declared
    `Made From:` field omits it.
    """
    text = normalized_attribute_text(*values)
    found = {token for token in text.split() if token in MADE_FROM_LEXICON}
    found.update(
        phrase for phrase in MADE_FROM_PHRASES
        if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text)
    )
    return frozenset(found)


def extract_flavor_tokens(*values: object) -> frozenset[str]:
    return flavor_tokens_from_text(normalized_attribute_text(*values))


def flavor_tokens_from_text(text: str) -> frozenset[str]:
    """Flavor tokens from ALREADY normalized text.

    normalized_attribute_text is idempotent over its own output alphabet, so a
    caller that already folded the text (extract_critical_claims does) must not
    pay to fold it again — that re-fold was 116k wasted calls over full titles.
    """
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
        "flavor": flavor_tokens_from_text(text) | extract_declared_flavor_tokens(*values),
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
    "flavor_tokens_from_text",
    "normalized_attribute_text",
    "sweetener_conflict",
    "volumes_compatible",
]
