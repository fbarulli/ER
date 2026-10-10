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
    from core.project_root import ProjectRoot

    root = ProjectRoot.find(Path(__file__).resolve())
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
# Precomputed token -> emitted-canonical dispatch table.  Reproduces the exact
# predicate "FLAVOR_ALIASES.get(token, token) in FLAVOR_LEXICON" (which used to
# call .get TWICE per whitespace token, ~194k calls per 10k cohort) in a single
# dict lookup.  A token that is an alias key whose value is NOT in the lexicon
# is deliberately absent (the alias wins over the raw token), and a lexicon
# token that is not an alias maps to itself.
_FLAVOR_TOKEN_MAP: dict[str, str] = {
    token: canonical
    for token, canonical in FLAVOR_ALIASES.items()
    if canonical in FLAVOR_LEXICON
}
_FLAVOR_TOKEN_MAP.update(
    {token: token for token in FLAVOR_LEXICON if token not in FLAVOR_ALIASES}
)
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

# ── hoisted module-level dispatch ──────────────────────────────────────────
# Every `re.search(pattern, text)` below is a call into re's pattern cache plus
# a wrapper; in the lane profile that dispatch alone was 1.9M `_compile` calls
# (1.07s self) across the tree. Compiled once here, called as a method.
_CAFFEINE_BAND = re.compile(r"\s*(\d+)")
# (phrase, boundary-anchored pattern) pairs. The boundary pattern is only ever
# needed once the phrase is known to occur at all: `(?<!\w)phrase(?!\w)` can
# only match text that CONTAINS the phrase verbatim, so a plain `in` test is an
# exact pre-filter. Measured over 4,000 lane rows: the 17 regex searches cost
# 0.3873s against 0.0148s for the 17 `in` tests, i.e. the search drops to
# 0.0149s when it runs only for phrases already known to be present.
_MADE_FROM_PHRASE_RES: tuple[tuple[str, re.Pattern], ...] = tuple(
    (phrase, re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"))
    for phrase in MADE_FROM_PHRASES
)
# `\b`/lookaround-free splitter for the declared-flavor field value; `re.escape`
# and the cache lookup used to run per row.
_DECLARED_FLAVOR_SPLIT = re.compile(r"[,/;&]")
# DECLARED_FLAVOR_FIELD_RE can only match at position 0 or just after a ";",
# so `";" in value` plus one anchored test at position 0 proves that no field
# can be present. Measured on the lane corpus: 76% of attribute cells and 98%
# of titles carry no flavor field at all, and every one of them was paying a
# full finditer scan.
_LEADING_DECLARED_FLAVOR_FIELD = re.compile(r"\s*flavou?r\s*:", re.IGNORECASE)

_DIET_RE = re.compile(r"\bdiet\b")
_NON_CARBONATED_RE = re.compile(
    r"\b(?:non carbonated|uncarbonated|not carbonated|without carbonic(?: acid)?|no bubbles?)\b")
# The carbonation scrub keeps "no bubbles?" in the boolean but not in the
# substitution (the original built two different patterns from one literal).
_NON_CARBONATED_SCRUB = re.compile(
    r"\b(?:non carbonated|uncarbonated|not carbonated|without carbonic(?: acid)?)\b")
_SODA_WORD_RE = re.compile(r"\b(?:baking|washing) soda\b")
_STILL_RE = re.compile(r"\bstill\b")
_CARBONATED_RE = re.compile(r"\b(?:carbonated|sparkling|fizzy|soda pop)\b")
_BARE_SODA_RE = re.compile(r"\bsoda\b")
_EFFERVESCENT_RE = re.compile(r"\beffervescent\b")
_EFFERVESCENT_TABLET_RE = re.compile(r"\beffervescent(?:\s+\w+){0,3}\s+(?:tablets?|tabs?)\b")
_NO_PULP_RE = re.compile(r"\b(?:no pulp|without pulp|pulp free|free of pulp)\b")
_WITH_PULP_RE = re.compile(
    r"\b(?:with (?:(?:extra|added|real|aloe vera|fruit) )?pulp|contains pulp|pulp yes|juice and pulp|juice with pulp|juice w pulp|juice e pulp|"
    r"(?:extra|light) pulp|pulp of|pulp aloe vera|(?:aloe vera|aloe|orange|coconut|fruit) pulp|orange juice pulp|concentrates and pulps?)\b")
_JUICE_RE = re.compile(r"\bjuice\b")
_WITH_BITS_RE = re.compile(r"\bwith bits\b")
_NO_BITS_RE = re.compile(r"\b(?:no bits|without bits)\b")
_SMOOTH_JUICE_RE = re.compile(r"\bsmooth(?:\s+\w+){0,3}\s+juice\b")
_ORGANIC_RE = re.compile(r"\b(?:organic|luomu)\b")


@lru_cache(maxsize=4096)
def _normalized_field_key(key: object) -> str:
    """`Key:` spelling folded once per process, not once per cell segment.

    `_field_tokens` asked for four keys of the SAME cell on every row, and each
    request re-folded the key through normalized_attribute_text — a full
    casefold + NFKD + join. The four keys are module constants in practice and
    the fold is deterministic, so one fold per distinct key is enough.
    """
    return normalized_attribute_text(key)


def _field_tokens(attribute: object, key: str) -> frozenset[str]:
    """Lowercased comma-split tokens of one `Key:` field in the attribute cell."""
    from core.text import attribute_field_value

    return frozenset(attribute_field_value(attribute, _normalized_field_key(key)))


def _field_tokens_many(attribute: object, keys: tuple[str, ...]) -> dict[str, frozenset[str]]:
    """The same tokens for several keys of ONE cell, in a single walk.

    `source_consistency_flags` reads four fields (free from, health claims, no
    artificial ingredients, caffeine) out of one attribute cell, and every
    `_field_tokens` call re-walked the whole cell through
    core.text.attribute_fields — which re-folds the key of EVERY segment. Four
    walks of the same string bought nothing: this is the identical token list
    per key (same `;` segments in the same document order, same comma split,
    same strip+lower), computed once.
    """
    from core.text import attribute_fields

    wanted = {_normalized_field_key(key): key for key in keys}
    cell = str(attribute or "")
    # A key survives the walk only if some segment's key folds to it, and for
    # ASCII text that fold (unicode_casefold then `[^a-z0-9]+` -> " ") maps
    # punctuation to a SPACE and never deletes a letter, so every word of the
    # key must occur in the cell as a substring. Cells that mention none of the
    # wanted fields therefore skip the walk completely: no segment is folded and
    # no key is compared. Guarded to ASCII because NFKD expands some non-ASCII
    # characters (ligature U+FB00 -> "ff"); non-ASCII cells take the old path.
    if cell.isascii():
        lowered = cell.lower()
        wanted = {name: key for name, key in wanted.items()
                  if all(word in lowered for word in name.split())}
        if not wanted:
            return {key: frozenset() for key in keys}
    collected: dict[str, list[str]] = {key: [] for key in keys}
    for name, raw_value in attribute_fields(attribute):
        key = wanted.get(name)
        if key is None:
            continue
        collected[key].extend(
            token.strip().lower() for token in raw_value.split(",") if token.strip()
        )
    return {key: frozenset(tokens) for key, tokens in collected.items()}


def _caffeine_positive(values: frozenset[str]) -> bool:
    """True when a caffeine band's lower bound is > 0 ("0-15 mg" is trace)."""
    for value in values:
        match = _CAFFEINE_BAND.match(value)
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
    key = _normalized_field_key(key)
    return ";".join(
        part
        for part in str(attribute or "").split(";")
        if not (":" in part and _normalized_field_key(part.split(":", 1)[0]) == key)
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
    fields = _field_tokens_many(attribute, ("free from", "health claims",
                                            "no artificial ingredients", "caffeine"))
    free_from = fields["free from"]
    claims = fields["health claims"]
    no_artificial = fields["no artificial ingredients"]
    caffeine = fields["caffeine"]
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
        phrase for phrase, pattern in _MADE_FROM_PHRASE_RES
        if phrase in text and pattern.search(text)
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
        _FLAVOR_TOKEN_MAP[token]
        for token in text.split()
        if token in _FLAVOR_TOKEN_MAP
    )


def _extract_declared_flavor_tokens_impl(*values: object) -> frozenset[str]:
    found: set[str] = set()
    for value in values:
        raw = str(value or "")
        if ";" not in raw and not _LEADING_DECLARED_FLAVOR_FIELD.match(raw):
            continue
        for field in DECLARED_FLAVOR_FIELD_RE.finditer(raw):
            for part in _DECLARED_FLAVOR_SPLIT.split(field.group(1)):
                candidate = normalized_attribute_text(part)
                if candidate in DECLARED_FLAVOR_LEXICON:
                    found.add(candidate)
    return frozenset(found)


@lru_cache(maxsize=131072)
def _extract_declared_flavor_tokens_cached(values: tuple[object, ...]) -> frozenset[str]:
    return _extract_declared_flavor_tokens_impl(*values)


def extract_declared_flavor_tokens(*values: object) -> frozenset[str]:
    """Accept reviewed flavor values only when the catalog declares the field.

    Memoized: the corpus repeats the same attribute cell across endpoints,
    variants and identity parsing, and the function is pure (it returns an
    immutable frozenset, so a cached result is safe to share).  Non-hashable
    arguments fall back to the uncached path unchanged.
    """
    try:
        key = tuple(values)
        return _extract_declared_flavor_tokens_cached(key)
    except TypeError:
        return _extract_declared_flavor_tokens_impl(*values)


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
_EFFERVESCENT_RE = re.compile(r"\beffervescent\b")
_EFFERVESCENT_TABLET_RE = re.compile(
    r"\beffervescent(?:\s+\w+){0,3}\s+(?:tablets?|tabs?)\b"
)
_NO_PULP_RE = re.compile(r"\b(?:no pulp|without pulp|pulp free|free of pulp)\b")
_WITH_PULP_RE = re.compile(
    r"\b(?:with (?:(?:extra|added|real|aloe vera|fruit) )?pulp|contains pulp|pulp yes|juice and pulp|juice with pulp|juice w pulp|juice e pulp|"
    r"(?:extra|light) pulp|pulp of|pulp aloe vera|(?:aloe vera|aloe|orange|coconut|fruit) pulp|orange juice pulp|concentrates and pulps?)\b"
)
_JUICE_RE = re.compile(r"\bjuice\b")
_WITH_BITS_RE = re.compile(r"\bwith bits\b")
_NO_BITS_RE = re.compile(r"\b(?:no bits|without bits)\b")
_SMOOTH_JUICE_RE = re.compile(r"\bsmooth(?:\s+\w+){0,3}\s+juice\b")
_ORGANIC_RE = re.compile(r"\b(?:organic|luomu)\b")


def extract_critical_claims(*values: object) -> dict[str, frozenset[str]]:
    """Extract explicit non-numeric critical claims from source text.

    Memoized over ``values``: every check runs at most once per distinct input
    tuple, and the result is returned as a FRESH dict each call (the pipeline
    mutates the dict it receives, so the cached dict itself must never leak).
    The values are frozensets, so sharing them is safe.  Non-hashable
    arguments fall back to the uncached path with identical semantics.
    """
    try:
        key = tuple(values)
        cached = _extract_critical_claims_cached(key)
    except TypeError:
        return _extract_critical_claims_impl(*values)
    return dict(cached)


@lru_cache(maxsize=131072)
def _extract_critical_claims_cached(values: tuple[object, ...]) -> dict[str, frozenset[str]]:
    return _extract_critical_claims_impl(*values)


@lru_cache(maxsize=131072)
def _non_flavor_claims_from_text(text: str) -> dict[str, frozenset[str]]:
    """The four non-flavor dimensions, keyed by ALREADY normalized text.

    Two raw value tuples can fold to the same text (a trailing mode_flavor
    column, a different column split), and the description lane and the
    title+attribute lane both consume this text.  Caching these ~15 pure
    regex scans on the folded text shares them across every such caller; the
    returned frozensets are immutable, so the cached dict can be copied out.
    The literal `in` gates below are the r17 pre-filters, so even a cold
    cache miss skips every scan whose required literal is absent.
    """
    # Every sugar/diet pattern requires the literal "sugar", the accepted
    # "dugar" typo, or "diet", so two `in` tests prove the whole block is a
    # no-op. Measured: the four searches cost 0.0889s per 4,000 rows against
    # 0.0019s for the gate.
    sweetener: set[str] = set()
    if "sugar" in text or "dugar" in text or "diet" in text:
        if NO_SUGAR_RE.search(text):
            sweetener.add("no_sugar")
        if NO_ADDED_SUGAR_RE.search(text):
            sweetener.add("no_added_sugar")
        if SUGAR_CLAIM_RE.search(text):
            sweetener.add("sugar")
        if _DIET_RE.search(text):
            sweetener.add("diet")

    carbonation: set[str] = set()
    if "still" in text and _STILL_RE.search(text):
        carbonation.add("still")
    # Remove explicit negative phrases before looking for positive
    # carbonation so "non-carbonated" cannot emit both states. The scrub and
    # the three positive checks are all literal, and every literal they need is
    # in this gate, so a gate miss means the block cannot add a state: the
    # scrub (two `sub` calls allocating a rewritten string) is now only paid
    # when it can change the answer.
    if ("carbonat" in text or "carbonic" in text or "bubbl" in text
            or "soda" in text or "sparkl" in text or "fizzy" in text
            or "syrup" in text or "concentrat" in text or "cordial" in text
            or "drink mix" in text or "powder" in text or "effervescent" in text):
        non_carbonated = bool(_NON_CARBONATED_RE.search(text))
        carbonation_text = _NON_CARBONATED_SCRUB.sub(" ", text)
        carbonation_text = _SODA_WORD_RE.sub(" ", carbonation_text)
        if non_carbonated:
            carbonation.add("still")
        # NOTE (audit 2026-09-28): only the unambiguous "soda pop" is an
        # unconditional carbonation claim. Bare "soda" fires only for
        # beverage-like products: syrups/concentrates/mixes (169 titles) and
        # still-declared drinks (282 titles, e.g. Sunny Delight) are excluded.
        # "still" in the set already covers the non-carbonated branch, since
        # that branch always records still.
        if _CARBONATED_RE.search(carbonation_text):
            carbonation.add("carbonated")
        if (
            _BARE_SODA_RE.search(carbonation_text)
            and "still" not in carbonation
            and not _SODA_DRY_PRODUCT_RE.search(carbonation_text)
        ):
            carbonation.add("carbonated")
        if _EFFERVESCENT_RE.search(carbonation_text) and not _EFFERVESCENT_TABLET_RE.search(
            carbonation_text
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
    # Both pulp patterns name "pulp" literally, so one `in` test replaces two
    # scans when a product never mentions it.
    if "pulp" in text:
        no_pulp = bool(_NO_PULP_RE.search(text))
        with_pulp = bool(_WITH_PULP_RE.search(text))
    else:
        no_pulp = with_pulp = False
    # Bits denotes juice pulp only in an explicit juice context. Smooth
    # alone can describe a smoothie or mouthfeel and is not a pulp claim.
    if "juice" in text and _JUICE_RE.search(text):
        if "bits" in text:
            with_pulp |= bool(_WITH_BITS_RE.search(text))
            no_pulp |= bool(_NO_BITS_RE.search(text))
        if "smooth" in text:
            no_pulp |= bool(_SMOOTH_JUICE_RE.search(text))
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
        frozenset({"organic"})
        if ("organic" in text or "luomu" in text) and _ORGANIC_RE.search(text)
        else frozenset()
    )

    return {
        "carbonation": frozenset(carbonation),
        "sweetener": frozenset(sweetener),
        "pulp": frozenset(pulp),
        "organic": organic,
    }


def _extract_critical_claims_impl(
    *values: object, _with_flavor: bool = True
) -> dict[str, frozenset[str]]:
    """Extract explicit non-numeric critical claims from source text.

    ``no added sugar`` is retained separately: it does not prove that a
    product contains no naturally occurring sugar.  ``diet`` is compatible
    with ``no_sugar`` but conflicts with an explicit ``sugar`` claim.
    """
    text = normalized_attribute_text(*values)
    base = _non_flavor_claims_from_text(text)
    if _with_flavor:
        # Key order is preserved exactly (flavor first) for callers that
        # serialize the mapping; the non-flavor four follow from the cache.
        return {
            "flavor": flavor_tokens_from_text(text)
            | extract_declared_flavor_tokens(*values),
            **base,
        }
    return dict(base)


@lru_cache(maxsize=131072)
def _extract_description_claims_cached(description: str) -> dict[str, frozenset[str]]:
    return _extract_critical_claims_impl(description, _with_flavor=False)


def extract_description_claims(description: object) -> dict[str, frozenset[str]]:
    """Extract only explicit match-relevant claims from catalog descriptions.

    Flavor is omitted: a long description can mention ingredients that are
    not the product's declared flavor.  The flavor branch is therefore never
    run here (it used to be computed and discarded), and the result is
    memoized on the description text; both are pure and re-returned as a
    fresh dict.
    """
    text = str(description or "")
    found = _extract_description_claims_cached(text)
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
