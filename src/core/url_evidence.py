"""core.url_evidence — product text carried by a listing URL.

WHY (owner ruling 2026-10-01). Every one of the 13 mapped columns is now
captured per title (config/paths.yaml `column_evidence`). This module is the
READER for the two URL columns, because a URL is not inert text: the path slug
is the retailer's own slugification of the product name. Measured on
dataset.csv:

    walmart.com/ip/Concord-Foods-Smoothie-Banana-Drink-Mixes-2-oz-Shelf-Stable
    riteaid.com/shop/sierra-mist-soda-lemon-lime-12-12-fl-oz-355-ml-cans-144-fl-oz

An earlier ruling excluded `url` and `image_url` as "listing identifiers with
no product semantics". That was wrong, and it was wrong because nobody read a
value before ruling on it — the slug carries flavour, volume and pack tokens
verbatim.

WHAT IT EXTRACTS, and what it refuses to. A URL is a mix of product text and
retailer scaffolding, so the normalizer must be as careful about what it drops
as about what it keeps:

  * scaffolding dropped — `ip`, `shop`, `media`, `catalog`, `product`,
    `cache`, `small_image`, `seo`, and the rest of PATH_SCHEMA_WORDS;
  * article numbers dropped — retailer ids like `17619697`; numbers inside
    explicit size/count spans survive because they describe the product;
  * hashes dropped — `9df78eab33525d08d6e5fb8d27136e95`, `d6bc7f7c`, `K6RMM`.
    This is why image_url is only PARTIALLY evidentiary: walmart's
    `seo/Concord-Foods-Smoothie-Banana-Drink-Mix-2-oz_d6bc7f7c` keeps its
    tokens, riteaid's `cache/1/small_image/220x/9df78eab.../0` is all
    scaffolding and hash. Measured: 5.6% of URLs clean to nothing.

A hash surviving into the token stream would be worse than dropping the column:
it becomes a confident-looking token that matches nothing and matches
everything. So the normalizer returns "" rather than noise, and callers must
treat "" as "no evidence", never as "no attributes".
"""

from __future__ import annotations

import re
from functools import lru_cache

from core.text import PACK_RE, extract_volume_evidence, normalize_text

__all__ = ["PATH_SCHEMA_WORDS", "UNITS", "is_evidentiary", "url_text"]

_SPEC_VIEW = None


def _spec():
    """The config-owned vocabulary + thresholds (config/paths.yaml url_evidence).

    Previously two module-level frozensets, PATH_SCHEMA_WORDS and _UNITS, held
    this vocabulary in code while config/paths.yaml held a copy of the same 85
    words — two sources of truth for one list, which is the duplication the
    SSOT rule exists to prevent. The config is now the ONLY copy; these two
    names are derived views of it, kept as module attributes because callers
    (and the regression tests) refer to them by name.

    The resolved view is memoized: this module already freezes the same view
    into PATH_SCHEMA_WORDS/UNITS at import (so it already treats the config as
    immutable for the life of the process), while _is_noise called the
    accessor once per token — 124,407 times in the lane profile, each paying
    an `import` statement plus a function call to re-read a constant.
    """
    spec = _SPEC_VIEW
    if spec is None:
        from core.common import data_cfg

        spec = data_cfg().url_evidence
        globals()["_SPEC_VIEW"] = spec
    return spec


# Derived, not authored: config/paths.yaml `url_evidence` is the single source.
PATH_SCHEMA_WORDS: frozenset[str] = frozenset(_spec().path_schema_words)
UNITS: frozenset[str] = frozenset(_spec().units)
# Longest-first, computed once. Equal-length units cannot both be a suffix of
# one token, so the (hash-order-dependent) tie order cannot matter.
_UNITS_BY_LENGTH: tuple[str, ...] = tuple(sorted(UNITS, key=len, reverse=True))

_EXTENSION = re.compile(r"\.(jpe?g|png|gif|webp|svg|html?|php|aspx|jsp)$", re.I)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
# A UUID/hash must be removed BEFORE hyphens become spaces, otherwise each of
# its groups survives as a bogus "word". Two details, both found by reading the
# function's OUTPUT rather than by reasoning about the pattern:
#   * the tail is `{6,}`, not a strict uuid's 12 — walmart's seo slugs carry an
#     8-4-4-4-8 hash (…-b86f-2de2f2d3) which a strict 12-char tail misses;
#   * there is NO leading \b — the slug arrives as "oz_d6bc7f7c-…", and \b does
#     not match between "_" and "d" because both are word characters, so a
#     \b-guarded pattern silently NEVER fires on the real data.
# 8-4-4-4-6 of hex cannot be a product phrase.
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{6,}", re.I
)
# A bare "NNNx" is an image-dimension spec from a media path
# (.../small_image/220x/…). Real pack notation carries its unit — "12x355ml"
# — so it survives the numeric-noise and dimension rules intact.
_IMAGE_DIM = re.compile(r"^\d+x$")
# The two "this token is a bare quantity" shapes _is_noise tests, as ONE
# compiled pattern: \d+(\.\d+)? and the media-dimension \d+x. Kept separate
# from _NUMERIC (below) because url_text's token filter must NOT treat "220x"
# as a quantity.
_QUANTITY_OR_DIM = re.compile(r"\d+(?:\.\d+)?|\d+x")
# url_text's numeric-token test: a bare decimal is product evidence ONLY inside
# a recognized quantity span, so it must stay narrower than _QUANTITY_OR_DIM.
_NUMERIC = re.compile(r"\d+(?:\.\d+)?")
# The same numeric shape, anchored by callers, for _has_unit_suffix.
_NUMERIC_FULL = re.compile(r"(?:\d+(?:\.\d+)?|\.\d+)")
_SLUG_SEPARATOR = re.compile(r"[-_]")
_SLUG_SEPARATOR_RUN = re.compile(r"[-_+]+")
_FL_OZ = re.compile(r"\bfl-oz\b", re.I)
_NON_SPACE = re.compile(r"\S+")
_CASE_OF = re.compile(r"\bcases?\s+of\s+\d+\b", re.I)
_IS_DIGIT = str.isdigit
# PACK_RE (core.text) can only match when one of its literals occurs in the
# slug, so this is a sound pre-filter, not a heuristic: "packet"/"packets" are
# covered by "pack" and "bottles" by "bottle". Measured on the lane corpus it
# skips the scan for 82% of slugs and cuts that scan's cost 2.9x.
# Observed slug decimals (zeroh 0-8l; Sierra 7-5oz) must survive number
# stripping. Require the complete configured volume-unit suffix before
# reconstructing punctuation; this is sanitation, not a conversion parser.
@lru_cache(maxsize=1)
def _slug_decimal_pattern():
    # A short fractional tail followed by an explicit unit is a slug decimal
    # (33-8-oz, 16-9-fl-oz). Three-digit tails (24-500ml) remain count lists.
    from core.text import _unit_spec

    units = '|'.join(entry.pattern for entry in _unit_spec().volume)
    return re.compile(r'(?<![\w.])(\d{1,3})-(\d{1,2})(?=-?(?:' + units + r')(?![a-z]))', re.I | re.X)


@lru_cache(maxsize=1)
def _slug_volume_scanner():
    """Every configured volume mention in a slug, as (value, unit-text)."""
    from core.text import _unit_spec

    units = '|'.join(entry.pattern for entry in _unit_spec().volume)
    return re.compile(r'(\d+(?:\.\d+)?)\s*-?(' + units + r')(?![a-z])', re.I | re.X)


@lru_cache(maxsize=1)
def _slug_unit_resolvers():
    """(compiled unit pattern, millilitres) pairs, longest first.

    A dict keyed on the matched text cannot work: the configured patterns
    include spelled forms (`milli\\s?lit(?:er|re)s?`) that never equal their
    own match, so resolve by re-matching instead.
    """
    from core.text import _unit_spec

    pairs = [(re.compile(entry.pattern, re.I), float(entry.ml_per_unit))
             for entry in _unit_spec().volume]
    return sorted(pairs, key=lambda item: -len(item[0].pattern))


@lru_cache(maxsize=1)
def _slug_unit_probes():
    """(unit-match probe, millilitres) pairs, longest first.

    `_reconstruct_slug_decimals` reads a candidate's own trailing unit with
    `(?:-?(?:unit))` anchored at the match end. The probe used to be BUILT as a
    string and handed to module-level `re.match` for every resolver tried, per
    candidate — the string concatenation, the `re` cache lookup and the
    wrapper call all repeated for a pattern that never changes.
    """
    from core.text import _unit_spec

    pairs = [(re.compile(r'(?:-?(?:' + entry.pattern + r'))', re.I),
              float(entry.ml_per_unit))
             for entry in _unit_spec().volume]
    return sorted(pairs, key=lambda item: -len(item[0].pattern))


def _slug_unit_ml(unit_text: str) -> float | None:
    for pattern, ml in _slug_unit_resolvers():
        if pattern.fullmatch(unit_text.strip()):
            return ml
    return None


# How closely a candidate must agree with the slug's OTHER volume mention.
# Measured on the real corpus, the pack and decimal readings are separated by
# far more than this: "12-12-fl-oz-355-ml-cans" reads 354.9 ml as a count
# (0.03% off the stated 355) against 358.4 ml as a decimal (0.97% off), while
# the confirmed decimals land inside 0.05% ("12-5floz-370mL" -> 369.7,
# "67-62oz" + "2l" -> 2000.0, "1-75l" -> 1750).
_SLUG_CORROBORATION_TOLERANCE = 0.005


def _reconstruct_slug_decimals(slug: str) -> str:
    """Resolve `NN-M-unit` against the slug's other volume evidence.

    Retailer slugs use one shape for two different things: `33-8-fl-oz` is a
    33.8 fl oz bottle, while `12-12-fl-oz-355-ml-cans` is a twelve-pack of
    12 fl oz cans. Blindly rebuilding the decimal turned the second into
    12.12 fl oz, which contradicts the 355 ml the same slug states.

    So when the slug carries ANOTHER volume mention, both readings are
    converted to millilitres and the one that agrees wins. With no second
    mention there is nothing to test against, so the decimal stands — which
    is the overwhelmingly common real case and keeps every audited decimal
    intact. Without this the bad value only ever died later at cross-source
    merge, so nothing upstream could see it.
    """
    pattern = _slug_decimal_pattern()
    scanner = _slug_volume_scanner()
    matches = list(pattern.finditer(slug))
    if not matches:
        return slug
    mentioned = [(float(scan.group(1)), scan.start(1), scan.end(1), scan.group(2))
                 for scan in scanner.finditer(slug)]
    if len(mentioned) < 2:
        return pattern.sub(r"\1.\2", slug)

    out: list[str] = []
    cursor = 0
    for match in matches:
        whole_str, tail_str = match.group(1), match.group(2)
        out.append(slug[cursor:match.start()])
        cursor = match.end()

        # This match's OWN unit, which the pattern's lookahead guarantees sits
        # immediately after it. The dash must be INSIDE the alternation:
        # `-?ltr|lt|l` binds -? to the first branch only, so `-l` (the exact
        # shape the lookahead allows) matched NO branch — measured: unit=''
        # killed corroboration for every late-branch unit (iper.it "25-05-l").
        unit_key = ''
        for probe_pattern, _ml_per_unit in _slug_unit_probes():
            # Sliced, not `match(slug, end)`: the slice is what the audited
            # behaviour was measured against, and a future configured unit
            # pattern carrying a leading \b or lookbehind would see one
            # character of context under `pos` that it does not see here.
            probe = probe_pattern.match(slug[match.end():])
            if probe:
                unit_key = probe.group(0).lstrip('-')
                break

        # Every OTHER mention in the slug, in millilitres. A mention inside
        # ANOTHER candidate's span is that candidate's own fractional tail
        # re-scanned as "6 fl oz" / "75 l" — not an independent measurement.
        # Measured on the real corpus: riteaid "67-6-fl-oz-2-qt-3-6-fl-oz",
        # aqua "33-8-fl-oz-1-qt-1-8-oz-1-lt" and mathem "1-75l-1-75l" each
        # matched their SIBLING's tail exactly (gap 0.0) and flipped audited
        # decimals (67.6->6, 33.8->8, 1.75->75) — cross-candidate self-
        # corroboration. Sibling-tail mentions are dropped; only free-standing
        # mentions (355-ml, 2-qt, 1-Gallon) corroborate.
        others = [
            value * factor
            for value, start, end, unit in mentioned
            if end <= match.start() or start >= match.end()
            if not any(o.start() <= start < o.end()
                       for o in matches if o is not match)
            for factor in (_slug_unit_ml(unit),) if factor is not None
        ]

        # 7-5 -> 7.5; 33-8 -> 33.8; 12-12 -> 12.12; zero pads survive:
        # the STRING captures, never the ints ("25-05" is 25.05, not 25.5).
        decimal_text = f"{whole_str}.{tail_str}"
        factor = _slug_unit_ml(unit_key)
        decimal_ml = None if factor is None else float(decimal_text) * factor
        count_ml = None if factor is None else float(tail_str) * factor

        if others and decimal_ml is not None and count_ml is not None and factor:
            gap_decimal = min(abs(decimal_ml - other) / max(abs(other), 1e-9)
                              for other in others)
            gap_count = min(abs(count_ml - other) / max(abs(other), 1e-9)
                            for other in others)
            # If the count reading is a significantly better match for the
            # other evidence, and is itself a good match (0.5%), use it.
            if gap_count < gap_decimal and gap_count <= _SLUG_CORROBORATION_TOLERANCE:
                out.append(tail_str)
                continue
        out.append(decimal_text)
    out.append(slug[cursor:])
    return ''.join(out)
_VOWELS = frozenset("aeiou")


@lru_cache(maxsize=8)
def _short_code_pattern(letters: int) -> re.Pattern:
    """Retailer media codes: a short run of letters welded to digits ("k6rmm",
    "ab12"). Built from the configured letter budget instead of a literal, so
    the threshold is tunable without touching code. Unit-bearing tokens are
    exempt from this rule (see _has_unit_suffix) because "250ml" is a size,
    not a media code."""
    return re.compile(rf"^(?:[a-z]{{1,{letters}}}\d+|\d+[a-z]{{1,{letters}}})$")


@lru_cache(maxsize=8192)
def _is_noise(token: str) -> bool:
    """True when ``token`` is storefront scaffolding rather than product text.

    Every threshold here comes from config (url_evidence in paths.yaml); this
    function holds the ORDER of the rules, not their values.

    The two rules that were wrong before this became config-driven, both found
    by diffing what the normalizer kept against what the raw slugs actually
    contain:

      * bare_hash — the old rule dropped any 8+ char alphanumeric run, with no
        requirement that it look random. "sparkling" (9 letters) and
        "strawberry" (10) matched and were deleted: 70 and 48 occurrences in
        8,000 sampled sku_url slugs. Now a run is only a hash if it also
        carries a digit, which keeps "9df78eab" dead and "sparkling" alive.

      * short_code — the old rule dropped any letter+digit token, so "250ml"
        and "2l" died (34 and 9 occurrences). Those are the size tokens the
        pack gate reads. Now a unit-bearing token is exempt, because the unit
        is evidence and the digits are quantity.

    Memoized: the rule set depends only on the config-owned vocabulary (already
    frozen into PATH_SCHEMA_WORDS/UNITS at import) and the token, and slug
    vocabulary is far smaller than slug length — 32,555 token classifications
    per lane pass over only 4,693 distinct tokens, an 85.6% hit rate.
    """
    spec = _spec()
    if len(token) < spec.min_token_length:
        # a bare path letter (peapod's ".../c/K6/…") is storefront scaffolding
        return True
    # Units are PRESERVED. They are checked before the scaffolding set because
    # the two are validated disjoint, so the order cannot matter in principle —
    # but units winning is the deliberate semantic, not an accident of order.
    if token in UNITS:
        return False
    if token in PATH_SCHEMA_WORDS:
        return True
    if _QUANTITY_OR_DIM.fullmatch(token):
        return True
    # UNIT CHECK FIRST. Order matters and the first attempt got it wrong:
    # "250ml" is 5 characters and contains no vowel, so the no-vowel rule ate
    # it before the unit exemption could run — the very token this whole fix
    # exists to preserve. A size token is evidence; noise rules never apply.
    if _has_unit_suffix(token, spec):
        return False
    if len(token) >= spec.no_vowel_min_length and not (set(token) & _VOWELS):
        # no vowel in a long token: a random code (k6rmm, 9df78eab…), not a
        # product word. Real words (sparkling, lemon) always have one.
        return True
    if _short_code_pattern(spec.short_code_letters).match(token):
        return True
    if len(token) >= spec.bare_hash_min_length and (
        sum(map(_IS_DIGIT, token)) >= spec.bare_hash_min_digits
    ):
        # long AND carries digits: a content hash. Long and all letters was the
        # old bug — "sparkling" is 9 letters and is a product word.
        return True
    return False


def _has_unit_suffix(token: str, spec) -> bool:
    """True when ``token`` is a SIZE token: a declared unit, with a quantity.

    Recognises the three shapes that actually appear in listing slugs:

        "ml" / "oz"      bare unit (kept — the unit itself is the evidence)
        "250ml", "2l"    quantity + unit
        "12x355ml"       pack notation, count x size

    The count separator is "x" (normalize_text has already folded the
    multiply sign \u00d7 to "x"), and each half may itself be a size token, so
    "12x8x355ml" reduces correctly.

    This is what the media-code rule must NOT eat. "250ml" and "2l" were being
    deleted as retailer codes — 34 and 9 occurrences in 8,000 sampled sku_url
    slugs — and those are precisely the size tokens the pack gate reads, so
    losing them silently removed pack evidence rather than noise.
    """
    for unit in _UNITS_BY_LENGTH:
        if not token.endswith(unit):
            continue
        head = token[: -len(unit)]
        if not head:
            # the unit itself: kept by the UNITS check in _is_noise
            return True
        if _NUMERIC_FULL.fullmatch(head):
            return True
        # pack notation: every "x"-separated part must itself be a size
        # token — integer, decimal, or unit-bearing. Decimal heads are
        # observed ("6x1.5l", "4x0.25l", "12x50.7oz", "12x0.33l": 26 sku_url
        # rows, measured 2026-10-02, 24 of 26 corroborated by the title);
        # the bare isdigit check dropped them as media codes.
        parts = head.split("x")
        if len(parts) > 1 and all(
            part.isdigit()
            or _NUMERIC_FULL.fullmatch(part) is not None
            or _has_unit_suffix(part, spec)
            for part in parts
        ):
            return True
    return False


def url_text(url: object) -> str:
    """The product-bearing prose of ``url``, or "" if it carries none.

    Deterministic and total: never raises, never returns scaffolding, a hash,
    or "nan" from a missing cell. Safe to call on the raw export without
    per-row guards.
    """
    if url is None:
        return ""
    if isinstance(url, float) and url != url:  # NaN without pandas
        return ""
    text = str(url).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.split("?", 1)[0].split("#", 1)[0]
    text = _SCHEME.sub("", text)
    parts = text.split("/", 1)
    slug = parts[1] if len(parts) > 1 else parts[0]
    slug = _EXTENSION.sub("", slug)
    # Opaque image basenames may contain a short unit-looking fragment
    # after punctuation (51AFLZI--8L._AC_US160_). Classify the basename
    # before splitting it, so the fragment cannot invent product volume.
    basename = slug.rsplit('/', 1)[-1]
    stem, marker, _transform = basename.partition('._')
    prefix = _SLUG_SEPARATOR.split(stem, 1)[0].lower()
    spec = _spec()
    if (
        marker and len(stem) >= spec.bare_hash_min_length
        and prefix[:1].isdigit() and any(char.isalpha() for char in prefix)
        and sum(map(_IS_DIGIT, stem)) >= spec.bare_hash_min_digits
        and not _has_unit_suffix(prefix, spec)
    ):
        slug = slug[:-len(basename)]
    slug = _UUID.sub(" ", slug)
    slug = _FL_OZ.sub('fl oz', slug)
    slug = _reconstruct_slug_decimals(slug)
    slug = _SLUG_SEPARATOR_RUN.sub(" ", slug)
    slug = normalize_text(slug)
    # A bare retailer id is noise; a number in an explicit measurement or
    # pack phrase is product evidence. Preserve only recognized spans, using
    # the same grammar as downstream readers (including multiword fl oz).
    quantity_spans = [(entry['start'], entry['end']) for entry in extract_volume_evidence(slug)]
    if ("x" in slug or "pack" in slug or "ct" in slug or "pk" in slug
            or "count" in slug or "pcs" in slug or "bottle" in slug):
        quantity_spans.extend(match.span() for match in PACK_RE.finditer(slug))
    # Retain the count in "case of 6 33.8 oz". Dropping the 6 previously
    # manufactured "case of 8 oz" after decimal fragments were filtered.
    quantity_spans.extend(match.span() for match in _CASE_OF.finditer(slug))
    tokens = [
        token
        for match in _NON_SPACE.finditer(slug)
        for token in [match.group().rstrip('.')]
        if (_NUMERIC.fullmatch(token)
            and any(start <= match.start() and match.end() <= end for start, end in quantity_spans))
        or not _is_noise(token)
    ]
    return " ".join(tokens)


def is_evidentiary(url: object) -> bool:
    """True when the URL yields at least one product token."""
    return bool(url_text(url))
