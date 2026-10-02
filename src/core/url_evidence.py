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

def _spec():
    """The config-owned vocabulary + thresholds (config/paths.yaml url_evidence).

    Previously two module-level frozensets, PATH_SCHEMA_WORDS and _UNITS, held
    this vocabulary in code while config/paths.yaml held a copy of the same 85
    words — two sources of truth for one list, which is the duplication the
    SSOT rule exists to prevent. The config is now the ONLY copy; these two
    names are derived views of it, kept as module attributes because callers
    (and the regression tests) refer to them by name.
    """
    from core.common import data_cfg

    return data_cfg().url_evidence


# Derived, not authored: config/paths.yaml `url_evidence` is the single source.
PATH_SCHEMA_WORDS: frozenset[str] = frozenset(_spec().path_schema_words)
UNITS: frozenset[str] = frozenset(_spec().units)

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
# Observed slug decimals (zeroh 0-8l; Sierra 7-5oz) must survive number
# stripping. Require the complete configured volume-unit suffix before
# reconstructing punctuation; this is sanitation, not a conversion parser.
@lru_cache(maxsize=1)
def _slug_decimal_pattern():
    # One digit on each side is an observed decimal shape. Longer count
    # heads (24-500ml) retain list semantics; unknown suffixes are not sizes.
    from core.text import _unit_spec

    units = '|'.join(entry.pattern for entry in _unit_spec().volume)
    return re.compile(r'(?<![\w.])(\d)-(\d)(?=(?:' + units + r')(?![a-z]))', re.I | re.X)
_VOWELS = frozenset("aeiou")


def _short_code_pattern(letters: int) -> re.Pattern:
    """Retailer media codes: a short run of letters welded to digits ("k6rmm",
    "ab12"). Built from the configured letter budget instead of a literal, so
    the threshold is tunable without touching code. Unit-bearing tokens are
    exempt from this rule (see _has_unit_suffix) because "250ml" is a size,
    not a media code."""
    return re.compile(rf"^(?:[a-z]{{1,{letters}}}\d+|\d+[a-z]{{1,{letters}}})$")


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
    if re.fullmatch(r'\d+(?:\.\d+)?', token) or _IMAGE_DIM.match(token):
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
        sum(ch.isdigit() for ch in token) >= spec.bare_hash_min_digits
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
    for unit in sorted(UNITS, key=len, reverse=True):
        if not token.endswith(unit):
            continue
        head = token[: -len(unit)]
        if not head:
            # the unit itself: kept by the UNITS check in _is_noise
            return True
        if re.fullmatch(r'(?:\d+(?:\.\d+)?|\.\d+)', head):
            return True
        # pack notation: every "x"-separated part must itself be a size
        # token — integer, decimal, or unit-bearing. Decimal heads are
        # observed ("6x1.5l", "4x0.25l", "12x50.7oz", "12x0.33l": 26 sku_url
        # rows, measured 2026-10-02, 24 of 26 corroborated by the title);
        # the bare isdigit check dropped them as media codes.
        parts = head.split("x")
        if len(parts) > 1 and all(
            part.isdigit()
            or re.fullmatch(r'(?:\d+(?:\.\d+)?|\.\d+)', part) is not None
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
    prefix = re.split(r'[-_]', stem, maxsplit=1)[0].lower()
    spec = _spec()
    if (
        marker and len(stem) >= spec.bare_hash_min_length
        and prefix[:1].isdigit() and any(char.isalpha() for char in prefix)
        and sum(char.isdigit() for char in stem) >= spec.bare_hash_min_digits
        and not _has_unit_suffix(prefix, spec)
    ):
        slug = slug[:-len(basename)]
    slug = _UUID.sub(" ", slug)
    slug = _slug_decimal_pattern().sub(r"\1.\2", slug)
    slug = re.sub(r"[-_+]+", " ", slug)
    slug = normalize_text(slug)
    # A bare retailer id is noise; a number in an explicit measurement or
    # pack phrase is product evidence. Preserve only recognized spans, using
    # the same grammar as downstream readers (including multiword fl oz).
    quantity_spans = [(entry['start'], entry['end']) for entry in extract_volume_evidence(slug)]
    quantity_spans.extend(match.span() for match in PACK_RE.finditer(slug))
    tokens = [
        token
        for match in re.finditer(r'\S+', slug)
        for token in [match.group().rstrip('.')]
        if (re.fullmatch(r'\d+(?:\.\d+)?', token)
            and any(start <= match.start() and match.end() <= end for start, end in quantity_spans))
        or not _is_noise(token)
    ]
    return " ".join(tokens)


def is_evidentiary(url: object) -> bool:
    """True when the URL yields at least one product token."""
    return bool(url_text(url))
