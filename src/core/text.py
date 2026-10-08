"""Single source of text/regex machinery for the euromonitor series.

All regexes and pure text-parsing helpers live here — volume extraction,
pack counting, flavor detection, nutrition-phrase stripping, unit
normalization, bucketing, and the disposition heuristics. Step scripts
(02/02b/03/03b) import from this module; there is exactly ONE definition
of every pattern so tuning is single-file and tests pin the behavior.

Extractor version note: canonical volume comes from title ONLY (description
is a separate low-confidence signal — its nutrition/serving/dilution prose
injects false volumes). Bare-oz is flagged ambiguous (weight vs fluid) and
callers gate on category before trusting it.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import pandas as pd  # pd.Series annotation in attributes_keys (F17)

# Module-scope compiled patterns (PERF r15).
#
# Every `re.sub(...)`/`re.search(...)`/`re.match(...)`/`re.fullmatch(...)` call
# written against the `re` module goes through `re._compile`, which re-hashes
# the pattern string and re-reads the module cache on EVERY call (measured
# 0.36 s of `re/__init__.py:_compile` + wrapper overhead in the text lane
# benchmark, 325,787 dispatches).  Binding the compiled pattern once at import
# removes that dispatch entirely; the match semantics are unchanged because the
# pattern text and flags are copied verbatim from the call sites below.
_NORM_KEEP_RE = re.compile(r"[^a-z0-9.\s]")
_SPACE_RUN_RE = re.compile(r"\s+")
_NON_ALNUM_RUN_RE = re.compile(r"[^a-z0-9]+")
_UNIT_SEP_RE = re.compile(r"[.\-\s]")
_GLUED_ZERO_RE = re.compile(r"0\d")


# Combining-mark strip: list comprehension + a LOCAL `combining` alias.
#
# MEASURED (269,867 chars from the smoke_500 catalog, best of 7):
#   generator + unicodedata.combining per char   0.02544 s
#   list comprehension + local combining alias   0.01615 s   <- kept
#   str.translate with a precomputed table       0.00290 s   <- REJECTED
#
# The str.translate variant was implemented, measured and REMOVED (r20).  It
# needs a table of every combining code point, which costs one full
# `unicodedata.combining` sweep of the code space (0.1408 s, 934 code points),
# so it was built lazily behind a character counter with a 2,097,152 threshold
# (just above the ~1.7 M break-even).  That machinery DOES fire in the
# single-process micro-benchmark (which is where an earlier -89 % casefold
# figure came from), but it does NOT fire in the real pipeline:
#
#   * the real 5k composition path is FORK-PARALLEL — `SkuTextPool` forks when
#     the frame is >= _INLINE_THRESHOLD (4096) rows, observed as
#     "payload: fork-parallel sku-text compose over 5,000 rows";
#   * measured on the real 5k cohort, the per-row call mix costs 419.5 chars
#     (`unicode_casefold` sees every brand/title/attribute/description fold);
#   * 3 workers x ~1,667 rows/worker x 419.5 chars = ~699,348 chars per worker
#     — a third of the 2,097,152 threshold.  At 10,000 rows it is ~1,398,613,
#     still under.  The counter is per-process, so no worker ever crosses.
#
# So the table would have been unexercised global mutable state on a per-row
# path in the workload that matters, while its 0.1408 s build cost is real.
# Only the unconditional list-comprehension win is kept.
def unicode_casefold(value: object) -> str:
    """Shared case/accent folding; punctuation and negation remain intact."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    combining = unicodedata.combining
    return "".join([char for char in text if not combining(char)])


def _normalize_text_impl(text: str) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN without pandas  # noqa: PLR0124
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.replace("\u00d7", "x")
    text = _NORM_KEEP_RE.sub(" ", text)
    return _SPACE_RUN_RE.sub(" ", text).strip()


@lru_cache(maxsize=131072)
def _normalize_text_cached(text: str) -> str:
    return _normalize_text_impl(text)


def normalize_text(text: str) -> str:
    """Lowercase a string to ``[a-z0-9. ]``, collapsing runs of space.

    LIVES HERE now: it used to live in pipeline.normalize_text, which meant
    every ``core`` module needing it imported the top-level pipeline module
    (and the pipeline import pulled the whole ML stack into tests that only
    wanted the cleaner). core.url_evidence must not import pipeline for the
    same reason: core <-> pipeline has to stay one-directional
    (pipeline imports core). One definition, three import paths.

    Memoized on the str fast path only (the pure per-value result is a str,
    so a cached return cannot be mutated); non-str inputs keep the exact
    original coercion path.
    """
    if isinstance(text, str):
        return _normalize_text_cached(text)
    return _normalize_text_impl(text)


def normalized_attribute_text(*values: object) -> str:
    """Shared attribute token normalization without dropping negation words.

    PERF r17: this is the most-called function in the text lane (1,577,690
    calls in the 10k-cohort profile) and it is a pure function of its
    arguments. Measured on 5,908 realistic calls built from the fixture
    catalog (every raw attribute key plus every row's title and attributes),
    1,439 of them are distinct — a 75.6 % repeat rate, because the same field
    names and the same cells recur across rows, endpoints and identity
    parsing. The fold is therefore looked up on the tuple of stringified
    arguments; the cached body only ever sees the exact strings the
    uncached body would have seen, so no object is stringified once and
    reused for a mutated receiver.

    PERF r22: the stringifying tuple build is itself per-call recomputation
    that a cache HIT does not need — it was rebuilt on every one of the 90 % of
    calls that pass a single string, only to be thrown away. That case now goes
    straight to a cache keyed on the string. `type(only) is str` (exact type,
    not isinstance) keeps str SUBCLASSES on the generic path, where they still
    round-trip through `str(value or "")` exactly as before; for an exact str,
    `str(value or "")` is the value itself, so the two paths agree by
    construction.
    """
    if len(values) == 1:
        only = values[0]
        if type(only) is str:
            return _normalized_attribute_text_one(only)
    return _normalized_attribute_text_cached(
        tuple(str(value or "") for value in values))


@lru_cache(maxsize=65536)
def _normalized_attribute_text_one(text: str) -> str:
    return _normalized_attribute_text_cached((text,))


@lru_cache(maxsize=16384)
def _normalized_attribute_text_cached(texts: tuple) -> str:
    # The second (whitespace-collapsing) substitution the uncached body used
    # to run here is a provable no-op and is gone: `[^a-z0-9]+` replaces every
    # MAXIMAL run of non-alphanumerics — Unicode whitespace included, since
    # `\s` is not in that class — with exactly one space, so two spaces can
    # never end up adjacent and no other whitespace can survive. Checked
    # against 200,000 random strings over a mixed ASCII/Unicode/whitespace
    # alphabet: byte-identical, 0 mismatches, and 5.64 us cheaper per call on
    # ~200-character inputs.
    text = unicode_casefold(" ".join(texts))
    return _NON_ALNUM_RUN_RE.sub(" ", text).strip()

# ---------------------------------------------------------------------------
# Pack-count phrases: "6-pack", "12 Pack", "12pcs", "10 Packets", "48 pk",
# "pack of 6", "( Pack of4)". Groups: (1) count-before-pack,
# (2) packet/bottle form, (3) pack-of form.
PACK_RE = re.compile(
    r"(?:(\d+)\s*(?:-|\s)?(?:pack|pk|pcs|count|ct|x)\b"
    r"|(\d+)\s*(?:packet|packets|bottle|bottles)\b"
    r"|(?:\d+\s*)?(?:pack|pk)\s*of\s*(\d+)\b)",
    re.IGNORECASE,
)

# Pack-count ONLY (03 reconcile): "6x1.5l", "4x 1 ltr", "10 pack",
# "10 Packets", "48 pk", "pack of 6", "case of 12". Deliberately separate
# from PACK_RE so reconciliation can be tuned/tested independently of volume
# extraction. `case of N` is a declared retail bundle exactly like "pack of
# N" — measured gap: Amazon case titles ('... 16oz Bottle ( Case of 12)')
# blocked under NO_PACK while their identity parser extracted {12}.
PACK_COUNT_RE = re.compile(
    r"""(?:
        (\d+)\s*x\s*\d+(?:[.,]\d+)?\s*(?:ml|l|ltr|cl|dl|oz|fl\.?\s?oz)\b
      | (\d+)\s*(?:-|\s)?(?:pack|pk|packets?|pcs|count|ct)\b
      | pack\s*of\s*(\d+)
      | cases?\s+of\s*(\d+)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Nutrition/serving/dilution prose that must NOT be read as package volume.
NUTRITION_RE = re.compile(
    r"per\s*(?:100|1)\s*(?:ml|g|gram|grams)|kcal\s*per|per\s*serving",
    re.IGNORECASE,
)
# Number part: a single-digit integer may carry a comma-decimal OR a bare
# space-decimal with optional whitespace ("0, 33l" = 0.33 L, "0 8l" = 0.8 L —
# EU comma AND the slug form where url_text has already folded the
# hyphen-decimal "0-8l" into "0 8l"); multi-digit integers may NOT ("case of
# 24, 500ml" is a count-list, not 24.5 ml). A DOT decimal never takes a
# space ("pH 9.0 bottle 600 ml" must stay 600, not "9. 600"). The
# (?<![0-9.])(?![0-9]) pair pins the single-digit branch to a REAL single
# digit, so "24, 500ml" is never read as "4, 500" and "24 500ml" is never a
# splittable decimal. A leading-dot decimal (".14 oz" = 0.14 oz, ".5 l" =
# 0.5 L) is a THIRD branch so the multi-digit branch never eats the digits
# after the dot and inflates the value 10x/100x (".14" must not read as 14).
_NUMBER_PART = (
    r"((?<![0-9.])\d(?![0-9])(?:,?\s*\d+|\.\d+)?"
    r"|(?<![0-9.])\d{2,}(?:[.,]\d+)?"
    r"|(?<![0-9])\.\d+)"
)

_UNIT_SPEC = None


def _unit_spec() -> "UnitsSpec":
    """The config-owned unit taxonomy (config/paths.yaml `units`).

    Previously core.text._TO_ML, AMBIGUOUS_UNITS, BUCKET, MIN_PACK/MAX_PACK
    and the unit alternation were module literals while the converter and the
    NER features held THIRD and FOURTH copies of the same table. Evidence
    forced the move: 'dl' parsed here but the converter had NO decilitre
    family and crashed on it (ValueError: unsupported volume unit 'dl'); 'cc'
    was invisible to VOLUME_RE (65 'Exotic ... 300 cc' titles lost their
    volume); and factors drifted between the two tables (29.5735 vs
    29.5735295625 for the same fl-oz spelling; gal/qt/pt alike). The tables
    below are DERIVED views of one source, resolved through module
    __getattr__ by the SAME names as before, so every existing reader keeps
    working (same lazy pattern as core.url_evidence's _spec()).

    PERF r16: `core.common.data_cfg()` returns the module-level validated
    singleton `_DATA_CFG` read once at import, so re-resolving it through a
    function-level import on every call only bought a sys.modules lookup and a
    module-attribute hop per call — and every volume candidate paid it. The
    import stays lazy (core.text must not import core.common at module scope:
    core.common imports core.text, so that would be a cycle); only the result
    is now remembered.
    """
    global _UNIT_SPEC
    spec = _UNIT_SPEC
    if spec is None:
        from core.common import data_cfg

        spec = _UNIT_SPEC = data_cfg().units
    return spec


@lru_cache(maxsize=1)
def _volume_views() -> tuple:
    """(to_ml, ambiguous, bucket, pack_min, pack_max, volume_re) from config."""
    spec = _unit_spec()
    to_ml: dict[str, float] = {}
    ambiguous: set[str] = set()
    for entry in spec.volume:
        for spelling in entry.spellings:
            to_ml[spelling.lower()] = entry.ml_per_unit
        if entry.ambiguous:
            ambiguous.update(entry.spellings)
    alternation = r"(?:" + r"  |  ".join(e.pattern for e in spec.volume) + r")"
    volume_re = re.compile(
        _NUMBER_PART + r"\s*(?<![a-zA-Z])" + alternation + r"(?![a-zA-Z])",
        re.IGNORECASE | re.VERBOSE,
    )
    return (to_ml, frozenset(ambiguous), spec.bucket_ml, spec.pack_min,
            spec.pack_max, volume_re, spec.glued_code_max_ml)


_LAZY_UNIT_ATTRS = {
    "VOLUME_RE": 5, "_TO_ML": 0, "AMBIGUOUS_UNITS": 1,
    "BUCKET": 2, "MIN_PACK": 3, "MAX_PACK": 4,
}


def _views():
    """In-module read of the derived unit views (avoids eager config import)."""
    return _volume_views()


def __getattr__(name: str):
    """Fallback for config-derived unit views consumed before binding (kept for
    importers that reference the names through the module object).

    core.text VOLUME_RE/_TO_ML/AMBIGUOUS_UNITS/BUCKET/MIN_PACK/MAX_PACK live
    in config/paths.yaml `units` now; this module only materializes them.
    """
    if name in _LAZY_UNIT_ATTRS:
        return _volume_views()[_LAZY_UNIT_ATTRS[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")




# Flavor vocabulary + FLAVOR_RE REMOVED (audit round 2 F18, round 3): zero
# consumers — the live flavor signal is the critical-claims extractor
# (core/critical_attributes.py extract_critical_claims, which parses the
# Flavor: key from the attributes blob). Same for DRY_MIX_HINTS and
# SUSPECT_ROUND below.



def normalize_retailer(name: str) -> str:
    """Canonical retailer identity key: accent-fold + casefold + punctuation
    and whitespace collapse.

    SSOT for every surface that compares retailer strings as identity
    (blocking multi-retailer grouping, kfold_gtins, record-linkage
    cross-retailer rule). Measured on the 61,529-row deduped export
    (280 distinct raw spellings reviewed): exactly three alias groups
    collapse under this fold — Voila/Voilà (1,533 rows), publix/Publix
    (938), El Corte Ingles/El Corte Inglés — and raw-string grouping
    counts one gtin (8432425093657) as multi-retailer on spelling
    alone, feeding a fake cross-source positive into eval pairs and
    k-folds. No semantic aliases (e.g. amazon/amazon.com) exist in the
    data; if one ever appears, add an explicit alias map to
    config/vocabulary.json rather than special-casing in code.

    Deliberately NOT pipeline.normalize_text: that strips non-ASCII
    characters outright ("Voilà" -> "voil"), which both fails to merge the
    alias and could collide distinct retailers on the truncated stem.
    """
    text = unicodedata.normalize("NFKD", str(name or "").casefold())
    text = "".join([char for char in text if not unicodedata.combining(char)])
    text = _NON_ALNUM_RUN_RE.sub(" ", text)
    return " ".join(text.split())


def norm_unit(token: str) -> str:
    """Normalize a unit token to a config-units key (tolerates spacing/punct).

    PERF r18: the unit surface handed in here comes out of the volume regex
    alternation, so it is one of the ~26 spellings the config taxonomy
    declares — measured 104,000 calls over the real spellings, 26 distinct
    (99.97 % repeat). The two substitutions are pure, so the lowered token is
    looked up instead: 0.723 us -> 0.064 us per call. `token.lower()` still
    runs first and outside the cache, exactly as before, so a non-string
    argument raises the same AttributeError rather than a TypeError.
    """
    return _norm_unit_cached(token.lower())


@lru_cache(maxsize=4096)
def _norm_unit_cached(lowered: str) -> str:
    t = _UNIT_SEP_RE.sub(" ", lowered).strip()
    t = _SPACE_RUN_RE.sub(" ", t)
    if t == "floz":
        return "fl oz"
    return t


def bucket_ml(ml: float) -> int:
    """Round ml to the nearest config bucket (half-up), never 0.

    Uses floor(ml / BUCKET + 0.5) instead of round(), whose banker's rounding
    (round-half-to-even) maps an exact x.5 bucket boundary inconsistently and
    can split two representations of the same volume into different buckets.
    """
    bucket = _views()[2]
    bucketed = math.floor(ml / bucket + 0.5) * bucket
    if bucketed == 0 and ml > 0:
        return math.floor(ml + 0.5)
    return bucketed


def extract_volume_measurement(text: str) -> tuple[float | None, str | None, bool]:
    """(value_in_unit, unit_token, is_ambiguous_unit) — extract_volume_match
    without the raw substring. The parse is ONE function; only the conversion
    differs per path (bucket_ml vs whole-ml canonical_volume_ml)."""
    if not isinstance(text, str):
        return None, None, False
    value, unit, ambiguous, _raw = extract_volume_match(text)
    return value, unit, ambiguous


_SPACE_THOUSANDS_RE = re.compile(r"\b(\d{1,3})[ \u00a0]+(000)(?=\s*ml\b)", re.IGNORECASE)

# _VolumeEvidenceReader's role ladder and number decode, compiled once (PERF
# r15).  Pattern text and flags are verbatim from the former module-level
# re.search/re.match/re.fullmatch/re.sub call sites, so `search`/`match`
# semantics per position are unchanged.
_DRY_PRODUCT_RE = re.compile(r"\b(?:powder(?:ed)?|dry mix|drink mix|tea bags?)\b", re.I)
_PER_TAIL_RE = re.compile(r"\bper\s*$", re.I)
_PER_SERVING_RE = re.compile(r"\s*(?:per\s+(?:serving|portion)|/\s*serving)\b", re.I)
_TOTAL_TAIL_RE = re.compile(r"\btotal\s*(?:of\s*)?[:=]?\s*$", re.I)
_TOTAL_LEAD_RE = re.compile(r"\s*(?:in\s+)?total\b", re.I)
_YIELD_TAIL_RE = re.compile(r"\b(?:makes?|yields?|dilutes?\s+to)\s*$", re.I)
_CONTAINS_TAIL_RE = re.compile(r"\bcontains?\s*$", re.I)
_INGREDIENT_LEAD_RE = re.compile(
    r"\s*(?:of\s+)?(?:juice|concentrate|syrup)\s+(?:in|per|within)\b", re.I)
_FRACTION_RE = re.compile(r'(?:(\d+)\s+)?(\d+)\s*/\s*(\d+)')
_NESTED_COUNT_RE = re.compile(r'\d+\s*[x×]\s*$', re.I)
_CASE_COUNT_RE = re.compile(r'\bcase\s+of\s*$|\d+\s*/\s*$', re.I)
_SLASH_NUMBER_RE = re.compile(r'\s/')
_COUNT_LIST_RE = re.compile(r'([1-9]\d*),\s+(\d{3,})')
# The units whose role depends on the dry-product pre-probe. Shared by the
# ladder rung in classify_role and by the lazy guard in read() so the two can
# never drift apart.
_OUNCE_UNITS = frozenset({'oz', 'ounce', 'ounces'})
_SEP_TIGHT_RE = re.compile(r'\s*([.,])\s*')


def _VOLUME_SEARCH(text: str):
    """The single volume probe shared by every conversion path.

    Applies the space thousands repair ("1 000 ml" -> "1000ml") before the
    pack/nutrition strip; restricting it to 000 groups is deliberate, because
    "24, 500ml" and "24 500ml" are count/volume pairs and must not become
    24,500ml.

    Candidates are scanned in order and an implausible CODE ARTIFACT skips to
    the next match rather than claiming its volume: digits glued to a
    preceding letter (alphanumeric product code, e.g. 'BG14980 L' = product
    code + brand initial read as 14,980 litres) AND canonical ml above
    config glued_code_max_ml (the measured 25,000 ml bulk ceiling). Glued
    but plausible sizes ('chinotto1 l', 'burst850ml') and unglued bulk
    yields ('makes 128 gal') are untouched.
    """
    cleaned = _SPACE_THOUSANDS_RE.sub(r"\1\2", text)
    cleaned = NUTRITION_RE.sub("", PACK_RE.sub("", cleaned))
    spec = _volume_views()
    to_ml, glued_max = spec[0], spec[6]
    for match in _volume_views()[5].finditer(cleaned):
        raw_value = match.group(1)
        unit = norm_unit(match.group(0)[len(raw_value):].strip())
        factor = to_ml.get(unit)
        if factor is None:
            continue
        glued = match.start() > 0 and cleaned[match.start() - 1].isalpha() and cleaned[match.start() - 1].casefold() != "x"
        try:
            ml = float(raw_value.replace(",", ".").replace(" ", "")) * factor
        except ValueError:
            continue
        if glued and ml > glued_max:
            continue
        return match
    return None



def _extract_volume_match_legacy(text: str) -> tuple:
    """Return (value_in_unit, unit_token, is_ambiguous_unit, raw_match) — the
    ONE parse; every volume conversion path reads it and only the conversion
    differs (bucket_ml vs whole-ml canonical_volume_ml)."""
    if not isinstance(text, str):
        return None, None, False, ""
    match = _VOLUME_SEARCH(text)
    if not match:
        return None, None, False, ""
    raw_value = match.group(1)
    unit = norm_unit(match.group(0)[len(raw_value):].strip())
    # PERF r18: one materialized-view read instead of two lru_cache hits.
    views = _volume_views()
    to_ml, ambiguous_units = views[0], views[1]
    if unit not in to_ml:
        return None, None, False, ""
    # European thousands separator: "1.000 ml" = 1000ml (fixes implausible
    # sub-20ml decimals only, keeping "500.0 ml" alone).
    if "." in raw_value:
        before, after = raw_value.split(".", 1)
        if (before and len(after) == 3 and after.isdigit()
                and float(raw_value.replace(",", ".")) * to_ml[unit] < 20):
            raw_value = before + after
    if " " in raw_value:
        raw_value = " ".join(raw_value.split())
    # EU-decimal-with-space vs count-list: "0, 33l" = 0.33 L; "24, 500ml" is a
    # count/volume pair, not 24.5 ml. Same plausibility rule as above.
    if "," in raw_value and " " in raw_value:
        head, tail = raw_value.split(",", 1)
        dec_ml = float((head + "." + tail).replace(" ", "")) * to_ml[unit]
        if float(head) >= 1 and dec_ml < 100:
            raw_value = tail.strip()
    elif " " in raw_value:
        # COMMA-LESS twin ("0 8l", the slug's hyphen-decimal after url_text) —
        # a zero head is never a count, so the decimal is the only plausible
        # reading: this is where the zeroh 8000ml lie becomes 800ml. A
        # NON-ZERO head is the opposite case and must NOT be decoded as a
        # decimal, nor by deleting the separator: "8 12 fl. oz." is a size
        # RANGE and "2 200 ml" a count head, and concatenating them read
        # 812 fl. oz. (24014ml) and 2200ml. The head is dropped instead, which
        # is what the range/list reading has always meant (and what the
        # pre-bare-space-decimal regex got by never matching the head).
        head, _, tail = raw_value.partition(" ")
        tail = tail.strip()
        if head == "0" and tail.isdigit():
            raw_value = head + "." + tail
        elif head.isdigit() and tail[:1].isdigit():
            raw_value = tail
    # Slug-fragment salvage (audit 2026-10-01, zeroh "0-8l"): URL slugs split
    # a decimal fraction on the hyphen, so the captured fragment can arrive
    # headless ("8l" for 0.8 l) or with the zero fused ("08l"). SAME REPAIR
    # CLASS as the EU-decimal-with-space branch above — a zero head is never
    # a count, so the only plausible reading is the decimal:
    #   fused   "08l"  -> 0.8 l  (exactly one digit after the zero; the
    #                           multi-digit run keeps the artifact refusal
    #                           so "0123" can never decode as 123.4),
    #   spaced  "0 8l" -> 0.8 l  via the single-digit branch below eating
    #                           the zero with its optional separator.
    if _GLUED_ZERO_RE.fullmatch(raw_value):
        raw_value = "0." + raw_value[1:]
    value = float(raw_value.replace(",", ".").replace(" ", ""))
    return value, unit, unit in ambiguous_units, match.group(0)


@lru_cache(maxsize=1)
def _measurement_candidate_re():
    units = r"(?:" + "|".join(entry.pattern for entry in _unit_spec().volume) + r")"
    number = r"(?:\d+[ \u00a0]+\d+\s*/\s*\d+|\d+\s*/\s*\d+|\d{1,3}[ \u00a0]+000|0\s+\d+|\d+(?:\s*[.,]\s*\d+)?|\.\d+)"
    return re.compile(
        r"(?<![\d.])(?P<number>" + number + r")\s*-?\s*(?P<unit>" + units + r")(?![^\W\d_])"
        r"|\b(?P<prefix_unit>ml|cl|ltr|lt|l)\.\s*(?P<prefix_number>\d+(?:[.,]\d+)?)(?![\d.])",
        re.IGNORECASE | re.VERBOSE,
    )


class _VolumeEvidenceReader:
    """One text's volume-evidence scan (roles and original spans preserved).

    A recipe yield, nutrition denominator, or dry weight is useful evidence,
    but cannot assert liquid package volume. Mixed and improper fractions are
    decoded before normalization; spaced ambiguous-ounce count/size notation
    such as ``24 / 2oz`` retains its package-size interpretation.

    SR phases, ONE fixed order in read(); the guarded candidate loop is the
    original body verbatim, so the candidate list is byte-identical.

    Phase map:
      prepare       — the dry-product pre-probe, one per text
      classify_role — the precedence ladder: nutrition denominator, stated
                      total, recipe yield, ingredient volume, dry net weight
      decode_number — the fraction branch (count/size interpretation inside)
      plausible     — the glued-code-artifact refusal vs the measured bulk
                      ceiling
    """

    def __init__(self, text: str) -> None:
        self._text = text


    def prepare(self) -> bool:
        return bool(_DRY_PRODUCT_RE.search(self._text))

    def classify_role(self, preceding: str, following: str, unit: str,
                      dry_product: bool) -> str:
        """The role ladder is precedence-ordered (first proof wins)."""
        if (_PER_TAIL_RE.search(preceding)
                or _PER_SERVING_RE.match(following)):
            role = 'nutrition'
        elif (_TOTAL_TAIL_RE.search(preceding)
              or _TOTAL_LEAD_RE.match(following)):
            role = 'total_volume'
        elif _YIELD_TAIL_RE.search(preceding):
            role = 'yield'
        elif (_CONTAINS_TAIL_RE.search(preceding)
              and _INGREDIENT_LEAD_RE.match(following)):
            role = 'ingredient_volume'
        elif dry_product and unit in _OUNCE_UNITS:
            role = 'net_weight'
        else:
            role = 'package_volume'
        return role

    def decode_number(self, number: str, unit: str, preceding: str):
        """The mixed/improper fraction decode with its count/size ladder.

        Returns a parsed 4-tuple, or None when the denominator is 0 (the
        candidate dies exactly as before)."""
        if '/' not in number:
            return None
        fraction = _FRACTION_RE.fullmatch(number)
        whole, numerator, denominator = (int(part or 0) for part in fraction.groups())
        if denominator == 0:
            return None
        nested_count = bool(_NESTED_COUNT_RE.search(preceding))
        case_count = bool(_CASE_COUNT_RE.search(preceding))
        # PERF r18: the ambiguous-unit set was read twice per fraction.
        ambiguous_units = _volume_views()[1]
        count_size = (nested_count or case_count or (not whole and numerator >= denominator
                      and unit in ambiguous_units and bool(_SLASH_NUMBER_RE.search(number))))
        value = float(denominator) if count_size else whole + numerator / denominator
        return value, unit, unit in ambiguous_units

    def plausible(self, glued: bool, value: float, parsed_unit: str) -> bool:
        """Digits glued to a preceding letter are CODE ARTIFACTS above the
        configured bulk ceiling; plausible glued sizes stay candidates.

        PERF r16: reads the bulk ceiling from the already-materialized view
        tuple (index 6 is `spec.glued_code_max_ml`, the same attribute) instead
        of re-entering the config resolver once per candidate.
        """
        return not (glued and value * _volume_views()[0][parsed_unit]
                    > _volume_views()[6])

    def read(self) -> list[dict]:
        """The guarded candidate loop (statements verbatim).

        PERF r21: `prepare()` — the dry-product pre-probe — was run once per
        text even though its result is read by exactly ONE rung of the role
        ladder (`dry_product and unit in _OUNCE_UNITS`).  Measured on the real
        5,000-row cohort's title/url/image mix it costs 4.80 us/text, 12.1 % of
        this function, and it was paid on texts that yield no candidate at all
        (61 % of them).  It is now resolved on first demand, and only for a
        candidate whose unit can actually reach that rung.  For every other
        unit the rung is False regardless of the probe, so passing False there
        is provably the same role; the candidate list this returns is
        byte-identical and is checked by the equivalence dump.
        """
        text = self._text
        candidates: list[dict] = []
        dry_product: bool | None = None  # None = probe not run yet
        for match in _measurement_candidate_re().finditer(text):
            number = match.group('number') or match.group('prefix_number')
            unit_surface = match.group('unit') or match.group('prefix_unit')
            unit = norm_unit(unit_surface)
            if dry_product is None and unit in _OUNCE_UNITS:
                dry_product = self.prepare()
            preceding = text[:match.start()].rstrip()
            following = text[match.end():]
            role = self.classify_role(preceding, following, unit, dry_product or False)
            if '/' in number:
                decoded = self.decode_number(number, unit, preceding)
                if decoded is None:
                    continue
                value, parsed_unit, ambiguous = decoded
            else:
                # Preserve comma decimal semantics. General normalization would
                # erase the comma and turn 1 ,25 litres into a count-list.
                # "concentrate 1 + 4, 200ml" is a dilution ratio followed by
                # bottle size, not a 4.2 ml package. Preserve genuine 0, 33 l
                # and 1,25 l decimals while separating integer/size lists.
                count_list = _COUNT_LIST_RE.fullmatch(number)
                numeric = count_list.group(2) if count_list else _SEP_TIGHT_RE.sub(r'\1', number)
                value, parsed_unit, ambiguous, _ = _extract_volume_match_legacy(numeric + ' ' + unit_surface)
            if value is None or parsed_unit is None or value <= 0:
                continue
            glued = match.start() > 0 and text[match.start() - 1].isalpha() and text[match.start() - 1].casefold() != 'x'
            if not self.plausible(glued, value, parsed_unit):
                continue
            candidates.append(dict(value=value, unit=parsed_unit, ambiguous=ambiguous,
                                   raw_match=match.group(0), start=match.start(), end=match.end(), role=role))
        return candidates


@lru_cache(maxsize=65536)
def _extract_volume_evidence_cached(text: str) -> list[dict]:
    return _VolumeEvidenceReader(text).read()


def extract_volume_evidence(text: str) -> list[dict]:
    """Keep measurement roles and original spans before punctuation cleanup —
    one phase-ordered scan on :class:`_VolumeEvidenceReader`.

    Memoized: the phase-ordered scan is a pure function of the text, and the
    same column string is re-scanned by several lanes within one listing's
    extraction. The returned entries are read-only at every call site (spread
    with ``**entry`` or field-read), so handing back the cached list preserves
    both bytes and semantics.
    """
    if not isinstance(text, str):
        return []
    return _extract_volume_evidence_cached(text)


@lru_cache(maxsize=65536)
def _extract_volume_match_cached(text: str) -> tuple:
    for candidate in _extract_volume_evidence_cached(text):
        if candidate['role'] == 'package_volume':
            return tuple(candidate[key] for key in ('value', 'unit', 'ambiguous', 'raw_match'))
    return None, None, False, ''


def extract_volume_match(text: str) -> tuple:
    """Select liquid package evidence, preserving non-package candidates separately."""
    if not isinstance(text, str):
        return None, None, False, ''
    return _extract_volume_match_cached(text)


@lru_cache(maxsize=1)
def _volume_spelling_index() -> dict:
    """post-norm_unit spelling -> the config entry declaring it.

    ONE index for one table: the parse capture keys both _TO_ML (float) and
    this attribution index off the very same spellings, so a config edit
    cannot update one half and strand the other.

    PERF r16: pure function of the config singleton, so it is built once.
    `pipeline.extract_volume_from_title` calls `_volume_entry` for EVERY title
    that mentions a volume, and each call used to rebuild this dict over the
    whole unit taxonomy (18,386 rebuilds / 0.689 s in the 10k-cohort profile).
    """
    index: dict = {}
    for entry in _unit_spec().volume:
        for spelling in entry.spellings:
            index.setdefault(spelling.lower(), []).append(entry)
    return index


def _volume_entry(unit: str):
    """The config VolumeUnitSpec for a post-norm_unit spelling; None when the
    taxonomy does not declare it — callers treat that as no_evidence."""
    matches = _volume_spelling_index().get(unit.lower(), ())
    return matches[0] if matches else None


def extract_volume_ml(text: str) -> tuple[int | None, bool]:
    """Return (canonical_ml, is_ambiguous_unit) — the ml projection of
    extract_volume measurement (kept for all existing callers/tests)."""
    value, unit, ambiguous = extract_volume_measurement(text)
    if value is None or unit is None:
        return None, False
    return bucket_ml(value * _volume_views()[0][unit]), ambiguous


# _LITER_UNITS, CATEGORY_MEASUREMENT_TYPE, DEFAULT_MEASUREMENT_TYPE and
# get_measurement_type REMOVED (audit round 2 F18, round 3): zero consumers
# anywhere (exhaustive grep) — the category->measurement-type taxonomy and
# the "single bottle > 10 L" rule that used the liter set are retired.


# MACRO_MAP REMOVED (SSOT move, this round): the category -> macro bucket
# taxonomy was judged DOMAIN DATA, not code — a curated mapping of the
# dataset's 24 strict categories to coarse macro buckets (the blocking
# layer's recall-first rollup; see 04b/04c) that the owner may tune
# without touching an import. It now lives in config/vocabulary.json
# category_macros: (required at load; SystemExit on a missing or malformed
# mapping). Every surface reads it through the same SSOT accessor,
# core.common.category_macros — the config yaml never held this key.
# Rationale: it maps DATA values (category names) to canonical forms —
# the same shape as column_mapping — rather than being regex-adjacent
# normalization logic that changes only alongside code. Consumers:
# src/training/blocking_audit.py, src/training/report_plots.py,
# src/core/hard_negatives.py. Scoring still uses the strict category
# (higher mutual information); only blocking rolls up to macro.


def extract_pack_counts(text: str) -> set[int]:
    """All plausible pack sizes mentioned in one product title."""
    counts = set()
    if not isinstance(text, str):
        return counts
    # PERF r18: index the view tuple directly — the slice built a 2-tuple per call.
    views = _volume_views()
    pack_min, pack_max = views[3], views[4]
    for m in PACK_COUNT_RE.finditer(text):
        for g in m.groups():
            if g:
                n = int(g)
                if pack_min <= n <= pack_max:
                    counts.add(n)
    return counts


# is_pack_multiple REMOVED (audit round 2 F18, round 3): zero consumers
# (grep-verified) — nothing in the tree calls it.


def attributes_keys(series: pd.Series, limit: int = 5000) -> Counter:
    """Extract `Key:` names from the ';'-delimited attributes strings."""
    keys: Counter = Counter()
    for value in series.fillna("").head(limit):
        for part in value.split(";"):
            part = part.strip()
            if ":" in part:
                keys[part.split(":", 1)[0].strip()] += 1
    return keys


def attribute_fields(cell: object, *, include_empty: bool = False) -> list[tuple[str, str]]:
    """One attributes cell -> [(normalized_key, raw_value), ...], deterministic.

    THE shared semantics for `Key: value` segment parsing (the docstring
    contract AttributeUniverse.parse spells out): split(';'), key before ':',
    key normalized through :func:`normalized_attribute_text`, value kept RAW
    (comma-splitting and lowercasing belong to the caller). Blank segments
    and segments without ':' are skipped — this is the six-call walk that
    used to be re-implemented in attribute_universe.parse, the universe
    census, audit_feature_capture, audit_attribute_readings._key_present and
    critical_attributes._field_tokens/_without_field; one implementation
    ends their drift. Callers that must CENSUS malformed segments (they are
    parser-gap evidence) keep their own walk — see
    product_dimensions.row_dimensions.

    Empty values are excluded from populated evidence by default; key-presence
    audits can retain explicit blank declarations with ``include_empty=True``.
    """
    found: list[tuple[str, str]] = []
    for part in str(cell or "").split(";"):
        if ":" not in part:
            continue
        raw_key, raw_value = part.split(":", 1)
        # PERF r18: the kept value used to be stripped twice (once for the
        # emptiness test, once for the tuple); strip once and reuse.
        value = raw_value.strip()
        if not include_empty and not value:
            continue
        found.append((normalized_attribute_text(raw_key), value))
    return found


def attribute_field_value(cell: object, key: str) -> list[str]:
    """Raw value tokens of one `Key:` field (stripped, comma-split)."""
    # Hoisted: the wanted key is normalized ONCE per call, not once per cell
    # segment. Callers ask for many keys of the same cell (see
    # critical_attributes._field_tokens), and this predicate used to re-fold an
    # already-normalized key for every segment of every one of those calls.
    wanted = normalized_attribute_text(key)
    return [
        token.strip().lower()
        for name, raw_value in attribute_fields(cell)
        if name == wanted
        for token in raw_value.split(",")
        if token.strip()
    ]
