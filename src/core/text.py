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


def unicode_casefold(value: object) -> str:
    """Shared case/accent folding; punctuation and negation remain intact."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    """Lowercase a string to ``[a-z0-9. ]``, collapsing runs of space.

    LIVES HERE now: it used to live in pipeline.normalize_text, which meant
    every ``core`` module needing it imported the top-level pipeline module
    (and the pipeline import pulled the whole ML stack into tests that only
    wanted the cleaner). core.url_evidence must not import pipeline for the
    same reason: core <-> pipeline has to stay one-directional
    (pipeline imports core). One definition, three import paths.
    """
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN without pandas  # noqa: PLR0124
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.replace("\u00d7", "x")
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_attribute_text(*values: object) -> str:
    """Shared attribute token normalization without dropping negation words."""
    text = unicode_casefold(" ".join(str(value or "") for value in values))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()

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
# "10 Packets", "48 pk", "pack of 6". Deliberately separate from PACK_RE
# so reconciliation can be tuned/tested independently of volume extraction.
PACK_COUNT_RE = re.compile(
    r"""(?:
        (\d+)\s*x\s*\d+(?:[.,]\d+)?\s*(?:ml|l|ltr|cl|dl|oz|fl\.?\s?oz)\b
      | (\d+)\s*(?:-|\s)?(?:pack|pk|packets?|pcs|count|ct)\b
      | pack\s*of\s*(\d+)
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
    """
    from core.common import data_cfg

    return data_cfg().units


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
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def norm_unit(token: str) -> str:
    """Normalize a unit token to a config-units key (tolerates spacing/punct)."""
    t = re.sub(r"[.\-\s]", " ", token.lower()).strip()
    t = re.sub(r"\s+", " ", t)
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
    to_ml, ambiguous_units = _volume_views()[0], _volume_views()[1]
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
    if re.fullmatch(r"0\d", raw_value):
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


def extract_volume_evidence(text: str) -> list[dict]:
    """Keep measurement roles and original spans before punctuation cleanup.

    A recipe yield, nutrition denominator, or dry weight is useful evidence,
    but cannot assert liquid package volume. Mixed and improper fractions are
    decoded before normalization; spaced ambiguous-ounce count/size notation
    such as ``24 / 2oz`` retains its package-size interpretation.
    """
    if not isinstance(text, str):
        return []
    candidates = []
    dry_product = bool(re.search(r"\b(?:powder(?:ed)?|dry mix|drink mix|tea bags?)\b", text, re.I))
    for match in _measurement_candidate_re().finditer(text):
        number = match.group('number') or match.group('prefix_number')
        unit_surface = match.group('unit') or match.group('prefix_unit')
        unit = norm_unit(unit_surface)
        preceding = text[:match.start()].rstrip()
        following = text[match.end():]
        role = 'package_volume'
        if (re.search(r"\bper\s*$", preceding, re.I)
                or re.match(r"\s*(?:per\s+(?:serving|portion)|/\s*serving)\b", following, re.I)):
            role = 'nutrition'
        elif (re.search(r"\btotal\s*(?:of\s*)?[:=]?\s*$", preceding, re.I)
              or re.match(r"\s*(?:in\s+)?total\b", following, re.I)):
            role = 'total_volume'
        elif re.search(r"\b(?:makes?|yields?|dilutes?\s+to)\s*$", preceding, re.I):
            role = 'yield'
        elif (re.search(r"\bcontains?\s*$", preceding, re.I)
              and re.match(r"\s*(?:of\s+)?(?:juice|concentrate|syrup)\s+(?:in|per|within)\b", following, re.I)):
            role = 'ingredient_volume'
        elif dry_product and unit in {'oz', 'ounce', 'ounces'}:
            role = 'net_weight'
        if '/' in number:
            fraction = re.fullmatch(r'(?:(\d+)\s+)?(\d+)\s*/\s*(\d+)', number)
            whole, numerator, denominator = (int(part or 0) for part in fraction.groups())
            if denominator == 0:
                continue
            nested_count = bool(re.search(r'\d+\s*[x×]\s*$', preceding, re.I))
            count_size = (nested_count or (not whole and numerator >= denominator
                          and unit in _volume_views()[1] and bool(re.search(r'\s/', number))))
            value = float(denominator) if count_size else whole + numerator / denominator
            parsed = (value, unit, unit in _volume_views()[1], match.group(0))
        else:
            # Preserve comma decimal semantics. General normalization would
            # erase the comma and turn 1 ,25 litres into a count-list.
            count_list = re.fullmatch(r'(\d{2,}),\s+(\d{3,})', number)
            numeric = count_list.group(2) if count_list else re.sub(r'\s*([.,])\s*', r'\1', number)
            parsed = _extract_volume_match_legacy(numeric + ' ' + unit_surface)
        value, parsed_unit, ambiguous, _ = parsed
        if value is None or parsed_unit is None or value <= 0:
            continue
        glued = match.start() > 0 and text[match.start() - 1].isalpha() and text[match.start() - 1].casefold() != 'x'
        if glued and value * _volume_views()[0][parsed_unit] > _unit_spec().glued_code_max_ml:
            continue
        candidates.append(dict(value=value, unit=parsed_unit, ambiguous=ambiguous,
                               raw_match=match.group(0), start=match.start(), end=match.end(), role=role))
    return candidates


def extract_volume_match(text: str) -> tuple:
    """Select liquid package evidence, preserving non-package candidates separately."""
    for candidate in extract_volume_evidence(text):
        if candidate['role'] == 'package_volume':
            return tuple(candidate[key] for key in ('value', 'unit', 'ambiguous', 'raw_match'))
    return None, None, False, ''


def _volume_spelling_index() -> dict:
    """post-norm_unit spelling -> the config entry declaring it.

    ONE index for one table: the parse capture keys both _TO_ML (float) and
    this attribution index off the very same spellings, so a config edit
    cannot update one half and strand the other.
    """
    index = {}
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
    pack_min, pack_max = _volume_views()[3:5]
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
        if not include_empty and not raw_value.strip():
            continue
        found.append((normalized_attribute_text(raw_key), raw_value.strip()))
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


# ---------------------------------------------------------------------------
# Noise-probe patterns (used by 01c_sparsity_noise.py — probes, not extraction)
# ---------------------------------------------------------------------------

# Solid-weight units in a beverage catalog are semantic noise: a "1 kg" or
# "10 lb" listing is a scraping/unit error or a non-beverage SKU. NOTES:
# - "mg" is deliberately NOT here — "L-Carnitine 2000mg" is a DOSE, not net
#   weight; including mg caused false-positive flags on supplement drinks.
# - zero values are excluded — "0g Added Sugar" is nutrition prose, not a
#   net weight; EU decimals like "0,5 kg" are still matched (0[.,]\d+).
WEIGHT_UNIT_RE = re.compile(
    r"\b(?:[1-9]\d*|0[.,]\d+)(?:[.,]\d+)?\s*"
    r"(?:kg|kgs?|g|grams?|grammes?|lb|lbs?|pounds?|pound)\b",
    re.IGNORECASE,
)

# MULTIPACK_RE REMOVED (audit round 2 F18, round 3): zero consumers —
# the "single bottle > 10 L" multipack false-positive rule that used it is
# retired with the rule above.

# Raw-HTML / entity artifacts from scraping.
NOISE_HTML_RE = re.compile(r"<[^>]+>|&nbsp;|&amp;|&quot;|&#\d+;", re.IGNORECASE)

# Placeholder / dummy values that carry no product information.
NOISE_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:n/?a|null|none|-|tbd|todo|to be updated|coming soon|"
    r"description|not available|n\.a\.?)\s*$",
    re.IGNORECASE,
)
