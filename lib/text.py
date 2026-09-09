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
from collections import Counter

import pandas as pd  # pd.Series annotation in attributes_keys (F17)

# ---------------------------------------------------------------------------
# Volume extraction (v2: broader units + decimal-comma + boundaries)
# ---------------------------------------------------------------------------
_UNIT_ALTERNATION = r"""
    (?:
        milli\s?lit(?:er|re)s?    |  ml
      | centi\s?lit(?:er|re)s?    |  cl
      | deci\s?lit(?:er|re)s?     |  dl
      | lit(?:er|re)s?            |  ltr | lt | l
      | fluid\s?ounces?           |  fl\.?\s?-?\s?oz\.?  | floz
      | gallons?                  |  gal
      | quarts?                   |  qt
      | pints?                    |  pt
      | ounces?                   |  oz
    )
"""

# Number part: a single-digit integer may carry a comma-decimal with optional
# whitespace ("0, 33l" = 0.33 L, "1, 5l" = 1.5 L — EU notation with a space);
# multi-digit integers may NOT ("case of 24, 500ml" is a count-list, not
# 24.5 ml). A DOT decimal never takes a space ("pH 9.0 bottle 600 ml" must
# stay 600, not "9. 600"). The (?<![0-9.])(?![0-9]) pair pins the single-digit
# branch to a REAL single digit, so "24, 500ml" is never read as "4, 500".
# A leading-dot decimal (".14 oz" = 0.14 oz, ".5 l" = 0.5 L) is a THIRD branch
# so the multi-digit branch never eats the digits after the dot and inflates
# the value 10x/100x (".14" must not read as 14).
_NUMBER_PART = (
    r"((?<![0-9.])\d(?![0-9])(?:,\s*\d+|\.\d+)?"
    r"|(?<![0-9.])\d{2,}(?:[.,]\d+)?"
    r"|(?<![0-9])\.\d+)"
)

VOLUME_RE = re.compile(
    _NUMBER_PART + r"\s*(?<![a-zA-Z])" + _UNIT_ALTERNATION + r"(?![a-zA-Z])",
    re.IGNORECASE | re.VERBOSE,
)

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

# Flavor vocabulary + FLAVOR_RE REMOVED (audit round 2 F18, round 3): zero
# consumers — the live flavor signal is data_pipe.FLAVOR_PATTERN (its own
# near-duplicate list, deliberately untouched). Same for DRY_MIX_HINTS and
# SUSPECT_ROUND below.

# Bare oz/ounce could mean weight (chips, protein powder) rather than fluid
# volume. Flag rather than silently trusting it.
AMBIGUOUS_UNITS = {"oz", "ounce", "ounces"}

BUCKET = 5  # nearest 5 ml absorbs 355 vs 355.0 vs 354 noise
MIN_PACK, MAX_PACK = 2, 100  # sane multipack range (excludes "1 count"/"365 days")

_TO_ML = {
    "ml": 1.0,
    "millilitre": 1.0,
    "milliliter": 1.0,
    "millilitres": 1.0,
    "milliliters": 1.0,
    "cl": 10.0,
    "centilitre": 10.0,
    "centiliter": 10.0,
    "centilitres": 10.0,
    "centiliters": 10.0,
    "dl": 100.0,
    "decilitre": 100.0,
    "deciliter": 100.0,
    "decilitres": 100.0,
    "deciliters": 100.0,
    "l": 1000.0,
    "ltr": 1000.0,
    "lt": 1000.0,
    "litre": 1000.0,
    "liter": 1000.0,
    "litres": 1000.0,
    "liters": 1000.0,
    "gal": 3785.41,
    "gallon": 3785.41,
    "gallons": 3785.41,
    "qt": 946.353,
    "quart": 946.353,
    "quarts": 946.353,
    "pt": 473.176,
    "pint": 473.176,
    "pints": 473.176,
    "fl oz": 29.5735,
    "floz": 29.5735,
    "fluid ounce": 29.5735,
    "fluid ounces": 29.5735,
    "oz": 29.5735,
    "ounce": 29.5735,
    "ounces": 29.5735,
}


def norm_unit(token: str) -> str:
    """Normalize a unit token to a _TO_ML key (tolerates spacing/punct)."""
    t = re.sub(r"[.\-\s]", " ", token.lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if t == "floz":
        return "fl oz"
    return t


def bucket_ml(ml: float) -> int:
    """Round ml to the nearest BUCKET (half-up), never 0 for a real detection.

    Uses floor(ml / BUCKET + 0.5) instead of round(), whose banker's rounding
    (round-half-to-even) maps an exact x.5 bucket boundary inconsistently and
    can split two representations of the same volume into different buckets.
    """
    bucketed = math.floor(ml / BUCKET + 0.5) * BUCKET
    if bucketed == 0 and ml > 0:
        return math.floor(ml + 0.5)
    return bucketed


def extract_volume_measurement(text: str) -> tuple[float | None, str | None, bool]:
    """Return (value_in_unit, unit_token, is_ambiguous_unit).

    The raw (value, unit) pair BEFORE canonical ml conversion — the columns a
    measurement validator needs. value_in_unit is the amount as written (350
    for "350 L"), unit_token the normalized unit ('l', 'ml', 'fl oz', 'oz',
    ...) or None when nothing is extracted. Same pipeline as extract_volume_ml
    (pack-count and nutrition phrases stripped first, symmetric word
    boundaries, EU decimal-comma, thousands-separator fix), so the two
    functions can never disagree on WHAT was found.
    """
    if not isinstance(text, str):
        return None, None, False
    cleaned = PACK_RE.sub("", text)
    cleaned = NUTRITION_RE.sub("", cleaned)
    match = VOLUME_RE.search(cleaned)
    if not match:
        return None, None, False
    raw_value = match.group(1)
    unit = norm_unit(match.group(0)[len(raw_value) :].strip())
    if unit not in _TO_ML:
        return None, None, False
    # European thousands separator: "1.000 ml" means 1000ml, not 1.0ml.
    # A '.' followed by EXACTLY 3 digits with no further digits, where the
    # decimal-interpretation would be implausibly small (< 20 ml).
    if "." in raw_value:
        before, after = raw_value.split(".", 1)
        # "before" must be non-empty: a leading-dot decimal (".500") is a
        # 0.500-style value, not a thousands-separated integer.
        if (
            before
            and len(after) == 3
            and after.isdigit()
            and float(raw_value.replace(",", ".")) * _TO_ML[unit] < 20
        ):
            raw_value = before + after  # "1.000" -> "1000"
    # EU-decimal-with-space vs count-list: "0, 33l" = 0.33 L (plausible), but
    # "1, 100cl" / "1 + 4, 200ml" / "33 x 4, 132 cl" are count-lists whose
    # decimal reading is an implausible volume. Reuse the plausibility idea
    # from the thousands rule: a leading count (>=1) with a sub-100ml decimal
    # reading means the number AFTER the comma is the real volume.
    if "," in raw_value and " " in raw_value:
        head, tail = raw_value.split(",", 1)
        dec_ml = float((head + "." + tail).replace(" ", "")) * _TO_ML[unit]
        if float(head) >= 1 and dec_ml < 100:
            raw_value = tail.strip()
    # Spaces inside the number ("0, 33" = 0.33) are EU-decimal formatting.
    value = float(raw_value.replace(",", ".").replace(" ", ""))
    ambiguous = unit in AMBIGUOUS_UNITS
    return value, unit, ambiguous


def extract_volume_ml(text: str) -> tuple[int | None, bool]:
    """Return (canonical_ml, is_ambiguous_unit) — the ml projection of
    extract_volume_measurement (kept for all existing callers/tests)."""
    value, unit, ambiguous = extract_volume_measurement(text)
    if value is None or unit is None:
        return None, False
    return bucket_ml(value * _TO_ML[unit]), ambiguous


# _LITER_UNITS, CATEGORY_MEASUREMENT_TYPE, DEFAULT_MEASUREMENT_TYPE and
# get_measurement_type REMOVED (audit round 2 F18, round 3): zero consumers
# anywhere (exhaustive grep) — the category->measurement-type taxonomy and
# the "single bottle > 10 L" rule that used the liter set are retired.


# Category -> macro bucket for BLOCKING (coarse, recall-first). Built from the
# dataset's 24 real categories: retailers categorize the same product
# consistently ~94% at the strict level, so the blocking layer rolls up to
# macro to avoid rejecting true matches (see 04b/04c). Scoring still uses the
# strict category (higher mutual information).
MACRO_MAP = {
    "Not from Concentrate 100% Juice": "JUICE",
    "Reconstituted 100% Juice": "JUICE",
    "Juice Drinks (up to 24% Juice)": "JUICE",
    "Nectars": "JUICE",
    "Coconut and Other Plant Waters": "JUICE",
    "Other Non-Cola Carbonates": "CARBONATES",
    "Orange Carbonates": "CARBONATES",
    "Regular Cola Carbonates": "CARBONATES",
    "Reduced Sugar Cola Carbonates": "CARBONATES",
    "Tonic Water/Mixers/Other Bitters": "CARBONATES",
    "Lemonade/Lime": "CARBONATES",
    "Energy Drinks": "ENERGY_SPORTS",
    "Sports Drinks": "ENERGY_SPORTS",
    "Still Bottled Water": "WATER",
    "Carbonated Bottled Water": "WATER",
    "Sparkling Flavoured Bottled Water": "WATER",
    "Still Flavoured Bottled Water": "WATER",
    "Functional Bottled Water": "WATER",
    "Still RTD Tea": "TEA_COFFEE",
    "Carbonated RTD Tea and Kombucha": "TEA_COFFEE",
    "RTD Coffee": "TEA_COFFEE",
    "Asian Speciality Drinks": "TEA_COFFEE",
    "Liquid Concentrates": "CONCENTRATES",
    "Powder Concentrates": "CONCENTRATES",
}


def extract_pack_counts(text: str) -> set[int]:
    """All plausible pack sizes mentioned in one product title."""
    counts = set()
    if not isinstance(text, str):
        return counts
    for m in PACK_COUNT_RE.finditer(text):
        for g in m.groups():
            if g:
                n = int(g)
                if MIN_PACK <= n <= MAX_PACK:
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
