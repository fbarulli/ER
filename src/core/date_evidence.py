"""Source dates are stock/review context, not product-identity contradictions."""
from datetime import date
import re


_CALENDAR = re.compile(r"(?<!\d)(?:\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}|\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{4})(?!\d)")
_EXPIRY = re.compile(r"\b(?:best\s+(?:before|by)|use\s+by|expir(?:y|ation|es)|tht|bbd|bbe|exp)\b[^;\n]{0,35}\Z", re.I)
_MANUFACTURE = re.compile(r"\b(?:manufactur(?:ed|e|ing)(?:\s+date)?|production\s+date|mfg|mfd)\b[^;\n]{0,35}\Z", re.I)
_MONTH = re.compile(r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{4})(?!\d)")
_SHORT_YEAR = re.compile(r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{2})(?!\d)")
_EXPIRY_CUE = re.compile(r"\b(?:best\s+(?:before|by)|use\s+by|expir(?:y|ation|es)|tht|bbd|bbe|exp)\b", re.I)
_SHELF_LIFE = re.compile(r"\bshelf\s+life\s*:\s*(\d+)\s*(days?|months?|years?)\b", re.I)

# Every date pattern except _EXPIRY_CUE requires a digit, and _EXPIRY_CUE is
# the only cue that can appear without one. Measured on the lane corpus
# (11,441 rows x 5 columns): 76% of titles, 65% of attribute cells, 83% of
# descriptions, 99.4% of breadcrumbs and 91.5% of categories contain NO digit
# at all, yet all six patterns were scanned over every one of them. The gate
# is exactly sound: it only skips scans that provably cannot match, and the
# surviving _EXPIRY_CUE pass keeps the discovery order the stable sort relies on.
_HAS_DIGIT = re.compile(r"\d").search
_HAS_4_DIGITS = re.compile(r"\d{4}").search
# _EXPIRY_CUE and _SHELF_LIFE are re.IGNORECASE, so their gate runs on a
# lowercased copy with plain `in` tests. Two measured dead ends first:
# an re.IGNORECASE regex hint is SLOWER than the pattern it would gate (0.1362s
# vs 0.1103s over 57,205 texts), and a case-SENSITIVE `in` hint silently missed
# 7 real matches ("Expiration Date: 1 Year", "Coffee with oat drink. ... Gluten
# free"). The gate is applied only to ASCII text, where str.lower() reproduces
# exactly the folding IGNORECASE performs over the letters in these keywords;
# IGNORECASE additionally folds U+017F, U+0130, U+0131 and U+212A, so anything
# else falls through to the unchanged scan.
# Hoisted module-level dispatch: each of these ran once per match (thousands of
# times), paying `re`'s pattern-cache lookup on every call.
_DATE_SEPARATOR = re.compile(r"[-/.]")
_NOT_AT_END = re.compile(r"\bnot\s*$", re.I)
_MEASUREMENT_TAIL = re.compile(
    r"\s*(?:days?|delivery|fl\b|oz\b|lbs?\b|ml\b|cl\b|kg\b|grams?\b)", re.I)
_FORMAT_GUIDANCE = re.compile(r"[^;\n]{0,35}\bDD\s*/\s*MM\s*/\s*YYYY", re.I)

_MONTH_NAMES = {name: month for month, names in enumerate((
    ("jan", "january"), ("feb", "february"), ("mar", "march"),
    ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
    ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
    ("nov", "november"), ("dec", "december"),
), 1) for name in names}
_NAMED_MONTH = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
_NAMED_CALENDAR = re.compile(
    r"\b(?:(?P<day_first>\d{1,2})(?:st|nd|rd|th)?\s+)?"
    r"(?P<month>" + _NAMED_MONTH + r")\.?\s+"
    r"(?:(?P<day_last>\d{1,2})(?:st|nd|rd|th)?,?\s+)?"
    r"(?P<year>\d{4})\b", re.I
)


def _role(prefix):
    expiry = _EXPIRY.search(prefix)
    if expiry and not _NOT_AT_END.search(prefix[:expiry.start()]):
        return "expiry"
    if _MANUFACTURE.search(prefix):
        return "manufacture"
    return "unspecified_calendar_date"


def _by_start(entry: dict) -> int:
    return entry["start"]


def extract_date_evidence(text: object) -> list[dict]:
    """Keep exact spans, date precision and possible orderings; never guess age.

    No scrape timestamp is available, so historical expiry text cannot prove
    that a present listing is expired. Different batches can share a GTIN.
    """
    if not isinstance(text, str):
        return []
    found = []
    occupied = []
    has_digit = _HAS_DIGIT(text) is not None
    # The loop ORDER is unchanged (the result is stable-sorted by start, so
    # equal-start entries keep discovery order); only scans that provably
    # cannot match are skipped. Every gate below is a necessary condition of
    # the pattern it guards, never a heuristic.
    if has_digit:
        # _CALENDAR/_MONTH/_SHORT_YEAR all require one of the three date
        # separators; _NAMED_CALENDAR requires a four-digit year. Measured on
        # the 9,739 digit-bearing texts: the separator gate is free (0.0009s)
        # and prunes 33%, the four-digit gate costs 0.0250s and prunes 84%
        # (NAMED_CALENDAR alone costs 0.1608s over those texts).
        has_separator = "-" in text or "/" in text or "." in text
    if has_digit and has_separator:
        for match in _CALENDAR.finditer(text):
            parts = [part.strip() for part in _DATE_SEPARATOR.split(match.group())]
            a, b, c = map(int, parts)
            tuples = [(a, b, c)] if len(parts[0]) == 4 else [(c, b, a), (c, a, b)]
            dates = set()
            for year, month, day in tuples:
                try:
                    dates.add(date(year, month, day).isoformat())
                except ValueError:
                    pass
            found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                          "role": _role(text[:match.start()]), "precision": "day",
                          "normalized_candidates": sorted(dates),
                          "parse_status": "invalid" if not dates else "ambiguous_order" if len(dates) > 1 else "parsed",
                          "gate_use": "stock_review_context"})
            occupied.append(match.span())
        for match in _MONTH.finditer(text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            role = _role(text[:match.start()])
            if role == "unspecified_calendar_date":
                continue
            month, year = map(int, match.groups())
            found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                          "role": role, "precision": "month",
                          "normalized_candidates": [f"{year:04d}-{month:02d}"] if 1 <= month <= 12 and 1 <= year <= 9999 else [],
                          "parse_status": "parsed" if 1 <= month <= 12 and 1 <= year <= 9999 else "invalid",
                          "gate_use": "stock_review_context"})
            occupied.append(match.span())
        for match in _SHORT_YEAR.finditer(text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            role = _role(text[:match.start()])
            if role == "unspecified_calendar_date":
                continue
            if _MEASUREMENT_TAIL.match(text[match.end():]):
                continue
            first, second = map(int, match.groups())
            partial_dates = set()
            for month, day in ((second, first), (first, second)):
                try:
                    date(2000, month, day)  # Validate components without choosing a year.
                    partial_dates.add(f"--{month:02d}-{day:02d}")
                except ValueError:
                    pass
            possible_month_year = 1 <= first <= 12
            status = ("ambiguous_components" if possible_month_year and partial_dates
                      else "ambiguous_century" if possible_month_year
                      else "missing_year" if partial_dates else "invalid")
            found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                          "role": role, "precision": "unknown" if possible_month_year and partial_dates
                              else "month" if possible_month_year else "day_month",
                          "normalized_candidates": [], "partial_candidates": sorted(partial_dates),
                          "parse_status": status, "gate_use": "stock_review_context"})
            occupied.append(match.span())
    if has_digit and _HAS_4_DIGITS(text):
        for match in _NAMED_CALENDAR.finditer(text):
            year = int(match.group("year"))
            month = _MONTH_NAMES[match.group("month").lower()]
            day_text = match.group("day_first") or match.group("day_last")
            candidates = []
            try:
                if match.group("day_first") and match.group("day_last"):
                    raise ValueError("two day components")
                normalized = date(year, month, int(day_text) if day_text else 1)
                candidates = [normalized.isoformat()] if day_text else [f"{year:04d}-{month:02d}"]
            except ValueError:
                pass
            found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                          "role": _role(text[:match.start()]), "precision": "day" if day_text else "month",
                          "normalized_candidates": candidates,
                          "parse_status": "parsed" if candidates else "invalid",
                          "gate_use": "stock_review_context"})
            occupied.append(match.span())
    low = text.lower() if text.isascii() else None
    if low is None or ("exp" in low or "tht" in low or "bbd" in low or "bbe" in low
                       or "best" in low or ("use" in low and "by" in low)):
        _scan_expiry_cue(text, found, occupied)
    if has_digit and (low is None or "shelf" in low):
        for match in _SHELF_LIFE.finditer(text):
            found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                          "role": "shelf_life", "precision": "duration", "normalized_candidates": [],
                          "duration_value": int(match.group(1)), "duration_unit": match.group(2).lower().rstrip('s'),
                          "parse_status": "parsed", "gate_use": "stock_review_context"})
    if not found:
        return []
    return sorted(found, key=_by_start)


def _scan_expiry_cue(text: str, found: list, occupied: list) -> None:
    for match in _EXPIRY_CUE.finditer(text):
        if any(match.end() <= start <= match.end() + 35 for start, end in occupied):
            continue
        negated = bool(_NOT_AT_END.search(text[:match.start()]))
        format_guidance = bool(_FORMAT_GUIDANCE.match(text[match.end():]))
        found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                      "role": "date_format_reference" if negated or format_guidance else "expiry_reference", "precision": "unknown", "normalized_candidates": [],
                      "parse_status": "reference_only", "gate_use": "stock_review_context"})
