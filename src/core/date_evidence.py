"""Source dates are stock/review context, not product-identity contradictions."""
from datetime import date
import re


_CALENDAR = re.compile(r"(?<!\d)(?:\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}|\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{4})(?!\d)")
_EXPIRY = re.compile(r"\b(?:best\s+(?:before|by)|use\s+by|expir(?:y|ation|es)|tht)\b[^;\n]{0,35}\Z", re.I)
_MANUFACTURE = re.compile(r"\b(?:manufactur(?:ed|e|ing)(?:\s+date)?|production\s+date)\b[^;\n]{0,35}\Z", re.I)
_MONTH = re.compile(r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{4})(?!\d)")
_SHORT_YEAR = re.compile(r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{2})(?!\d)")
_EXPIRY_CUE = re.compile(r"\b(?:best\s+(?:before|by)|use\s+by|expir(?:y|ation|es)|tht)\b", re.I)
_SHELF_LIFE = re.compile(r"\bshelf\s+life\s*:\s*(\d+)\s*(days?|months?|years?)\b", re.I)


def _role(prefix):
    expiry = _EXPIRY.search(prefix)
    if expiry and not re.search(r"\bnot\s*$", prefix[:expiry.start()], re.I):
        return "expiry"
    if _MANUFACTURE.search(prefix):
        return "manufacture"
    return "unspecified_calendar_date"


def extract_date_evidence(text: object) -> list[dict]:
    """Keep exact spans, date precision and possible orderings; never guess age.

    No scrape timestamp is available, so historical expiry text cannot prove
    that a present listing is expired. Different batches can share a GTIN.
    """
    if not isinstance(text, str):
        return []
    found = []
    occupied = []
    for match in _CALENDAR.finditer(text):
        parts = [part.strip() for part in re.split(r"[-/.]", match.group())]
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
                      "normalized_candidates": [f"{year:04d}-{month:02d}"] if 1 <= month <= 12 else [],
                      "parse_status": "parsed" if 1 <= month <= 12 else "invalid",
                      "gate_use": "stock_review_context"})
        occupied.append(match.span())
    for match in _SHORT_YEAR.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        role = _role(text[:match.start()])
        if role == "unspecified_calendar_date":
            continue
        if re.match(r"\s*(?:days?|delivery|fl\b|oz\b|lbs?\b|ml\b|cl\b|kg\b|grams?\b)", text[match.end():], re.I):
            continue
        month = int(match.group(1))
        found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                      "role": role, "precision": "month", "normalized_candidates": [],
                      "parse_status": "ambiguous_century" if 1 <= month <= 12 else "invalid",
                      "gate_use": "stock_review_context"})
        occupied.append(match.span())
    for match in _EXPIRY_CUE.finditer(text):
        if any(match.end() <= start <= match.end() + 35 for start, end in occupied):
            continue
        negated = bool(re.search(r"\bnot\s*$", text[:match.start()], re.I))
        format_guidance = bool(re.match(r"[^;\n]{0,35}\bDD\s*/\s*MM\s*/\s*YYYY", text[match.end():], re.I))
        found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                      "role": "date_format_reference" if negated or format_guidance else "expiry_reference", "precision": "unknown", "normalized_candidates": [],
                      "parse_status": "reference_only", "gate_use": "stock_review_context"})
    for match in _SHELF_LIFE.finditer(text):
        found.append({"raw_match": match.group(), "start": match.start(), "end": match.end(),
                      "role": "shelf_life", "precision": "duration", "normalized_candidates": [],
                      "duration_value": int(match.group(1)), "duration_unit": match.group(2).lower().rstrip('s'),
                      "parse_status": "parsed", "gate_use": "stock_review_context"})
    return sorted(found, key=lambda entry: entry["start"])
