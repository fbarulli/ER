"""pipeline.py — THE official data pipeline, all transformations in ONE
module (owner directive: smash the DATA_PIPE folder into one file).

Sections (in dependency order):
  1. extraction    — normalize_text, volume/pack/flavor extractors, extract_all
  2. gating        — three_way_gate (owner's second_gating.py, verbatim)
  3. similarity    — jaccard (the old embedding-similarity stub was dead:
                     zero consumers; the zero-shot lane is src/training/zero_shot_sims)
  4. canonical     — per-GTIN canonical generation + clean_sku_text +
                     load_canonical_map (owner's second_canonical.py)
  5. numbers       — number-token reference + strip (95.2% coverage)
  6. pipeline      — run_within_brand_pipeline (canonical + gate CSVs)
  7. pairs         — build_training_data (the OFFICIAL training pairs)

Public surface (old DATA_PIPE imports keep working):
  normalize_text, extract_all, three_way_gate, jaccard_similarity,
  generate_canonical, clean_sku_text,
  load_canonical_map, run_within_brand_pipeline, build_training_data,
  strip_number_tokens, build_reference, census_texts, token_verdict
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.columns import (
    CANONICAL_DATASET_REQUIRED_COLUMNS as CANONICAL_DATASET_REQUIRED_COLUMNS_REQUIRED,
)
from core.columns import COLUMN_ALIASES, DATA_PREP_REQUIRED_COLUMNS, source_row_pairs
from core.common import (
    DATA_DIR,
    RESULTS,
    F,
    data_cfg,
    load_config,
    training_cfg,
    vocabulary,
)
from core.run_log import RunLogger
from core.pair_identity import PairIdentity
from ner.ner_product_attributes import extract_title_attributes, parse_attribute_details
from core.schemas import (
    CanonicalRecord,
    ExtractedAttributes,
    GateResult,
    check_canonical_records_frame,
    check_gate_results_frame,
    check_verdict_map,
    require_populated_source_rows,
    upgrade_canonical_records_frame,
)
_LOG = RunLogger(__name__)

from core.critical_attributes import (
    CRITICAL_ATTRIBUTE_DIMENSIONS,
    categorical_conflict,
    extract_critical_claims,
    extract_description_claims,
    extract_made_from_tokens,
    source_consistency_flags,
    volumes_compatible,
)
from core.tracing import (
    CENSUS_TOP_N,
    DETAIL_CELL_CHARS,
    ENTITY_ROW_CAP,
    ENTITY_SAMPLE_PER_REASON,
    trace_path,
)

# Column contracts of the two frames this module consumes. They are DIFFERENT
# and the handoff between them is not a rename: stage 1 (run_within_brand_pipeline)
# reads the RAW export, stage 2 (build_training_data) reads the DEDUPED dataset
# whose columns were canonicalized by the dedupe tier. Both are declared here so
# the trace states, per run, which contract each stage actually received.
# The column contracts come from the SSOT (config/paths.yaml via
# core.columns), never from a literal tuple here. They were hardcoded and
# drifted from the mapping that already declared every column on both sides.
RAW_EXPORT_REQUIRED_COLUMNS = DATA_PREP_REQUIRED_COLUMNS
CANONICAL_DATASET_REQUIRED_COLUMNS = CANONICAL_DATASET_REQUIRED_COLUMNS_REQUIRED
# The per-entity caps live in core.tracing (one policy, one place) with their
# justification: exact per-reason census rows + a bounded stratified sample.
ENTITY_PER_REASON = ENTITY_SAMPLE_PER_REASON
ENTITY_TOTAL_CAP = ENTITY_ROW_CAP

# ── bounded trace cells ────────────────────────────────────────────────────
# The consolidated trace is a READABLE census, not a dump. Two shapes had grown
# unbounded here and blew the file up to ~12 MB for three runs (the contract's
# own justification is ~1.5 MB):
#   * a free-text reason used as a bucket label is a PER-PAIR literal (gate
#     reasons embed the values that produced them, "...: mode_flavor:orange|apple"),
#     so it minted one group row per pair — 589 rows / 257 singleton buckets on
#     the live census. The CATEGORY (text before the first ':') collapses those
#     to 9 stable, non-singleton buckets.
#   * one `count_rows(..., limit=None)` cell carried every distinct value; the
#     dimension-conflict cell alone was 1.07 MB (13,067 strings).
# So: bucket by category, list the top-N values, and state the remainder as
# NUMBERS (`distinct` / `others_buckets` / `others`) — nothing is hidden, but a
# wide distribution costs two integers instead of a megabyte.
#
# The two CAPS live once, in core.tracing (CENSUS_TOP_N / DETAIL_CELL_CHARS,
# imported at the top of this module): the pipeline reads them instead of
# re-spelling the literals, so the census caps can never drift between the two
# producers. CENSUS_CELL_BYTES is this module's historical name for the
# detail-cell byte cap, kept as an alias for its call sites and the tests.
CENSUS_CELL_BYTES = DETAIL_CELL_CHARS
REASON_LABEL_CHARS = 96


def _reason_category(reason: object) -> str:
    """The reason's CATEGORY: the text before its first ':' (values follow it).

    Gate reasons append the evidence that produced them ("Declared product
    identity differs or is incomplete: flavor,organic"), so the raw string is a
    per-pair literal. Measured on data/gate_results.csv: 138 distinct reasons ->
    9 categories, none of them a singleton.
    """
    text = str(reason or "").strip()
    return text.split(":", 1)[0].strip()[:REASON_LABEL_CHARS]


def _bounded_census(values, *, top_n: int = CENSUS_TOP_N) -> dict[str, object]:
    """The top-N `"value=n"` entries plus an explicit remainder, as NUMBERS.

    `distinct` and `others`/`others_buckets` are always stated, so a bounded cell
    still reports the exact shape of the distribution it summarises.
    """
    series = pd.Series(list(values))
    if series.empty:
        return {"top": [], "distinct": 0, "others_buckets": 0, "others": 0}
    couples = series.astype(str).value_counts()
    head = couples.head(int(top_n))
    return {
        "top": [f"{name}={int(count)}" for name, count in head.items()],
        "distinct": int(len(couples)),
        "others_buckets": int(max(0, len(couples) - len(head))),
        "others": int(couples.iloc[len(head):].sum()),
    }


def unit_change_counts(incoming: int, outgoing: int) -> tuple[int | None, int, bool]:
    """A step's counts, as a funnel ONLY when the units do not change.

    Some steps are unit CHANGES, not filters: one block expands into its
    candidate pairs, one pair into its two directions, one source row into a row
    plus its canonical, one listing into several vocabulary entries. Those
    legitimately emit MORE than they received, and stating an in/out pair for
    them makes `dropped_count` negative — which the row contract forbids
    (dropped = in - out, and a drop cannot be negative). Such a step therefore
    states only its output and flags the unit change.

    Public because it is the ONE declaration of this rule: pipeline's own steps
    use it directly, and modules that already depend on pipeline
    (training.build_reference) import it rather than restating the arithmetic.
    """
    incoming, outgoing = int(incoming), int(outgoing)
    if outgoing > incoming:
        return None, outgoing, True
    return incoming, outgoing, False


def _capped_detail(detail: dict, *, max_bytes: int = CENSUS_CELL_BYTES) -> dict:
    """Bound a detail payload's serialized size, stating exactly what was elided.

    Values keep their identity (a long string is truncated IN PLACE with its
    elided length recorded; a long list keeps its head with its elided item count
    recorded), so a cap never turns a readback into a different one. If even that
    is over budget the payload degrades to its own key list plus the byte count —
    a bounded cell that says so, never a megabyte.
    """
    text = json.dumps(detail, sort_keys=True, default=str)
    size = len(text.encode())
    if size <= int(max_bytes):
        return detail
    capped: dict[str, object] = {}
    for key, value in detail.items():
        if isinstance(value, str) and len(value) > 512:
            capped[key] = f"{value[:512]}…<elided {len(value) - 512} chars>"
        elif isinstance(value, list) and len(value) > CENSUS_TOP_N:
            capped[key] = [
                *value[:CENSUS_TOP_N],
                f"…<elided {len(value) - CENSUS_TOP_N} items>",
            ]
        else:
            capped[key] = value
    capped["detail_bytes_before"] = size
    capped["detail_truncated"] = True
    if len(json.dumps(capped, sort_keys=True, default=str).encode()) <= int(max_bytes):
        return capped
    return {
        "detail_keys": sorted(str(key) for key in detail),
        "detail_bytes_before": size,
        "detail_truncated": True,
        "note": "payload exceeded the trace cell budget; its keys are listed",
    }

# ============================================================================
# EXTRACTION
# ============================================================================


def _load_stopwords(key: str) -> set:
    """STOPWORDS / MINIMAL_STOPWORDS from config/vocabulary.json
    (SSOT via config/paths.yaml files.stopwords)."""
    values = vocabulary().get(key)
    if not isinstance(values, list):
        raise SystemExit(f"vocabulary.{key} is missing or malformed")
    return set(values)


def _load_concept_folds() -> dict[str, str]:
    """CONCEPT_FOLDS from config/vocabulary.json — SAME SSOT file as the
    word lists (owner directive 2026-09-08: folds belong with the stopwords,
    NOT in a config yaml). KEY wins; VALUE folds into it."""
    folds = vocabulary().get("CONCEPT_FOLDS")
    if folds is None:
        raise SystemExit(
            "CONCEPT_FOLDS missing from config/vocabulary.json — the concept-folding "
            "discipline requires it (no silent fallback to identity)"
        )
    return dict(folds)


STOPWORDS = _load_stopwords("STOPWORDS")

# normalize_text now LIVES in core.text (see its docstring for why it moved).
# Re-exported here, unaltered, because ten modules import it from this path
# (core.sku_identity, core.record_linkage, core.model_input, the training
# lane, tests). One definition, two import paths, no cycle.
from core.text import normalize_text  # noqa: E402,F401


def extract_volume_from_title(title: str) -> dict:
    """Adapter over core.text's single volume PARSE; attribution from config.

    WHY THIS EXISTS. This function used to carry its own two regexes
    (VOLUME_PATTERN_US_EXT / VOLUME_PATTERN_METRIC_EXT) plus its own
    "1 000 ml" repair — a second full parse of the same text core.text's
    VOLUME_RE already performed, so the two paths could silently disagree.
    Both patterns are gone; every field below derives from ONE parse
    (core.text.extract_volume_match) plus the SAME config unit table the
    converter (core.unit_canonicalization.canonical_volume_ml) reads:
      * conversion = whole-ml half-up, round 2 kept for stored identity —
        the shipped canonical volume_set stores values like 1893.0, not
        bucketed 5ml grid values, so identity is 1ml, not bucket;
      * confidence = config units.volume entry for the captured spelling,
        raised by decimal_confidence only when the value was written WITH a
        decimal separator (was inline if/elif literals only in this module);
      * parse_status = the entry's family name verbatim.
    The parse itself no longer duplicates; the conversion stays local on
    purpose (bucketed identity would lose the .5 of '87.5 Millilitre' the
    observed-notation contract requires) and is unit-tested in
    tests/test_sweetener_assignment.py against both spellings.
    """
    from core.text import extract_volume_match, _volume_entry
    from core.unit_canonicalization import canonical_volume_ml

    # Keep fractions, nutrition context, and original spans intact. The shared
    # reader handles separators before selecting a physical package volume.
    value, unit, _ambiguous, raw = extract_volume_match(str(title or ""))
    # A captured value of 0 is never a volume ("0.5 l" fragments, URL slugs
    # with dimension tokens) — canonical_volume_ml rejects it, so treat it
    # the same as no mention instead of raising mid-pipeline.
    if value is None or unit is None or float(value) <= 0:
        return {"volume_ml": 0.0, "confidence": 0.0, "raw_match": "",
                "parse_status": "no_volume_mention"}
    entry = _volume_entry(unit)
    if entry is None:
        return {"volume_ml": 0.0, "confidence": 0.0, "raw_match": "",
                "parse_status": "no_volume_mention"}
    ml = canonical_volume_ml(value, unit)
    if ml <= 0:
        return {"volume_ml": 0.0, "confidence": 0.0, "raw_match": "",
                "parse_status": "no_volume_mention"}
    # "0.98 if the value was written as a decimal" — read the numeric prefix,
    # not the whole match, so a '.' inside a unit cannot trip it.
    has_decimal = bool(_DECIMAL_PREFIX_RE.match(raw))
    confidence = entry.confidence
    if has_decimal and entry.decimal_confidence is not None:
        confidence = entry.decimal_confidence
    return {"volume_ml": ml, "confidence": confidence, "raw_match": raw,
            "parse_status": entry.family}


# ── pack-evidence scan vocabulary ──────────────────────────────────────────
# Every pattern _PackEvidenceReader scans with is a constant: the count/number
# vocabulary, the container and measurement tails, the phase patterns built
# from them, and the word-count table. Compiling them once at import keeps the
# ~1.25M re.finditer/re.search/re.match calls this family issues on a 10k
# catalog (per text field, per row) out of the module-level `re` wrapper and
# its per-call cache lookup, and drops the per-call pattern rebuilding in
# prepare(). Only the pattern OBJECTS move: the pattern source text is
# unchanged, so every match, span, raw_match and evidence record is unchanged.
_DECIMAL_PREFIX_RE = re.compile(r"\s*\d+(?:[.,]\d)")

_PACK_COUNT_TOKEN = r"([1-9]\d{0,2}(?:[.,]\d{3})+|\d+)(?!\d|[.,]\d)"
_PACK_NUMBER = r"(?<![\w$€£])(?<!\d[.,])" + _PACK_COUNT_TOKEN
_PACK_CONTAINERS = r"(?:bottles?|bt|cans?|tins?|cartons?|boxes?|packets?|sachets?|bags?)"
# The tail of an x-multiplier must be a RECOGNIZED measurement unit (or
# container word), never any letter, and one descriptive word may sit between
# the count and a MEASUREMENT unit: "12x1 mineralwasser"/"12x1 pet" emitted the
# false raw spans `12x1 m`/`12x1 p`, '12x1 pet bottles' is not a recognized
# span, and '6x20 organic cl' is.
_PACK_MEASUREMENT_TAIL = r"(?:fl\.?\s?oz\.?|ltr|lt|ml|cl|dl|cc|kcal|mg|kg|lbs?|gr|fz|g\b|oz\b|l\b)"
_PACK_MULTIPLIER_TAIL = (rf"(?:\s*(?:{_PACK_MEASUREMENT_TAIL}|{_PACK_CONTAINERS})\b"
                         rf"|\s+[a-z]+\s*{_PACK_MEASUREMENT_TAIL})")
_PACK_PATTERNS = tuple(
    (kind, re.compile(pattern, re.I), role) for kind, pattern, role in (
        ("nested", rf"{_PACK_NUMBER}\s*[x×]\s*(\d+)\s*(?:{_PACK_CONTAINERS}\s*)?(?:[x×]|/)\s*\d+(?:[.,]\d+|[.,]\s+\d{{1,2}})?{_PACK_MULTIPLIER_TAIL}", "unit_count"),
        ("multiplier", rf"{_PACK_NUMBER}\s*[x×]\s*(?:pack\s*)?\d+(?:[.,]\d+|[.,]\s+\d{{1,2}})?{_PACK_MULTIPLIER_TAIL}", "unit_count"),
        # Retail titles also use a terminal count without a unit size:
        # "Hip Pop - Blueberry Ginger - kombucha - 12x". Restrict it to
        # a suffix; model codes and unfinished size multipliers stay unknown.
        ("multiplier", rf"{_PACK_NUMBER}\s*[x×]\s*[)\]]?\s*$", "unit_count"),
        ("pack_of", rf"\b(?:packs?|packages?)\s+of\s*{_PACK_COUNT_TOKEN}\b", "unit_count"),
        ("pack_of", rf"\bcases?\s+of\s*{_PACK_COUNT_TOKEN}\b", "unit_count"),
        ("count", rf"{_PACK_NUMBER}\s*[- ]?\s*(?:pcs?|pieces?|packs?|packages?|pk|units?|ct|count)\b", "unit_count"),
        ("compact", rf"\bpack\s*[- ]?\s*{_PACK_COUNT_TOKEN}\b", "unit_count"),
        ("container", rf"{_PACK_NUMBER}\s*(?:glass\s*)?{_PACK_CONTAINERS}\b", "unit_count"),
        ("count", rf"{_PACK_NUMBER}\s*cases?\b", "outer_count"),
    ))
_PACK_CURRENCY_TAIL_RE = re.compile(r"[$€£]\s*$")
_PACK_WEIGHT_TAIL_RE = re.compile(r"(?:gross\s+)?weight\W*$", re.I)
_PACK_PREFIX_MULTIPLIER_RE = re.compile(rf"{_PACK_NUMBER}\s*[x×]\s+(?=[a-z])", re.I)
_PACK_DOSE_BETWEEN_RE = re.compile(r"[.;\n]|\b(?:dose|daily|times|servings?)\b", re.I)
_PACK_SET_OF_RE = re.compile(rf"\b(?:set|bundle)\s+of\s*{_PACK_COUNT_TOKEN}\b", re.I)
_PACK_SET_LIST_RE = re.compile(r"\s*(?:flavou?rs?|choices?|colou?rs?|options?)\b", re.I)
_PACK_UNIT_COUNT_RE = re.compile(r"\bunit count\s+(\d+)(?:\.0+)?\s+count\b", re.I)
_PACK_WORD_COUNTS = dict(zip(
    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"),
    range(1, 13), strict=True,
))
_PACK_WORD_PACKS_RE = re.compile(r"\b(" + "|".join(_PACK_WORD_COUNTS) + r")\s*[- ]\s*packs?\b", re.I)
_PACK_STICKS_PER_BOX_RE = re.compile(rf"(?<![\d.,]){_PACK_COUNT_TOKEN}\s*sticks?\s+per\s+box\b", re.I)
_PACK_BOXES_RE = re.compile(rf"{_PACK_NUMBER}\s*boxes\b", re.I)
_PACK_TOTAL_PREFIX_RE = re.compile(rf"{_PACK_NUMBER}\s+$")
_PACK_TOTAL_WORD_RE = re.compile(r"[ .()]*total\s*", re.I)
_PACK_OUTER_RE = re.compile(rf"\(\s*(?:pack(?:age)?\s+of\s*|[x×]\s*){_PACK_COUNT_TOKEN}\s*\)", re.I)
# Counts are parsed by deleting the metric decimal/thousands separators; a
# str.translate table is the same deletion as re.sub(r'[.,]', '', ...) without
# a regex call (this runs on every recognized match).
_PACK_DIGITS_STRIP = str.maketrans("", "", ".,")


@lru_cache(maxsize=8)
def _bulk_container_re(terms: tuple[str, ...]) -> re.Pattern[str]:
    """`\\b(?:term|...)\\b` over the configured bulk-container vocabulary.

    The vocabulary is config-owned and constant for the process, so the
    alternation, the escaping of every term and the pattern compilation are
    built once instead of on every fused row (11,441 calls on the 10k cohort).
    """
    escaped = "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in terms)
    return re.compile(r"\b(?:" + escaped + r")\b", re.I)


class _PackEvidenceReader:
    """One title's pack-evidence scan, phase by phase.

    The phases below run in ONE fixed order inside read(); the statements are
    the pre-refactor body verbatim, so the returned evidence list (order,
    keys, values, rule names) is byte-identical.

    Phase map:
      prepare            — text, volume measurements, package subset,
                           configured confidence, count regex vocabulary,
                           the family patterns
      scan_family_patterns — the ordered span-claimed pattern scan
      scan_prefix_multipliers — a multiplier preceding the product name
      scan_set_and_count_words — set/bundle + retail unit counts + word
                           counts + the sticks-per-box hierarchy
      reconcile_totals   — whitespace-multiplier proof via stated totals
      reinterpret_outer  — "(Pack of n)" outer/inner hierarchy rewrite
    """

    def __init__(self, title: str) -> None:
        self._title = title
        self.evidence: list[dict] = []
        self.occupied: list[tuple[int, int]] = []

    # ── phase: prepare ──────────────────────────────────────────────────────

    def prepare(self) -> dict:
        """Reader + regex vocabulary (shared by every scan phase)."""
        text = str(self._title or "")
        from core.text import extract_volume_evidence
        measurements = extract_volume_evidence(text)
        confidence = data_cfg().extraction.pack_confidence
        # Count tokens must include their entire number: decimal and price tails
        # cannot masquerade as integer quantities. Grouped thousands are counts.
        # The vocabulary and its patterns are module-level constants (_PACK_*):
        # they are identical for every title, so they are compiled once at
        # import instead of being rebuilt (9 f-strings) per title.
        return {
            "text": text,
            "measurements": measurements,
            "package_measurements": [m for m in measurements if m['role'] == 'package_volume'],
            "confidence": confidence,
            "count_token": _PACK_COUNT_TOKEN,
            "number": _PACK_NUMBER,
            "patterns": _PACK_PATTERNS,
        }

    # ── phase: the family scan ──────────────────────────────────────────────

    def scan_family_patterns(self, ctx: dict) -> None:
        """Ordered span-claimed scan over every recognized family."""
        text = ctx["text"]
        measurements = ctx["measurements"]
        confidence = ctx["confidence"]
        occupied = self.occupied
        evidence = self.evidence
        for kind, pattern, role in ctx["patterns"]:
            for match in pattern.finditer(text):
                start = match.start()
                if occupied and any(a <= start < b for a, b in occupied):
                    continue
                if kind == "compact" and any(
                    entry["start"] == match.start(1) for entry in measurements
                ):
                    continue
                if kind == "pack_of" and any(
                    entry["start"] == match.start(1) for entry in measurements
                ):
                    # "8 pack of 16 Fl Oz" states eight units, not sixteen.
                    # A quantity carrying a volume unit cannot be a pack count.
                    continue
                if kind == "compact" and text[match.end():].startswith(")") and re.search(
                    rf"\b{int(match.group(1).translate(_PACK_DIGITS_STRIP)) + 1}\)",
                    text[match.end() + 1:],
                ):
                    # "Combo Pack - 1) product A & 2) product B" is a list.
                    continue
                # Currency followed by whitespace still denotes a price.
                # MEASURED NEGATIVE (r17): bounding these end-anchored readers
                # with search(text, 0, start) instead of search(text[:start])
                # was 1.675 s -> 1.724 s on bench_pipeline (2000 rows), so the
                # prefix slice stays.
                if _PACK_CURRENCY_TAIL_RE.search(text[:start]):
                    continue
                # GDSN weight declarations ("gross weight: 527 unit (specific) …
                # centiliters") are prose measurements, not a retail bundle: a
                # count immediately preceded by a weight label is skipped.
                if _PACK_WEIGHT_TAIL_RE.search(text[:start]):
                    continue
                count = int(match.group(1).translate(_PACK_DIGITS_STRIP))
                if kind == "nested":
                    count *= int(match.group(2))
                if count <= 0:
                    continue
                occupied.append(match.span())
                evidence.append({"count": count, "confidence": confidence[kind],
                                 "role": role, "raw_match": match.group(0),
                                 "start": start, "end": match.end(), "rule": kind})

    # ── phase: multiplier before the product name ───────────────────────────

    def scan_prefix_multipliers(self, ctx: dict) -> None:
        """A multiplier can precede the product name, not just its unit size.

        Require a physical-package measurement after it and reject dosage-only
        text; bare model codes and unproved whitespace counts stay unknown."""
        text = ctx["text"]
        confidence = ctx["confidence"]
        package_measurements = ctx["package_measurements"]
        occupied = self.occupied
        evidence = self.evidence
        for match in _PACK_PREFIX_MULTIPLIER_RE.finditer(text):
            start = match.start()
            if occupied and any(a <= start < b for a, b in occupied):
                continue
            following = next((m for m in package_measurements
                              if match.end() <= m['start'] and m['start'] - match.end() <= 100), None)
            if following is None:
                continue
            between = text[match.end():following['start']]
            if _PACK_DOSE_BETWEEN_RE.search(between):
                continue
            end = following['end']
            evidence.append({'count': int(match.group(1).translate(_PACK_DIGITS_STRIP)),
                             'confidence': confidence['multiplier'], 'role': 'unit_count',
                             'raw_match': text[start:end], 'start': start,
                             'end': end, 'rule': 'multiplier'})
            occupied.append((start, end))

    # ── phase: set/bundle + retail counts + word counts + sticks hierarchy ──

    def scan_set_and_count_words(self, ctx: dict) -> None:
        """set/bundle lane (requires package measurements), retail unit-count
        lane, word-count lane, and the sticks-per-box hierarchy (deterministic
        append order)."""
        text = ctx["text"]
        confidence = ctx["confidence"]
        evidence = self.evidence
        if ctx["package_measurements"]:
            for match in _PACK_SET_OF_RE.finditer(text):
                if _PACK_SET_LIST_RE.match(text[match.end():]):
                    continue
                evidence.append({'count': int(match.group(1).translate(_PACK_DIGITS_STRIP)),
                                 'confidence': confidence['pack_of'], 'role': 'unit_count',
                                 'raw_match': match.group(0), 'start': match.start(),
                                 'end': match.end(), 'rule': 'pack_of'})
        # Retail metadata is count evidence only when its unit is Count, not
        # fluid ounces or a mass; decimal .00 is an integer count here.
        for match in _PACK_UNIT_COUNT_RE.finditer(text):
            if int(match.group(1)):
                evidence.append({'count': int(match.group(1)), 'confidence': confidence['count'],
                                 'role': 'unit_count', 'raw_match': match.group(0),
                                 'start': match.start(), 'end': match.end(), 'rule': 'count'})
        for match in _PACK_WORD_PACKS_RE.finditer(text):
            evidence.append({"count": _PACK_WORD_COUNTS[match.group(1).lower()],
                             "confidence": confidence["count"], "role": "unit_count",
                             "raw_match": match.group(0), "start": match.start(),
                             "end": match.end(), "rule": "count"})
        inner = _PACK_STICKS_PER_BOX_RE.search(text)
        if inner:
            inner_count = int(inner.group(1).translate(_PACK_DIGITS_STRIP))
            self.evidence.append({"count": inner_count, "confidence": confidence["count"],
                                  "role": "inner_count", "raw_match": inner.group(0),
                                  "start": inner.start(), "end": inner.end(), "rule": "count"})
            for entry in list(self.evidence):
                if entry["rule"] == "compact" and entry["role"] == "unit_count":
                    entry["role"] = "outer_count"
                    entry["hierarchy_ambiguous"] = True
                    start, end = min(entry["start"], inner.start()), max(entry["end"], inner.end())
                    self.evidence.append({"count": inner_count * entry["count"],
                                          "confidence": min(entry["confidence"], confidence["count"]),
                                          "role": "derived_inner_total", "hierarchy_ambiguous": True,
                                          "raw_match": text[start:end], "start": start, "end": end,
                                          "rule": "nested"})
            outer = _PACK_BOXES_RE.search(text)
            if outer:
                start, end = min(outer.start(), inner.start()), max(outer.end(), inner.end())
                self.evidence.insert(0, {"count": inner_count * int(outer.group(1).translate(_PACK_DIGITS_STRIP)),
                                         "confidence": confidence["nested"], "role": "unit_count",
                                         "raw_match": text[start:end], "start": start, "end": end,
                                         "rule": "nested"})

    # ── phase: whitespace-multiplier proof via stated totals ────────────────

    def reconcile_totals(self, ctx: dict) -> None:
        """Two passes over the measurements: the stated-total proof and the
        package/total ratio (the ratio needs a REAL total_volume)."""
        # Whitespace alone is not a multiplier. A nearby explicitly stated total
        # can prove the relation, e.g. "6 330 ml (Total 1980 ml)".
        from core.unit_canonicalization import canonical_volume_ml
        text = ctx["text"]
        confidence = ctx["confidence"]
        measurements = ctx["measurements"]
        occupied = self.occupied
        for unit_entry, total_entry in zip(measurements, measurements[1:]):
            prefix = _PACK_TOTAL_PREFIX_RE.search(text[:unit_entry["start"]])
            between = text[unit_entry["end"]:total_entry["start"]]
            if prefix is None or not _PACK_TOTAL_WORD_RE.fullmatch(between):
                continue
            if occupied and any(start <= prefix.start() < end for start, end in occupied):
                continue
            count = int(prefix.group(1).translate(_PACK_DIGITS_STRIP))
            unit_volume = canonical_volume_ml(unit_entry["value"], unit_entry["unit"])
            total_volume = canonical_volume_ml(total_entry["value"], total_entry["unit"])
            if count > 0 and unit_volume > 0 and math.isclose(count * unit_volume, total_volume):
                start, end = prefix.start(), total_entry["end"]
                self.evidence.append({"count": count, "confidence": confidence["multiplier"],
                                      "role": "unit_count", "raw_match": text[start:end],
                                      "start": start, "end": end, "rule": "multiplier"})
        for total_entry in measurements:
            if total_entry["role"] != "total_volume":
                continue
            for unit_entry in measurements:
                if (unit_entry["role"] != "package_volume"
                    or unit_entry["end"] >= total_entry["start"]
                    or unit_entry["unit"] != total_entry["unit"]
                    or unit_entry["value"] <= 0):
                    continue
                ratio = total_entry["value"] / unit_entry["value"]
                count = round(ratio)
                if count <= 1 or not math.isclose(ratio, count):
                    continue
                if any(entry["role"] == "unit_count" for entry in self.evidence):
                    break
                start, end = unit_entry["start"], total_entry["end"]
                self.evidence.append({"count": count, "confidence": confidence["multiplier"],
                                      "role": "unit_count", "raw_match": text[start:end],
                                      "start": start, "end": end, "rule": "multiplier"})
                break

    # ── phase: "(Pack of n)" outer reinterpretation ─────────────────────────

    def reinterpret_outer(self, ctx: dict) -> None:
        """Rewrite the hierarchy when an outer pack surrounds inner counts."""
        # "4 x 250ml (Pack of 2)" describes two inner four-packs. Preserve
        # levels and a proven physical-unit total instead of picking inner four.
        text = ctx["text"]
        confidence = ctx["confidence"]
        outer = _PACK_OUTER_RE.search(text)
        inner_units = [e for e in self.evidence if e['role'] == 'unit_count' and e['rule'] == 'multiplier'
                       and (outer is None or e['end'] <= outer.start())]
        if outer and inner_units:
            inner = inner_units[0]
            outer_count = int(outer.group(1).translate(_PACK_DIGITS_STRIP))
            start, end = inner['start'], outer.end()
            for entry in self.evidence:
                if entry['role'] == 'unit_count':
                    entry['role'] = 'outer_count' if entry['start'] >= outer.start() else 'inner_count'
            self.evidence.insert(0, {'count': inner['count'] * outer_count,
                                     'confidence': confidence['nested'], 'role': 'unit_count',
                                     'raw_match': text[start:end], 'start': start, 'end': end,
                                     'rule': 'nested'})
        elif len({e['count'] for e in self.evidence if e['role'] == 'unit_count'}) > 1:
            # Unresolved competing counts are not a license to select whichever
            # happens to match another record.
            for entry in self.evidence:
                if entry['role'] == 'unit_count':
                    entry['hierarchy_ambiguous'] = True

    # ── orchestration ───────────────────────────────────────────────────────

    def read(self) -> list[dict]:
        """Run the load-bearing scan order."""
        ctx = self.prepare()
        self.scan_family_patterns(ctx)
        self.scan_prefix_multipliers(ctx)
        self.scan_set_and_count_words(ctx)
        self.reconcile_totals(ctx)
        self.reinterpret_outer(ctx)
        return self.evidence


@lru_cache(maxsize=65536)
def _extract_pack_evidence_cached(title: str) -> list[dict]:
    return _PackEvidenceReader(title).read()


def extract_pack_evidence(title: str) -> list[dict]:
    """Retain physical-unit and outer-package quantities with original spans —
    see _PackEvidenceReader.read (phases, order, output bytes identical).

    Memoized: the scan is a pure function of ``title`` and the same column is
    re-scanned by several phases (volume resolution, evidence ledger). Callers
    only read/spread the entries, so the cached list is safe to share.
    """
    if not isinstance(title, str):
        return _PackEvidenceReader(title).read()
    return _extract_pack_evidence_cached(title)


@lru_cache(maxsize=65536)
def _extract_pack_from_title_cached(title: str) -> tuple:
    evidence = _extract_pack_evidence_cached(title)
    units = [entry for entry in evidence if entry["role"] == "unit_count"]
    if units:
        return units[0]["count"], units[0]["confidence"]
    ambiguous_outer = [entry for entry in evidence if entry.get("hierarchy_ambiguous") and entry["role"] == "outer_count"]
    if ambiguous_outer:
        return ambiguous_outer[0]["count"], ambiguous_outer[0]["confidence"]
    # Outer cases do not state the number of consumer units in each case.
    return 1, 0.0


# Attribute-cell readers are called once per row; their patterns are constant,
# so they are compiled once here instead of through the module-level `re`
# wrapper (and its per-call cache lookup) on every row.
_ATTR_VOLUME_RE = re.compile(
    r"\bVolume:\s*(.*?)(?=;|\n|\s+[A-Za-z][A-Za-z ]*:|$)", re.IGNORECASE)
_ATTR_VOLUME_NUMBER_RE = re.compile(r"\d+(?:[.,]\s*\d+)?")
_ATTR_COUNT_PER_UNIT_RE = re.compile(r"Count per Unit:\s*(\d+)(?!\d|[.,/]\s*\d)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def extract_pack_from_title(title: str) -> tuple:
    if not isinstance(title, str):
        return _extract_pack_from_title_cached(str(title))
    return _extract_pack_from_title_cached(title)


def parse_attribute_volume_pack(
    attr_str: str,
) -> tuple[float, float, int, float]:  # (vol_ml, vol_conf, pack_qty, pack_conf)
    from core.unit_canonicalization import canonical_pack_count, canonical_volume_ml

    vol_ml = 0.0
    vol_conf = 0.0
    pack_qty = 1
    pack_conf = 0.0
    if not attr_str or attr_str == "nan":
        return vol_ml, vol_conf, pack_qty, pack_conf
    from core.text import extract_volume_evidence
    m_vol = _ATTR_VOLUME_RE.search(attr_str)
    if m_vol:
        declared = m_vol.group(1).strip()
        measurements = extract_volume_evidence(declared)
        if measurements and measurements[0]["start"] == 0:
            entry = measurements[0]
            vol_ml = canonical_volume_ml(entry["value"], entry["unit"])
        elif _ATTR_VOLUME_NUMBER_RE.fullmatch(declared):
            # Historical export convention: unitless declared Volume is ml.
            number = _WHITESPACE_RE.sub("", declared)
            if float(number.replace(",", ".")) > 0:
                vol_ml = canonical_volume_ml(number, "ml")
        if vol_ml > 0:
            vol_conf = 0.9
    m_pack = _ATTR_COUNT_PER_UNIT_RE.search(attr_str)
    if m_pack and int(m_pack.group(1)) > 0:
        # zero-guard: export noise remains unknown rather than a count.
        pack_qty = canonical_pack_count(m_pack.group(1))
        pack_conf = data_cfg().extraction.pack_confidence["attribute"]
    return vol_ml, vol_conf, pack_qty, pack_conf


# Packaging LEVEL is an identity dimension independent of pack COUNT: a
# 12-pack sold as a retail pack and the same 12 units sold as a shipping case
# are distinct GS1 trade items with distinct GTINs, so a level change must
# separate them exactly as a count change does. Count alone cannot express
# this — "Case of 12 / 7.5 fl oz" and a 12-pack both extract to 12.
#
# Only a POSITIVE, packaging-context match yields "case". A bare \bcase\b
# matches prose ("in this case", "NOT A CASE") and would manufacture false
# splits, so the negative lookarounds are load-bearing, not defensive
# decoration. "case of", "case:", "/ case", "master case", "case pack".
_CASE_LEVEL = re.compile(
    r"\b(?:master\s+case|retail\s+case|case\s+pack|case\s+of|case)\s*"
    r"(?:of|[:/])?\s*\d*"
    r"|\bcases?\s+of\s+\d+"
    r"|/\s*case\b",
    re.IGNORECASE,
)
# Contexts that are NOT a packaging level.
_CASE_FALSE = re.compile(
    r"not\s+a\s+case|in\s+this\s+case|in\s+case|any\s+case|case[-\s]?insensitive",
    re.IGNORECASE,
)


def extract_packaging_level(title: str) -> set[str]:
    """Packaging level asserted by a title, as a claim set.

    Returns {"case"} for a case-level listing and an EMPTY set otherwise.
    The empty set is deliberately NOT {"single"}: absence of a case marker
    is absence of evidence, and the gate's conflict rules require BOTH sides
    to be populated before declaring a mismatch. Encoding "single" here would
    make every title that simply omits the word "case" conflict against a
    real case listing, splitting genuine duplicates.
    """
    text = str(title or "")
    if not text.strip():
        return set()
    # Strip an explicit merchandising negation before testing, so
    # "(NOT A CASE) Juice Lemon" reads as no claim rather than a case claim.
    if _CASE_FALSE.search(text):
        return set()
    return {"case"} if _CASE_LEVEL.search(text) else set()


# extract_salient_tokens REMOVED (audit 2026-09-09): zero callers across
# the repo (verified by grep). Its "salient token" job is done by the
# NgramIDF discriminative extractor; this legacy variant duplicated a
# volume/pack regex inline (a second declaration the config cannot steer).


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURED EVIDENCE SECTION — attribute-cell capture (measured high-yield
# rows, layouts.attribute_universe_census): pack material type 51,703
# rows / 5 value-sets / 9.64% same-GTIN conflict (inside the VETO BAND
# 2.5%-15%, census-verified); juice content 63,117 rows / 27 numeric bands;
# carbonization 56,125 rows (prose claims already flow through
# extract_critical_claims — this section mirrors the value vocabulary);
# water type 18,850 / naturally derived 28,637 / made from 19,338. Made from
# now HAS a canonical set column (`made_from_set`, fcc2c07: title+attribute
# lexicon capture) — but the attribute-cell capture here stays, and it stays
# a supporting-review (never model-visible) channel: consumption is
# config-owned (training.yaml
# rand_matching.targeted_veto_gates.supporting_feature_review_dimensions,
# read via attribute_conflicts._universe_value "made from"). The veto LIST
# itself stays config-owned
# (training.yaml rand_matching.targeted_veto_gates.veto_dimensions) — this
# capture is evidence, config-owned wiring decides consumption.
# ═══════════════════════════════════════════════════════════════════════════
ATTRIBUTE_UNIVERSE_CAPTURE_KEYS: tuple[str, ...] = (
    "pack material type",
    "juice content",
    "carbonization",
    "naturally derived",
    "water type",
    "made from",
)

# Ordered percent/mg band canon, byte-identical to
# core.attribute_universe._canonical_band (the census lane's own normalizer:
# "0-2 %" -> "0-2%", "200 + mg" -> "200+mg", off-vocabulary text unchanged).
# Reproduced here as a static device because pipeline is a hot per-row path
# that must not load the census module for every cell; the mirrored behaviour
# is pinned bidirectionally in tests/test_universe_capture_wiring.py (any
# drift on either side fails loudly there, not silently here).
_BAND_RANGE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*[-]\s*(\d+(?:\.\d+)?)\s*(%|mg)$")
_BAND_PLUS_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*\+\s*(%|mg)$")
_BAND_EXACT_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(%|mg)$")


def _canonical_band(token: str) -> str:
    """Normalise one ordered band token to its canonical band text."""
    plain = str(token or "").strip().lower().replace("\u2013", "-").replace(" ", "")
    match = _BAND_RANGE_RE.match(plain)
    if match:
        return f"{match.group(1)}-{match.group(2)}{match.group(3)}"
    match = _BAND_PLUS_RE.match(plain)
    if match:
        return f"{match.group(1)}+{match.group(2)}"
    match = _BAND_EXACT_RE.match(plain)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return str(token or "").strip().lower()


def capture_universe_attributes(attribute: object) -> dict[str, frozenset[str]]:
    """Parse the raw attribute cell into the capture keys, census-named.

    Same split discipline the census lane declares (split on ';', key before
    ':', comma-joined values lowered and stripped; keys normalized through
    core.text.normalized_attribute_text so the parity contract
    tests/test_universe_capture_wiring.py can diff this output against
    AttributeUniverse.parse cell-by-cell). Juice-content values go through
    the band canon; every other key keeps its raw lower tokens. Fail-safe
    input shape: unparseable parts contribute nothing — an empty cell or a
    foreign key is deterministic absence, never an invention.
    """
    from core.text import normalized_attribute_text

    captured: dict[str, set[str]] = {key: set() for key in ATTRIBUTE_UNIVERSE_CAPTURE_KEYS}
    for part in str(attribute or "").split(";"):
        if ":" not in part:
            continue
        raw_key, raw_value = part.split(":", 1)
        key = normalized_attribute_text(raw_key)
        if key not in captured:
            continue
        tokens = tuple(
            token.strip().lower() for token in raw_value.split(",") if token.strip()
        )
        if not tokens:
            continue
        if key == "juice content":
            captured[key].update(_canonical_band(token) for token in tokens)
        else:
            captured[key].update(tokens)
    return {key: frozenset(values) for key, values in captured.items()}


class ProductTypeMatcher:
    """Reads `type` and `subtype` for a normalized text off the config SSOT.

    List order is precedence: specific subtypes are matched before family
    words regardless of position (latte -> coffee even when the title never
    says coffee), and phrases ("coconut water") before their containing
    word ("water").
    """

    def __init__(self, spec) -> None:
        # subtype patterns first (specificity), then family words — both in
        # registry order so a config edit steers the matching as data.
        self._subtype_entries: list[tuple[str, str, re.Pattern]] = [
            (type_name, sub, re.compile(r"\b" + re.escape(sub) + r"\b"))
            for type_name, entry in spec.types.items()
            for sub in entry.subtypes
        ]
        self._word_entries: list[tuple[str, re.Pattern]] = [
            (type_name, re.compile(r"\b" + re.escape(word) + r"\b"))
            for type_name, entry in spec.types.items()
            for word in entry.words
        ]

    def match(self, text: str) -> tuple[str, str]:
        """(type, subtype) — ("", "") when the text carries neither."""
        for type_name, sub, pattern in self._subtype_entries:
            if pattern.search(text):
                # when the family word is also unambiguous in the text, the
                # card keeps the SPECIFIC subtype and the coarse family name
                return type_name, sub
        for type_name, pattern in self._word_entries:
            if pattern.search(text):
                return type_name, ""
        return "", ""


_PRODUCT_TYPE_MATCHER: ProductTypeMatcher | None = None


def _product_type_matcher() -> ProductTypeMatcher:
    """The card's product-type matcher (config/paths.yaml product_types),
    parsed once — every extract_all call shares one compiled instance."""
    global _PRODUCT_TYPE_MATCHER
    if _PRODUCT_TYPE_MATCHER is None:
        _PRODUCT_TYPE_MATCHER = ProductTypeMatcher(data_cfg().product_types)
    return _PRODUCT_TYPE_MATCHER


def fuse_confidence(claims: list[tuple[float, float, str]]) -> float:
    """Fuse independent source groups; every contradictory reader caps trust.

    Copied title, URL, and image surfaces share the configured listing group,
    so repetitions do not invent independent corroboration. Disagreement uses
    every observed reader, regardless of its position in the precedence list.
    """
    if not claims:
        return 0.0
    values = {float(value) for value, _, _ in claims}
    if len(values) == 1:
        groups: dict[str, float] = {}
        configured = data_cfg().extraction.source_groups
        for _, conf, source in claims:
            group = configured.get(source, source)
            groups[group] = max(groups.get(group, 0.0), float(conf))
        confidence = 1.0
        for conf in groups.values():
            confidence *= (1.0 - conf)
        return min(1.0, 1.0 - confidence)
    return min(float(conf) for _, conf, _ in claims)


def extract_all(sku_name_eng: str, attribute: str, description_short_eng: str = "",
     sku_url: str = "", image_url: str = "", breadcrumbs_eng: str = "",
     category: str = "") -> dict:
    """Extract structured fields plus salient tokens from a single SKU row.

    Evidence is drawn from ALL available columns — sku_name_eng, attribute,
    description_short_eng, URL slug, image filename, breadcrumbs_eng, and category —
    so the gate sees every product-bearing signal before deciding. The phases
    run on a ListingCardBuilder; this delegator keeps the documented call.
    """
    return ListingCardBuilder(
        sku_name_eng, attribute, description_short_eng,
        sku_url=sku_url, image_url=image_url,
        breadcrumbs_eng=breadcrumbs_eng, category=category,
    ).execute()


class ListingCardBuilder:
    """One listing's extract_all execution, phase by phase.

    Single responsibility per phase; the phases run in ONE fixed order inside
    execute() so the evidence-ledger append order and the consistency-flag
    population stay byte-identical to the pre-refactor linear body
    (the serialized ledger order is part of canonical_records.csv's bytes).

    Phases (order = load-bearing):
      harvest_consumer_tokens -> url/image/category token lanes
      harvest_evidence_channels -> critical/description claims, every
        per-column evidence ledger, cross-source contradiction flags
      resolve_product_type -> config product-type ladder + category fallback
      resolve_volume_and_pack -> column precedence chain, corroboration
        fusion, plausibility bounds
      assemble_card -> boundary validation (ExtractedAttributes) + additive
        evidence keys
    """

    def __init__(self, sku_name_eng, attribute, description_short_eng="",
                 sku_url="", image_url="", breadcrumbs_eng="", category=""):
        from core.sweetener_values import declared_sweeteners
        from core.url_evidence import url_text

        # ── column inputs ──
        self._sku_name_eng = sku_name_eng
        self._attribute = attribute
        self._description_short_eng = description_short_eng
        self._sku_url = sku_url
        self._image_url = image_url
        self._breadcrumbs_eng = breadcrumbs_eng
        self._category = category
        self.url_tokens = url_text(sku_url)
        self.img_tokens = url_text(image_url)
        # ── shared card state (append order is the byte contract) ──
        self.ledger: list[dict] = []
        self.flags: set[str] = set()
        # ── phase outputs ──
        self.sweeteners = declared_sweeteners(attribute)
        self.url_norm = ""
        self.img_norm = ""
        self.cat_tokens = ""
        self.title_norm = ""
        self.critical = {}
        self.description_claims = {}
        self.negative_ingredients: set[str] = set()
        self.date_evidence = []
        self.measurement_evidence = []
        self.pack_evidence = []
        self.declared_identity = None
        self.flavor_set: set[str] = set()
        self.made_from_set: set[str] = set()
        self.ptype = ""
        self.subtype = ""
        # volume + pack resolution chain
        self.title_vol = {}
        self.pack_title, self.pack_conf_title = (1, 0.0)
        self.pack_description, self.pack_conf_description = (1, 0.0)
        self.attr_vol, self.attr_vol_conf, self.attr_pack, self.attr_pack_conf = (0.0, 0.0, 1, 0.0)
        self.vol_url: dict = {}
        self.pack_url, self.pack_conf_url = (1, 0.0)
        self.vol_img: dict = {}
        self.pack_img, self.pack_conf_img = (1, 0.0)
        self.title_vol_ml = 0.0
        self.volume_ml = 0.0
        self.volume_conf = 0.0
        self.volume_raw = ""
        self.volume_status = ""
        self.pack_qty = 1
        self.pack_conf = 0.0
        self.vol_claims: list[tuple[float, float, str]] = []
        self.pack_claims: list[tuple[float, float, str]] = []
        self.universe_evidence: dict[str, frozenset[str]] = {}
        self.package_types: list[str] = []
        self.packaging_levels: set[str] = set()
        self.package_materials: list[str] = []

    # ── phase 1: consumer token lanes ──────────────────────────────────────

    def harvest_consumer_tokens(self) -> None:
        """Sweetener(title/URL/image) union + normalized URL-side tokens."""
        from core.sweetener_values import (
            extract_sweetening_status,
            title_sweetener_types,
        )
        from core.text import normalize_text

        self.sweeteners["sweetener_type"].update(title_sweetener_types(self._sku_name_eng))
        self.sweeteners["sweetener_type"].update(title_sweetener_types(self._description_short_eng))
        self.sweeteners["sweetening"].update(extract_sweetening_status(
            self._sku_name_eng, self._attribute, self._description_short_eng
        ))
        self.title_norm = normalize_text(self._sku_name_eng)
        # URL tokens: product-bearing prose from the listing slug.
        # Fed into volume/pack extraction when title/attributes are silent.
        # url_text is the reader for BOTH URL columns (docstring, url_evidence.py):
        # image filenames go through the same normalizer — hashes, media dims and
        # scaffolding fall out; size tokens ("250ml") survive.
        self.url_norm = normalize_text(self.url_tokens)
        self.img_norm = normalize_text(self.img_tokens)
        self.sweeteners["sweetener_type"].update(title_sweetener_types(self.url_tokens))
        self.sweeteners["sweetener_type"].update(title_sweetener_types(self.img_tokens))
        # Category evidence: breadcrumbs_eng and category provide
        # product-type signals (flavor hints, carbonation clues)
        # that title/attributes may miss.
        cat_tokens = normalize_text(self._breadcrumbs_eng) + " " + normalize_text(self._category)
        self.cat_tokens = cat_tokens.strip()

    # ── phase 2: evidence channels ─────────────────────────────────────────

    def harvest_evidence_channels(self) -> None:
        """Critical claims + every per-column ledger entry + contradiction flags.

        The statements here run in exactly the original order: the ledger is
        serialized later and its order is data, not decoration.
        """
        from core.date_evidence import extract_date_evidence
        from core.product_selection import selected_identity_inputs
        from core.sweetener_values import negated_sweetener_types
        from core.text import extract_volume_evidence

        identity_title, identity_attributes, _selected_variant = selected_identity_inputs(
            self._sku_name_eng, self._attribute)
        self.critical = extract_critical_claims(identity_title, identity_attributes)
        self.description_claims = extract_description_claims(self._description_short_eng)
        self.flags.update(self.sweeteners["consistency_flags"])
        self.negative_ingredients = negated_sweetener_types(
            self._sku_name_eng, self._attribute, self._description_short_eng,
            self.url_tokens, self.img_tokens,
        )
        self.flags.update(
            f"sweetener_source_conflict:{ingredient}"
            for ingredient in self.negative_ingredients & self.sweeteners["sweetener_type"]
        )
        # Product card evidence ledger: every claim the columns yield, recorded
        # with its source at the moment of extraction (surface-one-by-one
        # ruling 2026-10-01). Rides the result dict additively, like
        # attribute_universe_evidence — schema stays extra="forbid".
        if self.negative_ingredients:
            self.ledger.append({"field": "negated_sweetener_type", "column": "sku_name_eng+attribute+description_short_eng",
                                "value": sorted(self.negative_ingredients)})
        self.date_evidence = [
            {"column": column, **entry}
            for column, text in (
                ("sku_name_eng", self._sku_name_eng), ("attribute", self._attribute),
                ("description_short_eng", self._description_short_eng), ("breadcrumbs_eng", self._breadcrumbs_eng),
                ("category", self._category),
            )
            for entry in extract_date_evidence(str(text or ""))
        ]
        for entry in self.date_evidence:
            self.ledger.append({"field": "source_date", "column": entry["column"], "value": entry})
        self.measurement_evidence = [
            {"column": column, **entry}
            for column, text in (("sku_name_eng", str(self._sku_name_eng or "")), ("sku_url", self.url_tokens), ("image_url", self.img_tokens))
            for entry in extract_volume_evidence(text)
        ]
        for entry in self.measurement_evidence:
            self.ledger.append({"field": "measurement", "column": entry["column"], "value": entry})
        self.pack_evidence = [
            {"column": column, **entry}
            for column, text in (("sku_name_eng", str(self._sku_name_eng or "")), ("description_short_eng", str(self._description_short_eng or "")), ("sku_url", self.url_tokens), ("image_url", self.img_tokens))
            for entry in extract_pack_evidence(text)
        ]
        if any(entry.get("hierarchy_ambiguous") for entry in self.pack_evidence):
            self.flags.add("pack_hierarchy_ambiguous")
        for entry in self.pack_evidence:
            self.ledger.append({"field": "pack_quantity", "column": entry["column"], "value": entry})
        self._merge_description_claims()
        # "Made From" base ingredients, title+attribute aware (the declared field
        # alone misses a title that names the ingredient, e.g. "ginger-turmeric"
        # with `Made From: lemon, ginger`). Vocabulary is config-owned.
        self.made_from_set = set(extract_made_from_tokens(self._sku_name_eng, self._attribute))
        if self.made_from_set:
            self.ledger.append({"field": "made_from", "column": "title+attributes",
                                "value": sorted(self.made_from_set)})
        # Source contradictions / implausible declarations (measured 2026-10-03):
        # the extractor is faithful, so these flag SOURCE defects for review.
        self.flags.update(
            source_consistency_flags(self._attribute, self._sku_name_eng,
                                     self.sweeteners["sweetener_type"])
        )
        # Categories classify products; they do not declare a SKU's flavor.
        # Broad "Lemonade/Lime" and negated "Non-Cola" categories previously
        # invented identity agreement between distinct variants.
        from core.declared_identity import listing_identity
        self.declared_identity = listing_identity(self._sku_name_eng, self._attribute, self._description_short_eng)
        if self.declared_identity:
            self.ledger.append({"field": "declared_identity", "column": "sku_name_eng+attribute+description_short_eng",
                                "value": self.declared_identity})

    def _merge_description_claims(self) -> None:
        """Fold description-only claims into the critical sets, contradiction-
        aware (exact original semantics and flag vocabulary)."""
        opposing_values = {
            "carbonation": (("carbonated", "still"),),
            "sweetener": (("sugar", "no_sugar"), ("sugar", "diet")),
            "pulp": (("with_pulp", "no_pulp"),),
            "organic": (("organic", "not_organic"),),
        }
        for dimension in ("carbonation", "sweetener", "pulp", "organic"):
            base = set(self.critical[dimension])
            described = set(self.description_claims[dimension])
            if not base:
                self.critical[dimension] = frozenset(described)
            elif described:
                inconsistent = any(
                    (left in base and right in described) or (right in base and left in described)
                    for left, right in opposing_values[dimension]
                )
                if inconsistent:
                    self.flags.add(f"description_conflict:{dimension}")
                else:
                    self.critical[dimension] = frozenset(base | described)
        if {"unsweetened", "sweetened"} <= self.sweeteners["sweetening"]:
            self.flags.add("sweetening_status_conflict")
        if "no_added_sugar" in self.critical["sweetener"] and "cane_sugar" in self.sweeteners["sweetener_type"]:
            self.flags.add("no_added_sugar_with_cane_sugar")
        self.flavor_set = set(self.critical["flavor"])
        if self.flavor_set:
            self.ledger.append({"field": "flavor", "column": "title+attributes",
                                "value": sorted(self.flavor_set)})

    # ── phase 3: product type ──────────────────────────────────────────────

    def resolve_product_type(self) -> None:
        """Scalar flavor summary, per-dimension ledgers, then the config
        product-type ladder on the title with the category fallback."""
        # Scalar flavor summary (deterministic first value) preserves the
        # historical CSV contract for the downstream readers.
        self._flavor_hint = sorted(self.flavor_set)[0] if self.flavor_set else ""
        for dimension in ("carbonation", "sweetener", "pulp", "organic"):
            if self.critical[dimension]:
                self.ledger.append({"field": dimension, "column": "title+attributes",
                                    "value": sorted(self.critical[dimension])})
            if self.description_claims[dimension]:
                self.ledger.append({"field": dimension, "column": "description_short_eng",
                                    "value": sorted(self.description_claims[dimension])})
        # Product type + subtype: config SSOT (config/paths.yaml product_types),
        # read once. Title first, then the category-lane fallback; the subtype
        # (latte, kombucha, ale...) is the finer axis the differentiation lane
        # consumes and is recorded per column like every claim.
        self.ptype, self.subtype = _product_type_matcher().match(self.title_norm)
        if self.ptype:
            self.ledger.append({"field": "type", "column": "sku_name_eng", "value": self.ptype})
        if self.subtype:
            self.ledger.append({"field": "subtype", "column": "sku_name_eng", "value": self.subtype})
        # Fallback: category tokens may carry the product type
        # when the title is too generic (e.g. "Product" with no type word).
        if not self.ptype and self.cat_tokens:
            cat_ptype, cat_subtype = _product_type_matcher().match(self.cat_tokens)
            if cat_ptype:
                self.ptype = cat_ptype
                self.subtype = self.subtype or cat_subtype
                self.ledger.append({"field": "type", "column": "category", "value": cat_ptype})
                if cat_subtype:
                    self.ledger.append({"field": "subtype", "column": "category", "value": cat_subtype})

    # ── phase 4: volume + pack resolution ──────────────────────────────────

    def resolve_volume_and_pack(self) -> None:
        """Column precedence chain, corroboration fusion, plausibility bounds.

        Winner chain: attribute over title unless they disagree by a large
        factor; URL and image URL lanes follow. Confidence is then fused INDEPENDENTLY
        of the winner (the card's confidence is a property of the CLAIM —
        agreeing readers pool upward, disagreeing readers cap at the weakest).
        """
        # Volume and pack from title
        self.title_vol = extract_volume_from_title(self._sku_name_eng)
        self.pack_title, self.pack_conf_title = extract_pack_from_title(self._sku_name_eng)
        self.pack_description, self.pack_conf_description = extract_pack_from_title(self._description_short_eng)

        # Attribute parsing
        self.attr_vol, self.attr_vol_conf, self.attr_pack, self.attr_pack_conf = parse_attribute_volume_pack(
            self._attribute
        )

        # URL evidence: product tokens from the listing slug.
        # Used when title/attributes are silent on volume/pack.
        self.vol_url = extract_volume_from_title(self.url_norm)
        self.pack_url, self.pack_conf_url = extract_pack_from_title(self.url_norm)
        self.vol_img = extract_volume_from_title(self.img_norm)
        self.pack_img, self.pack_conf_img = extract_pack_from_title(self.img_norm)

        # Combine: prefer attribute if present, but default to title when
        # the two disagree by 10x+ (title misparses "0, 33l" as 33000ml
        # vs attribute 330ml — the title is the correct unit here).
        self.title_vol_ml = float(self.title_vol["volume_ml"] or 0.0)
        if self.attr_vol > 0:
            self.ledger.append({"field": "volume_ml", "column": "attribute",
                                "value": self.attr_vol, "confidence": self.attr_vol_conf})
        if self.title_vol_ml > 0:
            self.ledger.append({"field": "volume_ml", "column": "sku_name_eng", "value": self.title_vol_ml,
                                "confidence": self.title_vol["confidence"]})
        if self.vol_url["volume_ml"] > 0:
            self.ledger.append({"field": "volume_ml", "column": "sku_url",
                                "value": self.vol_url["volume_ml"], "confidence": self.vol_url["confidence"]})
        if self.vol_img["volume_ml"] > 0:
            self.ledger.append({"field": "volume_ml", "column": "image_url",
                                "value": self.vol_img["volume_ml"], "confidence": self.vol_img["confidence"]})
        if self.attr_pack > 1 or self.attr_pack_conf > 0:
            self.ledger.append({"field": "pack_qty", "column": "attribute",
                                "value": self.attr_pack, "confidence": self.attr_pack_conf})
        if self.pack_title > 1 or self.pack_conf_title > 0:
            self.ledger.append({"field": "pack_qty", "column": "sku_name_eng",
                                "value": self.pack_title, "confidence": self.pack_conf_title})
        if self.pack_conf_description > 0:
            self.ledger.append({"field": "pack_qty", "column": "description_short_eng",
                                "value": self.pack_description, "confidence": self.pack_conf_description})
        if self.pack_url > 1 or self.pack_conf_url > 0:
            self.ledger.append({"field": "pack_qty", "column": "sku_url",
                                "value": self.pack_url, "confidence": self.pack_conf_url})
        if self.pack_img > 1 or self.pack_conf_img > 0:
            self.ledger.append({"field": "pack_qty", "column": "image_url",
                                "value": self.pack_img, "confidence": self.pack_conf_img})
        self._volume_precedence_chain()
        self._pack_precedence_chain()
        self._fuse_and_bound()

    def _volume_precedence_chain(self) -> None:
        """The documented volume winner chain (attr/title/url/img)."""
        if self.attr_vol > 0 and self.title_vol_ml > 0:
            ratio = max(self.attr_vol, self.title_vol_ml) / min(self.attr_vol, self.title_vol_ml)
            if ratio >= data_cfg().extraction.title_attribute_override_ratio:
                self.volume_ml = self.title_vol_ml
                self.volume_conf = self.title_vol["confidence"]
                self.volume_raw = self.title_vol["raw_match"]
                self.volume_status = self.title_vol["parse_status"]
                self.flags.add("volume_inconsistency")
            else:
                self.volume_ml = self.attr_vol
                self.volume_conf = self.attr_vol_conf
                self.volume_raw = f"attribute: {self.attr_vol}"
                self.volume_status = "attribute_volume"
        elif self.attr_vol > 0:
            self.volume_ml = self.attr_vol
            self.volume_conf = self.attr_vol_conf
            self.volume_raw = f"attribute: {self.attr_vol}"
            self.volume_status = "attribute_volume"
        elif self.title_vol_ml > 0:
            self.volume_ml = self.title_vol_ml
            self.volume_conf = self.title_vol["confidence"]
            self.volume_raw = self.title_vol["raw_match"]
            self.volume_status = self.title_vol["parse_status"]
        elif self.vol_url["volume_ml"] > 0:
            self.volume_ml = self.vol_url["volume_ml"]
            self.volume_conf = self.vol_url["confidence"]
            self.volume_raw = self.vol_url["raw_match"]
            self.volume_status = self.vol_url["parse_status"]
            self.flags.add("volume_from_url")
        elif self.vol_img["volume_ml"] > 0:
            self.volume_ml = self.vol_img["volume_ml"]
            self.volume_conf = self.vol_img["confidence"]
            self.volume_raw = self.vol_img["raw_match"]
            self.volume_status = self.vol_img["parse_status"]
            self.flags.add("volume_from_image_url")
        else:
            self.volume_ml = self.title_vol["volume_ml"]
            self.volume_conf = self.title_vol["confidence"]
            self.volume_raw = self.title_vol["raw_match"]
            self.volume_status = self.title_vol["parse_status"]

    def _pack_precedence_chain(self) -> None:
        """Pack qty resolved early for ambiguous_volume check. Slugs can be
        truncated ("16-9-Count") or omit separators: keep their disagreement
        in the ledger, but don't overwrite an explicit title count with
        URL/image-derived numbers."""
        if self.attr_pack > 1 or self.attr_pack_conf > 0:
            self.pack_qty = self.attr_pack
            self.pack_conf = self.attr_pack_conf
        elif self.pack_conf_title > 0:
            self.pack_qty = self.pack_title
            self.pack_conf = self.pack_conf_title
        elif self.pack_url > 1 or self.pack_conf_url > 0:
            self.pack_qty = self.pack_url
            self.pack_conf = self.pack_conf_url
        elif self.pack_img > 1 or self.pack_conf_img > 0:
            self.pack_qty = self.pack_img
            self.pack_conf = self.pack_conf_img
        else:
            self.pack_qty = self.pack_description
            self.pack_conf = self.pack_conf_description

    def _fuse_and_bound(self) -> None:
        """Corroboration fusion + source-disagreement flags + plausibility bounds."""
        # CORROBORATION FUSION (2026-10-01 ruling): the card's confidence is a
        # property of the CLAIM, not of the winning column — agreeing
        # independent readers pool upward, disagreeing readers cap the card at
        # the weaker one. The winner chain above decides VALUE + precedence;
        # this only changes confidence.
        self.vol_claims = [
            (value, conf, column)
            for value, conf, column in (
                (self.attr_vol, self.attr_vol_conf, "attribute"),
                (self.title_vol_ml, self.title_vol["confidence"], "sku_name_eng"),
                (self.vol_url["volume_ml"], self.vol_url["confidence"], "sku_url"),
                (self.vol_img["volume_ml"], self.vol_img["confidence"], "image_url"),
            )
            if value > 0 and conf > 0
        ]
        self.pack_claims = [
            (value, conf, column)
            for value, conf, column in (
                (self.attr_pack, self.attr_pack_conf, "attribute"),
                (self.pack_title, self.pack_conf_title, "sku_name_eng"),
                (self.pack_description, self.pack_conf_description, "description_short_eng"),
                (self.pack_url, self.pack_conf_url, "sku_url"),
                (self.pack_img, self.pack_conf_img, "image_url"),
            )
            if value > 0 and conf > 0
        ]
        self.volume_conf = fuse_confidence(self.vol_claims)
        self.pack_conf = fuse_confidence(self.pack_claims)
        gate_cfg = training_cfg().gate
        if self.vol_claims and any(
            not volumes_compatible({left[0]}, {right[0]},
                                   volume_relative_tolerance=float(gate_cfg.vol_tolerance),
                                   volume_absolute_tolerance_ml=float(gate_cfg.vol_abs_tolerance))
            for index, left in enumerate(self.vol_claims) for right in self.vol_claims[index + 1:]
        ):
            self.flags.add("volume_sources_disagree")
        if self.pack_claims and len({value for value, _, _ in self.pack_claims}) > 1:
            self.flags.add("pack_sources_disagree")
        # Bounds apply to the selected physical-package size. A count of packages
        # does not make an implausible per-package size legitimate; named bulk
        # containers use the separately configured ceiling.
        extraction_policy = data_cfg().extraction
        bulk_container = bool(_bulk_container_re(tuple(extraction_policy.bulk_container_terms)).search(
            f"{self._sku_name_eng} {self._attribute}"))
        volume_max = extraction_policy.bulk_volume_max_ml if bulk_container else extraction_policy.volume_max_ml
        if self.volume_ml > 0 and not extraction_policy.volume_min_ml <= self.volume_ml <= volume_max:
            self.flags.add("ambiguous_volume")

    # ── phase 5: card assembly ─────────────────────────────────────────────

    def assemble_card(self) -> dict:
        """Boundary validation + additive evidence keys (exact original order)."""
        # BOUNDARY CONTRACT (lib.schemas): the extracted-attribute dict is the
        # input to BOTH the canonical build and the gate — validate the shape
        # once here so a confidence out of [0,1] or a pack_qty < 1 crashes at
        # the transform, not downstream in the gate's comparisons.
        title_attributes = extract_title_attributes(self._sku_name_eng)
        package_types = title_attributes["package_types"]
        if not package_types:
            package_types = parse_attribute_details(self._attribute).get("attribute_package_types", [])
        # Title-only, and deliberately so: the raw `attributes` field carries no
        # packaging-level key at all (measured 2026-09-30 — `attributes` holds
        # Volume/Pack Type/Flavour/... and zero case-quantity columns), so the
        # title is the only place this claim exists.
        packaging_levels = extract_packaging_level(self._sku_name_eng)
        # Structured evidence section (census script): pack material type is the
        # measured 9.64% within-GTIN conflict band, so the attribute cell is now
        # an ELIGIBLE material source: the title NER scrape keeps its exact
        # convention (list order and values byte-unchanged) and the
        # attribute-only values are appended, sorted, after it. Title-scraped
        # values win duplicates by construction; a set union in
        # generate_canonical package_material_set is what the gate and the
        # structured channel actually read, so no material evidence is lost —
        # only where the model VISIBLY can see it: "paper / carton" is
        # byte-identical to the census value kept here (raw lower tokens, same
        # semantics the census measured). Juice content bands live in the
        # evidence section (numeric 27-band vocabulary, attribute_universe SSOT
        # canon); the remaining three high-yield keys have no set field —
        # captured for census→wiring parity, deliberately not wired.
        self.universe_evidence = capture_universe_attributes(self._attribute)
        title_materials = title_attributes["package_materials"]
        material_seen = {value.casefold() for value in title_materials}
        package_materials = list(title_materials) + sorted(
            value
            for value in self.universe_evidence["pack material type"]
            if value.casefold() not in material_seen
        )
        result = ExtractedAttributes(
            flavor=self._flavor_hint,
            type=self.ptype,
            volume_ml=self.volume_ml,
            volume_confidence=self.volume_conf,
            volume_raw=self.volume_raw,
            volume_status=self.volume_status,
            pack_qty=self.pack_qty,
            pack_confidence=self.pack_conf,
            package_types=package_types,
            package_materials=package_materials,
            packaging_levels=packaging_levels,
            flavor_set=self.flavor_set,
            made_from_set=self.made_from_set,
            carbonation_set=set(self.critical["carbonation"]),
            sweetener_set=set(self.critical["sweetener"]),
            sweetener_type_set=self.sweeteners["sweetener_type"],
            sweetening_set=self.sweeteners["sweetening"],
            attribute_consistency_flags=self.flags,
            pulp_set=set(self.critical["pulp"]),
            organic_set=set(self.critical["organic"]),
        ).model_dump()
        # The extract dict is a plain dict after the boundary validation, so the
        # evidence section rides ADDITIVELY beside the model dump. Old consumers
        # iterate the named fields, the model channel reads the two wired keys,
        # the census parity test reads the whole section. Sorted lists, never
        # sets — byte-determinism (PYTHONHASHSEED) is the contract here too.
        result["attribute_universe_evidence"] = {
            key: sorted(values) for key, values in self.universe_evidence.items() if values
        }
        result["evidence_ledger"] = self.ledger
        result["date_evidence"] = self.date_evidence
        result["measurement_evidence"] = self.measurement_evidence
        result["pack_evidence"] = self.pack_evidence
        result["negated_sweetener_type_set"] = sorted(self.negative_ingredients)
        return result

    # ── orchestration ──────────────────────────────────────────────────────

    def execute(self) -> dict:
        """Run the load-bearing phase order, then assemble the card."""
        self.harvest_consumer_tokens()
        self.harvest_evidence_channels()
        self.resolve_product_type()
        self.resolve_volume_and_pack()
        return self.assemble_card()


# ============================================================================
# GATING
# ============================================================================
def _attribute_flags(obj: object) -> set[str]:
    """Read consistency flags from extracted objects or serialized canonicals."""
    if isinstance(obj, dict):
        value = obj.get("attribute_consistency_flags")
    else:
        value = getattr(obj, "attribute_consistency_flags", None)
    if value is None:
        return set()
    if isinstance(value, (set, frozenset, list, tuple)):
        return {str(item).strip() for item in value}
    text = str(value).strip()
    if not text:
        return set()
    if text in {"set()", "frozenset()"}:
        return set()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid attribute consistency flags: {value!r}") from exc
    if not isinstance(parsed, (set, frozenset, list, tuple)):
        raise ValueError(f"attribute consistency flags are not a sequence: {value!r}")
    return {str(item).strip() for item in parsed}


def _has_attribute_flag(obj: object, flag: str) -> bool:
    return flag in _attribute_flags(obj)


class _GateEvaluator:
    """Decide whether two record sides' known identity attributes conflict.

    Single responsibility: one pack_gate verdict from the two sides' evidence.
    The semantic score is accepted for a stable gate-call interface but is
    deliberately not used: a high semantic score cannot override a known pack,
    package-type, or volume conflict.

    EVIDENCE TRUST (audit 2026-09-15). A parsed attribute is only comparable
    when the parser reported enough confidence to be believed. A trust
    threshold below 1.0-or-None makes the volume/pack comparison evidence-
    aware: a side below the bar is treated exactly like a missing side, so it
    stays *unknown* and reaches the confidence/fallback lane instead of being
    fabricated into a hard rejection.

    PACK SEMANTICS: a canonical keeps EVERY pack count observed across its
    titles, so a multi-title canonical legitimately holds ``{12, 24}``. A
    shared count is therefore positive evidence of compatibility and only a
    genuinely disjoint pair conflicts — the rule the canonical writer
    documents ("gate logic intersects them").
    """

    def __init__(
        self,
        *,
        volume_relative_tolerance: float,
        volume_absolute_tolerance_ml: float,
        trust_threshold: float | None,
        check_categorical: bool,
    ) -> None:
        self._volume_relative_tolerance = volume_relative_tolerance
        self._volume_absolute_tolerance_ml = volume_absolute_tolerance_ml
        self._trust_threshold = trust_threshold
        self._check_categorical = check_categorical
        self._veto_dimensions = frozenset(
            training_cfg().rand_matching.targeted_veto_gates.veto_dimensions
        )

    # -- field readers ------------------------------------------------------

    @staticmethod
    def _value(obj: object, *names: str):
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
        else:
            for name in names:
                if hasattr(obj, name):
                    return getattr(obj, name)
        return None

    @staticmethod
    def _set(value: object) -> set:
        if value is None or value == "":
            return set()
        if isinstance(value, (set, frozenset, list, tuple)):
            return set(value)
        return {value}

    def _trusted(self, obj: object, *names: str) -> bool:
        """Whether the named confidence field clears the caller's bar.

        An absent field means the lane never carried a confidence observation;
        the caller's own confidence lane owns that case, so it is not second
        guessed here.
        """
        dimension = "volume" if "volume_confidence" in names else "pack"
        if _has_attribute_flag(obj, f"{dimension}_sources_disagree") or (
            dimension == "pack" and _has_attribute_flag(obj, "pack_hierarchy_ambiguous")
        ):
            return False
        raw = self._value(obj, *names)
        if raw is None or raw == "":
            return True
        if self._trust_threshold is None:
            return True
        try:
            value = float(raw)
            return math.isfinite(value) and 0.0 <= value <= 1.0 and value >= float(self._trust_threshold)
        except (TypeError, ValueError):
            return False

    def _claim_set(self, obj: object, dimension: str) -> set[str]:
        explicit = self._value(obj, f"{dimension}_set")
        if explicit:
            return self._set(explicit)
        found = extract_critical_claims(str(self._value(obj, "canonical") or ""))[dimension]
        return set(found)

    # -- conflict tests (one per identity dimension) -------------------------

    def _pack_count_conflict(self, left_pack: set, right_pack: set, sku_a: object, sku_b: object) -> bool:
        """PACK COUNT: shared evidence agrees; disjoint counts conflict.

        No count on one side is unknown, not an assertion of single-unit
        packaging. The caller's confidence/review lane owns missing evidence.
        """
        return (
            "pack" in self._veto_dimensions
            and left_pack
            and right_pack
            and not (left_pack & right_pack)
            and self._trusted(sku_a, "pack_confidence")
            and self._trusted(sku_b, "pack_confidence")
        )

    def _package_type_conflict(self, left_type: set, right_type: set) -> bool:
        """PACKAGE TYPE: disjoint categorical evidence conflicts."""
        return ("package_type" in self._veto_dimensions
                and left_type and right_type and not (left_type & right_type))

    def _volume_conflict(self, left_volume: set, right_volume: set, sku_a: object, sku_b: object) -> bool:
        """VOLUME: both trusted and incompatible under either configured cut."""
        return (
            "volume" in self._veto_dimensions
            and left_volume
            and right_volume
            and self._trusted(sku_a, "volume_confidence")
            and self._trusted(sku_b, "volume_confidence")
            and not volumes_compatible(
                left_volume,
                right_volume,
                volume_relative_tolerance=self._volume_relative_tolerance,
                volume_absolute_tolerance_ml=self._volume_absolute_tolerance_ml,
            )
        )

    def _categorical_conflict(self, sku_a: object, sku_b: object) -> bool:
        """Critical categorical dimensions (carbonation/sweetener/pulp)."""
        for dimension in sorted(self._veto_dimensions & {"carbonation", "sweetener", "pulp"}):
            if not self._check_categorical:
                continue
            left = self._claim_set(sku_a, dimension)
            right = self._claim_set(sku_b, dimension)
            if left and right and categorical_conflict(
                dimension, {dimension: left}, {dimension: right}
            ):
                return True
        return False

    # -- verdict -------------------------------------------------------------

    def skus_compatible(self, sku_a: object, sku_b: object) -> bool:
        left_pack = self._set(self._value(sku_a, "pack_size", "pack_set", "pack_qty"))
        right_pack = self._set(self._value(sku_b, "pack_size", "pack_set", "pack_qty"))
        if self._pack_count_conflict(left_pack, right_pack, sku_a, sku_b):
            return False
        left_type = self._set(self._value(sku_a, "package_type", "package_type_set"))
        right_type = self._set(self._value(sku_b, "package_type", "package_type_set"))
        if self._package_type_conflict(left_type, right_type):
            return False
        left_volume = (
            set()
            if _has_attribute_flag(sku_a, "ambiguous_volume")
            else self._set(self._value(sku_a, "volume", "volume_set", "volume_ml"))
        )
        right_volume = (
            set()
            if _has_attribute_flag(sku_b, "ambiguous_volume")
            else self._set(self._value(sku_b, "volume", "volume_set", "volume_ml"))
        )
        if self._volume_conflict(left_volume, right_volume, sku_a, sku_b):
            return False
        return not self._categorical_conflict(sku_a, sku_b)


def pack_gate(
    score: float,
    sku_a: object,
    sku_b: object,
    *,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
    trust_threshold: float | None = None,
    check_categorical: bool = True,
) -> bool:
    """Return whether known package identity attributes are compatible.

    ``score`` is accepted for a stable gate-call interface but is deliberately
    not used: a high semantic score cannot override a known pack, package-type,
    or volume conflict. The entire verdict is the _GateEvaluator's.
    """
    del score
    return _GateEvaluator(
        volume_relative_tolerance=volume_relative_tolerance,
        volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
        trust_threshold=trust_threshold,
        check_categorical=check_categorical,
    ).skus_compatible(sku_a, sku_b)


def attribute_gate_census_column_names(registry=None) -> list[str]:
    """The ALL-DIMENSIONS census audit-column contract (deterministic order).

    One `<key>_state` column per AttributeUniverse-registered key plus the
    five-state rollup columns. Unit-tested against the registry so a new
    registered field can never silently miss its census column.
    """
    from core.attribute_universe import attribute_registry

    keys = dict(registry) if registry is not None else attribute_registry()
    if not keys:
        raise ValueError("attribute_gate_census_column_names requires a non-empty registry")
    return [
        f"{key.replace(' ', '_')}_state" for key in sorted(keys)
    ] + [
        "dimension_conflicts",
        "dimension_conflict_count",
        "dimension_missing_left",
        "dimension_missing_right",
        "dimension_missing_both",
        "dimension_unknown_parse",
    ]


def attribute_gate_universe_scope_detail() -> dict[str, object]:
    """Trace detail for the attribute-gate decision-scope evidence row.

    States the owner ruling, which dimensions DECIDE today, which census
    states exist, what each state MEANS (absence stays unknown, a conflict
    votes only where the config permits) and the volume tolerances the census
    applied. Same-config read only, no writes.
    """
    from core.attribute_conflicts import DIMENSION_STATES

    from core.attribute_universe import attribute_registry

    return {
        "owner_ruling": "ALL ATTRIBUTES are used to make ALL DECISIONS",
        "registry_size": len(attribute_registry()),
        "decision_dimensions": list(CRITICAL_ATTRIBUTE_DIMENSIONS),
        "census_states": sorted(DIMENSION_STATES),
        "census_column_contract": attribute_gate_census_column_names(),
        "state_semantics": {
            "agree": "both sides populated, no conflict under the field's own measured semantics",
            "conflict": "both sides populated and genuinely incompatible (votes only where config permits)",
            "missing_left": "right side populated only — stays UNKNOWN, never vetoed",
            "missing_right": "left side populated only — stays UNKNOWN, never vetoed",
            "missing_both": "no evidence either side — UNKNOWN",
            "unknown_parse": "populated but unclassifiable (band grammar miss) — UNKNOWN",
        },
        "volume_census_tolerances": {
            "relative": float(training_cfg().gate.vol_tolerance),
            "absolute_ml": float(training_cfg().gate.vol_abs_tolerance),
            "applied": "whichever cut is wider (core.critical_attributes.volumes_compatible)",
            "source": "config/training.yaml gate.vol_tolerance + gate.vol_abs_tolerance",
        },
    }


class _ThreeWayGate:
    """One pair's three_way_gate decision, phase by phase.

    The phases count and order are the ORIGINAL decision table, statement for
    statement — same decisions, same reasons, same config reads; nothing is
    reordered (several placements are load-bearing measured history, see the
    inline comments). This class is the SR split of the former 310-line body;
    the module-level three_way_gate keeps its documented call.

    Phase map:
      resolve_thresholds -> config blocks behind every None argument
      pack_file_vetoes   -> pack_gate + the package_type/material/level hard nos
      decision_engine    -> census evaluation + claim/engine categorical vetoes
      fallback_lanes     -> source disagreement / ambiguity / confidence gates
      overlap_lanes      -> volume + pack overlap hard nos
      packaging_identity -> one-sided packaging level, consistency, supporting
                            attributes, mode_flavor, declared identity, policy
    """

    def __init__(self, attrs1: dict, attrs2: dict, vol_tolerance: float,
                 raw_conf_threshold: float, consistency_fallback_threshold: float,
                 vol_abs_tolerance: float) -> None:
        self.attrs1 = attrs1
        self.attrs2 = attrs2
        self.vol_tolerance = vol_tolerance
        self.raw_conf_threshold = raw_conf_threshold
        self.consistency_fallback_threshold = consistency_fallback_threshold
        self.vol_abs_tolerance = vol_abs_tolerance

    @classmethod
    def from_config(cls, attrs1: dict, attrs2: dict, *,
                    vol_tolerance: float | None, raw_conf_threshold: float | None,
                    consistency_fallback_threshold: float | None,
                    vol_abs_tolerance: float | None) -> "_ThreeWayGate":
        """NO-FALLBACK SSOT (audit round 2, F01): the decision thresholds live
        in config/training.yaml `gate:` and are read through training_cfg() —
        the old signature defaults (0.05/0.85/0.3) were a second declaration
        the config could not steer. Passing a value explicitly still wins
        (selftest pins known-good gate behavior with explicit values)."""
        if (
            vol_tolerance is None
            or raw_conf_threshold is None
            or consistency_fallback_threshold is None
            or vol_abs_tolerance is None
        ):
            _g = training_cfg().gate
            if vol_tolerance is None:
                vol_tolerance = float(_g.vol_tolerance)
            if vol_abs_tolerance is None:
                vol_abs_tolerance = float(_g.vol_abs_tolerance)
            if raw_conf_threshold is None:
                raw_conf_threshold = float(_g.raw_conf_threshold)
            if consistency_fallback_threshold is None:
                consistency_fallback_threshold = float(
                    _g.consistency_fallback_threshold
                )
        return cls(attrs1, attrs2, float(vol_tolerance), float(raw_conf_threshold),
                   float(consistency_fallback_threshold), float(vol_abs_tolerance))

    # -- phase: pack + file vetoes ------------------------------------------

    def pack_file_vetoes(self) -> dict | None:
        """pack_gate + the package_type/material/level hard no vetoes."""
        _r = training_cfg().gate.reasons
        if not pack_gate(
            0.0,
            self.attrs1,
            self.attrs2,
            volume_relative_tolerance=float(self.vol_tolerance),
            volume_absolute_tolerance_ml=float(self.vol_abs_tolerance),
            trust_threshold=float(self.raw_conf_threshold),
            check_categorical=False,
        ):
            return GateResult(
                decision="hard_no",
                reason=_r.pack_blocker,
            ).model_dump()
        veto_dimensions = self._veto_dimensions
        for field, dimension, reason in (
            ("package_type_set", "package_type", _r.package_type_mismatch),
            ("package_material_set", "pack_material", _r.package_material_mismatch),
            ("packaging_level_set", None, _r.packaging_level_mismatch),
        ):
            if dimension is not None and dimension not in veto_dimensions:
                continue
            if dimension is not None and any(_has_attribute_flag(record, f"categorical_source_conflict:{dimension}") for record in (self.attrs1, self.attrs2)):
                continue
            left, right = set(self.attrs1.get(field, set())), set(self.attrs2.get(field, set()))
            if left and right and not (left & right):
                return GateResult(decision="hard_no", reason=reason).model_dump()
        return None

    # -- phase: the single decision engine -----------------------------------

    def decision_engine(self) -> dict | None:
        """The census-engine lane: claim conflicts + engine conflicts."""
        # Every explicit categorical conflict uses THE SINGLE DECISION ENGINE
        # (owner directive: ALL attributes × ALL metrics for the ENTIRE decision
        # process). The engine evaluates the three critical-categorical channels
        # with the whole ordered stack (negation hard-veto, alias-folded
        # equality, set overlaps, fuzzy surface) — so unclear spellings rescue
        # instead of riding bare inequality, while a negation conflict stays a
        # definite negative. Unknown stays unknown here; it is not fabricated
        # into a conflict or an agreement.
        from core.attribute_conflicts import (
            CRITICAL_NAME_BY_CENSUS_KEY,
            canonical_attribute_info,
        )
        from core.attribute_decision import AttributeDecisionEngine
        from core.attribute_universe import attribute_registry

        left_info, right_info = canonical_attribute_info(self.attrs1), canonical_attribute_info(self.attrs2)
        source_flags = _attribute_flags(self.attrs1) | _attribute_flags(self.attrs2)
        sweetener_source_conflict = any(
            flag.startswith("sweetener_source_conflict:") for flag in source_flags
        )
        uncertain_categorical_dimensions = {
            flag.split(":", 1)[1] for flag in source_flags
            if flag.startswith(("description_conflict:", "categorical_source_conflict:"))
        }
        if sweetener_source_conflict or source_flags & {
            "unsweetened_with_declared_sweetener", "sweetening_status_conflict",
            "no_added_sugar_with_cane_sugar",
        }:
            uncertain_categorical_dimensions.add("sweetener")
        # Pulp has no registry key; the registry sweetener key owns ingredient
        # identity, not sugar/no-sugar claims. Preserve these separate explicit
        # claim predicates and report their actual dimensions.
        claim_conflicts = sorted(
            dimension for dimension in (self._veto_dimensions & {"sweetener", "pulp"}) - uncertain_categorical_dimensions
            if categorical_conflict(dimension, left_info, right_info)
        )
        if claim_conflicts:
            _r = training_cfg().gate.reasons
            return GateResult(
                decision="hard_no",
                reason=f"{_r.categorical_mismatch} " + ",".join(claim_conflicts),
            ).model_dump()
        categorical_dimensions = self._veto_dimensions - {
            "volume", "pack", "package_type", "pack_material"
        }
        # Evaluate the complete registry; configured vetoes and review policy
        # consume this same evidence rather than projecting away attributes.
        evidence = AttributeDecisionEngine(
            volume_relative_tolerance=float(self.vol_tolerance),
            volume_absolute_tolerance_ml=float(self.vol_abs_tolerance),
        ).evaluate(left_info, right_info, left_raw=self.attrs1, right_raw=self.attrs2)
        categorical_conflicts = sorted(
            CRITICAL_NAME_BY_CENSUS_KEY[key] for key in evidence.conflicts
            if CRITICAL_NAME_BY_CENSUS_KEY.get(key) in
            categorical_dimensions - uncertain_categorical_dimensions
        )
        uncertain_categorical_dimensions.update(
            CRITICAL_NAME_BY_CENSUS_KEY.get(key, key)
            for key, entry in evidence.dimensions.items()
            if entry.fallback_from in {"source_conflict", "claim_conflict"}
        )
        if sweetener_source_conflict:
            categorical_conflicts = [name for name in categorical_conflicts if name != "sweetener"]
        if categorical_conflicts:
            _r = training_cfg().gate.reasons
            return GateResult(
                decision="hard_no",
                reason=f"{_r.categorical_mismatch} " + ",".join(categorical_conflicts),
            ).model_dump()

        # carried into the later phases: the fallback gates need the uncertainty
        # ledger and the summary evidences exactly as evaluated here.
        self._uncertain_categorical = uncertain_categorical_dimensions
        self._left_info = left_info
        self._right_info = right_info
        self._evidence = evidence
        self._source_flags = source_flags
        return None

    # -- phase: fallback lanes ------------------------------------------------

    def primary_fallback_lanes(self) -> dict | None:
        """Source disagreement + ambiguity + the confidence gates."""
        _r = training_cfg().gate.reasons
        if self._uncertain_categorical or self._source_flags & {"volume_sources_disagree", "pack_sources_disagree", "pack_hierarchy_ambiguous"}:
            return GateResult(decision="fallback", reason=_r.source_conflict).model_dump()

        if _has_attribute_flag(self.attrs1, "ambiguous_volume") or _has_attribute_flag(
            self.attrs2, "ambiguous_volume"
        ):
            return GateResult(
                decision="fallback", reason=_r.ambiguous_volume
            ).model_dump()

        # raw confidence check
        if (
            not self.attrs1["volume_set"]
            or not self.attrs2["volume_set"]
            or not self._reliable(self.attrs1["volume_confidence"], self.raw_conf_threshold)
            or not self._reliable(self.attrs2["volume_confidence"], self.raw_conf_threshold)
        ):
            return GateResult(
                decision="fallback", reason=_r.low_volume_confidence
            ).model_dump()
        # Pack confidence: skip when both sides have no pack evidence
        # (single-unit products with no "Count per Unit" in source attributes).
        # pack_gate already treats low-confidence pack evidence as unknown.
        if self.attrs1["pack_set"] or self.attrs2["pack_set"]:
            if (
                not self.attrs1["pack_set"]
                or not self.attrs2["pack_set"]
                or not self._reliable(self.attrs1["pack_confidence"], self.raw_conf_threshold)
                or not self._reliable(self.attrs2["pack_confidence"], self.raw_conf_threshold)
            ):
                return GateResult(
                    decision="fallback", reason=_r.low_pack_confidence
                ).model_dump()
        return None

    # -- phase: overflow lanes -------------------------------------------------

    def overlap_lanes(self) -> dict | None:
        """Volume-overlap and pack-overlap hard nos (same predicates as every
        other lane)."""
        _r = training_cfg().gate.reasons
        # volume overlap
        vol_overlap = False
        for v1 in self.attrs1["volume_set"]:
            for v2 in self.attrs2["volume_set"]:
                if v1 == 0 or v2 == 0:
                    continue
                # Same predicate as every other lane (SSOT, audit 2026-09-15):
                # whichever of the two configured cuts is wider applies. The
                # hand-rolled relative-only ratio this replaces disagreed with
                # the veto lane at small volumes.
                if volumes_compatible(
                    {v1},
                    {v2},
                    volume_relative_tolerance=float(self.vol_tolerance),
                    volume_absolute_tolerance_ml=float(self.vol_abs_tolerance),
                ):
                    vol_overlap = True
                    break
            if vol_overlap:
                break
        if "volume" in self._veto_dimensions and not vol_overlap:
            return GateResult(decision="hard_no", reason=_r.no_volume_overlap).model_dump()

        # pack overlap: skip when both sides have no pack evidence
        # (single-unit products with no "Count per Unit" in source).
        if self.attrs1["pack_set"] or self.attrs2["pack_set"]:
            pack_overlap = self.attrs1["pack_set"] & self.attrs2["pack_set"]
            if "pack" in self._veto_dimensions and not pack_overlap:
                return GateResult(decision="hard_no", reason=_r.no_pack_overlap).model_dump()
        return None

    # -- phase: packaging level + identity + review lanes -----------------------

    def identity_review_lanes(self) -> dict | None:
        """Packaging level, consistency, supporting dims, mode_flavor, declared
        identity, pair policy. Placement after the categorical conflict check
        is load-bearing measured history (see the inline comments)."""
        _r = training_cfg().gate.reasons
        # PACKAGING LEVEL is one-sided in practice (measured 2026-09-30: 217 of
        # 13,250 records assert a level, and ZERO pairs have it populated on both
        # sides), so the both-populated rule above can never fire for it. That is
        # deliberate, not an oversight: a missing marker is absence of evidence,
        # not an affirmative "retail" claim, so this CANNOT be a hard_no without
        # inventing a negative from silence.
        #
        # PLACED AFTER the categorical conflict check on purpose (measured
        # 2026-09-30): an earlier placement downgraded 79 genuine flavour
        # conflicts from hard_no to fallback, because a one-sided level claim
        # is WEAKER evidence than a two-sided attribute conflict. A definite
        # negative must always win over a review flag.
        #
        # It still must not be a silent PROCEED. A case listing and a retail pack
        # are distinct GS1 trade items carrying distinct GTINs, so merging them
        # trains the linker to violate that. One-sided evidence is exactly what
        # the fallback bucket is for: a human applies the rule, the model is not
        # asked to guess. Measured impact: 109 proceed -> fallback, 0 hard_no.
        _lvl_a, _lvl_b = set(self.attrs1.get("packaging_level_set", set())), set(
            self.attrs2.get("packaging_level_set", set())
        )
        if _lvl_a and not _lvl_b or _lvl_b and not _lvl_a:
            return GateResult(
                decision="fallback",
                reason=_r.packaging_level_review,
            ).model_dump()

        # consistency check
        if (
            not self._reliable(self.attrs1["volume_consistency"], self.consistency_fallback_threshold)
            or not self._reliable(self.attrs2["volume_consistency"], self.consistency_fallback_threshold)
            or not self._reliable(self.attrs1["pack_consistency"], self.consistency_fallback_threshold)
            or not self._reliable(self.attrs2["pack_consistency"], self.consistency_fallback_threshold)
        ):
            return GateResult(
                decision="fallback", reason=_r.low_consistency
            ).model_dump()

        # Supporting attributes can require review when the critical flavor
        # evidence is incomplete. They never acquire a hard-veto permission.
        if not self._left_info.get("flavor_set") or not self._right_info.get("flavor_set"):
            from core.attribute_conflicts import _universe_value
            from core.attribute_universe import attribute_registry
            supporting = training_cfg().rand_matching.targeted_veto_gates.supporting_feature_review_dimensions
            specs = attribute_registry()
            differing_support = []
            for dimension in supporting:
                spec = specs[dimension]
                left = set(_universe_value(self._left_info, dimension, spec))
                right = set(_universe_value(self._right_info, dimension, spec))
                if left and right and not (left <= right or right <= left):
                    differing_support.append(dimension)
            if differing_support:
                return GateResult(
                    decision="fallback",
                    reason=_r.supporting_feature_review + " " + ",".join(sorted(differing_support)),
                ).model_dump()

        # MODE_FLAVOR SURFACE LANE (JEV audit, 2026-10-02): reached only when
        # no census conflict vetoed the pair. When the canonical mode_flavor
        # values are both populated and different, this is not a clean
        # proceed — the mode is the deterministic per-listing consensus and it
        # disagrees. Measured on the JEV-graded slice: 216/523 falsified
        # positives carry differing modes (plus 69 with one empty side, not
        # reachable by this two-sided rule). REVIEW only, never a hard_no: a
        # mode is weaker evidence than a census conflict, matching the
        # packaging-level doctrine.
        mf1 = str(self.attrs1.get("mode_flavor", "") or "").strip().lower()
        mf2 = str(self.attrs2.get("mode_flavor", "") or "").strip().lower()
        equal_full_flavor = bool(self._left_info.get("flavor_set")) and self._left_info.get("flavor_set") == self._right_info.get("flavor_set")
        if mf1 and mf2 and mf1 != mf2 and not equal_full_flavor:
            return GateResult(
                decision="fallback",
                reason=_r.supporting_feature_review + " mode_flavor:" + mf1 + "|" + mf2,
            ).model_dump()

        # Exact-product approval requires consistency of declared identity, not
        # merely an absence of conflicts in generic or missing attribute sets.
        # Review additions/subsets and named distinctions; existing configured
        # hard vetoes retain precedence above this supplementary review lane.
        from core.declared_identity import identity_review_dimensions
        identity_differences = identity_review_dimensions(self.attrs1, self.attrs2)
        if identity_differences:
            return GateResult(
                decision="fallback",
                reason=_r.declared_identity_review + " " + ",".join(identity_differences),
            ).model_dump()

        from core.pair_policy import assess_pair
        policy = assess_pair(self._evidence, self.attrs1, self.attrs2)
        if policy['review']:
            return GateResult(
                decision="fallback",
                reason=_r.supporting_feature_review + " full_evidence:" + ",".join(policy['review']),
            ).model_dump()

        return GateResult(
            decision="proceed", reason=_r.clean_proceed
        ).model_dump()

    # -- shared helper ---------------------------------------------------------

    @staticmethod
    def _reliable(value: object, threshold: float) -> bool:
        """NaN bypasses ordinary less-than checks; invalid evidence is unknown."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number) and 0.0 <= number <= 1.0 and number >= threshold

    def decide(self) -> dict:
        """Run the ORIGINAL phase order; the first veto/fallback verdict wins."""
        self._veto_dimensions = frozenset(
            training_cfg().rand_matching.targeted_veto_gates.veto_dimensions
        )
        verdict = self.pack_file_vetoes()
        if verdict is not None:
            return verdict
        verdict = self.decision_engine()
        if verdict is not None:
            return verdict
        verdict = self.primary_fallback_lanes()
        if verdict is not None:
            return verdict
        verdict = self.overlap_lanes()
        if verdict is not None:
            return verdict
        return self.identity_review_lanes()


def three_way_gate(
    attrs1: dict,
    attrs2: dict,
    vol_tolerance: float | None = None,
    raw_conf_threshold: float | None = None,
    consistency_fallback_threshold: float | None = None,
    vol_abs_tolerance: float | None = None,
) -> dict:
    """Deterministic volume/pack/flavor gate — one phase-ordered decision
    table on _ThreeWayGate (same decisions, same reasons as before)."""
    gate = _ThreeWayGate.from_config(
        attrs1, attrs2,
        vol_tolerance=vol_tolerance,
        raw_conf_threshold=raw_conf_threshold,
        consistency_fallback_threshold=consistency_fallback_threshold,
        vol_abs_tolerance=vol_abs_tolerance,
    )
    return gate.decide()
# ============================================================================
# SIMILARITY
# ============================================================================
def jaccard_similarity(str1: str, str2: str) -> float:
    """Word-set Jaccard overlap (0.0 when either side is empty)."""
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


# ============================================================================
# CANONICAL
# ============================================================================


MINIMAL_STOPWORDS = _load_stopwords("MINIMAL_STOPWORDS")

# CONCEPT FOLDS (owner directive 2026-09-08: "only one instance of each
# concept"): synonym/singular-plural pairs that are THE SAME product
# concept — 'sparkling'+'carbonated' co-occurred in 836 canonicals,
# singular+plural ('mineral'+'minerals') in 196. The KEY is the canonical
# representative; every VALUE folds into it before the word-once passes.
# SSOT: stopwords.json CONCEPT_FOLDS. Keep-tokens stay atomic (no_sugar
# never folds).
_CONCEPT_FOLDS: dict[str, str] = _load_concept_folds()


def _fold_concept(word: str) -> str:
    """Fold a word to its canonical concept representative (SSOT map)."""
    return _CONCEPT_FOLDS.get(word, word)


# -----------------------------------------------------------------------------
# Minimal stopwords: only truly non‑semantic tokens (not product attributes)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Critical single tokens that must always be considered when present
# -----------------------------------------------------------------------------
KEEP_TOKENS = {
    "zero",
    "light",
    "diet",
    "carbonated",
    "still",
    "sparkling",
    "pulp",
    "sugar",
    "no_sugar",
    "no_added_sugar",
    "added_sugar",
    "with_pulp",
    "no_pulp",
}

# PHRASE VARIATIONS (owner ruling 2026-09-07): one diet-variant concept,
# many retail phrasings. A keep-token matches when ANY variant regex fires
# on the normalized doc text — hyphenated ("sugar-free"), fused
# ("sugarfree"), reversed ("free sugar"), of-linked ("free of sugar"),
# sweetener-synonym ("sugarless", "without sugar", "zero sugar") all map
# to the SAME canonical token so the diet variant stays identity-bearing
# in the canonical. Census on the deduped corpus: "sugar free" 1,589 /
# "sugarfree" 124 / "sugarless" 13 / "free sugar" 9 (all are "…calorie
# free SUGAR FREE…" — two adjacent compounds) / "free of sugar" 0 (not in
# this export but covered by the ruling) / "no sugar" 7,123 / "no added
# sugar" 4,039 / "without added sugar" 39.
# All sugar-free spellings use one ``no_sugar`` token. ``no added sugar`` is
# deliberately separate because it does not prove absence of natural sugar.
PHRASE_VARIANTS = {
    "no_sugar": [
        re.compile(r"\bsugar\s*[- ]?\s*free\b"),          # sugar free / sugar-free / sugarfree
        re.compile(r"\bsugarfree\b"),                     # fused (no separator survived)
        re.compile(r"\bsugarless\b"),                     # sweetener synonym
        re.compile(r"\bfree\s+(?:of\s+)?sugar\b"),        # free sugar / free of sugar
        re.compile(r"\bwithout\s+sugar\b"),               # without sugar
        re.compile(r"\bzero\s+sugar\b"),                  # zero sugar
        re.compile(r"\bno\s+sugar\b"),                    # no sugar IS sugar-free
    ],
    "no_added_sugar": [
        re.compile(r"\bno\s+added\s+sugar\b"),
        re.compile(r"\bwithout\s+added\s+sugar\b"),
    ],
    "added_sugar": [
        re.compile(r"\bwith\s+added\s+sugar\b"),          # the POSITIVE claim only
    ],
    # pulp variants: 'with' and 'no' are stopworded before the bigram forms,
    # so with_pulp/no_pulp NEVER fired (dead keep-tokens) — and raw bigram
    # 'cola_pulp' is IDENTICAL for both claims, so only the phrase layer can
    # keep them distinct.
    "with_pulp": [
        re.compile(r"\bwith\s+(?:extra\s+)?pulp\b"),      # with pulp / with extra pulp
    ],
    "no_pulp": [
        re.compile(r"\b(?:no|without)\s+pulp\b"),          # no pulp / without pulp
        re.compile(r"\bpulp\s*[- ]?\s*free\b"),            # pulp free / pulp-free
        re.compile(r"\bfree\s+(?:of\s+)?pulp\b"),          # free pulp / free of pulp
    ],
}


# -----------------------------------------------------------------------------
# N‑gram IDF calculator (global or within‑brand)
# -----------------------------------------------------------------------------
class NgramIDF:
    """Compute document frequency of n‑grams (1-4) across a set of GTIN documents."""

    def __init__(self, rows_by_gtin):
        self.N = len(rows_by_gtin)
        self.df = Counter()
        self._build(rows_by_gtin)

    def _build(self, rows_by_gtin):
        for rows in rows_by_gtin.values():
            doc_ngrams = set()
            tokens = []
            for sku, attr in rows:
                text = normalize_text(sku) + " " + normalize_text(attr)
                # Remove volume/pack numbers and unit words
                text = re.sub(
                    r"\b\d+(\.\d+)?\s*(ml|l|lt|ltr|liter|litre|cl|centiliter|oz|fl oz|qt|gal|ounce|fluid ounce|pack|case|pcs?|pieces?|units?|x)\b",
                    " ",
                    text,
                    flags=re.IGNORECASE,
                )
                toks = text.split()
                toks = [
                    tok for tok in toks if tok not in MINIMAL_STOPWORDS and len(tok) > 1
                ]
                tokens.extend(toks)
            # Generate n‑grams (1,2,3,4)
            for n in (1, 2, 3, 4):
                for i in range(len(tokens) - n + 1):
                    ngram = " ".join(tokens[i : i + n])
                    doc_ngrams.add(ngram)
            for ngram in doc_ngrams:
                self.df[ngram] += 1

    def idf(self, ngram):
        """Inverse document frequency with smoothing."""
        return math.log((self.N + 1) / (self.df.get(ngram, 0) + 1)) + 1.0


# -----------------------------------------------------------------------------
# N‑gram generation
# -----------------------------------------------------------------------------
def generate_ngrams(tokens: list[str], n: int) -> list[str]:
    """Contiguous n-gram strings (space-joined) over a token list."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# -----------------------------------------------------------------------------
# Discriminative n‑gram extraction
# -----------------------------------------------------------------------------
class _SalientNgramScorer:
    """Select a document's discriminative n-grams (1-4) by TF-IDF.

    Single responsibility per phase; score() runs them in ONE fixed order and
    the statements are the pre-refactor body verbatim, so the selected list
    (and its ordering) is byte-identical.

    Phase map:
      tokenize       — normalized title/attribute streams: filtered unigram
                       token lane + PRE-stopword phrase lane (the phrase
                       regexes must see 'no'/'with'/'of')
      tfidf_select   — candidate n-grams, global x brand IDF scoring (with the
                       all-GTIN penalty and the length bonus), descending cut
      keep_token_merge — KEEP_TOKENS/PHRASE_VARIANTS rescue lane (atomic and
                       bigram forms, sorted for PYTHONHASHSEED determinism)
    """

    def __init__(self, titles: list[str], attributes: list[str],
                 brand_tokens: set[str], global_idf: 'NgramIDF',
                 brand_idf: 'NgramIDF', top_k: int) -> None:
        self._titles = titles
        self._attributes = attributes
        self._brand_tokens = brand_tokens
        self._global_idf = global_idf
        self._brand_idf = brand_idf
        self._top_k = top_k

    # ── phase: tokenize ─────────────────────────────────────────────────────

    def tokenize(self) -> tuple[list[str], list[str]]:
        """Combine all text into token list; carry the PRE-stopword phrase
        parts alongside the filtered stream."""
        tokens = []
        phrase_parts = []  # PRE-stopword text: phrase regexes must see 'no',
        # 'with', 'of' — MINIMAL_STOPWORDS deletes them before the keep-token
        # check could ever fire (the live miss on "no sugar"/"free of sugar")
        for title, attr in zip(self._titles, self._attributes, strict=True):
            text = normalize_text(title) + " " + normalize_text(attr)
            text = re.sub(
                r"\b\d+(\.\d+)?\s*(ml|l|lt|ltr|liter|litre|cl|centiliter|oz|fl oz|qt|gal|ounce|fluid ounce|pack|case|pcs?|pieces?|units?|x)\b",
                " ",
                text,
                flags=re.IGNORECASE,
            )
            toks = text.split()
            phrase_parts.append(text)
            toks = [tok for tok in toks if tok not in MINIMAL_STOPWORDS and len(tok) > 1]
            tokens.extend(toks)
        return tokens, phrase_parts

    # ── phase: TF-IDF selection ─────────────────────────────────────────────

    def tfidf_select(self, tokens: list[str]) -> list[str]:
        """Candidate n-grams (1-4) scored against global + brand IDF; the
        highest-scored ``top_k`` (underscore-joined) survive."""
        candidates = []
        for n in (1, 2, 3, 4):
            candidates.extend(generate_ngrams(tokens, n))

        if not candidates:
            return []

        tf = Counter(candidates)
        total = len(candidates)

        # Number of GTINs in the brand
        N_brand = self._brand_idf.N if self._brand_idf else 1

        scores = {}
        for ngram, count in tf.items():
            tf_val = count / total if total else 0
            g_idf = self._global_idf.idf(ngram)
            b_idf = self._brand_idf.idf(ngram) if self._brand_idf else 1.0

            # Strong penalty for n‑grams present in ALL brand GTINs (not discriminative)
            if self._brand_idf:
                df_brand = self._brand_idf.df.get(ngram, 0)
                if df_brand == N_brand:
                    b_idf = 0.05  # almost zero

            num_words = len(ngram.split())
            # Length bonus: longer n‑grams are more specific, but we include unigrams with slight penalty
            if num_words == 1:
                length_bonus = 0.8
            else:
                length_bonus = 1.0 + 0.1 * (num_words - 1)

            score = tf_val * g_idf * b_idf * length_bonus
            scores[ngram] = score

        sorted_ngrams = sorted(scores.items(), key=lambda x: -x[1])
        return [ngram.replace(" ", "_") for ngram, _ in sorted_ngrams[:self._top_k]]

    # ── phase: keep-token merge ─────────────────────────────────────────────

    def keep_token_merge(self, tokens: list[str], phrase_parts: list[str],
                         selected: list[str]) -> list[str]:
        """Add KEEP_TOKENS that appear in the document but may not be top.

        Compound keepers ('no_sugar', 'with_pulp') are stored underscore-joined
        and used to be checked against SPACE-joined doc text — they could never
        match (dead entries). Check the compound's WORDS as a contiguous bigram
        instead ('no' is stopworded away, so 'sugar_free' matches 'sugar free').
        The PRE-stopword text feeds the phrase regexes."""
        bigrams = {
            f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)
        }
        # PRE-stopword text: 'no sugar'/'free of sugar'/'with added sugar' die
        # in the MINIMAL_STOPWORDS filter before the keep check — the phrase
        # regexes see the raw normalized text, the token/bigram checks keep
        # using the filtered stream (unchanged behavior for plain keepers).
        doc_text = " ".join(phrase_parts)
        # DETERMINISM (reproducibility contract): iterating a SET of strings is
        # process-random (PYTHONHASHSEED) — keep-tokens appended in a different
        # order per run and canonical_records.csv drifted. sorted() pins it.
        for keep in sorted(KEEP_TOKENS):
            # PHRASE VARIATIONS (owner ruling 2026-09-07): one concept, many
            # phrasings — a keep-token matches when ANY of its regex variants
            # fires on the doc text (hyphens/fused/reversed/of-linked word
            # orders all map to the SAME canonical token; census: sugar free
            # 1,589 / sugarfree 124 / sugarless 13 / free sugar 9 / free of
            # sugar 0-but-covered / no sugar 7,123 / no added sugar 4,039).
            if keep in PHRASE_VARIANTS:
                hit = any(p.search(doc_text) for p in PHRASE_VARIANTS[keep])
            else:
                hit = (keep in tokens) if "_" not in keep else (keep in bigrams)
            if hit and keep not in selected:
                selected.append(keep)
                if len(selected) >= self._top_k + 3:
                    break
        return selected

    # ── orchestration ───────────────────────────────────────────────────────

    def score(self) -> list[str]:
        """Run the load-bearing phase order."""
        tokens, phrase_parts = self.tokenize()
        selected = self.tfidf_select(tokens)
        return self.keep_token_merge(tokens, phrase_parts, selected)


def extract_discriminative_ngrams(
    titles: list[str],
    attributes: list[str],
    brand_tokens: set[str],
    global_idf: NgramIDF,
    brand_idf: NgramIDF,
    top_k: int = 5,
) -> list[str]:
    """Select n‑grams (1‑4) with highest TF‑IDF, considering global and
    within‑brand IDF — see _SalientNgramScorer.score."""
    return _SalientNgramScorer(
        titles, attributes, brand_tokens, global_idf, brand_idf, top_k
    ).score()


# -----------------------------------------------------------------------------
# Canonical generation (now uses n‑grams)
# -----------------------------------------------------------------------------
def generate_canonical(
    gtin: str,
    brand: str,
    rows: list[tuple[str, str]],
    global_idf: NgramIDF,
    brand_idf: NgramIDF | None,
    *,
    descriptions: list[str] | None = None,
    urls: list[str] | None = None,
    image_urls: list[str] | None = None,
    breadcrumbs_engs: list[str] | None = None,
    categories: list[str] | None = None,
    countries: list[str] | None = None,
    retailers: list[str] | None = None,
) -> dict:  # CanonicalRecord.model_dump() — validated shape, plain dict
    """Compose one GTIN's canonical record — see CanonicalCardComposer.compose."""
    return CanonicalCardComposer(
        gtin, brand, rows, global_idf, brand_idf,
        descriptions=descriptions, urls=urls, image_urls=image_urls,
        breadcrumbs_engs=breadcrumbs_engs, categories=categories,
        countries=countries, retailers=retailers,
    ).compose()


class CanonicalCardComposer:
    """One GTIN's canonical record, phase by phase.

    Single responsibility per phase; compose() runs them in ONE fixed order.
    Aggregate-set construction, conflict-flag vocabulary, the token-once
    canonical text discipline, the universe-evidence JSON and the additive
    persistence keys are byte-identical to the pre-refactor linear body.

    Phases (order = load-bearing):
      accept_rows -> per-listing extract_all cards
      aggregate_sets -> every canonical set column + contradiction flags
      confidences_and_consistency -> mean confidences, mode-share consistency
      token_once_text -> strict-novelty n-grams + word-once final pass
      universe_evidence -> census-SSOT parse, one JSON string
      record -> boundary validation (CanonicalRecord) + additive keys
    """

    def __init__(self, gtin, brand, rows, global_idf: 'NgramIDF',
                 brand_idf: 'NgramIDF | None', *, descriptions=None, urls=None,
                 image_urls=None, breadcrumbs_engs=None, categories=None,
                 countries=None, retailers=None):
        self._gtin = gtin
        self._brand = brand
        self._rows = rows
        self._global_idf = global_idf
        self._brand_idf = brand_idf
        self._descriptions = descriptions
        self._urls = urls
        self._image_urls = image_urls
        self._breadcrumbs_engs = breadcrumbs_engs
        self._categories = categories
        self._countries = countries
        self._retailers = retailers
        # phase outputs
        self.titles: list[str] = []
        self.attributes: list[str] = []
        self.extracted: list[dict] = []
        self.brand_norm = ""
        self.brand_tokens: set[str] = set()
        self.mode_flavor = ""
        self.mode_type = ""
        self.salient_ngrams: list[str] = []
        self.raw_salient_ngrams: list[str] = []
        self.volume_set: set[float] = set()
        self.pack_set: set[int] = set()
        self.package_type_set: set[str] = set()
        self.packaging_level_set: set[str] = set()
        self.package_material_set: set[str] = set()
        self.flavor_set: set[str] = set()
        self.made_from_set: set[str] = set()
        self.carbonation_set: set[str] = set()
        self.sweetener_set: set[str] = set()
        self.sweetener_type_set: set[str] = set()
        self.sweetening_set: set[str] = set()
        self.attribute_consistency_flags: set[str] = set()
        self.pulp_set: set[str] = set()
        self.organic_set: set[str] = set()
        self.vol_conf = 0.0
        self.pack_conf = 0.0
        self.n_titles = 0
        self.volume_consistency = 1.0
        self.pack_consistency = 1.0
        self.canonical = ""
        self.kept_ngrams: list[str] = []
        self.universe_evidence_json = ""

    # ── phase 1: rows ──────────────────────────────────────────────────────

    def accept_rows(self) -> None:
        """Materialize the per-listing cards (extract_all) in row order."""
        self.titles = [sku for sku, attr in self._rows]
        self.attributes = [attr for sku, attr in self._rows]
        descriptions = self._descriptions or [""] * len(self._rows)
        urls = self._urls or [""] * len(self._rows)
        image_urls = self._image_urls or [""] * len(self._rows)
        breadcrumbs_engs = self._breadcrumbs_engs or [""] * len(self._rows)
        categories = self._categories or [""] * len(self._rows)
        countries = self._countries or [""] * len(self._rows)
        retailers = self._retailers or [""] * len(self._rows)
        self.extracted = [
            extract_all(
                sku, attr,
                "" if pd.isna(desc) else str(desc),
                url, img_url, cat_path, cat,
            )
            for (sku, attr), desc, url, img_url, cat_path, cat
            in zip(self._rows, descriptions, urls, image_urls, breadcrumbs_engs, categories, strict=True)
        ]
        del countries, retailers
        self.brand_norm = normalize_text(spell_numeric_brand(self._brand))
        self.brand_tokens = set(self.brand_norm.split())

    # ── phase 2: sets + flags ──────────────────────────────────────────────

    def aggregate_sets(self) -> None:
        """Per-GTIN union of every extracted set column + contradiction flags."""
        flavors = [x["flavor"] for x in self.extracted if x["flavor"]]
        types = [x["type"] for x in self.extracted if x["type"]]
        self.mode_flavor = Counter(flavors).most_common(1)[0][0] if flavors else ""
        self.mode_type = Counter(types).most_common(1)[0][0] if types else ""

        # Get discriminative n‑grams
        self.salient_ngrams = extract_discriminative_ngrams(
            self.titles, self.attributes, self.brand_tokens,
            self._global_idf, self._brand_idf, top_k=5,
        )

        # Volume and pack sets
        self.volume_set = {round(x["volume_ml"], 2) for x in self.extracted if x["volume_ml"] > 0}
        # A parser-safe quantity of one is not evidence of a single-item pack.
        # Keep only rows with explicit pack evidence in the canonical attribute
        # set; otherwise missing pack data becomes a false pack conflict.
        self.pack_set = {
            x["pack_qty"] for x in self.extracted if x["pack_confidence"] > 0
        }
        self.package_type_set = {value for x in self.extracted for value in x["package_types"]}
        self.packaging_level_set = {value for x in self.extracted for value in x["packaging_levels"]}
        self.package_material_set = {value for x in self.extracted for value in x["package_materials"]}
        self.flavor_set = {value for x in self.extracted for value in x["flavor_set"]}
        self.made_from_set = {value for x in self.extracted for value in x["made_from_set"]}
        self.carbonation_set = {value for x in self.extracted for value in x["carbonation_set"]}
        self.sweetener_set = {value for x in self.extracted for value in x["sweetener_set"]}
        self.sweetener_type_set = {value for x in self.extracted for value in x["sweetener_type_set"]}
        self.sweetening_set = {value for x in self.extracted for value in x["sweetening_set"]}
        self.attribute_consistency_flags = {value for x in self.extracted for value in x["attribute_consistency_flags"]}
        # Negations can be on a different listing of the same GTIN from the
        # affirmative ingredient. Preserve that contradiction at aggregation.
        negative_ingredients = {
            value for x in self.extracted for value in x.get("negated_sweetener_type_set", ())
        }
        self.attribute_consistency_flags.update(
            f"sweetener_source_conflict:{ingredient}"
            for ingredient in negative_ingredients & self.sweetener_type_set
        )
        self.pulp_set = {value for x in self.extracted for value in x["pulp_set"]}
        self.organic_set = {value for x in self.extracted for value in x.get("organic_set") or set()}
        self._contradiction_flags()

    def _contradiction_flags(self) -> None:
        """Cross-check aggregated sets; the flag vocabulary is unchanged."""
        for dimension, values, opposites in (
            ("sweetener", self.sweetener_set, (("sugar", "no_sugar"), ("sugar", "diet"))),
            ("carbonation", self.carbonation_set, (("still", "carbonated"),)),
            ("pulp", self.pulp_set, (("no_pulp", "with_pulp"),)),
            ("organic", self.organic_set, (("organic", "not_organic"),)),
        ):
            if any({left, right} <= values for left, right in opposites):
                self.attribute_consistency_flags.add(f"categorical_source_conflict:{dimension}")

        gate_cfg = training_cfg().gate
        observed_volumes = sorted(self.volume_set)
        if any(not volumes_compatible({left}, {right},
                                      volume_relative_tolerance=float(gate_cfg.vol_tolerance),
                                      volume_absolute_tolerance_ml=float(gate_cfg.vol_abs_tolerance))
               for i, left in enumerate(observed_volumes) for right in observed_volumes[i + 1:]):
            self.attribute_consistency_flags.add('volume_sources_disagree')
        if len(self.pack_set) > 1:
            self.attribute_consistency_flags.add('pack_sources_disagree')
        if len(self.package_type_set) > 1:
            self.attribute_consistency_flags.add('categorical_source_conflict:package_type')
        if len(self.package_material_set) > 1:
            self.attribute_consistency_flags.add('categorical_source_conflict:pack_material')

    # ── phase 3: confidence / consistency ──────────────────────────────────

    def confidences_and_consistency(self) -> None:
        """Mean per-card confidences + mode-share consistency (scale-free)."""
        vol_confs = [x["volume_confidence"] for x in self.extracted if x["volume_ml"] > 0]
        pack_confs = [x["pack_confidence"] for x in self.extracted if x["pack_confidence"] > 0]
        self.vol_conf = sum(vol_confs) / len(vol_confs) if vol_confs else 0.0
        self.pack_conf = sum(pack_confs) / len(pack_confs) if pack_confs else 0.0
        self.n_titles = len(self.extracted)
        # consistency = share of rows agreeing with the MOST COMMON value.
        # The old formula divided conflicts by ROW COUNT n, so a 41k-row group
        # with 2,000 distinct volumes scored 0.95 "consistent" — more rows made
        # contradiction look BETTER. Mode-share is scale-free and monotone.
        vol_mode = Counter(x["volume_ml"] for x in self.extracted if x["volume_ml"] > 0)
        pack_mode = Counter(
            x["pack_qty"] for x in self.extracted if x["pack_confidence"] > 0
        )
        # mode share over rows that HAVE a volume (unknown-volume rows don't vote)
        self.volume_consistency = (
            (vol_mode.most_common(1)[0][1] / sum(vol_mode.values())) if vol_mode else 1.0
        )
        n_known_pack = sum(pack_mode.values())
        self.pack_consistency = (
            pack_mode.most_common(1)[0][1] / n_known_pack
            if n_known_pack
            else 1.0
        )

    # ── phase 4: the canonical text ────────────────────────────────────────

    def token_once_text(self) -> None:
        """Build the canonical string under the WORD-ONCE discipline (all
        statements and pass order byte-identical to the original body)."""
        # Build canonical string
        # TOKEN-ONCE DISCIPLINE (owner directive 2026-09-08): a canonical must
        # carry each WORD at most once — the concatenation of brand + flavor +
        # type + salient n-grams used to repeat 'water' up to 5x (unigram from
        # mode_type AND inside 4 different IDF compounds) in 1,489 canonicals;
        # pure noise for both Jaccard and the embedding payload. Dedup works
        # on UNDERSCORE-PARTS across ALL parts: an n-gram compound is dropped
        # when EVERY word in it already appeared earlier (fully redundant);
        # a compound carrying at least one new word stays (partial novelty —
        # dropping only the repeated words would mutate the compound into a
        # string that no longer corresponds to any real n-gram). First
        # occurrence wins; deterministic by construction order.
        parts = [self.brand_norm]
        if self.mode_flavor:
            parts.append(self.mode_flavor)
        if self.mode_type:
            parts.append(self.mode_type)
        # Explicit categorical fields are spoken before free-form n-grams. This
        # guarantees that polarity survives canonical generation even when its
        # source phrase is not among the top TF-IDF n-grams.
        parts.extend(sorted(self.carbonation_set))
        parts.extend(sorted(self.sweetener_set))
        parts.extend(sorted(self.pulp_set))
        parts.extend(sorted(self.flavor_set - ({self.mode_flavor} if self.mode_flavor else set())))
        # token-once: filter the salient n-grams against every word already
        # spoken (brand/flavor/type parts + earlier n-grams). STRICT novelty
        # (owner directive 2026-09-08: 'we cant have repeated strings'): the
        # IDF top-5 are a sliding-window ladder (kr_white_grape_flavored,
        # white_grape_flavored_sparkling, grape_flavored_sparkling_bottled —
        # one phrase, three compounds, words repeated 3x) — an n-gram is kept
        # only when EVERY word it carries is new; the highest-IDF member of
        # each phrase family wins and the ladder's redundant echo dies.
        spoken: set[str] = set()
        for p in parts:
            if p:
                spoken.update(
                    _fold_concept(w) for t in p.split() for w in t.split("_")
                )
        kept_ngrams: list[str] = []
        for ng in self.salient_ngrams:
            # concept-fold each word before novelty: a compound carrying only
            # folded echoes of already-spoken concepts ('sparkling' after
            # 'carbonated') is redundant, not new
            words = [w for w in ng.split("_") if w]
            folded = [_fold_concept(w) for w in words]
            if folded and all(f not in spoken for f in folded):
                kept_ngrams.append("_".join(folded))
                spoken.update(folded)
        # fallback: if strict novelty dropped EVERYTHING (top-5 all one family
        # and mode parts already spoke the words), keep the first n-gram that
        # carries ANY new word — but contribute ONLY its new words (owner
        # directive: no repeated strings; 'berry_acai' after mode 'berry'
        # contributes 'acai', not the echo). If truly nothing is new, keep the
        # single top n-gram (canonical never ends up bare brand+flavor+type).
        if not kept_ngrams and self.salient_ngrams:
            for ng in self.salient_ngrams:
                new_words = [
                    _fold_concept(w)
                    for w in ng.split("_")
                    if w and _fold_concept(w) not in spoken
                ]
                if new_words:
                    kept_ngrams = ["_".join(new_words)]
                    spoken.update(new_words)
                    break
            else:
                kept_ngrams = [self.salient_ngrams[0]]
                spoken.update(
                    w for w in self.salient_ngrams[0].split("_") if w
                )
        # raw list kept for the strip-audit visibility (what token-once removed)
        self.raw_salient_ngrams = list(self.salient_ngrams)
        self.salient_ngrams = kept_ngrams
        self.kept_ngrams = kept_ngrams
        parts.extend(self.salient_ngrams)
        self._final_word_once_pass(parts)

    def _final_word_once_pass(self, parts: list[str]) -> None:
        """Rewrite compounds under the concept-fold word-once rule."""
        # FINAL WORD-ONCE PASS (owner directive: 'it should have only a single
        # instance of each word'): the strict novelty pass runs on COMPOUND
        # granularity (an n-gram survives or dies whole), so a kept compound
        # can still repeat a word internally (source text 'Vitamin B12 Vitamin
        # 6' -> vitamin_b12_vitamin_b6) or against a later keep-token
        # (sugar_sugar_calories_water then no_sugar). This pass rewrites the
        # compounds themselves: every compound keeps only its first-seen
        # CONCEPT-folded words (sparkling==carbonated, minerals==mineral —
        # SSOT stopwords.json CONCEPT_FOLDS). Keep-tokens are ALWAYS atomic
        # (no_sugar never folds or fragments). A compound reduced to zero
        # words drops; unigrams follow the same rule. First occurrence wins;
        # deterministic.
        seen_words: set[str] = set()
        final_tokens: list[str] = []
        for t in " ".join(parts).split():
            # keep-tokens are ATOMIC: no_sugar / with_pulp never fold or
            # fragment — they carry the phrase-variant concept as one unit
            is_keep = t in KEEP_TOKENS  # atomic — never folded, never split
            if "_" in t and not is_keep:
                kept = []
                for w in t.split("_"):
                    fw = _fold_concept(w)
                    if fw and fw not in seen_words:
                        seen_words.add(fw)
                        kept.append(fw)
                if kept:
                    final_tokens.append("_".join(kept))
            elif is_keep:
                if t not in seen_words:
                    # atomic: mark the whole token, not its parts
                    seen_words.add(t)
                    final_tokens.append(t)
            else:
                ft = _fold_concept(t)
                if ft and ft not in seen_words:
                    seen_words.add(ft)
                    final_tokens.append(ft)
        self.canonical = " ".join(final_tokens)

    # ── phase 5: universe evidence ─────────────────────────────────────────

    def universe_evidence(self) -> None:
        """Parse the GTIN's raw attribute cells with the SAME census SSOT
        parser and UNION the token sets per registered key."""
        # CANONICAL-SIDE UNIVERSE EVIDENCE (owner ruling 2026-10-01, closes the
        # wiring-agent's reported gap): canonical_records.csv previously carried
        # NO universe_evidence column, so the CANONICAL side of the per-pair
        # attribute census (core.attribute_conflicts.full_dimension_states ->
        # canonical_attribute_info._universe_evidence_of) could only populate the
        # critical channels; the SKU side already parses all 37 registered keys
        # from the raw attribute cell (parse_universe_cell). Parse each of the
        # GTIN's raw attribute cells with the SAME census SSOT parser
        # (parse_universe_cell -> AttributeUniverse.parse) and UNION the token
        # sets per registered key — same key normalization
        # (core.text.normalized_attribute_text), same token lowercase/strip, same
        # band canon. Persisted per key as SORTED value lists under ONE
        # deterministic JSON string (sorted keys), matching the CSV writer's
        # sorted-set convention: ast.literal_eval round-trips it in
        # _universe_evidence_of exactly like the other canonical set columns.
        # Two keys are deliberately NOT persisted:
        #   * "volume" — the parse emits float ml values; the canonical volume
        #     channel is volume_set (read directly by _universe_value), so floats
        #     here would be unused noise in the CSV;
        #   * "unclassified_keys" — key NAMES, not values; the reader keeps
        #     registered keys only, and the SKU side owns the unclassified bucket.
        # Existing columns are untouched (additive column at the frame's end —
        # CANONICAL_RECORDS_COLUMNS updated deliberately, never silently).
        canonical_universe_evidence: dict[str, set[str]] = {}
        from core.attribute_conflicts import parse_universe_cell
        for attr in self.attributes:
            parsed = parse_universe_cell(attr)
            parsed.pop("unclassified_keys", None)
            parsed.pop("volume", None)
            for key, values in parsed.items():
                if values:
                    canonical_universe_evidence.setdefault(key, set()).update(
                        str(token) for token in values
                    )
        self.universe_evidence_json = json.dumps(
            {key: sorted(values) for key, values in sorted(canonical_universe_evidence.items())}
        )

    # ── phase 6: the validated record ──────────────────────────────────────

    def record(self) -> dict:
        """One validated CanonicalRecord + the additive persistence keys."""
        # BOUNDARY CONTRACT (lib.schemas): one validated record per canonical.
        # brand NaN-guard: a group whose brand column is all-NaN would carry a
        # float NaN into mode_brand (pandas would write ""), which pydantic's
        # str field would coerce to "nan" — the exact title-poisoning bug class
        # the lane fixed for titles. Clean it here so the RECORD is honest.
        brand_clean = self._brand if isinstance(self._brand, str) else ""
        rec = CanonicalRecord(
            gtin=self._gtin,
            canonical=self.canonical,
            mode_brand=brand_clean,
            mode_flavor=self.mode_flavor,
            mode_type=self.mode_type,
            salient_ngrams=self.salient_ngrams,
            dropped_redundant_ngrams=[
                ng for ng in self.raw_salient_ngrams if ng not in set(self.kept_ngrams)
            ],
            # NOTE: kept as SETS here — gate logic intersects them (pack_set &
            # pack_set). data_prep sorts them AT THE CSV WRITE so the display
            # is deterministic (PYTHONHASHSEED-proof) without touching logic.
            volume_set=self.volume_set,
            pack_set=self.pack_set,
            packaging_level_set=self.packaging_level_set,
            package_type_set=self.package_type_set,
            package_material_set=self.package_material_set,
            flavor_set=self.flavor_set,
            made_from_set=self.made_from_set,
            carbonation_set=self.carbonation_set,
            sweetener_set=self.sweetener_set,
            sweetener_type_set=self.sweetener_type_set,
            sweetening_set=self.sweetening_set,
            attribute_consistency_flags=self.attribute_consistency_flags,
            pulp_set=self.pulp_set,
            organic_set=self.organic_set,
            volume_confidence=round(self.vol_conf, 3),
            pack_confidence=round(self.pack_conf, 3),
            volume_consistency=round(self.volume_consistency, 3),
            pack_consistency=round(self.pack_consistency, 3),
            n_titles=self.n_titles,
        )
        # Additive persistence key (post-dump, like description_evidence before
        # it was a model field): the canonical record model stays extra='forbid'
        # for its gate-facing fields; the universe evidence rides the CSV
        # contract next to them as the one rendered JSON string.
        out = rec.model_dump()
        out["universe_evidence"] = self.universe_evidence_json
        # GTIN CARD (evidence ledger, one per listing): every claim any of the
        # gtin's listing cards recorded, with listing origin kept so the surface
        # can walk a card listing by listing. Ordered PER ATTRIBUTE — sorted by
        # (field, source, value) via whole-entry JSON — deterministic across
        # hash seeds; exact repeats (two listings extracting the identical
        # claim) collapse via the same serialization.
        json_entries = [
            json.dumps({"listing": listing_index, **entry}, sort_keys=True)
            for listing_index, per_listing in enumerate(self.extracted)
            for entry in (per_listing.get("evidence_ledger") or [])
        ]
        out["evidence_ledger"] = json.dumps(
            [json.loads(e) for e in sorted(dict.fromkeys(json_entries))]
        )
        return out

    # ── orchestration ──────────────────────────────────────────────────────

    def compose(self) -> dict:
        self.accept_rows()
        self.aggregate_sets()
        self.confidences_and_consistency()
        self.token_once_text()
        self.universe_evidence()
        return self.record()


# ═══════════════════════════════════════════════════════════════════════════
# OFFICIAL training-text transformations (owner spec 2026-09-06):
# the model trains on the CLEANED sku text and the CANONICAL gtin text —
# (clean_sku_text, canonical_gtin) positive, (clean_sku_text,
# canonical_other_gtin) negative.
# ═══════════════════════════════════════════════════════════════════════════

# volume/pack numbers + unit words stripped — the exact regex used inside the
# IDF build above (single SSOT copy, referenced by both call sites)
_VOLUME_PACK_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(ml|l|lt|ltr|liter|litre|cl|centiliter|oz|fl oz|qt|gal|ounce|fluid ounce|pack|case|pcs?|pieces?|units?|x)\b",
    re.IGNORECASE,
)


def clean_sku_text(
    title: str,
    attribute: str = "",
    brand: str = "",
    description: str = "",
    category: str = "",
    breadcrumbs: str = "",
) -> str:
    """The OFFICIAL cleaned sku text the model sees.

    normalize(title) + ' ' + normalize(attribute), volume/pack tokens
    stripped, MINIMAL_STOPWORDS + single-char tokens removed, then the
    number-token reference strip (bare volumes/counts/multipliers/codes
    removed; name-embedded digits b12/o2/alkaline88 and this row's numeric
    brand tokens survive — data/number_tokens_reference.csv, 95.2% coverage).
    """
    context = " ".join(
        part for part in (
            f"brand {brand}" if brand else "",
            f"description {description}" if description else "",
            f"category {category}" if category else "",
            f"breadcrumbs {breadcrumbs}" if breadcrumbs else "",
        ) if part
    )
    text = normalize_text(title) + " " + normalize_text(attribute or "") + " " + normalize_text(context)
    text = _VOLUME_PACK_RE.sub(" ", text)
    toks = [t for t in text.split() if t not in MINIMAL_STOPWORDS and len(t) > 1]
    return strip_number_tokens(" ".join(toks), spell_numeric_brand(brand or ""))


def load_canonical_map() -> dict[str, str]:
    """gtin -> canonical string, from the pipeline's canonical_records.csv
    (file name via config/paths.yaml SSOT).

    Reads the file with the SAME string semantics the matching lane's own
    read uses (dtype=str, keep_default_na=False — rand_matching's
    record_map): an empty gtin cell becomes the plain key "" in BOTH maps,
    never the literal "nan" / float-NaN key pandas otherwise fabricates.
    Blank/whitespace gtin keys are rejected loudly — a "" canonical key
    would silently capture every gtin-less row in the payload space.
    """
    df = pd.read_csv(
        RESULTS / F["canonical_records"],
        dtype=str,
        keep_default_na=False,
    )
    df = upgrade_canonical_records_frame(df)
    check_canonical_records_frame(df)
    from core.identity_policy import exclude_reviewed_rows
    df = exclude_reviewed_rows(df, column="gtin")
    blank = df.index[df["gtin"].astype(str).str.strip().eq("")]
    if len(blank):
        raise ValueError(
            f"canonical_records.csv has {len(blank)} blank/whitespace gtin "
            f"rows (row indices "
            f"{list(blank[:5])}{' ...' if len(blank) > 5 else ''}) — refusing "
            f"to key the canonical map by an empty gtin"
        )
    return dict(zip(df["gtin"], df["canonical"], strict=True))


# tokens that are SEMANTIC despite carrying digits (same whitelist the
# number-reference build measured: b12/o2/alkaline88 brands etc.) survive;
# everything else with a digit is stripped from the MODEL-side canonical.
_CANON_KEEP_DIGIT = re.compile(
    r"^(?:b\d+|o2|co2|h2o?|ph\d*(?:\.\d+)?|\d+(?:\.\d+)?ph\.?)$", re.IGNORECASE
)


def canonical_evidence_text(value: object) -> str:
    """Render canonical evidence as deterministic plain model text.

    ``description_evidence`` and ``breadcrumb_evidence`` are stored as lists
    in the canonical record model, but CSV round-trips load those cells as
    strings containing Python-list reprs.  Never pass either representation
    directly to the encoder: list brackets, quotes, and escape syntax are
    serialization artifacts rather than product evidence.

    The parser accepts both in-memory sequences and the legacy CSV repr.  A
    scalar is treated as one evidence value, and sequence values are sorted
    by their normalized text so the model input is byte-stable regardless of
    source ordering.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = value
        if isinstance(parsed, (list, tuple, set, frozenset)):
            values = parsed
        else:
            values = (value,)
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        values = (value,)

    normalized = {
        normalize_text(item).strip()
        for item in values
        if item is not None and str(item).strip()
    }
    return " ".join(sorted(normalized))


def canonical_model_text(canonical: str) -> str:
    """Number-free base canonical for the MODEL payload.

    The gate's canonical (canonical_records.csv) keeps volumes/packs/metric
    mentions — that file drives hard_no decisions. Strip every digit token
    that is not a semantic nutrient/brand whitelist member here; the training
    payload then appends normalized ``volume_ml_*`` and ``pack_qty_*`` tokens
    from the structured sets.
    """
    if not re.search(r"\d", canonical):
        return canonical
    out = []
    for t in canonical.split():
        if not re.search(r"\d", t):
            out.append(t)
            continue
        if _CANON_KEEP_DIGIT.match(t):
            out.append(t)  # semantic: b12, o2, alkaline88-style whitelist
            continue
        # compound n-gram: keep it only if the alpha-only remainder is
        # still meaningful (>= 2 word parts)
        parts = [p for p in t.split("_") if p and not re.search(r"\d", p)]
        if len(parts) >= 2:
            out.append("_".join(parts))
        # else: pure numeric/short residue dies (volume_591, 24, 100...)
    return " ".join(out)


# ── attribute-schema boilerplate (stage-2 of the residual n-gram census) ────
# These label words come from the attributes blob's structure (type/content/
# material/...) and repeat on nearly every row — zero pairwise discriminative
# signal, a constant offset diluting cosine differences. SSOT for the model
# payload cleaning; the GATE canonical keeps them (its CSV is unchanged).
SCHEMA_WORDS = frozenset(
    {
        # attribute-blob column labels
        "type", "content", "material", "carbonization", "health", "claims",
        "features", "ingredients", "sourcing", "geographic", "flavours",
        "fragrances", "flavour", "flavor", "volume", "pack",
        "format", "feature",
        # doubled-label artifacts seen in the census
        "artificial",
        # NOTE: "sweetener" REMOVED (owner ruling, stage-3): it is a
        # diet-variant signal, not a schema label.
    }
)

# ── curated soft stop list (owner ruling, stage-3) ──────────────────────────
# Stage-2 left a residual constant offset (top-20 tokens ≈ 36% of payload
# mass). Curated ruling: strip packaging materials (rarely change product
# identity), marketing/greenwashing boilerplate, and product-type words
# redundant after type extraction (juice/drink/beverage/water).
# KEPT as variant-defining signals: still/carbonated/sparkling (carbonation),
# sugar/sweetener (diet variants), concentrate/powder/syrup (format),
# vitamins/mineral/calories/antioxidants (fortification). "concentrate"
# deliberately NOT here — it is a format-variant signal (owner ruling).
# Applied at the same composition point as SCHEMA_WORDS: model payload ONLY.
# The gate canonical and the number-token reference census are untouched.
MODEL_PAYLOAD_SOFT_STOP = frozenset(
    {
        # attribute schema labels
        "type", "volume", "content", "material", "carbonization",
        "flavour", "flavours", "ingredients", "health", "claims",
        "features", "packtype", "packmaterial",
        # packaging materials (rarely distinguish SKU variants)
        "plastic", "metal", "paper", "carton", "glass", "aluminum",
        # greenwashing / marketing boilerplate
        "naturally", "derived", "artificial", "immune", "sustainable",
        "support", "sourcing", "environmentally", "friendly",
        "additives", "preservatives", "natural", "organic",
        # redundant after type extraction
        "juice", "drink", "beverage", "water",
    }
)

# one union, one composition point — no asymmetry between sku and canonical
_MODEL_STOP = SCHEMA_WORDS | MODEL_PAYLOAD_SOFT_STOP


def strip_schema_words(text: str) -> str:
    """Remove schema boilerplate + curated soft stops from a MODEL-side text.

    Strips SCHEMA_WORDS | MODEL_PAYLOAD_SOFT_STOP (stage-2 census list +
    stage-3 curated ruling). Kept ON PURPOSE (discriminative, owner
    ruling): carbonation words (still/carbonated/sparkling), diet/format
    variants (sugar/sweetener/concentrate/powder/syrup), fortification
    terms (vitamins/mineral/calories/antioxidants), brand names, flavor
    words. Model payload ONLY — gate canonicals and the number-token
    reference keep every token (their CSVs are unchanged).
    """
    return " ".join(t for t in text.split() if t not in _MODEL_STOP)


# ============================================================================
# NUMBERS
# ============================================================================


# ── unit words that follow a number (weight/volume/energy/pack) ────────────
_NUM_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?(?:mg|g|kg|oz|ml|l|cl|lt|dl|cc|kcal|lb|lbs|pcs|ct|pk|gr|fz)\b\.?",
    re.IGNORECASE,
)
# ── pack multipliers: 6x330ml, 12x0., x12, 0.33lx6 ─────────────────────────
_MULTIPLIER_RE = re.compile(
    r"(?:\d+(?:\.\d+)?[a-z]?)?x\d+(?:\.\d+)?[a-z]*\.?", re.IGNORECASE
)
# ── nutrient/chemical codes that are semantic (vitamin b12, o2, ph) ───────
_NUTRIENT_RE = re.compile(
    r"^(?:b\d+|o2|co2|h2o?|ph\d*(?:\.\d+)?|\d+(?:\.\d+)?ph\.?)$", re.IGNORECASE
)
# ── product/retailer codes: 1-3 letters + 3+ digits (l9044, gp0027, u0026) ─
_CODE_RE = re.compile(r"^[a-z]{0,3}\d{3,}[a-z]{0,3}$", re.IGNORECASE)


def _classify(tok: str) -> str:
    if _NUM_UNIT_RE.fullmatch(tok):
        return "num_unit"
    if _MULTIPLIER_RE.fullmatch(tok):
        return "multiplier"
    if re.fullmatch(r"\d+", tok):
        return "pure_int"
    if re.fullmatch(r"\d+\.\d+", tok):
        return "decimal"
    if re.fullmatch(r"\d+%", tok):
        return "percent"
    if re.fullmatch(r"(?:no\.|n°)\s?\d+", tok, re.IGNORECASE):
        return "product_number"
    if re.search(r"[a-z]\d|\d[a-z]", tok, re.IGNORECASE):
        return "word_embedded"
    return "other"


def _name_embedded(tok: str) -> bool:
    """A digit genuinely inside a NAME: letters (>=3) on BOTH flanks of the
    digit block (careh2o, fruit2go, good4you, 226ers, alkaline88, bolt24),
    or a digit block glued to a >=3-letter word START (3in1, es6, v2u,
    sub9, og2). Trailing digit runs (cranberry750, hollerstrauchsaft250,
    20floz, 500mlx12) are size/count tails -> NOT name-embedded.
    """
    core = re.sub(r"[^a-z0-9]", "", tok.lower())
    m = re.search(r"\d+", core)
    if not m:
        return False
    left = core[: m.start()]
    right = core[m.end() :]
    if len(left) >= 3 and len(right) >= 3:
        return True  # digit block inside the word: careh2o, fruit2go
    if not right:
        return False  # trailing digits: cranberry750 -> size tail
    if not left:
        # leading digit + word right: a name only if the right flank is a
        # real word (>=2 letters, not unit/x glue): 3in1 -> in1 (in=word),
        # 500mlx24 -> mlx24 (ml=unit glue) strips, 5electrolytes -> electrolytes
        alpha_right = re.sub(r"[0-9]", "", right)
        if re.match(
            r"^(?:ml|cl|lt?r?|oz|fl|fz|pk|cc|gr|kcal|lb)", alpha_right, re.IGNORECASE
        ):
            return False  # unit glue right of leading digits: size tail
        # 4er/6er/20ea/1pt/12shots are count phrases; 5electrolytes is a word
        return len(alpha_right) >= 4 and not alpha_right.lower().startswith("x")
    return False


def token_verdict(tok: str, brand_vocab: set[str]) -> tuple[str, str]:
    """Return (class, verdict) for one digit-token.

    verdicts: strip | keep_brand | keep_nutrient | keep_name
    """
    cls = _classify(tok)
    core = re.sub(r"[^a-z0-9]", "", tok.lower())
    if tok.lower() in brand_vocab or core in brand_vocab:
        return cls, "keep_brand"
    if _NUTRIENT_RE.match(tok):
        return cls, "keep_nutrient"
    if cls == "word_embedded" and _name_embedded(tok) and not _CODE_RE.match(tok):
        return cls, "keep_name"
    if cls == "word_embedded" and re.search(
        r"(?:floz|fl\.oz|fz|\dml|ml\d|x\d|pks|packs?|pk\d|oz|lx\d|\dlx|adet|gallon|stick|ounces|liters?|pint)",
        tok,
        re.IGNORECASE,
    ):
        return cls, "strip"  # volume/pack composite tails: 8.5floz 500mlx24 12packs
    return cls, "strip"


class NumberTokenAuthority:
    """The number-token reference: census, verdict map, per-row strip.

    Single responsibility per method: corpus text census (census_corpus),
    the digit-token reference frame (reference_frame), the cached SSOT read
    (verdicts), and the model-side strip (strip). The verdict cache and the
    unseen-token counter stay MODULE-LEVEL on purpose: prepare_all resets
    them to force a fresh read between stages (pipeline._VERDICTS_CACHE /
    _VERDICTS_LOADED are the pinned external interface, never copied).
    """

    def census_corpus(self, df: pd.DataFrame) -> list[str]:
        """Pre-number-strip sku texts (the census input): the official cleaning
        WITHOUT the final number-token strip, so every digit token in the corpus
        appears in the reference."""
        out = []
        for t, a in zip(df["sku_name_eng"].fillna(""), df["attribute"].fillna(""), strict=True):
            text = normalize_text(t) + " " + normalize_text(a or "")
            text = _VOLUME_PACK_RE.sub(" ", text)
            toks = [x for x in text.split() if x not in MINIMAL_STOPWORDS and len(x) > 1]
            out.append(" ".join(toks))
        return out

    def reference_frame(self, texts: list[str], brand_vocab: set[str]) -> pd.DataFrame:
        """Census every digit-token in the corpus with its verdict.

        One row per distinct token: token, class, n_occurrences, verdict, rule.
        """
        tokens = Counter()
        for tx in texts:
            for t in tx.split():
                if re.search(r"\d", t):
                    tokens[t] += 1
        rows = []
        for tok, n in sorted(tokens.items(), key=lambda x: (-x[1], x[0])):
            cls, v = token_verdict(tok, brand_vocab)
            rows.append(
                {
                    "token": tok,
                    "class": cls,
                    "n_occurrences": n,
                    "verdict": v,
                    "rule": v,
                }
            )
        return pd.DataFrame(rows)

    def verdicts(self) -> dict[str, str] | None:
        """token -> verdict map from the reference CSV (None if not built yet).

        Cached at module level: strip_number_tokens calls this once PER TOKEN
        (up to 41k digit-texts × 2.7ms CSV re-read ≈ 112s of pure re-read per
        full-corpus pass). One read, memoized for the process lifetime — the
        CSV is written by src/training/build_reference.py, not mutated mid-run.
        """
        global _VERDICTS_CACHE, _VERDICTS_LOADED
        if _VERDICTS_LOADED:
            return _VERDICTS_CACHE
        p = reference_path()
        if not p.exists():
            # SSOT missing: SAY IT — the caller falls back to regex rules only
            # (95.2% coverage comes from the CSV; regex-only is a degradation)
            print(
                f"[numbers] reference CSV missing ({p}) — regex-rule fallback only",
                flush=True,
            )
            _VERDICTS_LOADED = True
            return None
        df = pd.read_csv(p, dtype={"token": str})
        # BOUNDARY CONTRACT (lib.schemas): every verdict must be in the
        # strip/keep_* vocabulary — a typo'd CSV value would silently never
        # match the startswith("keep") branch in strip_number_tokens.
        _VERDICTS_CACHE = check_verdict_map(dict(zip(df["token"], df["verdict"], strict=True)))
        _VERDICTS_LOADED = True
        return _VERDICTS_CACHE

    def strip(self, text: str, brand: str = "") -> str:
        """Remove number tokens from an already-clean sku text.

        `brand` is the ROW's brand string: numeric brand tokens ("28" in
        "28 Black") survive only when that number appears in this row's own
        brand; the same number in another row's title (e.g. a 24-pack of a
        different brand) is stripped. A digit token that IS this row's spelled
        numeric brand (1724 → seventeen) is replaced by the spelled form.
        Everything else resolves via the reference CSV (SSOT) with the
        regex-rule fallback.
        """
        global _UNSEEN_TOKEN_TOTAL
        if not re.search(r"\d", text):
            return text
        verdicts = self.verdicts()
        # brand arrives PRE-SPELLED from clean_sku_text; recover the digit key
        # (if this row's brand was numeric) so the keep-brand test still matches
        brand_l = (brand or "").lower()
        digit_key = next((k for k, w in NUMERIC_BRAND_WORDS.items() if w == brand_l), "")
        spelled = NUMERIC_BRAND_WORDS.get(digit_key, "")
        out = []
        # AUDIT 2026-09-09 (visibility): tokens missing from the reference CSV
        # fall through to regex rules with an EMPTY brand vocab — correct by
        # design (the CSV is the SSOT for seen tokens), but the count of
        # unseen-token resolutions was invisible. Count them per call.
        n_unseen = 0
        for t in text.split():
            if not re.search(r"\d", t):
                out.append(t)
                continue
            v = (verdicts or {}).get(t)
            if v is None:
                n_unseen += 1
                _, v = token_verdict(t, set())
            if v == "keep_brand":
                # numeric brand token: keep only if THIS row's brand carries it.
                # this row's numeric brand (1724/1642) emits its WORD form; other
                # numeric brands (28 in "28 Black") keep their digit form
                core = re.sub(r"[^a-z0-9]", "", t.lower())
                if spelled and core == digit_key:
                    out.extend(spelled.split())
                elif core and core in brand_l:
                    out.append(t)
            elif v.startswith("keep"):
                out.append(t)
        if n_unseen:
            _UNSEEN_TOKEN_TOTAL += n_unseen
        return " ".join(out)


_NUMBERS = NumberTokenAuthority()


def census_texts(df: pd.DataFrame) -> list[str]:
    """Pre-number-strip sku texts (the census input) — see _NUMBERS.census_corpus."""
    return _NUMBERS.census_corpus(df)


def build_reference(texts: list[str], brand_vocab: set[str]) -> pd.DataFrame:
    """Census every digit-token in the corpus with its verdict —
    see _NUMBERS.reference_frame."""
    return _NUMBERS.reference_frame(texts, brand_vocab)


def reference_path() -> Path:
    """DATA_DIR / F["number_reference"] — the token-verdict CSV (SSOT)."""
    return DATA_DIR / F["number_reference"]


_VERDICTS_CACHE: dict[str, str] | None = None
_VERDICTS_LOADED = False
# AUDIT 2026-09-09: process-lifetime count of digit tokens that fell through
# to the regex fallback because the reference CSV did not carry them —
# printed at data-prep exit so the degradation is visible, not silent.
_UNSEEN_TOKEN_TOTAL = 0


# Pure-numeric BRAND values are year-styled names ("1642", "1724") — the
# products spell them out in their own titles ("SEVENTEEN ginger beer").
# Owner rule: numeric brand names get SPELLED OUT everywhere (brand string
# AND its digit token inside the sku text), never dropped — the digit form
# collides with size/quantity tokens in embedding space.
NUMERIC_BRAND_WORDS: dict[str, str] = {
    "1642": "sixteen forty two",
    "1724": "seventeen",
}


def spell_numeric_brand(value: str) -> str:
    """Replace a pure-numeric brand value with its spelled-out form."""
    key = re.sub(r"\s+", " ", (value or "").strip().lower())
    return NUMERIC_BRAND_WORDS.get(key, value or "")


def strip_number_tokens(text: str, brand: str = "") -> str:
    """Remove number tokens from an already-clean sku text — see _NUMBERS.strip."""
    return _NUMBERS.strip(text, brand)


# ============================================================================
# PIPELINE
# ============================================================================
from collections import defaultdict


# PER-TITLE ORIGINAL EVIDENCE (owner ruling 2026-10-01).
#
# WHY THIS EXISTS. The decision engine's stage-7 clarification re-reads the
# ORIGINAL columns (title / attributes / description) whenever a dimension
# comes back INCONCLUSIVE. It was wired but INERT: three_way_gate handed it a
# canonical record (canonical, flavor_set, volume_set, mode_brand), and
# _fallback_reparse looks for `attributes`/`title`/`description` — none of
# which exist on a canonical. It therefore returned None for all 37 keys and
# the clarification pass never fired. The fix is to CARRY the original
# columns instead of re-deriving them downstream.
#
# PER-TITLE, NOT PER-CANONICAL: measured 26,211 titles behind 13,216
# canonicals (4.77 rows each). Flat columns would force a pick-one and
# reintroduce exactly the loss that made stage 7 unreachable.
#
# WHICH COLUMNS is declared in config/paths.yaml `source_row_fields`, each with
# a required non-empty reason, and read through core.columns — deliberately
# NOT a list here. A list in code is steered by nothing and drifts from the
# mapping that already names every column on both sides; the exclusions
# (url/image_url/price) are auditable in the config instead of re-argued by
# taste. Module-level so the writer can be tested directly against the frame
# validator that has to accept its output.
def _source_rows_for(frame: pd.DataFrame) -> str:
    """Serialize one title's original columns as deterministic JSON.

    Empty and null cells are dropped rather than written as "", because "" is
    not evidence of absence — it is absence of evidence, and stage 7 must not
    read a blank as a negative claim.
    """
    fields = source_row_pairs()
    entries = []
    for record in frame.to_dict("records"):
        entry = {}
        for target, source in fields:
            value = record.get(source)
            if value is None or pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                entry[target] = text
        if entry:
            entries.append(entry)
    # Sorted keys for byte-determinism; the per-title ORDER follows the
    # frame's own row order so extract_all stays fed in the same order.
    return json.dumps(entries, sort_keys=True)


class GtinGroupAssembler:
    """Box the raw export into one payload row per GS1-valid gtin.

    Single responsibility: turn the post-guard frame into
    (grouped frame, rows-by-gtin zip source). The grouped frame feeds the
    canonical card pool; the rows-by-gtin dict feeds the IDF indexes. The
    output columns and ordering are byte-identical to the pre-refactor
    vectorized re-implementation (gtin-sorted keys via groupby's sorted
    .indices, first-occurrence tie-break for the dominant brand).
    """

    _COLUMNS = ("sku_name_eng", "attribute", "description_short_eng", "sku_url",
                "image_url", "breadcrumbs_eng", "category", "country", "retailer",
                "brand")

    def assemble(self, df_full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        columns = self._COLUMNS
        group_indices = df_full.groupby("gtin").indices
        member_frames = {gtin: df_full.iloc[indices] for gtin, indices
                         in _LOG.progress(sorted(group_indices.items()),
                                          desc='gtin_groups', unit='gtin')}
        frame_rows = []
        rows_by_gtin: dict[str, list[tuple[str, str]]] = {}
        for gtin, member in _LOG.progress(member_frames.items(), desc='assemble_groups',
                                          unit='gtin'):
            columns_of = {column: list(member[column]) for column in columns}
            group_rows = list(zip(columns_of["sku_name_eng"],
                                  columns_of["attribute"], strict=True))
            rows_by_gtin[gtin] = group_rows
            frame_rows.append({
                "gtin": gtin,
                "rows": group_rows,
                "descriptions": columns_of["description_short_eng"],
                "urls": columns_of["sku_url"],
                "image_urls": columns_of["image_url"],
                "breadcrumbs_engs": columns_of["breadcrumbs_eng"],
                "categories": columns_of["category"],
                "countries": columns_of["country"],
                "retailers": columns_of["retailer"],
                "brand": _dominant_brand(columns_of["brand"]),
                "description_evidence": _source_evidence_values(columns_of["description_short_eng"]),
                "breadcrumb_evidence": _source_evidence_values(columns_of["breadcrumbs_eng"]),
                "source_rows": _source_rows_for(member),
            })
        grouped = pd.DataFrame(frame_rows)
        return grouped, rows_by_gtin


_GROUP_ASSEMBLER = GtinGroupAssembler()


def _assemble_gtin_groups(df_full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Vectorized re-implementation of the groupby.agg boxing block.

    Same output columns and ordering (gtin-sorted keys via groupby's
    sorted .indices), no per-group function dispatch from pandas internals.
    Returns (grouped frame, rows-by-gtin zip source).
    """
    return _GROUP_ASSEMBLER.assemble(df_full)


def _source_evidence_values(values: list) -> list[str]:
    """The in-scope _source_evidence for a pre-assembled member list."""
    return sorted({str(value).strip() for value in values
                   if pd.notna(value) and str(value).strip()})


def _dominant_brand(values: list) -> str:
    """The group's most-common brand (first-occurrence tie-break, as before)."""
    return Counter(values).most_common(1)[0][0]


class CanonicalCardPool:
    """Build every per-GTIN canonical record: inline or fork-parallel.

    One owner of the worker-pool lifecycle. The per-run IDF state is published
    to the pool (fork COW share); each task computes ONE canonical record from
    its pre-assembled group payload only, so the split is deterministic either
    way — pool.map yields results in submission order, and every task is a
    pure function of payload + inherited IDF maps.
    """

    _INLINE_TASK_THRESHOLD = 256
    _CHUNKSIZE = 16

    def __init__(self) -> None:
        self._state: dict[str, object] = {}

    def bind(self, global_idf: 'NgramIDF',
             brand_idf_map: dict[str, 'NgramIDF']) -> None:
        """Publish the per-run IDF state to the worker pool (fork COW share)."""
        self._state.clear()
        self._state['global_idf'] = global_idf
        self._state['brand_idf_map'] = brand_idf_map

    def record_for_task(self, task: tuple) -> dict:
        """One canonical record, computed from pre-assembled group payload only."""
        gtin, brand, rows, descriptions, urls, image_urls, breadcrumbs_engs, \
            categories, description_evidence, breadcrumb_evidence, source_rows = task
        record = generate_canonical(
            gtin, brand, rows,
            self._state['global_idf'],
            self._state['brand_idf_map'][brand.lower().strip()],
            descriptions=descriptions, urls=urls, image_urls=image_urls,
            breadcrumbs_engs=breadcrumbs_engs, categories=categories,
        )
        record['description_evidence'] = description_evidence
        record['breadcrumb_evidence'] = breadcrumb_evidence
        record['source_rows'] = source_rows
        return record

    def build(self, grouped: pd.DataFrame, global_idf: 'NgramIDF',
              brand_idf_map: dict[str, 'NgramIDF']) -> pd.DataFrame:
        self.bind(global_idf, brand_idf_map)
        tasks = list(
            (row.gtin, row.brand, row.rows, row.descriptions, row.urls,
             row.image_urls, row.breadcrumbs_engs, row.categories,
             row.description_evidence, row.breadcrumb_evidence, row.source_rows)
            for row in grouped.itertuples(index=False)
        )
        if len(tasks) < self._INLINE_TASK_THRESHOLD:
            records = [
                self.record_for_task(task)
                for task in _LOG.progress(tasks, desc='cards_inline', unit='gtin')
            ]
            return pd.DataFrame(records)
        _LOG.info(f"canon: fork-parallel canonical build over {len(tasks):,} gtins")
        import concurrent.futures
        from multiprocessing import get_context
        records: list[dict] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(2, os.cpu_count() - 1), mp_context=get_context('fork'),
        ) as pool:
            bar = _LOG.bar(total=len(tasks), desc='canonical_cards', unit='gtin')
            try:
                for record in pool.map(self.record_for_task, tasks, chunksize=self._CHUNKSIZE):
                    records.append(record)
                    bar.update()
            finally:
                bar.close()
        return pd.DataFrame(records)


_CARD_POOL = CanonicalCardPool()


def _canonical_records_df(grouped: pd.DataFrame, global_idf: 'NgramIDF',
                          brand_idf_map: dict[str, 'NgramIDF']) -> pd.DataFrame:
    """Build every canonical record: parallel across cores when worthwhile.

    Deterministic either way: imap yields results in submission order, and
    every task is a pure function of its payload + fork-inherited IDF maps.
    """
    return _CARD_POOL.build(grouped, global_idf, brand_idf_map)


class _PipelineSteering:
    """Own one stage-1 data-prep steering pass: raw export in, two CSVs out.

    Every phase below is a single responsibility and runs in ONE fixed order
    inside steer(); the statements are the pre-refactor body verbatim, so the
    printed guard lines, the trace rows, the timing marks (byte-identical
    labels: guards_grouping_idf / canonical_cards / pre_gate_census /
    gate_loop / csv_write) and the CSV bytes all survive untouched.

    Phase map:
      column_backfill      — evidence columns guaranteed on the raw export
      open_stage           — consolidated trace + Timing("data_prep.pipeline")
      gtin_guard           — identity links, NaN + GS1-checksum quarantine
      grouping_and_idf     — gtin boxing, global + per-brand NgramIDF
      canonical_stage      — the fork-parallel card pool + identity-close &
                             universe-evidence census trace rows
      pre_gate_census      — attribute-gate doctrine evidence rows
      brand_blocking_gate  — same-brand pair space + the scalar gate loop
      sort_validate_write  — PYTHONHASHSEED-safe display sort, frame
                             contracts, atomic CSV writes
      gate_trace_rows      — decision/reason census + bounded pair samples
    """

    def __init__(self, df_full: pd.DataFrame, trace=None) -> None:
        self.df_full = df_full
        # An EXTERNALLY owned trace writer (training.data_prep owns the stage so
        # the manifest's row accounting lands in the SAME run as the pipeline's
        # step rows — one writer per stage, core.tracing's contract). None keeps
        # the historical behaviour: this class creates and commits its own.
        self._external_trace = trace
        self.trace = None
        self.timing = None
        self.grouped: pd.DataFrame | None = None
        self.global_idf: NgramIDF | None = None
        self.brand_to_gtins: dict[str, list[str]] = {}
        self.brand_idf_map: dict[str, NgramIDF] = {}
        self.df_canon: pd.DataFrame | None = None
        self.candidate_pairs: set[tuple[str, str]] = set()
        self.gtin_to_canon: dict[str, dict] = {}
        self.results: list[dict] = []
        self.gate_vis: list[dict] = []
        self.results_df: pd.DataFrame | None = None

    # ── phase: columns + trace + timing ────────────────────────────────────

    def column_backfill(self) -> None:
        """Preserve the evidence-bearing source fields as canonical-level
        inputs. They remain OUTSIDE the frozen canonical text until a
        component-safe ablation establishes their value."""
        for column in (
            "description_short_eng", "breadcrumbs_eng",
            "sku_url", "image_url", "category", "country", "retailer",
        ):
            if column not in self.df_full:
                self.df_full = self.df_full.assign(**{column: ""})

    def open_stage(self) -> None:
        """The consolidated-trace writer + the pipeline Timing instance."""
        # ── CONSOLIDATED TRACE: stage 1 writer ────────────────────────────
        # ONE writer for the whole stage, committed once at the end of the
        # function. The first row records the COLUMN CONTRACT this frame arrived
        # with, because the frame is loaded separately from the one stage 2 builds
        # and that handoff used to be invisible until a KeyError fired somewhere
        # downstream.
        from core.tracing import TraceRun

        self.trace = (
            self._external_trace
            if self._external_trace is not None
            else TraceRun("data_prep")
        )
        self.trace.add_column_contract(
            self.df_full,
            contract="raw_export (core.common.load_raw_export)",
            required=RAW_EXPORT_REQUIRED_COLUMNS,
            note=(
                "stage 2 (build_training_data) does NOT consume this frame: it "
                "reloads the deduped dataset through load_dataset_deduped(). Both "
                "stages use the raw export's column names (c698200), so the two "
                "contracts are identical by construction — they are separate "
                "datasets joined at canonical_records.csv + gate_results.csv, not "
                "separate column vocabularies"
            ),
        )

        from core.timing import Timing

        self.timing = Timing("data_prep.pipeline")

    # ── phase: gtin guard ──────────────────────────────────────────────────

    def gtin_guard(self) -> None:
        """NaN/invalid identity quarantine + the guard trace row."""
        # NaN/empty GTINs must NOT form a group: 41,545 rows (58% of the corpus)
        # share gtin=NaN and used to collapse into ONE canonical record with an
        # arbitrary mode-brand — poisoning canonical_records.csv AND the global
        # IDF every other GTIN was scored against. Drop them explicitly.
        from core.identity_policy import apply_identity_links
        self.df_full = apply_identity_links(self.df_full)
        n_before = len(self.df_full)
        gtin_valid = (
            self.df_full["gtin"].notna()
            & (self.df_full["gtin"].astype(str).str.strip() != "")
            & (self.df_full["gtin"].astype(str).str.lower() != "nan")
        )
        # Checksum enforcement (owner ruling): 1,747 of 14,997 distinct gtins
        # (3,715 rows) FAIL the GS1 check digit — retailer-export noise. An
        # invalid gtin must not assert product identity: no canonical forms
        # on it, so no (sku, canonical) positive pairs and no false labels leak
        # into training/eval. The ROWS survive (corpus unchanged); only the
        # identity claim dies. Loud per lane doctrine — never silent.
        from core.gtin import gtin_validity

        bc_valid = gtin_validity(self.df_full["gtin"].fillna("").astype(str).str.strip())
        from core.identity_policy import reviewed_row_mask
        reviewed = reviewed_row_mask(self.df_full)
        bc_valid &= ~reviewed
        checksum_bad = gtin_valid & ~bc_valid & ~reviewed
        n_checksum_dropped = int(checksum_bad.sum())
        self.df_full = self.df_full[gtin_valid & bc_valid]
        if reviewed.any():
            print(f"[gtin-guard] excluded {int(reviewed.sum()):,} identity-review rows (GLN or unresolved formulation)", flush=True)
        if n_checksum_dropped:
            print(
                f"[gtin-guard] dropped {n_checksum_dropped:,} rows whose gtin "
                f"FAILS the GS1 check digit (no canonical/labels form on a "
                f"gtin that cannot be trusted as identity)",
                flush=True,
            )
        if n_before != len(self.df_full):
            print(
                f"[gtin-guard] total dropped {n_before - len(self.df_full):,} rows "
                f"(missing/NaN gtin or failed checksum) — they cannot be "
                f"grouped by product",
                flush=True,
            )
        # CONSOLIDATED TRACE (§gtin-guard): the guard is where identity dies, so
        # both populations are recorded with the reason that removed them.
        self.trace.add(
            "gtin_guard",
            "identity_claims_evaluated",
            in_count=n_before,
            out_count=len(self.df_full),
            reason="rows keep identity only with a present, GS1-valid gtin",
            detail={
                "gtin_missing_or_nan": int((~gtin_valid).sum()),
                "gs1_checksum_failed": n_checksum_dropped,
                "identity_review_quarantined": int(reviewed.sum()),
                "rows_retained": int(len(self.df_full)),
            },
            source="raw export",
        )

    # ── phase: grouping + IDF ──────────────────────────────────────────────

    def grouping_and_idf(self) -> None:
        """Vectorized gtin boxing + the global and per-brand IDF indexes."""
        # Group by GTIN (vectorized index assembly; per-group python lambdas
        # through .agg are ~3x slower than one pass of dict-of-lists, and every
        # column below is exactly a per-order-group collection).
        self.grouped, _rows_by_gtin_source = _assemble_gtin_groups(self.df_full)
        rows_by_gtin = dict(zip(self.grouped["gtin"], self.grouped["rows"], strict=True))
        self.global_idf = NgramIDF(rows_by_gtin)

        # Precompute within-brand IDF per brand
        self.brand_to_gtins = defaultdict(list)
        for gtin, brand in zip(self.grouped["gtin"], self.grouped["brand"], strict=True):
            self.brand_to_gtins[brand.lower().strip()].append(gtin)

        # For each brand, build an IDF from that brand's GTINs
        self.brand_idf_map = {}
        for brand, gtins in self.brand_to_gtins.items():
            brand_rows = {gtin: rows_by_gtin[gtin] for gtin in gtins}
            self.brand_idf_map[brand] = NgramIDF(brand_rows)

        self.timing.mark("guards_grouping_idf")

    # ── phase: canonical cards ─────────────────────────────────────────────

    def canonical_stage(self) -> None:
        """The card pool + the row-identity close and universe census rows."""
        # Generate canonical records
        self.df_canon = _canonical_records_df(
            self.grouped, self.global_idf, self.brand_idf_map
        )
        self.timing.mark("canonical_cards")

        # ── CONSOLIDATED TRACE: the row identity closes here ──────────────
        # Every GS1-valid row is either promoted to its gtin's canonical record or
        # collapsed into it (kept and aggregated — a distinct destiny from the
        # guard's two drop populations). With this row the trace alone closes
        #   rows_in == canonical_records + collapsed_same_gtin
        #              + gtin_missing_or_nan + gs1_checksum_failed
        # which is what core.tracing.accounting() recomputes from the file.
        self.trace.add(
            "canonical",
            "records_built",
            in_count=len(self.df_full),
            out_count=len(self.df_canon),
            reason=(
                "one canonical record per distinct GS1-valid gtin; the other rows "
                "collapse into their own gtin's record (kept and aggregated, not "
                "dropped)"
            ),
            detail={
                "distinct_gtins": int(len(self.df_canon)),
                "collapsed_same_gtin": int(len(self.df_full) - len(self.df_canon)),
                "brands": int(self.grouped["brand"].nunique()) if len(self.grouped) else 0,
                "brands_with_pairs": int(
                    sum(1 for gtins in self.brand_to_gtins.values() if len(gtins) > 1)
                ),
            },
            source="raw export",
        )

        self._universe_evidence_census()

    def _universe_evidence_census(self) -> None:
        """Canonical-side universe-evidence census row (audit readback only)."""
        # ── CONSOLIDATED TRACE: canonical-side universe-evidence census ──
        # Closure evidence for the wiring gap this column closes: how many
        # canonicals carry ANY universe evidence, per registered key. Audit
        # readback only — no decision reads this row.
        if not len(self.df_canon):
            return
        from core.attribute_conflicts import _universe_evidence_of

        _evid = [ _universe_evidence_of(row) for row in self.df_canon.to_dict("records") ]
        _per_key: Counter[str] = Counter()
        for evidence in _evid:
            for key in evidence:
                _per_key[key] += 1
        self.trace.add(
            "canonical",
            "universe_evidence_census",
            in_count=int(len(self.df_canon)),
            out_count=int(sum(1 for item in _evid if item)),
            reason="canonicals persisting non-empty universe_evidence (per-key counts in detail)",
            detail={
                "canonicals_with_evidence": int(
                    sum(1 for item in _evid if item)
                ),
                "per_key_populated": {
                    key: count for key, count in sorted(_per_key.items())
                },
            },
            source="canonical_records.csv (in-memory frame)",
        )

    # ── phase: pre-gate census rows ────────────────────────────────────────

    def pre_gate_census(self) -> None:
        """Attribute-gate doctrine evidence rows (read-only census + config)."""
        # ── CONSOLIDATED TRACE: attribute-gate evidence sections (owner ruling
        # 2026-10-01, "ALL ATTRIBUTES are used to make ALL DECISIONS") ──────
        # The registry census (layouts.attribute_universe_census) measured
        # every raw key; the decision layer (core.attribute_conflicts
        # full_attribute_evaluation) now evaluates ALL of them per pair with each
        # field's own measured conflict semantics, while the VETO still votes only
        # where config permits (vetoes stay absence-blind and config-owned). These
        # two run-scope rows sit next to the canonical row so the decision
        # doctrine and its evidence live in one file. The detail builders live at
        # module scope (attribute_gate_* below) so the contracts are unit-testable
        # without a full data-prep run. The ledger reads ONLY the census artifact
        # + the current config (never writes either).
        from core.attribute_conflicts import veto_eligibility_ledger

        self.trace.add(
            "attribute_gate",
            "universe_decision_scope",
            reason="every AttributeUniverse-registered dimension enters pair-level evaluation; absence never vetoes",
            detail=attribute_gate_universe_scope_detail(),
            source="core.attribute_universe census artifact",
        )
        self.trace.add(
            "attribute_gate",
            "veto_eligibility",
            reason="per-dimension veto-eligibility ledger: evidence class + CURRENT config state + the exact owner delta",
            detail={"ledger": veto_eligibility_ledger()},
            source="core.attribute_universe census + config/training.yaml (both read-only)",
        )

        self.timing.mark("pre_gate_census")

    # ── phase: blocking + gate loop ────────────────────────────────────────

    def brand_blocking_gate(self) -> None:
        """Same-brand candidate space + the deliberate scalar gate loop."""
        # Brand blocking
        candidate_pairs = set()
        for brand, gtins in self.brand_to_gtins.items():
            if len(gtins) < 2:
                continue
            for i in range(len(gtins)):
                for j in range(i + 1, len(gtins)):
                    candidate_pairs.add((gtins[i], gtins[j]))

        # Gate and similarity
        self.gtin_to_canon = dict(zip(self.df_canon["gtin"], self.df_canon.to_dict(orient="records"), strict=True))
        self.results = []
        self.candidate_pairs = candidate_pairs
        # GATE VISIBILITY (owner directive 2026-09-07): every gate call logs
        # exactly what it SAW (both sides' volume/pack/flavor + confidences)
        # next to what it DECIDED — auditable inputs→outputs, rewritten every
        # run. Since 2026-09-15 these rows land in the ONE consolidated trace
        # (core.tracing) instead of a per-stage results/logs CSV, so a decision
        # and its readback are never in two places. Full census, not a sample:
        # the whole point is no invisibility.
        from core.attribute_conflicts import (
            canonical_attribute_info,
            full_attribute_evaluation,
        )

        self.gate_vis = []
        # VECTORIZATION RULING (audit close, 2026-09-10): this per-pair Python
        # loop is deliberately kept scalar. "Optimize and vectorize wherever
        # possible" reaches HOT paths; this is not one — it runs ONCE per
        # data-prep regeneration (src/training/data_prep.py is the sole caller) and
        # no training/eval step executes it (they consume the CSVs it writes).
        # three_way_gate is the label source — every training label flows
        # through its decision table — so an equivalent-but-restructured
        # rewrite puts all pinned counts (135,769 / 92,650 / 29,351 / 13,768)
        # at risk for seconds saved on a one-time run. Two vectorization
        # attempts were abandoned for exactly this risk/benefit. If this ever
        # becomes a hot path, vectorize with the equivalence protocol:
        # pinned counts + diagonal crosstab vs the previous CSV + 0-tolerance
        # confidence match, revert on ANY divergence.
        for g1, g2 in _LOG.progress(sorted(candidate_pairs), desc="gate", unit="pair"):
            self._gate_one_pair(g1, g2, canonical_attribute_info, full_attribute_evaluation)
        self.timing.mark("gate_loop")

    def _gate_one_pair(self, g1: str, g2: str, canonical_attribute_info, full_attribute_evaluation) -> None:
        """One candidate pair: gate verdict + Jaccard + the visibility row."""
        from core.pair_policy import identity_similarity

        a1 = self.gtin_to_canon[g1]
        a2 = self.gtin_to_canon[g2]
        gate = three_way_gate(a1, a2)
        sim = identity_similarity(a1["canonical"], a2["canonical"])
        evaluation = full_attribute_evaluation(
            canonical_attribute_info(a1),
            canonical_attribute_info(a2),
            # BOTH cuts, so the census records the same volume verdict the
            # gate just decided on. Relative-only here minted `conflict` on
            # small-volume pairs the gate had accepted (the relative cut is
            # the stricter of the two below ~100ml).
            volume_relative_tolerance=float(
                training_cfg().gate.vol_tolerance
            ),
            volume_absolute_tolerance_ml=float(
                training_cfg().gate.vol_abs_tolerance
            ),
        )
        self.results.append(
            {
                "gtin1": g1,
                "gtin2": g2,
                "canon1": a1["canonical"],
                "canon2": a2["canonical"],
                "gate_decision": gate["decision"],
                "gate_reason": gate["reason"],
                "similarity": sim,
            }
        )
        self._visibility_row(g1, g2, a1, a2, gate, sim, evaluation)

    def _visibility_row(self, g1: str, g2: str, a1: dict, a2: dict, gate: dict, sim: float, evaluation: dict) -> None:
        """The gate-visibility row the consolidated trace samples from."""
        self.gate_vis.append(
            {
                "gtin1": g1,
                "gtin2": g2,
                "vol_set1": sorted(a1["volume_set"]),
                "vol_set2": sorted(a2["volume_set"]),
                "vol_conf1": a1["volume_confidence"],
                "vol_conf2": a2["volume_confidence"],
                "vol_consist1": a1["volume_consistency"],
                "vol_consist2": a2["volume_consistency"],
                "pack_set1": sorted(a1["pack_set"]),
                "pack_set2": sorted(a2["pack_set"]),
                "pack_conf1": a1["pack_confidence"],
                "pack_conf2": a2["pack_confidence"],
                "package_types1": sorted(a1["package_type_set"]),
                "package_types2": sorted(a2["package_type_set"]),
                "package_materials1": sorted(a1["package_material_set"]),
                "package_materials2": sorted(a2["package_material_set"]),
                "flavor1": a1.get("mode_flavor", ""),
                "flavor2": a2.get("mode_flavor", ""),
                "decision": gate["decision"],
                "reason": gate["reason"],
                "jaccard_short_tokens": sim,
                # FULL-ATTRIBUTES per-pair census (owner ruling 2026-10-01):
                # evaluated on the SAME canonical evidence the gate consumed,
                # outside the decision table — zero influence on
                # gate_decision/gate_reason (the vetoes stay config-owned).
                # Stored as a JSON string (sorted keys) so the trace
                # readback is byte-deterministic; carrier column only, never
                # written to gate_results.csv (that frame contract is fixed).
                "dimension_census": json.dumps(
                    evaluation["dimension_states"], sort_keys=True
                ),
                "dimension_conflicts": ",".join(
                    key.replace(" ", "_")
                    for key in evaluation["dimension_conflicts"]
                ),
            }
        )

    # ── phase: sort + validate + write ─────────────────────────────────────

    def sort_validate_write(self) -> None:
        """Display sort, frame contracts, atomic CSV writes."""
        self.results_df = pd.DataFrame(self.results)

        # TRAIN_GPU writes ONLY inside its own tree (lib.common RESULTS —
        # the repo's results dir must never be touched by the standalone lane).
        # DETERMINISM: set->display columns render in
        # PYTHONHASHSEED-random order otherwise; sort the DISPLAY (after all
        # gate logic consumed the real sets) so the CSV is byte-reproducible.
        for _col in (
            "volume_set",
            "pack_set",
            "package_type_set",
            "packaging_level_set",
            "package_material_set",
            "flavor_set",
            "made_from_set",
            "carbonation_set",
            "sweetener_set",
            "sweetener_type_set",
            "sweetening_set",
            "attribute_consistency_flags",
            "pulp_set",
        ):
            self.df_canon[_col] = self.df_canon[_col].map(lambda s: sorted(s))
        # Same for the pair ROW ORDER: candidate_pairs is a SET, so iteration
        # order is process-random. Gate decisions themselves are order-free —
        # only the CSV row sequence drifted. Sort on the identity columns.
        self.results_df = self.results_df.sort_values(
            ["gtin1", "gtin2"], kind="stable"
        ).reset_index(drop=True)
        # THE pair key (core.pair_identity SSOT): direction-independent, so the
        # gate row joins the labeled, validation and prediction rows that name
        # the same pair with the endpoints swapped.
        self.results_df["pair_id"] = PairIdentity.column(
            self.results_df["gtin1"], self.results_df["gtin2"]
        )
        RESULTS.mkdir(parents=True, exist_ok=True)
        # FRAME CONTRACTS (lib.schemas): column sets, decision domain, similarity
        # bounds, GTIN endpoints — asserted at the WRITE boundary so a corrupted
        # transform can never land in the CSVs every downstream step reads.
        require_populated_source_rows(check_canonical_records_frame(self.df_canon))
        check_gate_results_frame(self.results_df)
        # SILENT_DROPS task 6: every CSV write goes through the atomic
        # mechanism (tmp sibling + fsync + rename) so an interrupt can never
        # leave a truncated artifact for downstream steps to read.
        from core.manifest import atomic_write_csv

        atomic_write_csv(self.df_canon, RESULTS / F["canonical_records"], index=False)
        atomic_write_csv(self.results_df, RESULTS / F["gate_results"], index=False)
        self.timing.mark("csv_write")

    # ── phase: gate trace census + close ───────────────────────────────────

    def gate_trace_rows(self) -> None:
        """Decision/reason census rows, bounded pair samples, trace close."""
        # ── CONSOLIDATED TRACE: gate stage ─────────────────────────────────
        # One CSV carries the whole story: run-scope funnels (candidate census →
        # decision census → complete reason census), one exact group row per
        # (decision, reason) bucket, then a bounded stratified SAMPLE of pairs with
        # the literal readback. Every pair's decision and reason is counted exactly
        # in the census rows; the sample exists so the evidence can be eyeballed
        # without opening gate_results.csv. Replaces the former
        # results/logs/gate_visibility.csv.
        gate_frame = (
            pd.DataFrame(self.gate_vis).sort_values(["gtin1", "gtin2"], kind="stable")
            if self.gate_vis
            else pd.DataFrame()
        )
        vis_counts = (
            gate_frame["decision"].value_counts().to_dict() if len(gate_frame) else {}
        )
        vis_reasons = (
            gate_frame["reason"].value_counts().to_dict() if len(gate_frame) else {}
        )
        self.trace.add(
            "gate",
            "candidates_gated",
            in_count=len(self.candidate_pairs),
            out_count=len(self.results_df),
            reason="every same-brand pair receives exactly one decision; none is dropped",
            detail={"decisions": {str(k): int(v) for k, v in vis_counts.items()}},
            source="canonical_records.csv (in-memory frame)",
        )
        self._decision_bucket_rows(gate_frame)
        self._dimension_rollup_row(gate_frame)
        self._reason_census_and_samples(gate_frame, vis_reasons)
        if self._external_trace is not None:
            # The owner of this stage commits the rows (it still has the manifest
            # accounting to add), so this writer must not commit a partial stage.
            return
        self.trace.write()
        print(
            f"[trace] data_prep steps written -> {trace_path()} | "
            f"gate decisions: {vis_counts}",
            flush=True,
        )

    def _decision_bucket_rows(self, gate_frame: pd.DataFrame) -> None:
        """One group row per decision: exact population + reason distribution.

        One group row per decision: its EXACT population plus the reason
        distribution inside it, as a BOUNDED top-N census (the raw reason is a
        per-pair literal; see :func:`_bounded_census`). "Which pairs got which
        decision and why" is still answered here, at category grain, with the
        exact bucket/remainder counts beside it.
        """
        for decision in ("hard_no", "fallback", "proceed"):
            subset = gate_frame[gate_frame["decision"] == decision] if len(gate_frame) else gate_frame
            categories = subset["reason"].map(_reason_category) if len(subset) else subset
            self.trace.add(
                "gate",
                f"decision_{decision}",
                scope="group",
                in_count=len(self.candidate_pairs),
                out_count=int(len(subset)),
                reason=f"gate_decision == {decision}",
                detail=_capped_detail(
                    {
                        "reasons": _bounded_census(categories),
                        "reason_category_chars": REASON_LABEL_CHARS,
                    }
                    if len(subset)
                    else {"reasons": _bounded_census([])}
                ),
                source="gate_results.csv",
            )

    def _dimension_rollup_row(self, gate_frame: pd.DataFrame) -> None:
        """FULL-ATTRIBUTES pair census rollup over the whole gated population."""
        # FULL-ATTRIBUTES pair census rollup (owner ruling 2026-10-01): a
        # run-scope row over the whole gated population states exactly how many
        # pairs carried at least one recorded dimension conflict and which
        # dimensions are the loud ones — per-pair detail rides the sampled
        # pair_decision rows (bounded sample, see core.tracing) and this row
        # carries the exact counts. The conflict cell is a TOP-N census, not the
        # full 13k-string dump it used to be (1.07 MB in one cell).
        if not len(gate_frame):
            return
        conflicts = gate_frame["dimension_conflicts"].astype(str)
        recorded = conflicts[conflicts != ""]
        self.trace.add(
            "attribute_gate",
            "pair_dimension_census",
            scope="group",
            in_count=int(len(gate_frame)),
            out_count=int((conflicts != "").sum()),
            reason="pairs with at least one recorded dimension conflict (absence stays unknown, never minted)",
            detail=_capped_detail(
                {
                    "pairs": int(len(gate_frame)),
                    "conflict_paired": _bounded_census(recorded),
                    "no_conflict": int((conflicts == "").sum()),
                }
            ),
            source="gate_stage in-memory readback",
        )

    def _reason_census_and_samples(self, gate_frame: pd.DataFrame, vis_reasons: dict) -> None:
        """Cross-decision reason census + the bounded per-pair readback."""
        if not len(gate_frame):
            return
        # Named `reason_census`, NOT `decision_reasons`: every group step
        # starting with "gate.decision_" is a decision bucket and is summed by
        # core.tracing.accounting(), so this cross-decision row must not share
        # that prefix.
        self.trace.add(
            "gate",
            "reason_census",
            scope="group",
            in_count=int(len(gate_frame)),
            out_count=int(len(vis_reasons)),
            reason="complete reason census over every decision, not just hard_no",
            detail=_capped_detail(
                {
                    "reasons": _bounded_census(
                        gate_frame["reason"].map(_reason_category)
                    ),
                    "pairs": int(len(gate_frame)),
                    "reason_category_chars": REASON_LABEL_CHARS,
                }
            ),
            source="gate_results.csv",
        )
        # Full per-pair readback: exactly what the gate SAW on both sides
        # (volume/pack/package sets + their confidences and consistency) next
        # to what it DECIDED and the similarity downstream mining bands on.
        # The inputs are JSON so one cell stays machine-readable. Bucketed by
        # (decision :: reason CATEGORY): the raw reason embeds the values that
        # produced it, so using it as a label minted one group row per pair
        # (589 rows / 257 singletons on the live census); the category keeps the
        # census and the sampled rows joining on a bounded label, and each
        # sampled row's detail still carries the FULL reason.
        self.trace.add_entities(
            "pair_decision",
            list(gate_frame.itertuples(index=False)),
            key_of=lambda r: f"{r.gtin1}|{r.gtin2}",
            reason_of=lambda r: f"{r.decision} :: {_reason_category(r.reason)}",
            detail_of=lambda r: json.dumps(
                {
                    "decision": str(r.decision),
                    "reason": str(r.reason),
                    "similarity": round(float(r.jaccard_short_tokens), 6),
                    "volume_a": list(r.vol_set1),
                    "volume_b": list(r.vol_set2),
                    "volume_confidence_a": float(r.vol_conf1),
                    "volume_confidence_b": float(r.vol_conf2),
                    "volume_consistency_a": float(r.vol_consist1),
                    "volume_consistency_b": float(r.vol_consist2),
                    "pack_a": list(r.pack_set1),
                    "pack_b": list(r.pack_set2),
                    "pack_confidence_a": float(r.pack_conf1),
                    "pack_confidence_b": float(r.pack_conf2),
                    "package_type_a": list(r.package_types1),
                    "package_type_b": list(r.package_types2),
                    "package_material_a": list(r.package_materials1),
                    "package_material_b": list(r.package_materials2),
                    "flavor_a": str(r.flavor1),
                    "flavor_b": str(r.flavor2),
                    # FULL-ATTRIBUTES census (owner ruling 2026-10-01): the
                    # per-pair all-dimension states, one `<key>_state` entry
                    # per AttributeUniverse-registered field.
                    "dimension_census": json.loads(str(r.dimension_census)),
                    "dimension_conflicts": str(r.dimension_conflicts),
                },
                sort_keys=True,
            ),
            source="gate_results.csv",
            per_reason=ENTITY_PER_REASON,
            total_cap=ENTITY_TOTAL_CAP,
        )

    # ── orchestration ──────────────────────────────────────────────────────

    def steer(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the load-bearing phase order, return (gate results, canonicals)."""
        self.column_backfill()
        self.open_stage()
        self.gtin_guard()
        self.grouping_and_idf()
        self.canonical_stage()
        self.pre_gate_census()
        self.brand_blocking_gate()
        self.sort_validate_write()
        self.gate_trace_rows()
        self.timing.dump_if_requested()

        return self.results_df, self.df_canon


def run_within_brand_pipeline(
    df_full: pd.DataFrame,
    trace=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # (gate results, canonical records)
    """Stage 1 of data prep: canonical cards + gate decisions over the RAW export.

    ``trace`` is an optional EXTERNALLY owned ``core.tracing.TraceRun`` for this
    stage. Supplying it makes the caller the stage's single writer (training.data_prep
    does, so the manifest's row accounting and the flag census join the pipeline's
    steps in one commit). With none, this function keeps its historical
    one-writer behaviour.
    """
    return _PipelineSteering(df_full, trace).steer()


# ============================================================================
# PAIRS
# ============================================================================
class SkuTextBuilder:
    """One payload variant's sku texts + structured info (SSOT loop owner).

    SR phases: variant_frame (full = the deduped frame itself, title_only =
    descriptor columns blanked through the config column contract), then the
    documented core.model_input composition (itself the SkuPayloadComposer).
    """

    def __init__(self, df: pd.DataFrame, payload_variant: str,
                 *, structured_enabled: bool, timing=None) -> None:
        self._df = df
        self._variant = str(payload_variant)
        self._structured_enabled = structured_enabled
        self._timing = timing

    def variant_frame(self) -> pd.DataFrame:
        """The frame whose columns the text lane may consume."""
        # ── clean sku text per row (variant: full = title+attr, title_only) ──
        # schema words (type/content/material/...) die on the MODEL side only —
        # the gate's inputs are untouched (owner 2026-09-07: stage-2 strip).
        # Both variants go through core.model_input, the shared builder.
        # The per-row composition loop is the SSOT core.model_input.build_sku_texts
        # (was inlined here and in predict_items / rand_matching / record_linkage).
        if self._variant == "full":
            return self._df
        if self._variant == "title_only":
            model_frame = self._df.copy()
            # The descriptor columns blanked for the title-only variant come from
            # the config column contract (core.columns.COLUMN_ALIASES carries the
            # raw-export aliases, e.g. "attr" for "attribute") — not a hand-typed
            # tuple that drifts from paths.yaml.
            for column in ("attribute", "description_short_eng",
                           *COLUMN_ALIASES.get("attribute", ())):
                if column in model_frame.columns:
                    model_frame[column] = ""
            return model_frame
        raise SystemExit(f"unknown payload variant: {self._variant}")

    def build(self) -> tuple[list[str], list[dict[str, set]]]:
        from core.model_input import build_sku_texts

        texts, infos = build_sku_texts(
            self.variant_frame(), structured_enabled=self._structured_enabled
        )
        if self._timing is not None:
            self._timing.mark("sku_texts")
        return texts, infos


class CanonicalPayloadBuilder:
    """One run's canonical payload population (record map + structured info
    + model texts), in sorted-gtin order.

    SR phases: read_records (migrate+validate the artifact frame, build the
    gtin->record map), structured_infos, encode_texts (the documented
    build_canonical_text per canonical). The bars keep their labels.
    """

    def __init__(self, *, structured_enabled: bool) -> None:
        self._structured_enabled = structured_enabled

    def read_records(self) -> tuple[pd.DataFrame, dict[str, dict]]:
        # MODEL payload: schema-free canonical variant plus normalized structured
        # volume/pack/package-type tokens. The gate's CSV keeps the original values and schema
        # labels for decisions; the model receives the stable normalized tokens
        # explicitly so those attributes are no longer discarded.
        canonical_records = pd.read_csv(
            RESULTS / F["canonical_records"], dtype={"gtin": str}, keep_default_na=False
        )
        # Same read contract as the other canonical lanes: migrate a stale
        # artifact (core.schemas.upgrade_canonical_records_frame) and validate it
        # before building the payload — a malformed file fails here, loudly.
        canonical_records = upgrade_canonical_records_frame(canonical_records)
        check_canonical_records_frame(canonical_records)
        # One row-listing pass replaces per-row Series construction (iterrows
        # builds a Series + index per row): record_map values keep their
        # (column -> value) dict shape, unchanged for every reader.
        record_map = {
            str(record["gtin"]): record
            for record in _LOG.progress(
                canonical_records.to_dict("records"),
                total=len(canonical_records),
                unit="record",
                desc="canon-records",
            )
        }
        return canonical_records, record_map

    def structured_infos(self, canonical_gtins: list[str],
                         record_map: dict[str, dict]) -> list[dict[str, set]]:
        from core.model_input import model_input_info
        from core.structured_features import (
            canonical_info as canonical_structured_info,
        )

        empty = {"volume": set(), "pack": set(), "package_type": set()}
        return [
            model_input_info(canonical_structured_info(record_map.get(g, {})))
            if self._structured_enabled
            else dict(empty)
            for g in _LOG.progress(
                canonical_gtins,
                total=len(canonical_gtins),
                unit="canon",
                desc="canon-info",
            )
        ]

    def encode_texts(self, canonical_gtins: list[str],
                     record_map: dict[str, dict],
                     infos: list[dict[str, set]]) -> list[str]:
        from core.model_input import build_canonical_text

        return [
            build_canonical_text(record_map.get(g, {}), info)
            for g, info in _LOG.progress(
                zip(canonical_gtins, infos, strict=True),
                total=len(canonical_gtins),
                unit="canon",
                desc="canon-text",
            )
        ]

    def build(self, canonical_gtins: list[str]) -> tuple[pd.DataFrame, dict[str, dict], list[dict[str, set]], list[str]]:
        records, record_map = self.read_records()
        infos = self.structured_infos(canonical_gtins, record_map)
        texts = self.encode_texts(canonical_gtins, record_map, infos)
        return records, record_map, infos, texts


class _PairBundleBuilder:
    """Own one stage-2 build: deduped dataset + gate CSVs -> TrainingData.

    Every phase below is a single responsibility and runs in ONE fixed order
    inside build(); the statements are the pre-refactor body verbatim, so the
    printed stage lines, the trace rows, the timing marks (config_and_trace /
    read_gate_results / sku_texts / canonical_payload / pairs_and_similarity /
    representative_row / pairs_and_mining) and the validated bundle survive
    untouched.

    Phase map:
      prepare_stage_frame   — reviewed-row exclusion + the stage's column
                              contract row
      open_config           — payload thresholds, structured-feature policy
      load_artifacts        — canonical map + gate_results read
      materialize_sku_payload — the variant frame through core.model_input
      materialize_canonical_payload — canonical texts + token budget +
                              structured features
      positives_stage       — row -> canonical pairs (empty-text guarded)
      representative_rows   — longest-title row per GTIN
      baseline_negatives    — gate hard_no band, both directions
      mined_lanes           — targeted-attribute + cross-brand miners
      finalize_bundle       — stats, TrainingData validation, the whole
                              consolidated trace close
    """

    def __init__(self, df: pd.DataFrame, payload_variant: str = "full") -> None:
        self._df = df
        self._payload_variant = payload_variant
        self.df = None
        self.cfg: dict = {}
        self.structured_cfg: dict = {}
        self.structured_enabled = False
        self.thr_pos = 0.0
        self.thr_neg = 0.0
        self.timing = None
        self.canon_map: dict[str, str] = {}
        self.gates: pd.DataFrame | None = None
        self.bc = None
        self.title = None
        self.attrs = None
        self.sku_texts: list[str] = []
        self.sku_structured: list[dict] = []
        self.payload: list[str] = []
        self.row_bc: list[str] = []
        self.canon_gtins: list[str] = []
        self.gtin_to_canon_idx: dict[str, int] = {}
        self.canonical_records: pd.DataFrame | None = None
        self.canonical_record_map: dict[str, dict] = {}
        self.canon_structured: list[dict] = []
        self.canon_texts: list[str] = []
        self.structured_features: list[list[float]] = []
        self.empty_sku: set[int] = set()
        self.empty_canon_idx: set[int] = set()
        self.cand_pos: list[tuple[int, int]] = []
        self.pos = np.empty((0, 2), dtype=int)
        self.gtin_to_row: dict[str, int] = {}
        self.hard_no_band = None
        self.same_canonical = None
        self.neg_mask = None
        self.neg_gates = None
        self.proceeded = None
        self.fell_back = None
        self.fwd = np.empty((0, 2), dtype=int)
        self.rev = np.empty((0, 2), dtype=int)
        self.a = self.b = self.ca = self.cb = None
        self.neg = np.empty((0, 2), dtype=int)
        self.targeted_cfg: dict = {}
        self.mining_funnel = None
        self.targeted_attribute_neg = np.empty((0, 2), dtype=int)
        self.targeted_attribute_scores = np.empty((0,), dtype=float)
        self.cross_cfg: dict = {}
        self.cross_brand_funnel = None
        self.cross_brand_neg = np.empty((0, 2), dtype=int)
        self.cross_brand_scores = np.empty((0,), dtype=float)

    # ── phase: stage frame ─────────────────────────────────────────────────

    def prepare_stage_frame(self) -> None:
        from core.identity_policy import exclude_reviewed_rows
        from core.tracing import TraceRun

        self.df = exclude_reviewed_rows(self._df).reset_index(drop=True)
        print(
            f"[payload-stage] building variant={self._payload_variant} rows={len(self.df):,}",
            flush=True,
        )
        # ── CONSOLIDATED TRACE: stage 2 writer ────────────────────────
        # Created here so the column contract of the frame THIS stage received is
        # the first row of the stage — see run_within_brand_pipeline for the other
        # half of the two-stage handoff.
        self._trace = TraceRun("pairs")
        self._trace.add_column_contract(
            self.df,
            contract="canonical dataset (core.common.load_dataset_deduped)",
            required=CANONICAL_DATASET_REQUIRED_COLUMNS,
            note=(
                "stage 1 (run_within_brand_pipeline) loads the RAW export "
                "separately — both stages now share one column vocabulary "
                "(c698200) and are joined by canonical_records.csv + "
                "gate_results.csv, never by passing this frame between them"
            ),
        )
        # ── config + thresholds ──
        self.cfg = load_config()
        self.structured_cfg = self.cfg["training"]["structured_features"]
        self.structured_enabled = bool(self.structured_cfg["enabled"])
        # The encoder text this stage materializes is an INPUT CONTRACT for every
        # downstream artifact (embeddings, ANN index, checkpoints, reports), so the
        # active composition is recorded on the run before any text is built — a
        # reader can then tell which composition produced what, after the fact.
        from core.model_input import model_input_composition
        self._trace.add(
            "payload",
            "model_input_composition",
            detail=model_input_composition().model_dump(),
        )
        self.thr_pos = float(self.cfg["pairs"]["proceed_sim_threshold"])
        self.thr_neg = float(self.cfg["pairs"]["hardneg_sim_threshold"])
        from core.timing import Timing as _Timing
        self.timing = _Timing("pipeline.build_training_data")
        self.timing.mark("config_and_trace")

    # ── phase: artifacts ───────────────────────────────────────────────────

    def load_artifacts(self) -> None:
        """The canonical map + the gate_results frame."""
        self.canon_map = load_canonical_map()
        self.gates = pd.read_csv(
            RESULTS / F["gate_results"],
            dtype={"gtin1": str, "gtin2": str},
            keep_default_na=False,
        )
        self.timing.mark("read_gate_results")
        self.bc = self.df["gtin"].fillna("").astype(str).str.strip()
        self.title = self.df["sku_name_eng"].fillna("")
        self.attrs = self.df["attribute"].fillna("")

    # ── phase: sku payload ─────────────────────────────────────────────────

    def materialize_sku_payload(self) -> None:
        """Variant frame -> core.model_input sku texts (+ structured sku info)."""
        self.sku_texts, self.sku_structured = SkuTextBuilder(
            self.df, self._payload_variant,
            structured_enabled=self.structured_enabled, timing=self.timing,
        ).build()

    # ── phase: canonical payload ───────────────────────────────────────────

    def materialize_canonical_payload(self) -> None:
        """Canonical texts (sorted-gtin order) + budget + structured features."""
        # ── payload: sku rows + canonical entries (in sorted-gtin order) ──
        self.payload = list(self.sku_texts)
        self.row_bc = [str(x) for x in self.bc]
        self.canon_gtins = sorted(self.canon_map)
        canon_start = len(self.payload)
        self.gtin_to_canon_idx = {g: canon_start + i for i, g in enumerate(self.canon_gtins)}
        builder = CanonicalPayloadBuilder(structured_enabled=self.structured_enabled)
        self.canonical_records, self.canonical_record_map, self.canon_structured,             self.canon_texts = builder.build(self.canon_gtins)
        self.payload.extend(self.canon_texts)
        self.timing.mark("canonical_payload")

        self._token_budget_row()
        self._structured_features()

    def _token_budget_row(self) -> None:
        """Measure the assembled payload; record the budget on the trace."""
        # The structured tail is appended LAST, so at max_seq_length it is the
        # first thing truncated. Measure the assembled payload and record it, so a
        # dropped field group is a named number in the run trace rather than an
        # invisible shortening. Reuses the tracing SSOT and the config SSOT.
        from transformers import AutoTokenizer

        from core.common import resolve_model, runtime
        from core.model_input import token_budget_report

        budget = token_budget_report(
            self.payload,
            tokenizer=AutoTokenizer.from_pretrained(
                str(resolve_model(str(runtime("base_model"))))
            ),
            max_seq_length=int(runtime("max_seq_length")),
        )
        self._trace.add("payload", "token_budget", detail=budget.model_dump())
        if budget.n_field_groups_dropped:
            print(
                f"    [token-budget] WARNING: {budget.n_over_budget:,}/"
                f"{budget.n_records:,} payload records exceed max_seq_length="
                f"{budget.max_seq_length}; dropped field groups: "
                f"{budget.dropped_groups}",
                flush=True,
            )
        self.row_bc.extend(self.canon_gtins)
        print(
            f"[payload-stage] materialized sku_payload={len(self.sku_texts):,} "
            f"canonical_payload={len(self.canon_texts):,}",
            flush=True,
        )

    def _structured_features(self) -> None:
        """Normalized numeric features for every payload entry."""
        from core.structured_features import vector as structured_vector

        structured_infos = self.sku_structured + self.canon_structured
        self.structured_features = [
            structured_vector(
                info,
                volume_scale_ml=float(self.structured_cfg["volume_scale_ml"]),
                pack_scale=float(self.structured_cfg["pack_scale"]),
                max_set_size=int(self.structured_cfg["max_set_size"]),
            )
            for info in structured_infos
        ]

    # ── phase: positives ───────────────────────────────────────────────────

    def positives_stage(self) -> None:
        """Row -> canonical positive pairs, empty-text guarded."""
        # ── empty-text guard (stage-3 soft stop) ──────────────────────────
        # Low-signal rows ("Single 2 Liter Bottle", "water 1.5 lt pack of 6")
        # strip to "". An empty string must not train as a positive — it pulls
        # a garbage vector onto its canonical. Counted in stats (lane doctrine:
        # nothing drops silently). Negatives keep empty texts: a weak
        # in-batch negative is harmless, a positive is not.
        self.empty_sku = {i for i, s in enumerate(self.sku_texts) if not s}
        self.empty_canon_idx = {
            self.gtin_to_canon_idx[g] for g, s in zip(self.canon_gtins, self.canon_texts, strict=True) if not s
        }

        # ── positives: every row whose gtin has a canonical ──
        self.cand_pos = [
            (i, self.gtin_to_canon_idx[g]) for i, g in enumerate(self.bc) if g in self.gtin_to_canon_idx
        ]
        pos_pairs = [
            (i, j)
            for i, j in self.cand_pos
            if i not in self.empty_sku and j not in self.empty_canon_idx
        ]
        self.pos = np.array(pos_pairs, dtype=int).reshape(-1, 2)
        self.timing.mark("pairs_and_similarity")

    # ── phase: representative rows ─────────────────────────────────────────

    def representative_rows(self) -> None:
        """Longest-title row per GTIN (length rank, index tie-break)."""
        # ── representative row per GTIN (longest title — most signal) ──
        # UNEXPECTED-BEHAVIOR FIX: the old code sorted titles
        # lexicographically DESCENDING and called it "longest" — a short
        # z-titled row won over a long a-titled one. Rank by title LENGTH;
        # ties break by row index (stable, reproducible).
        t_len = self.title.astype(str).str.len().to_numpy()
        order = np.lexsort((np.arange(len(t_len)), -t_len))
        seen: set[str] = set()
        self.gtin_to_row = {}
        bc_arr = self.bc.to_numpy() if hasattr(self.bc, "to_numpy") else list(self.bc)
        for i in _LOG.progress(
            order,
            total=len(order),
            unit="row",
            desc="canon-row",
        ):
            g = bc_arr[i]
            if g and g not in seen:
                seen.add(g)
                self.gtin_to_row[g] = i
        self.timing.mark("representative_row")

    # ── phase: baseline negatives ──────────────────────────────────────────

    def baseline_negatives(self) -> None:
        """Gate hard_no band -> index-resolved negative pairs, both directions."""
        # ── negatives: gate hard-no pairs (both directions) ──
        # A hard_no gate decision is not sufficient for training: separate GTINs
        # can still resolve to the same canonical item.  Those rows are true
        # matches and must never be emitted as label-0 pairs.
        gate_canon1 = self.gates["gtin1"].map(self.canon_map)
        gate_canon2 = self.gates["gtin2"].map(self.canon_map)
        self.same_canonical = (
            gate_canon1.notna()
            & gate_canon2.notna()
            & gate_canon1.eq(gate_canon2)
        )
        self.hard_no_band = (self.gates["gate_decision"] == "hard_no") & (
            self.gates["similarity"] >= self.thr_neg
        )
        self.neg_mask = self.hard_no_band & ~self.same_canonical
        self.neg_gates = self.gates[self.neg_mask]
        # The other two gate outcomes, kept as masks so the label-destiny census
        # below accounts for EVERY candidate pair rather than only the negatives.
        self.proceeded = self.gates["gate_decision"] == "proceed"
        self.fell_back = self.gates["gate_decision"] == "fallback"
        self.a = self.neg_gates["gtin1"].map(self.gtin_to_row)
        self.b = self.neg_gates["gtin2"].map(self.gtin_to_row)
        self.ca = self.neg_gates["gtin2"].map(self.gtin_to_canon_idx)
        self.cb = self.neg_gates["gtin1"].map(self.gtin_to_canon_idx)
        ok1 = self.a.notna() & self.ca.notna()
        ok2 = self.b.notna() & self.cb.notna()
        self.fwd = np.stack([self.a[ok1].astype(int), self.ca[ok1].astype(int)], axis=1)
        self.rev = np.stack([self.b[ok2].astype(int), self.cb[ok2].astype(int)], axis=1)
        self.neg = np.vstack([self.fwd, self.rev]) if len(self.fwd) or len(self.rev) else np.empty((0, 2), dtype=int)

    # ── phase: mined lanes ─────────────────────────────────────────────────

    def mined_lanes(self) -> None:
        """Targeted-attribute + cross-brand hard-negative miners (their own
        funnels stay the attrition authority)."""
        self._targeted_attribute_lane()
        self._cross_brand_lane()

    def _targeted_attribute_lane(self) -> None:
        from core.hard_negatives import (
            MiningFunnel,
            mine_targeted_attribute_negatives,
        )

        self.targeted_cfg = self.cfg["mining"]["attribute_conflict"]
        self.mining_funnel = (
            MiningFunnel() if bool(self.targeted_cfg["same_product_name"]) else None
        )
        self.targeted_attribute_neg, self.targeted_attribute_scores = (
            mine_targeted_attribute_negatives(
                self.df,
                self.gates,
                self.canonical_records,
                self.gtin_to_row,
                self.gtin_to_canon_idx,
                existing=self.neg,
                n_target=int(self.targeted_cfg["target"]),
                min_similarity=float(self.targeted_cfg["min_similarity"]),
                # Same volume tolerance the training-label gate uses, so a pair the
                # gate calls compatible can never be mined here as a conflict.
                # BOTH cuts: the relative-only cut rejected small-volume pairs the
                # gate accepts, which would mine a true match as a hard negative.
                volume_relative_tolerance=float(training_cfg().gate.vol_tolerance),
                volume_absolute_tolerance_ml=float(
                    training_cfg().gate.vol_abs_tolerance
                ),
                # Same canonical-identity rule the baseline negative lane already
                # applies: a same-canonical pair is a true match, not a label-0 row.
                canonical_map=self.canon_map,
                funnel=self.mining_funnel,
            )
            if bool(self.targeted_cfg["same_product_name"])
            else (np.empty((0, 2), dtype=int), np.empty((0,), dtype=float))
        )

    def _cross_brand_lane(self) -> None:
        from core.hard_negatives import (
            CrossBrandMiningFunnel,
            mine_cross_brand_negatives,
        )

        self.cross_cfg = self.cfg["mining"]["cross_brand"]
        self.cross_brand_funnel = (
            CrossBrandMiningFunnel() if bool(self.cross_cfg["enabled"]) else None
        )
        self.cross_brand_neg, self.cross_brand_scores = (
            mine_cross_brand_negatives(
                self.df,
                self.canonical_records,
                self.gtin_to_row,
                self.gtin_to_canon_idx,
                existing=self.neg,
                n_target=int(self.cross_cfg["target"]),
                require_agreement=tuple(str(d) for d in self.cross_cfg["require_agreement"]),
                min_similarity=float(self.cross_cfg["min_similarity"]),
                max_per_canonical=int(self.cross_cfg["max_per_canonical"]),
                max_per_brand=int(self.cross_cfg["max_per_brand"]),
                # Same volume tolerance the training-label gate uses, so a pair the
                # gate calls compatible can never be mined here as a conflict.
                # BOTH cuts (see the targeted lane above).
                volume_relative_tolerance=float(training_cfg().gate.vol_tolerance),
                volume_absolute_tolerance_ml=float(
                    training_cfg().gate.vol_abs_tolerance
                ),
                funnel=self.cross_brand_funnel,
            )
            if bool(self.cross_cfg["enabled"])
            else (np.empty((0, 2), dtype=int), np.empty((0,), dtype=float))
        )

    # ── phase: final bundle + trace close ──────────────────────────────────

    def finalize_bundle(self) -> dict:
        """Stats + prints + validated TrainingData + the consolidated trace."""
        n_forward_source_unresolved = int(self.a.isna().sum())
        n_forward_target_unresolved = int(self.ca.isna().sum())
        n_reverse_source_unresolved = int(self.b.isna().sum())
        n_reverse_target_unresolved = int(self.cb.isna().sum())
        n_resolution_dropped = int(len(self.neg_gates) * 2 - len(self.neg))
        stats = {
            "n_rows": len(self.df),
            "n_sku_with_canonical": len(self.cand_pos),
            "n_pos_empty_dropped": len(self.cand_pos) - len(self.pos),
            "n_empty_sku_texts": len(self.empty_sku),
            "n_empty_canon_texts": len(self.empty_canon_idx),
            "n_canonicals": len(self.canon_gtins),
            "n_pos_gate_rows": int(
                (
                    (self.gates["gate_decision"] == "proceed") & (self.gates["similarity"] >= self.thr_pos)
                ).sum()
            ),
            "n_neg_same_canonical_dropped": int((self.hard_no_band & self.same_canonical).sum()),
            "n_neg_hard_no_band": int(self.hard_no_band.sum()),
            "n_neg_gate_rows": int(self.neg_mask.sum()),
            "n_neg_resolved": len(self.neg),
            "n_neg_forward_resolved": int(len(self.fwd)),
            "n_neg_reverse_resolved": int(len(self.rev)),
            "n_neg_forward_source_unresolved": n_forward_source_unresolved,
            "n_neg_forward_target_unresolved": n_forward_target_unresolved,
            "n_neg_reverse_source_unresolved": n_reverse_source_unresolved,
            "n_neg_reverse_target_unresolved": n_reverse_target_unresolved,
            "n_neg_resolution_dropped": n_resolution_dropped,
            "n_neg_dropped": n_resolution_dropped,
            "n_targeted_attribute_candidates": int(len(self.targeted_attribute_scores)),
            "n_targeted_attribute_resolved": int(len(self.targeted_attribute_neg)),
            # Candidates ENTERING the cross-brand funnel (the pairs its
            # require_agreement blocking generated) and the pair rows it emitted.
            # The two differ by the funnel's own attrition, which the trace
            # records step by step.
            "n_cross_brand_candidates": int(
                self.cross_brand_funnel.candidates_in_blocks
                if self.cross_brand_funnel is not None
                else 0
            ),
            "n_cross_brand_resolved": int(len(self.cross_brand_neg)),
        }
        print(
            f"[payload-stage] pairs resolved positives={len(self.pos):,} "
            f"hard_negatives={len(self.neg):,} unresolved_or_dropped={n_resolution_dropped:,}",
            flush=True,
        )
        print(
            f"[targeted-attribute-negatives] {len(self.targeted_attribute_neg):,} "
            f"same-brand/name explicit-conflict pairs with gate similarity "
            f"> {float(self.targeted_cfg['min_similarity']):.2f}",
            flush=True,
        )
        if self.cross_brand_funnel is not None:
            _cb = self.cross_brand_funnel
            print(
                f"[cross-brand-negatives] {len(self.cross_brand_neg):,} label-0 pair rows "
                f"from {_cb.accepted_candidates:,} candidates "
                f"({_cb.candidates_in_blocks:,} generated -> "
                f"{_cb.passed_candidates:,} survived every filter; "
                f"target {int(self.cross_cfg['target']):,}, "
                f"reached={_cb.emitted_pairs >= int(self.cross_cfg['target']) > 0})",
                flush=True,
            )
        else:
            print(
                "[cross-brand-negatives] disabled by mining.cross_brand.enabled",
                flush=True,
            )
        self._validated_bundle(stats)
        self._trace_close()
        return self._bundle_dump

    def _validated_bundle(self, stats: dict) -> None:
        """Payload-dump rows + the boundary TrainingData validation."""
        # ── EXACT MODEL PAYLOAD DUMP (owner directive 2026-09-07) ──────
        # Every pair the model trains on, with the LITERAL texts it ingests.
        # The rows are recorded in the ONE consolidated trace (core.tracing) below,
        # replacing the former per-stage payload_pairs.csv.
        _rows = []
        for i, j in self.pos:
            _rows.append(
                {
                    "kind": "pos",
                    "payload_idx_a": int(i),
                    "payload_idx_b": int(j),
                    "gtin_a": self.row_bc[i],
                    "gtin_b": self.row_bc[j],
                    "text_a": self.payload[i],
                    "text_b": self.payload[j],
                }
            )
        for i, j in self.neg:
            _rows.append(
                {
                    "kind": "neg_hard",
                    "payload_idx_a": int(i),
                    "payload_idx_b": int(j),
                    "gtin_a": self.row_bc[i],
                    "gtin_b": self.row_bc[j],
                    "text_a": self.payload[i],
                    "text_b": self.payload[j],
                }
            )
        for i, j in self.targeted_attribute_neg:
            _rows.append(
                {
                    "kind": "neg_targeted_attribute",
                    "payload_idx_a": int(i),
                    "payload_idx_b": int(j),
                    "gtin_a": self.row_bc[i],
                    "gtin_b": self.row_bc[j],
                    "text_a": self.payload[i],
                    "text_b": self.payload[j],
                }
            )
        for i, j in self.cross_brand_neg:
            _rows.append(
                {
                    "kind": "neg_cross_brand",
                    "payload_idx_a": int(i),
                    "payload_idx_b": int(j),
                    "gtin_a": self.row_bc[i],
                    "gtin_b": self.row_bc[j],
                    "text_a": self.payload[i],
                    "text_b": self.payload[j],
                }
            )
        self._rows = _rows
        self._kinds = {
            "pos": int(len(self.pos)),
            "neg_hard": int(len(self.neg)),
            "neg_targeted_attribute": int(len(self.targeted_attribute_neg)),
            "neg_cross_brand": int(len(self.cross_brand_neg)),
        }
        # BOUNDARY CONTRACT (lib.schemas.TrainingData): payload/row_bc locked,
        # every pos/neg index in range, gtin_to_row targets valid — the bundle
        # crosses into src/training/train + src/training/training; a shape break must die
        # HERE with a named field, not as an IndexError in a fold.
        from core.schemas import TrainingData as _TrainingData

        _bundle = _TrainingData(
            payload=self.payload,
            structured_features=self.structured_features,
            row_bc=np.array(self.row_bc),
            pos=self.pos,
            neg=self.neg,
            targeted_attribute_neg=self.targeted_attribute_neg,
            cross_brand_neg=self.cross_brand_neg,
            gtin_to_row=self.gtin_to_row,
            stats=stats,
        )
        self._bundle = _bundle
        # ── CONSOLIDATED TRACE: pairs + mining funnel ──────────────────────
        # The former per-stage files (negative_resolution_manifest.csv,
        # payload_pairs.csv) folded into the ONE trace (core.tracing). Stage 2
        # commits onto stage 1's rows for the same run, so the file reads as one
        # continuous flow. This block used to sit AFTER `return _bundle.model_dump()`
        # and was therefore dead: stage 2 ran, printed its counts, and wrote nothing
        # to the trace. The bundle is validated first (fail fast on a shape break),
        # then every step is recorded, then the bundle is returned.
        self._bundle_dump = _bundle.model_dump()

    def _trace_close(self) -> None:
        """Label-destiny census, funnel audits, payload census, entity samples."""
        trace = self._trace
        # EVERY candidate pair gets exactly one label destiny. Summing these group
        # rows reproduces len(gates) exactly, which is the pair-side accounting
        # identity: no gate pair is unaccounted for, and each row states in words
        # why that population did or did not become a training label.
        n_gates = int(len(self.gates))
        label_buckets = [
            (
                "proceed_not_a_training_pair",
                int(self.proceeded.sum()),
                "gate says same product: a PROCEED pair yields no label here — "
                "positives come from the row→canonical relation, not the pair",
                {"gate_decision": "proceed"},
            ),
            (
                "fallback_unresolved",
                int(self.fell_back.sum()),
                "gate could not resolve the pair: neither a verified match nor a "
                "hard no, so neither mining lane may use it",
                {"gate_decision": "fallback"},
            ),
            (
                "negative_hard",
                int(self.neg_mask.sum()),
                "hard_no inside the mining similarity band and NOT same-canonical: "
                "emitted as a label-0 pair in both directions",
                {
                    "gate_decision": "hard_no",
                    "similarity_threshold": float(self.thr_neg),
                    "directions_per_pair": 2,
                },
            ),
            (
                "true_match_same_canonical_excluded",
                int((self.hard_no_band & self.same_canonical).sum()),
                "hard_no in band but both gtins share one canonical record: a true "
                "match, so label 0 would be wrong",
                {"gate_decision": "hard_no"},
            ),
            (
                "hard_no_below_similarity_floor",
                int(((self.gates["gate_decision"] == "hard_no") & ~self.hard_no_band).sum()),
                "hard_no below the mining similarity floor: a valid hard no that "
                "this lane's negative mining does not reach",
                {"gate_decision": "hard_no", "similarity_threshold": float(self.thr_neg)},
            ),
        ]
        for bucket, population, why, extra in label_buckets:
            trace.add(
                "labels",
                f"destiny_{bucket}",
                scope="group",
                in_count=n_gates,
                out_count=population,
                reason=why,
                detail={"population": population, **extra},
                source="gate_results.csv",
            )
        trace.add(
            "labels",
            "every_pair_accounted",
            in_count=n_gates,
            out_count=int(sum(population for _, population, _, _ in label_buckets)),
            reason="every candidate pair carries exactly one label destiny",
            detail={
                "gate_pairs": n_gates,
                "destinies": {
                    bucket: population for bucket, population, _, _ in label_buckets
                },
            },
            source="gate_results.csv",
        )
        trace.add(
            "positives",
            "sku_to_canonical",
            scope="group",
            in_count=int(len(self.cand_pos)),
            out_count=int(len(self.pos)),
            reason="a row with a resolvable canonical and non-empty model text is a positive",
            detail={
                "rows": int(len(self.df)),
                "sku_with_canonical": int(len(self.cand_pos)),
                "dropped_empty_sku_text": int(len(self.empty_sku)),
                "dropped_empty_canon_text": int(len(self.empty_canon_idx)),
                "canonicals": int(len(self.canon_gtins)),
                "gate_proceed_rows": int(
                    (
                        (self.gates["gate_decision"] == "proceed")
                        & (self.gates["similarity"] >= self.thr_pos)
                    ).sum()
                ),
            },
            source="canonical_records.csv",
        )
        trace.add(
            "negatives",
            "gate_hard_no_band",
            scope="group",
            in_count=n_gates,
            out_count=int(len(self.neg_gates)),
            reason=(
                "hard_no with similarity >= the mining threshold; same-canonical "
                "pairs are true matches and are excluded here"
            ),
            detail={
                "hard_no_and_in_band": int(self.hard_no_band.sum()),
                "dropped_same_canonical": int((self.hard_no_band & self.same_canonical).sum()),
                "similarity_threshold": float(self.thr_neg),
                "both_directions": int(len(self.neg_gates) * 2),
            },
            source="gate_results.csv",
        )
        trace.add(
            "negatives",
            "index_resolution",
            scope="group",
            in_count=int(len(self.neg_gates) * 2),
            out_count=int(len(self.neg)),
            reason="both endpoints must resolve to a payload row index",
            detail={
                "forward_resolved": int(len(self.fwd)),
                "reverse_resolved": int(len(self.rev)),
                "forward_source_unresolved": int(self.a.isna().sum()),
                "forward_target_unresolved": int(self.ca.isna().sum()),
                "reverse_source_unresolved": int(self.b.isna().sum()),
                "reverse_target_unresolved": int(self.cb.isna().sum()),
            },
            source="model payload index maps (rows + canonicals)",
        )
        self._funnel_rows(trace)
        # A unit CHANGE, not a filter: one source row becomes a payload row plus
        # one canonical entry, so the payload is legitimately LARGER than the row
        # population. Stating an in/out pair here made dropped_count negative,
        # which the row contract forbids; the unit change is flagged instead.
        _payload_in, _payload_out, _payload_unit_change = unit_change_counts(
            int(len(self.df)), int(len(self.payload))
        )
        trace.add(
            "payload",
            "materialized",
            in_count=_payload_in,
            out_count=_payload_out,
            reason="every source row plus one canonical per GTIN",
            detail={
                "sku_payload": int(len(self.df)),
                "canonical_payload": int(len(self.canon_gtins)),
                "unit_change": _payload_unit_change,
                "structured_feature_dim": int(len(self.structured_features[0])),
                "structured_encode": bool(self.structured_cfg.get("enabled", True)),
            },
            source="canonical_records.csv",
        )
        trace.add(
            "payload",
            "pair_census",
            scope="group",
            in_count=int(
                len(self.pos) + len(self.neg) + len(self.targeted_attribute_neg) + len(self.cross_brand_neg)
            ),
            out_count=int(len(self._rows)),
            reason="final label populations handed to training",
            detail={
                "pos": int(len(self.pos)),
                "neg_hard": int(len(self.neg)),
                "neg_targeted_attribute": int(len(self.targeted_attribute_neg)),
                "neg_cross_brand": int(len(self.cross_brand_neg)),
                "text_columns": ["text_a", "text_b"],
            },
            source="model payload",
        )
        self._pair_payload_entities(trace)
        trace.write()
        print(
            f"[trace] pairs steps written -> {trace_path()} | {self._kinds}",
            flush=True,
        )
        self.timing.mark("pairs_and_mining")
        self.timing.dump_if_requested()

    def _funnel_rows(self, trace) -> None:
        """One row per real filter, from the miner's OWN funnel accounting."""
        # ONE row per real filter, taken from the miner's OWN funnel accounting.
        # The previous single row restated "gate rows above the similarity floor"
        # as the input and claimed the filters generically, which hid that ~39,896
        # candidates die at the name filter and made the lane's ceiling
        # unanswerable from the trace. The miner stays the label authority; this
        # only records what it did.
        #
        # The whole funnel is stated ONCE per lane (`*.census`), not repeated in
        # every step's detail: at ~3.7 KB per copy and ~30 steps that was
        # O(steps^2) bytes per row set. A step row now carries its own numbers,
        # the shared thresholds, and a pointer to the census row.
        if self.mining_funnel is not None:
            trace.add(
                "mining",
                "targeted_attribute_funnel.census",
                reason=(
                    "the targeted lane's funnel, stated ONCE: every per-step row "
                    "below carries only that step's own numbers"
                ),
                detail=_capped_detail({"funnel": self.mining_funnel.to_dict()}),
                source="gate_results.csv",
            )
            for _step, _in, _out, _why in self.mining_funnel.stages():
                _in_count, _out_count, _unit_change = unit_change_counts(_in, _out)
                trace.add(
                    "mining",
                    f"targeted_attribute_funnel.{_step}",
                    in_count=_in_count,
                    out_count=_out_count,
                    reason=_why,
                    detail=_capped_detail({
                        "gate_similarity_floor": float(self.targeted_cfg["min_similarity"]),
                        "volume_tolerance": float(training_cfg().gate.vol_tolerance),
                        "target": int(self.targeted_cfg["target"]),
                        "unit_change": _unit_change,
                        "funnel_census": "mining.targeted_attribute_funnel.census",
                    }),
                    source="gate_results.csv",
                )
        else:
            trace.add(
                "mining",
                "targeted_attribute_funnel.disabled",
                in_count=0,
                out_count=0,
                reason="mining.attribute_conflict.same_product_name is false",
                detail={"target": int(self.targeted_cfg["target"])},
                source="config/training.yaml",
            )
        # The cross-brand lane's own funnel, in the same shape as the targeted one:
        # its generation step (blocking census) then every filter's attrition. A
        # lane whose population is generated rather than handed in cannot be
        # audited from its output count alone, so the census is the trace's job.
        if self.cross_brand_funnel is not None:
            trace.add(
                "mining",
                "cross_brand_funnel.census",
                reason=(
                    "the cross-brand lane's funnel, stated ONCE: every per-step row "
                    "below carries only that step's own numbers"
                ),
                detail=_capped_detail({"funnel": self.cross_brand_funnel.to_dict()}),
                source="canonical_records.csv + gate_results.csv",
            )
            for _step, _in, _out, _why in self.cross_brand_funnel.stages():
                _in_count, _out_count, _unit_change = unit_change_counts(_in, _out)
                trace.add(
                    "mining",
                    f"cross_brand_funnel.{_step}",
                    in_count=_in_count,
                    out_count=_out_count,
                    reason=_why,
                    detail=_capped_detail({
                        "target": int(self.cross_cfg["target"]),
                        "min_similarity": float(self.cross_cfg["min_similarity"]),
                        "require_agreement": list(self.cross_cfg["require_agreement"]),
                        "volume_tolerance": float(training_cfg().gate.vol_tolerance),
                        "unit_change": _unit_change,
                        "funnel_census": "mining.cross_brand_funnel.census",
                    }),
                    source="canonical_records.csv + gate_results.csv",
                )
        else:
            trace.add(
                "mining",
                "cross_brand_funnel.disabled",
                in_count=0,
                out_count=0,
                reason="mining.cross_brand.enabled is false",
                detail={"target": int(self.cross_cfg["target"])},
                source="config/training.yaml",
            )

    def _pair_payload_entities(self, trace) -> None:
        """Bounded per-pair readback with the LITERAL model texts."""
        # Bounded per-pair readback with the LITERAL model texts, so the trace is
        # a sample of the training input rather than only a count of it. Bucketed
        # by label kind, so the three label populations above and the sampled rows
        # below join on the same label.
        trace.add_entities(
            "pair_payload",
            self._rows,
            key_of=lambda r: f"{r['kind']}|{r['gtin_a']}|{r['gtin_b']}",
            reason_of=lambda r: r["kind"],
            detail_of=lambda r: json.dumps(
                {
                    "payload_idx_a": r["payload_idx_a"],
                    "payload_idx_b": r["payload_idx_b"],
                    "gtin_a": r["gtin_a"],
                    "gtin_b": r["gtin_b"],
                    "text_a": r["text_a"],
                    "text_b": r["text_b"],
                },
                sort_keys=True,
            ),
            source="model payload",
            per_reason=ENTITY_PER_REASON,
            total_cap=ENTITY_TOTAL_CAP,
        )

    # ── orchestration ──────────────────────────────────────────────────────

    def build(self) -> dict:
        """Run the load-bearing phase order; return the validated bundle dump."""
        self.prepare_stage_frame()
        self.load_artifacts()
        self.materialize_sku_payload()
        self.materialize_canonical_payload()
        self.positives_stage()
        self.representative_rows()
        self.baseline_negatives()
        self.mined_lanes()
        return self.finalize_bundle()


def build_training_data(
    df: pd.DataFrame,
    *,
    payload_variant: str = "full",
) -> dict:
    """Build payload + pos/neg pairs from the deduped dataset + gate results.

    Returns dict with:
        payload : list[str]  — clean sku text per row + one canonical per GTIN
        structured_features : list[list[float]] — normalized numeric features
        row_bc  : np.ndarray — gtin per payload entry (gtin for canonicals)
        pos     : np.ndarray (N,2) — (sku_row, canon_idx) for every row whose
                  gtin has a canonical
        neg     : np.ndarray (M,2) — (rep_row(g1), canon(g2)) and mirror, for
                  every gate hard-no pair with similarity >= threshold
        stats   : dict — counts (nothing dropped silently)
    """
    # Contract preserved: the caller receives the TrainingData bundle.
    return _PairBundleBuilder(df, payload_variant=payload_variant).build()
