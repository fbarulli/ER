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
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from core.columns import (
    CANONICAL_DATASET_REQUIRED_COLUMNS as CANONICAL_DATASET_REQUIRED_COLUMNS_REQUIRED,
)
from core.columns import DATA_PREP_REQUIRED_COLUMNS, source_row_pairs
from core.common import (
    DATA_DIR,
    RESULTS,
    F,
    data_cfg,
    load_config,
    training_cfg,
    vocabulary,
)
from core.schemas import (
    CanonicalRecord,
    ExtractedAttributes,
    GateResult,
    check_canonical_records_frame,
    check_gate_results_frame,
    check_verdict_map,
    require_populated_source_rows,
)
from ner.ner_product_attributes import extract_title_attributes, parse_attribute_details
from core.critical_attributes import (
    CRITICAL_ATTRIBUTE_DIMENSIONS,
    categorical_conflict,
    extract_critical_claims,
    extract_description_claims,
    extract_flavor_tokens,
    volumes_compatible,
)
from core.tracing import (
    ENTITY_ROW_CAP,
    ENTITY_SAMPLE_PER_REASON,
    count_rows,
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
# (core.product_identity, core.record_linkage, core.model_input, the training
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
    has_decimal = bool(re.match(r"\s*\d+(?:[.,]\d)", raw))
    confidence = entry.confidence
    if has_decimal and entry.decimal_confidence is not None:
        confidence = entry.decimal_confidence
    return {"volume_ml": ml, "confidence": confidence, "raw_match": raw,
            "parse_status": entry.family}


def extract_pack_evidence(title: str) -> list[dict]:
    """Retain physical-unit and outer-package quantities with original spans."""
    text = str(title or "")
    from core.text import extract_volume_evidence
    measurements = extract_volume_evidence(text)
    confidence = data_cfg().extraction.pack_confidence
    # Count tokens must include their entire number: decimal and price tails
    # cannot masquerade as integer quantities. Grouped thousands are counts.
    count_token = r"([1-9]\d{0,2}(?:[.,]\d{3})+|\d+)(?!\d|[.,]\d)"
    number = r"(?<![\w$€£])(?<!\d[.,])" + count_token
    containers = r"(?:bottles?|bt|cans?|tins?|cartons?|boxes?|packets?|sachets?|bags?)"
    patterns = (
        ("nested", rf"{number}\s*[x×]\s*(\d+)\s*(?:{containers}\s*)?(?:[x×]|/)\s*\d+(?:[.,]\d+|[.,]\s+\d{{1,2}})?\s*[a-z]", "unit_count"),
        ("multiplier", rf"{number}\s*[x×]\s*(?:pack\s*)?\d+(?:[.,]\d+|[.,]\s+\d{{1,2}})?\s*[a-z]", "unit_count"),
        ("pack_of", rf"\b(?:packs?|packages?)\s+of\s*{count_token}\b", "unit_count"),
        ("pack_of", rf"\bcases?\s+of\s*{count_token}\b", "unit_count"),
        ("count", rf"{number}\s*[- ]?\s*(?:pcs?|pieces?|packs?|packages?|pk|units?|ct|count)\b", "unit_count"),
        ("compact", rf"\bpack\s*[- ]?\s*{count_token}\b", "unit_count"),
        ("container", rf"{number}\s*(?:glass\s*)?{containers}\b", "unit_count"),
        ("count", rf"{number}\s*cases?\b", "outer_count"),
    )
    evidence = []
    occupied = []
    for kind, pattern, role in patterns:
        for match in re.finditer(pattern, text, re.I):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            if kind == "compact" and any(
                entry["start"] == match.start(1) for entry in measurements
            ):
                continue
            if kind == "compact" and text[match.end():].startswith(")") and re.search(
                rf"\b{int(re.sub(r'[.,]', '', match.group(1))) + 1}\)",
                text[match.end() + 1:],
            ):
                # "Combo Pack - 1) product A & 2) product B" is a list.
                continue
            # Currency followed by whitespace still denotes a price.
            if re.search(r"[$€£]\s*$", text[:match.start()]):
                continue
            count = int(re.sub(r"[.,]", "", match.group(1)))
            if kind == "nested":
                count *= int(match.group(2))
            if count <= 0:
                continue
            occupied.append(match.span())
            evidence.append({"count": count, "confidence": confidence[kind],
                             "role": role, "raw_match": match.group(0),
                             "start": match.start(), "end": match.end(), "rule": kind})
    word_counts = dict(zip(
        ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"),
        range(1, 13), strict=True,
    ))
    for match in re.finditer(r"\b(" + "|".join(word_counts) + r")\s*[- ]\s*packs?\b", text, re.I):
        evidence.append({"count": word_counts[match.group(1).lower()],
                         "confidence": confidence["count"], "role": "unit_count",
                         "raw_match": match.group(0), "start": match.start(),
                         "end": match.end(), "rule": "count"})
    inner = re.search(rf"(?<![\d.,]){count_token}\s*sticks?\s+per\s+box\b", text, re.I)
    if inner:
        inner_count = int(re.sub(r"[.,]", "", inner.group(1)))
        evidence.append({"count": inner_count, "confidence": confidence["count"],
                         "role": "inner_count", "raw_match": inner.group(0),
                         "start": inner.start(), "end": inner.end(), "rule": "count"})
        for entry in list(evidence):
            if entry["rule"] == "compact" and entry["role"] == "unit_count":
                entry["role"] = "outer_count"
                entry["hierarchy_ambiguous"] = True
                start, end = min(entry["start"], inner.start()), max(entry["end"], inner.end())
                evidence.append({"count": inner_count * entry["count"],
                                 "confidence": min(entry["confidence"], confidence["count"]),
                                 "role": "derived_inner_total", "hierarchy_ambiguous": True,
                                 "raw_match": text[start:end], "start": start, "end": end,
                                 "rule": "nested"})
        outer = re.search(rf"{number}\s*boxes\b", text, re.I)
        if outer:
            start, end = min(outer.start(), inner.start()), max(outer.end(), inner.end())
            evidence.insert(0, {"count": inner_count * int(re.sub(r"[.,]", "", outer.group(1))),
                                "confidence": confidence["nested"], "role": "unit_count",
                                "raw_match": text[start:end], "start": start, "end": end,
                                "rule": "nested"})
    # Whitespace alone is not a multiplier. A nearby explicitly stated total
    # can prove the relation, e.g. "6 330 ml (Total 1980 ml)".
    from core.unit_canonicalization import canonical_volume_ml
    for unit_entry, total_entry in zip(measurements, measurements[1:]):
        prefix = re.search(rf"{number}\s+$", text[:unit_entry["start"]])
        between = text[unit_entry["end"]:total_entry["start"]]
        if prefix is None or not re.fullmatch(r"[ .()]*total\s*", between, re.I):
            continue
        if any(start <= prefix.start() < end for start, end in occupied):
            continue
        count = int(re.sub(r"[.,]", "", prefix.group(1)))
        unit_volume = canonical_volume_ml(unit_entry["value"], unit_entry["unit"])
        total_volume = canonical_volume_ml(total_entry["value"], total_entry["unit"])
        if count > 0 and unit_volume > 0 and math.isclose(count * unit_volume, total_volume):
            start, end = prefix.start(), total_entry["end"]
            evidence.append({"count": count, "confidence": confidence["multiplier"],
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
            if any(entry["role"] == "unit_count" for entry in evidence):
                break
            start, end = unit_entry["start"], total_entry["end"]
            evidence.append({"count": count, "confidence": confidence["multiplier"],
                             "role": "unit_count", "raw_match": text[start:end],
                             "start": start, "end": end, "rule": "multiplier"})
            break
    return evidence


def extract_pack_from_title(title: str) -> tuple:
    evidence = extract_pack_evidence(title)
    units = [entry for entry in evidence if entry["role"] == "unit_count"]
    if units:
        return units[0]["count"], units[0]["confidence"]
    ambiguous_outer = [entry for entry in evidence if entry.get("hierarchy_ambiguous") and entry["role"] == "outer_count"]
    if ambiguous_outer:
        return ambiguous_outer[0]["count"], ambiguous_outer[0]["confidence"]
    # Outer cases do not state the number of consumer units in each case.
    return 1, 0.0


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
    m_vol = re.search(
        r"\bVolume:\s*(.*?)(?=;|\n|\s+[A-Za-z][A-Za-z ]*:|$)",
        attr_str, re.IGNORECASE,
    )
    if m_vol:
        declared = m_vol.group(1).strip()
        measurements = extract_volume_evidence(declared)
        if measurements and measurements[0]["start"] == 0:
            entry = measurements[0]
            vol_ml = canonical_volume_ml(entry["value"], entry["unit"])
        elif re.fullmatch(r"\d+(?:[.,]\s*\d+)?", declared):
            # Historical export convention: unitless declared Volume is ml.
            number = re.sub(r"\s+", "", declared)
            if float(number.replace(",", ".")) > 0:
                vol_ml = canonical_volume_ml(number, "ml")
        if vol_ml > 0:
            vol_conf = 0.9
    m_pack = re.search(r"Count per Unit:\s*(\d+)(?!\d|[.,/]\s*\d)", attr_str, re.IGNORECASE)
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
# rows, results/attribute_universe_census.json): pack material type 51,703
# rows / 5 value-sets / 9.64% same-GTIN conflict (inside the VETO BAND
# 2.5%-15%, census-verified); juice content 63,117 rows / 27 numeric bands;
# carbonization 56,125 rows (prose claims already flow through
# extract_critical_claims — this section mirrors the value vocabulary);
# water type 18,850 / naturally derived 28,637 / made from 19,338 (no set
# field exists — captured for the census→wiring parity check only, never a
# model-visible field). The veto LIST itself stays config-owned
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


def extract_all(sku_name: str, attribute: str, description: str = "",
     url: str = "", image_url: str = "", category_path: str = "",
     category: str = "") -> dict:
    """Extract structured fields plus salient tokens from a single SKU row.

    Evidence is drawn from ALL available columns — title, attributes,
    description, URL slug, image filename, category path, and category —
    so the gate sees every product-bearing signal before deciding.
    """
    from core.sweetener_values import declared_sweeteners, extract_sweetening_status, title_sweetener_types, negated_sweetener_types
    from core.text import extract_volume_evidence
    from core.url_evidence import url_text

    sweeteners = declared_sweeteners(attribute)
    sweeteners["sweetener_type"].update(title_sweetener_types(sku_name))
    sweeteners["sweetener_type"].update(title_sweetener_types(description))
    sweeteners["sweetening"].update(extract_sweetening_status(sku_name, attribute, description))
    t = normalize_text(sku_name)
    # URL tokens: product-bearing prose from the listing slug.
    # Fed into volume/pack extraction when title/attributes are silent.
    # url_text is the reader for BOTH URL columns (docstring, url_evidence.py):
    # image filenames go through the same normalizer — hashes, media dims and
    # scaffolding fall out; size tokens ("250ml") survive.
    url_tokens = url_text(url)
    img_tokens = url_text(image_url)
    url_norm = normalize_text(url_tokens)
    img_norm = normalize_text(img_tokens)
    # Category evidence: category_path and category provide
    # product-type signals (flavor hints, carbonation clues)
    # that title/attributes may miss.
    cat_tokens = normalize_text(category_path) + " " + normalize_text(category)
    cat_tokens = cat_tokens.strip()
    # Critical categorical evidence is parsed once for the canonical, model,
    # mining, and inference lanes.  Keep the historical scalar flavor as a
    # deterministic first value for compatibility with existing CSV readers.
    critical = extract_critical_claims(sku_name, attribute)
    description_claims = extract_description_claims(description)
    consistency_flags = set(sweeteners["consistency_flags"])
    negative_ingredients = negated_sweetener_types(sku_name, attribute, description)
    consistency_flags.update(
        f"sweetener_source_conflict:{ingredient}"
        for ingredient in negative_ingredients & sweeteners["sweetener_type"]
    )
    # Product card evidence ledger: every claim the columns yield, recorded
    # with its source at the moment of extraction (surface-one-by-one
    # ruling 2026-10-01). Rides the result dict additively, like
    # attribute_universe_evidence — schema stays extra="forbid".
    ledger: list[dict] = []
    if negative_ingredients:
        ledger.append({"field": "negated_sweetener_type", "column": "title+attributes+description",
                       "value": sorted(negative_ingredients)})
    from core.date_evidence import extract_date_evidence
    date_evidence = [
        {"column": column, **entry}
        for column, text in (
            ("title", sku_name), ("attributes", attribute),
            ("description", description), ("category_path", category_path),
            ("category", category),
        )
        for entry in extract_date_evidence(str(text or ""))
    ]
    for entry in date_evidence:
        ledger.append({"field": "source_date", "column": entry["column"], "value": entry})
    measurement_evidence = [
        {"column": column, **entry}
        for column, text in (("title", str(sku_name or "")), ("sku_url", url_tokens), ("image_url", img_tokens))
        for entry in extract_volume_evidence(text)
    ]
    for entry in measurement_evidence:
        ledger.append({"field": "measurement", "column": entry["column"], "value": entry})
    pack_evidence = [
        {"column": column, **entry}
        for column, text in (("title", str(sku_name or "")), ("sku_url", url_tokens), ("image_url", img_tokens))
        for entry in extract_pack_evidence(text)
    ]
    if any(entry.get("hierarchy_ambiguous") for entry in pack_evidence):
        consistency_flags.add("pack_hierarchy_ambiguous")
    for entry in pack_evidence:
        ledger.append({"field": "pack_quantity", "column": entry["column"], "value": entry})
    opposing_values = {
        "carbonation": (("carbonated", "still"),),
        "sweetener": (("sugar", "no_sugar"), ("sugar", "diet")),
        "pulp": (("with_pulp", "no_pulp"),),
        "organic": (("organic", "not_organic"),),
    }
    for dimension in ("carbonation", "sweetener", "pulp", "organic"):
        base = set(critical[dimension])
        described = set(description_claims[dimension])
        if not base:
            critical[dimension] = frozenset(described)
        elif described:
            inconsistent = any(
                (left in base and right in described) or (right in base and left in described)
                for left, right in opposing_values[dimension]
            )
            if inconsistent:
                consistency_flags.add(f"description_conflict:{dimension}")
            else:
                critical[dimension] = frozenset(base | described)
    if {"unsweetened", "sweetened"} <= sweeteners["sweetening"]:
        consistency_flags.add("sweetening_status_conflict")
    if "no_added_sugar" in critical["sweetener"] and "cane_sugar" in sweeteners["sweetener_type"]:
        consistency_flags.add("no_added_sugar_with_cane_sugar")
    flavor_set = set(critical["flavor"])
    if flavor_set:
        ledger.append({"field": "flavor", "column": "title+attributes",
                       "value": sorted(flavor_set)})
    # Category tokens may carry flavor evidence
    # (e.g. "Orange Juice" in category path) when title/attributes
    # are silent on flavor.
    cat_flavors = extract_flavor_tokens(cat_tokens) if cat_tokens else frozenset()
    if cat_flavors:
        ledger.append({"field": "flavor", "column": "category",
                       "value": sorted(cat_flavors)})
        flavor_set.update(cat_flavors)
    flavor = sorted(flavor_set)[0] if flavor_set else ""
    for dimension in ("carbonation", "sweetener", "pulp", "organic"):
        if critical[dimension]:
            ledger.append({"field": dimension, "column": "title+attributes",
                           "value": sorted(critical[dimension])})
        if description_claims[dimension]:
            ledger.append({"field": dimension, "column": "description",
                           "value": sorted(description_claims[dimension])})
    # Product type + subtype: config SSOT (config/paths.yaml product_types),
    # read once. Title first, then the category-lane fallback; the subtype
    # (latte, kombucha, ale...) is the finer axis the differentiation lane
    # consumes and is recorded per column like every claim.
    ptype, subtype = _product_type_matcher().match(t)
    if ptype:
        ledger.append({"field": "type", "column": "title", "value": ptype})
    if subtype:
        ledger.append({"field": "subtype", "column": "title", "value": subtype})
    # Fallback: category tokens may carry the product type
    # when the title is too generic (e.g. "Product" with no type word).
    if not ptype and cat_tokens:
        cat_ptype, cat_subtype = _product_type_matcher().match(cat_tokens)
        if cat_ptype:
            ptype = cat_ptype
            subtype = subtype or cat_subtype
            ledger.append({"field": "type", "column": "category", "value": cat_ptype})
            if cat_subtype:
                ledger.append({"field": "subtype", "column": "category", "value": cat_subtype})

    # Volume and pack from title
    vol_title = extract_volume_from_title(sku_name)
    pack_title, pack_conf_title = extract_pack_from_title(sku_name)

    # Attribute parsing
    attr_vol, attr_vol_conf, attr_pack, attr_pack_conf = parse_attribute_volume_pack(
        attribute
    )

    # URL evidence: product tokens from the listing slug.
    # Used when title/attributes are silent on volume/pack.
    vol_url = extract_volume_from_title(url_norm)
    pack_url, pack_conf_url = extract_pack_from_title(url_norm)
    vol_img = extract_volume_from_title(img_norm)
    pack_img, pack_conf_img = extract_pack_from_title(img_norm)

    # Combine: prefer attribute if present, but default to title when
    # the two disagree by 10x+ (title misparses "0, 33l" as 33000ml
    # vs attribute 330ml — the title is the correct unit here).
    title_vol = float(vol_title["volume_ml"] or 0.0)
    if attr_vol > 0:
        ledger.append({"field": "volume_ml", "column": "attributes",
                       "value": attr_vol, "confidence": attr_vol_conf})
    if title_vol > 0:
        ledger.append({"field": "volume_ml", "column": "title", "value": title_vol,
                       "confidence": vol_title["confidence"]})
    if vol_url["volume_ml"] > 0:
        ledger.append({"field": "volume_ml", "column": "sku_url",
                       "value": vol_url["volume_ml"], "confidence": vol_url["confidence"]})
    if vol_img["volume_ml"] > 0:
        ledger.append({"field": "volume_ml", "column": "image_url",
                       "value": vol_img["volume_ml"], "confidence": vol_img["confidence"]})
    if attr_pack > 1 or attr_pack_conf > 0:
        ledger.append({"field": "pack_qty", "column": "attributes",
                       "value": attr_pack, "confidence": attr_pack_conf})
    if pack_title > 1 or pack_conf_title > 0:
        ledger.append({"field": "pack_qty", "column": "title",
                       "value": pack_title, "confidence": pack_conf_title})
    if pack_url > 1 or pack_conf_url > 0:
        ledger.append({"field": "pack_qty", "column": "sku_url",
                       "value": pack_url, "confidence": pack_conf_url})
    if pack_img > 1 or pack_conf_img > 0:
        ledger.append({"field": "pack_qty", "column": "image_url",
                       "value": pack_img, "confidence": pack_conf_img})
    if attr_vol > 0 and title_vol > 0:
        ratio = max(attr_vol, title_vol) / min(attr_vol, title_vol)
        if ratio >= data_cfg().extraction.title_attribute_override_ratio:
            volume_ml = title_vol
            volume_conf = vol_title["confidence"]
            volume_raw = vol_title["raw_match"]
            volume_status = vol_title["parse_status"]
            consistency_flags.add("volume_inconsistency")
        else:
            volume_ml = attr_vol
            volume_conf = attr_vol_conf
            volume_raw = f"attribute: {attr_vol}"
            volume_status = "attribute_volume"
    elif attr_vol > 0:
        volume_ml = attr_vol
        volume_conf = attr_vol_conf
        volume_raw = f"attribute: {attr_vol}"
        volume_status = "attribute_volume"
    elif title_vol > 0:
        volume_ml = title_vol
        volume_conf = vol_title["confidence"]
        volume_raw = vol_title["raw_match"]
        volume_status = vol_title["parse_status"]
    elif vol_url["volume_ml"] > 0:
        volume_ml = vol_url["volume_ml"]
        volume_conf = vol_url["confidence"]
        volume_raw = vol_url["raw_match"]
        volume_status = vol_url["parse_status"]
        consistency_flags.add("volume_from_url")
    elif vol_img["volume_ml"] > 0:
        volume_ml = vol_img["volume_ml"]
        volume_conf = vol_img["confidence"]
        volume_raw = vol_img["raw_match"]
        volume_status = vol_img["parse_status"]
        consistency_flags.add("volume_from_image_url")
    else:
        volume_ml = vol_title["volume_ml"]
        volume_conf = vol_title["confidence"]
        volume_raw = vol_title["raw_match"]
        volume_status = vol_title["parse_status"]
    # Pack qty resolved early for ambiguous_volume check
    if attr_pack > 1 or attr_pack_conf > 0:
        pack_qty = attr_pack
        pack_conf = attr_pack_conf
    elif pack_url > 1 or pack_conf_url > 0:
        pack_qty = pack_url
        pack_conf = pack_conf_url
    elif pack_img > 1 or pack_conf_img > 0:
        pack_qty = pack_img
        pack_conf = pack_conf_img
    else:
        pack_qty = pack_title
        pack_conf = pack_conf_title
    # CORROBORATION FUSION (2026-10-01 ruling): the card's confidence is a
    # property of the CLAIM, not of the winning column — agreeing
    # independent readers pool upward, disagreeing readers cap the card at
    # the weaker one. The winner chain above decides VALUE + precedence;
    # this only changes confidence.
    vol_claims = [
        (value, conf, column)
        for value, conf, column in (
            (attr_vol, attr_vol_conf, "attributes"),
            (title_vol, vol_title["confidence"], "title"),
            (vol_url["volume_ml"], vol_url["confidence"], "sku_url"),
            (vol_img["volume_ml"], vol_img["confidence"], "image_url"),
        )
        if value > 0 and conf > 0
    ]
    pack_claims = [
        (value, conf, column)
        for value, conf, column in (
            (attr_pack, attr_pack_conf, "attributes"),
            (pack_title, pack_conf_title, "title"),
            (pack_url, pack_conf_url, "sku_url"),
            (pack_img, pack_conf_img, "image_url"),
        )
        if value > 0 and conf > 0
    ]
    volume_conf = fuse_confidence(vol_claims)
    pack_conf = fuse_confidence(pack_claims)
    gate_cfg = training_cfg().gate
    if vol_claims and any(
        not volumes_compatible({left[0]}, {right[0]},
                               volume_relative_tolerance=float(gate_cfg.vol_tolerance),
                               volume_absolute_tolerance_ml=float(gate_cfg.vol_abs_tolerance))
        for index, left in enumerate(vol_claims) for right in vol_claims[index + 1:]
    ):
        consistency_flags.add("volume_sources_disagree")
    if pack_claims and len({value for value, _, _ in pack_claims}) > 1:
        consistency_flags.add("pack_sources_disagree")
    # Bounds apply to the selected physical-package size. A count of packages
    # does not make an implausible per-package size legitimate; named bulk
    # containers use the separately configured ceiling.
    extraction_policy = data_cfg().extraction
    bulk_terms = "|".join(re.escape(term).replace(r"\ ", r"\s+") for term in extraction_policy.bulk_container_terms)
    bulk_container = bool(re.search(rf"\b(?:{bulk_terms})\b", f"{sku_name} {attribute}", re.I))
    volume_max = extraction_policy.bulk_volume_max_ml if bulk_container else extraction_policy.volume_max_ml
    if volume_ml > 0 and not extraction_policy.volume_min_ml <= volume_ml <= volume_max:
        consistency_flags.add("ambiguous_volume")

    # BOUNDARY CONTRACT (lib.schemas): the extracted-attribute dict is the
    # input to BOTH the canonical build and the gate — validate the shape
    # once here so a confidence out of [0,1] or a pack_qty < 1 crashes at
    # the transform, not downstream in the gate's comparisons.
    title_attributes = extract_title_attributes(sku_name)
    package_types = title_attributes["package_types"]
    if not package_types:
        package_types = parse_attribute_details(attribute).get("attribute_package_types", [])
    # Title-only, and deliberately so: the raw `attributes` field carries no
    # packaging-level key at all (measured 2026-09-30 — `attributes` holds
    # Volume/Pack Type/Flavour/... and zero case-quantity columns), so the
    # title is the only place this claim exists.
    packaging_levels = extract_packaging_level(sku_name)
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
    universe_evidence = capture_universe_attributes(attribute)
    title_materials = title_attributes["package_materials"]
    material_seen = {value.casefold() for value in title_materials}
    package_materials = list(title_materials) + sorted(
        value
        for value in universe_evidence["pack material type"]
        if value.casefold() not in material_seen
    )
    result = ExtractedAttributes(
        flavor=flavor,
        type=ptype,
        volume_ml=volume_ml,
        volume_confidence=volume_conf,
        volume_raw=volume_raw,
        volume_status=volume_status,
        pack_qty=pack_qty,
        pack_confidence=pack_conf,
        package_types=package_types,
        package_materials=package_materials,
        packaging_levels=packaging_levels,
        flavor_set=flavor_set,
        carbonation_set=set(critical["carbonation"]),
        sweetener_set=set(critical["sweetener"]),
        sweetener_type_set=sweeteners["sweetener_type"],
        sweetening_set=sweeteners["sweetening"],
        attribute_consistency_flags=consistency_flags,
        pulp_set=set(critical["pulp"]),
        organic_set=set(critical["organic"]),
    ).model_dump()
    # The extract dict is a plain dict after the boundary validation, so the
    # evidence section rides ADDITIVELY beside the model dump. Old consumers
    # iterate the named fields, the model channel reads the two wired keys,
    # the census parity test reads the whole section. Sorted lists, never
    # sets — byte-determinism (PYTHONHASHSEED) is the contract here too.
    result["attribute_universe_evidence"] = {
        key: sorted(values) for key, values in universe_evidence.items() if values
    }
    result["evidence_ledger"] = ledger
    result["date_evidence"] = date_evidence
    result["measurement_evidence"] = measurement_evidence
    result["pack_evidence"] = pack_evidence
    result["negated_sweetener_type_set"] = sorted(negative_ingredients)
    return result


# ============================================================================
# CARD SURFACE
# ============================================================================
def surface_card(extracted: dict) -> str:
    """One listing's card as text: the card fill, then evidence one by one.

    ``extracted`` is an extract_all() result dict. The fill is what the card
    actually carries (winners from the precedence chain); the ledger beneath
    repeats EVERY claim any column yielded, winner or loser, with its source.
    """
    lines = [
        f"PRODUCT  {extracted.get('type', '') or '?'}"
        f"  flavor={extracted.get('flavor', '') or '?'}",
        f"VOLUME   {extracted.get('volume_ml', 0.0) or '?'} ml"
        f"  (conf {extracted.get('volume_confidence', 0.0):.2f},"
        f" {extracted.get('volume_status', '')}; raw: {extracted.get('volume_raw', '')})",
        f"PACK     {extracted.get('pack_qty', 1)}"
        f"  (conf {extracted.get('pack_confidence', 0.0):.2f})",
    ]
    for value in extracted.get("package_types") or []:
        if isinstance(value, str) and value:
            lines.append(f"PKG-TYPE {value}")
    for value in extracted.get("package_materials") or []:
        lines.append(f"PKG-MAT  {value}")
    flags = extracted.get("attribute_consistency_flags") or []
    for flag in sorted(flags):
        lines.append(f"FLAG     {flag}")
    lines.append("EVIDENCE (one per source claim)")
    for item in extracted.get("evidence_ledger") or []:
        lines.append(
            f"  [{item['column']}] {item['field']}"
            f" = {item['value']} (conf {item.get('confidence', '')})"
        )
    return "\n".join(lines)


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
    or volume conflict.

    EVIDENCE TRUST (audit 2026-09-15). A parsed attribute is only comparable
    when the parser reported enough confidence to be believed. Passing
    ``trust_threshold`` makes the volume/pack comparison evidence-aware: a
    side below the bar is treated exactly like a missing side, so it stays
    *unknown* and reaches the confidence/fallback lane instead of being
    fabricated into a hard rejection. Callers that leave it ``None`` keep the
    pure structural semantics used by the I/O lanes.

    PACK SEMANTICS: a canonical keeps EVERY pack count observed across its
    titles, so a multi-title canonical legitimately holds ``{12, 24}``. A
    shared count is therefore positive evidence of compatibility and only a
    genuinely disjoint pair conflicts — the rule the canonical writer
    documents ("gate logic intersects them"). Requiring set equality here
    would reject a `{12, 24}` canonical against a `{12}` one that shares 12.
    """
    del score
    veto_dimensions = frozenset(
        training_cfg().rand_matching.targeted_veto_gates.veto_dimensions
    )

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

    def _set(value: object) -> set:
        if value is None or value == "":
            return set()
        if isinstance(value, (set, frozenset, list, tuple)):
            return set(value)
        return {value}

    def _trusted(obj: object, *names: str) -> bool:
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
        raw = _value(obj, *names)
        if raw is None or raw == "":
            return True
        if trust_threshold is None:
            return True
        try:
            value = float(raw)
            return math.isfinite(value) and 0.0 <= value <= 1.0 and value >= float(trust_threshold)
        except (TypeError, ValueError):
            return False

    # PACK COUNT: shared evidence agrees; disjoint counts conflict.
    left_pack = _set(_value(sku_a, "pack_size", "pack_set", "pack_qty"))
    right_pack = _set(_value(sku_b, "pack_size", "pack_set", "pack_qty"))
    if (
        "pack" in veto_dimensions
        and left_pack
        and right_pack
        and not (left_pack & right_pack)
        and _trusted(sku_a, "pack_confidence")
        and _trusted(sku_b, "pack_confidence")
    ):
        return False
    # No count on one side is unknown, not an assertion of single-unit
    # packaging. The caller's confidence/review lane owns missing evidence.

    # PACKAGE TYPE: disjoint categorical evidence conflicts.
    left_type = _set(_value(sku_a, "package_type", "package_type_set"))
    right_type = _set(_value(sku_b, "package_type", "package_type_set"))
    if "package_type" in veto_dimensions and left_type and right_type and not (left_type & right_type):
        return False

    left_volume = (
        set()
        if _has_attribute_flag(sku_a, "ambiguous_volume")
        else _set(_value(sku_a, "volume", "volume_set", "volume_ml"))
    )
    right_volume = (
        set()
        if _has_attribute_flag(sku_b, "ambiguous_volume")
        else _set(_value(sku_b, "volume", "volume_set", "volume_ml"))
    )
    if (
        "volume" in veto_dimensions
        and left_volume
        and right_volume
        and _trusted(sku_a, "volume_confidence")
        and _trusted(sku_b, "volume_confidence")
        and not volumes_compatible(
            left_volume,
            right_volume,
            volume_relative_tolerance=volume_relative_tolerance,
            volume_absolute_tolerance_ml=volume_absolute_tolerance_ml,
        )
    ):
        return False

    def _claim_set(obj: object, dimension: str) -> set[str]:
        explicit = _value(obj, f"{dimension}_set")
        if explicit:
            return _set(explicit)
        found = extract_critical_claims(str(_value(obj, "canonical") or ""))[dimension]
        return set(found)

    for dimension in sorted(veto_dimensions & {"carbonation", "sweetener", "pulp"}):
        if not check_categorical:
            continue
        left = _claim_set(sku_a, dimension)
        right = _claim_set(sku_b, dimension)
        if left and right and categorical_conflict(
            dimension, {dimension: left}, {dimension: right}
        ):
            return False
    return True


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


def three_way_gate(
    attrs1: dict,
    attrs2: dict,
    vol_tolerance: float | None = None,
    raw_conf_threshold: float | None = None,
    consistency_fallback_threshold: float | None = None,
    vol_abs_tolerance: float | None = None,
) -> dict:
    """Deterministic volume/pack/flavor gate.

    NO-FALLBACK SSOT (audit round 2, F01): the decision thresholds live in
    config/training.yaml `gate:` and are read through training_cfg() — the
    old signature defaults (0.05/0.85/0.3) were a second declaration the
    config could not steer. Passing a value explicitly still wins (selftest
    pins known-good gate behavior with explicit values).

    VOLUME TOLERANCE (owner ruling 2026-10-01): BOTH cuts are read from the
    gate block and threaded to every downstream volume comparison — the
    pack_gate call, the inline overlap loop, the critical-7 evaluation and
    the decision engine. The block previously carried only the relative cut,
    so the absolute one stayed at its 0.0 parameter default here while the
    veto lane applied it; since the relative cut is the stricter of the two
    at small volumes, the gate and the veto lane then disagreed about the
    same pair. volumes_compatible applies whichever cut is wider.
    """
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
    _r = training_cfg().gate.reasons
    if not pack_gate(
        0.0,
        attrs1,
        attrs2,
        volume_relative_tolerance=float(vol_tolerance),
        volume_absolute_tolerance_ml=float(vol_abs_tolerance),
        trust_threshold=float(raw_conf_threshold),
        check_categorical=False,
    ):
        return GateResult(
            decision="hard_no",
            reason=_r.pack_blocker,
        ).model_dump()
    veto_dimensions = frozenset(
        training_cfg().rand_matching.targeted_veto_gates.veto_dimensions
    )
    for field, dimension, reason in (
        ("package_type_set", "package_type", _r.package_type_mismatch),
        ("package_material_set", "pack_material", _r.package_material_mismatch),
        ("packaging_level_set", None, _r.packaging_level_mismatch),
    ):
        if dimension is not None and dimension not in veto_dimensions:
            continue
        left, right = set(attrs1.get(field, set())), set(attrs2.get(field, set()))
        if left and right and not (left & right):
            return GateResult(decision="hard_no", reason=reason).model_dump()


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

    left_info, right_info = canonical_attribute_info(attrs1), canonical_attribute_info(attrs2)
    source_flags = _attribute_flags(attrs1) | _attribute_flags(attrs2)
    sweetener_source_conflict = any(
        flag.startswith("sweetener_source_conflict:") for flag in source_flags
    )
    uncertain_categorical_dimensions = {
        flag.split(":", 1)[1] for flag in source_flags
        if flag.startswith(("description_conflict:", "categorical_source_conflict:"))
    }
    if sweetener_source_conflict or source_flags & {
        "unsweetened_with_declared_sweetener", "sweetening_status_conflict"
    }:
        uncertain_categorical_dimensions.add("sweetener")
    # Pulp has no registry key; the registry sweetener key owns ingredient
    # identity, not sugar/no-sugar claims. Preserve these separate explicit
    # claim predicates and report their actual dimensions.
    claim_conflicts = sorted(
        dimension for dimension in (veto_dimensions & {"sweetener", "pulp"}) - uncertain_categorical_dimensions
        if categorical_conflict(dimension, left_info, right_info)
    )
    if claim_conflicts:
        return GateResult(
            decision="hard_no",
            reason=f"{_r.categorical_mismatch} " + ",".join(claim_conflicts),
        ).model_dump()
    categorical_dimensions = veto_dimensions - {
        "volume", "pack", "package_type", "pack_material"
    }
    # The gate consumes only configured categorical verdicts. Per-key engine
    # evaluation is independent, so project the registry before costly raw
    # re-parsing. The census still evaluates the complete universe separately.
    categorical_registry = {
        key: spec for key, spec in attribute_registry().items()
        if CRITICAL_NAME_BY_CENSUS_KEY.get(key) in categorical_dimensions - uncertain_categorical_dimensions
    }
    categorical_conflicts = []
    if categorical_registry:
        evidence = AttributeDecisionEngine(
            volume_relative_tolerance=float(vol_tolerance),
            volume_absolute_tolerance_ml=float(vol_abs_tolerance),
        ).evaluate(
            left_info, right_info, left_raw=attrs1, right_raw=attrs2,
            registry=categorical_registry,
        )
        categorical_conflicts = sorted(
            CRITICAL_NAME_BY_CENSUS_KEY[key] for key in evidence.conflicts
        )
    if sweetener_source_conflict:
        categorical_conflicts = [name for name in categorical_conflicts if name != "sweetener"]
    if categorical_conflicts:
        return GateResult(
            decision="hard_no",
            reason=f"{_r.categorical_mismatch} " + ",".join(categorical_conflicts),
        ).model_dump()

    if uncertain_categorical_dimensions or source_flags & {"volume_sources_disagree", "pack_sources_disagree", "pack_hierarchy_ambiguous"}:
        return GateResult(decision="fallback", reason=_r.source_conflict).model_dump()

    if _has_attribute_flag(attrs1, "ambiguous_volume") or _has_attribute_flag(
        attrs2, "ambiguous_volume"
    ):
        return GateResult(
            decision="fallback", reason=_r.ambiguous_volume
        ).model_dump()
    def _reliable(value: object, threshold: float) -> bool:
        # NaN bypasses ordinary less-than checks; invalid evidence is unknown.
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number) and 0.0 <= number <= 1.0 and number >= threshold

    # raw confidence check
    if (
        not attrs1["volume_set"]
        or not attrs2["volume_set"]
        or not _reliable(attrs1["volume_confidence"], raw_conf_threshold)
        or not _reliable(attrs2["volume_confidence"], raw_conf_threshold)
    ):
        return GateResult(
            decision="fallback", reason=_r.low_volume_confidence
        ).model_dump()
    # Pack confidence: skip when both sides have no pack evidence
    # (single-unit products with no "Count per Unit" in source attributes).
    # pack_gate already treats low-confidence pack evidence as unknown.
    if attrs1["pack_set"] or attrs2["pack_set"]:
        if (
            not attrs1["pack_set"]
            or not attrs2["pack_set"]
            or not _reliable(attrs1["pack_confidence"], raw_conf_threshold)
            or not _reliable(attrs2["pack_confidence"], raw_conf_threshold)
        ):
            return GateResult(
                decision="fallback", reason=_r.low_pack_confidence
            ).model_dump()

    # volume overlap
    vol_overlap = False
    for v1 in attrs1["volume_set"]:
        for v2 in attrs2["volume_set"]:
            if v1 == 0 or v2 == 0:
                continue
            # Same predicate as every other lane (SSOT, audit 2026-09-15):
            # whichever of the two configured cuts is wider applies. The
            # hand-rolled relative-only ratio this replaces disagreed with
            # the veto lane at small volumes.
            if volumes_compatible(
                {v1},
                {v2},
                volume_relative_tolerance=float(vol_tolerance),
                volume_absolute_tolerance_ml=float(vol_abs_tolerance),
            ):
                vol_overlap = True
                break
        if vol_overlap:
            break
    if "volume" in veto_dimensions and not vol_overlap:
        return GateResult(decision="hard_no", reason=_r.no_volume_overlap).model_dump()

    # pack overlap: skip when both sides have no pack evidence
    # (single-unit products with no "Count per Unit" in source).
    if attrs1["pack_set"] or attrs2["pack_set"]:
        pack_overlap = attrs1["pack_set"] & attrs2["pack_set"]
        if "pack" in veto_dimensions and not pack_overlap:
            return GateResult(decision="hard_no", reason=_r.no_pack_overlap).model_dump()

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
    _lvl_a, _lvl_b = set(attrs1.get("packaging_level_set", set())), set(
        attrs2.get("packaging_level_set", set())
    )
    if _lvl_a and not _lvl_b or _lvl_b and not _lvl_a:
        return GateResult(
            decision="fallback",
            reason=_r.packaging_level_review,
        ).model_dump()

    # consistency check
    if (
        not _reliable(attrs1["volume_consistency"], consistency_fallback_threshold)
        or not _reliable(attrs2["volume_consistency"], consistency_fallback_threshold)
        or not _reliable(attrs1["pack_consistency"], consistency_fallback_threshold)
        or not _reliable(attrs2["pack_consistency"], consistency_fallback_threshold)
    ):
        return GateResult(
            decision="fallback", reason=_r.low_consistency
        ).model_dump()

    return GateResult(
        decision="proceed", reason=_r.clean_proceed
    ).model_dump()


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
def extract_discriminative_ngrams(
    titles: list[str],
    attributes: list[str],
    brand_tokens: set[str],
    global_idf: NgramIDF,
    brand_idf: NgramIDF,
    top_k: int = 5,
) -> list[str]:
    """
    Select n‑grams (1‑4) with highest TF‑IDF, considering global and within‑brand IDF.
    """
    # Combine all text into token list
    tokens = []
    phrase_parts = []  # PRE-stopword text: phrase regexes must see 'no',
    # 'with', 'of' — MINIMAL_STOPWORDS deletes them before the keep-token
    # check could ever fire (the live miss on "no sugar"/"free of sugar")
    for title, attr in zip(titles, attributes, strict=True):
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

    candidates = []
    for n in (1, 2, 3, 4):
        candidates.extend(generate_ngrams(tokens, n))

    if not candidates:
        return []

    tf = Counter(candidates)
    total = len(candidates)

    # Number of GTINs in the brand
    N_brand = brand_idf.N if brand_idf else 1

    scores = {}
    for ngram, count in tf.items():
        tf_val = count / total if total else 0
        g_idf = global_idf.idf(ngram)
        b_idf = brand_idf.idf(ngram) if brand_idf else 1.0

        # Strong penalty for n‑grams present in ALL brand GTINs (not discriminative)
        if brand_idf:
            df_brand = brand_idf.df.get(ngram, 0)
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
    selected = [ngram.replace(" ", "_") for ngram, _ in sorted_ngrams[:top_k]]

    # Add KEEP_TOKENS that appear in the document but may not be top.
    # Compound keepers ('no_sugar', 'with_pulp') are stored underscore-joined
    # and used to be checked against SPACE-joined doc text — they could never
    # match (dead entries). Check the compound's WORDS as a contiguous bigram
    # instead ('no' is stopworded away, so 'sugar_free' matches 'sugar free').
    doc_tokens = tokens
    bigrams = {
        f"{doc_tokens[i]}_{doc_tokens[i + 1]}" for i in range(len(doc_tokens) - 1)
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
            hit = (keep in doc_tokens) if "_" not in keep else (keep in bigrams)
        if hit and keep not in selected:
            selected.append(keep)
            if len(selected) >= top_k + 3:
                break

    return selected[: top_k + 3]


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
    category_paths: list[str] | None = None,
    categories: list[str] | None = None,
    countries: list[str] | None = None,
    retailers: list[str] | None = None,
) -> dict:  # CanonicalRecord.model_dump() — validated shape, plain dict
    titles = [sku for sku, attr in rows]
    attributes = [attr for sku, attr in rows]
    descriptions = descriptions or [""] * len(rows)
    urls = urls or [""] * len(rows)
    image_urls = image_urls or [""] * len(rows)
    category_paths = category_paths or [""] * len(rows)
    categories = categories or [""] * len(rows)
    countries = countries or [""] * len(rows)
    retailers = retailers or [""] * len(rows)
    extracted = [
        extract_all(
            sku, attr,
            "" if pd.isna(desc) else str(desc),
            url, img_url, cat_path, cat,
        )
        for (sku, attr), desc, url, img_url, cat_path, cat
        in zip(rows, descriptions, urls, image_urls, category_paths, categories, strict=True)
    ]

    brand_norm = normalize_text(spell_numeric_brand(brand))
    brand_tokens = set(brand_norm.split())

    flavors = [x["flavor"] for x in extracted if x["flavor"]]
    types = [x["type"] for x in extracted if x["type"]]
    mode_flavor = Counter(flavors).most_common(1)[0][0] if flavors else ""
    mode_type = Counter(types).most_common(1)[0][0] if types else ""

    # Get discriminative n‑grams
    salient_ngrams = extract_discriminative_ngrams(
        titles, attributes, brand_tokens, global_idf, brand_idf, top_k=5
    )

    # Volume and pack sets
    volume_set = {round(x["volume_ml"], 2) for x in extracted if x["volume_ml"] > 0}
    # A parser-safe quantity of one is not evidence of a single-item pack.
    # Keep only rows with explicit pack evidence in the canonical attribute
    # set; otherwise missing pack data becomes a false pack conflict.
    pack_set = {
        x["pack_qty"] for x in extracted if x["pack_confidence"] > 0
    }
    package_type_set = {value for x in extracted for value in x["package_types"]}
    packaging_level_set = {value for x in extracted for value in x["packaging_levels"]}
    package_material_set = {value for x in extracted for value in x["package_materials"]}
    flavor_set = {value for x in extracted for value in x["flavor_set"]}
    carbonation_set = {value for x in extracted for value in x["carbonation_set"]}
    sweetener_set = {value for x in extracted for value in x["sweetener_set"]}
    sweetener_type_set = {value for x in extracted for value in x["sweetener_type_set"]}
    sweetening_set = {value for x in extracted for value in x["sweetening_set"]}
    attribute_consistency_flags = {value for x in extracted for value in x["attribute_consistency_flags"]}
    # Negations can be on a different listing of the same GTIN from the
    # affirmative ingredient. Preserve that contradiction at aggregation.
    negative_ingredients = {
        value for x in extracted for value in x.get("negated_sweetener_type_set", ())
    }
    attribute_consistency_flags.update(
        f"sweetener_source_conflict:{ingredient}"
        for ingredient in negative_ingredients & sweetener_type_set
    )
    pulp_set = {value for x in extracted for value in x["pulp_set"]}
    organic_set = {value for x in extracted for value in x.get("organic_set") or set()}

    for dimension, values, opposites in (
        ("sweetener", sweetener_set, (("sugar", "no_sugar"), ("sugar", "diet"))),
        ("carbonation", carbonation_set, (("still", "carbonated"),)),
        ("pulp", pulp_set, (("no_pulp", "with_pulp"),)),
        ("organic", organic_set, (("organic", "not_organic"),)),
    ):
        if any({left, right} <= values for left, right in opposites):
            attribute_consistency_flags.add(f"categorical_source_conflict:{dimension}")

    # Confidence / consistency
    vol_confs = [x["volume_confidence"] for x in extracted if x["volume_ml"] > 0]
    pack_confs = [x["pack_confidence"] for x in extracted if x["pack_confidence"] > 0]
    vol_conf = sum(vol_confs) / len(vol_confs) if vol_confs else 0.0
    pack_conf = sum(pack_confs) / len(pack_confs) if pack_confs else 0.0
    n = len(extracted)
    # consistency = share of rows agreeing with the MOST COMMON value.
    # The old formula divided conflicts by ROW COUNT n, so a 41k-row group
    # with 2,000 distinct volumes scored 0.95 "consistent" — more rows made
    # contradiction look BETTER. Mode-share is scale-free and monotone.
    vol_mode = Counter(x["volume_ml"] for x in extracted if x["volume_ml"] > 0)
    pack_mode = Counter(
        x["pack_qty"] for x in extracted if x["pack_confidence"] > 0
    )
    # mode share over rows that HAVE a volume (unknown-volume rows don't vote)
    volume_consistency = (
        (vol_mode.most_common(1)[0][1] / sum(vol_mode.values())) if vol_mode else 1.0
    )
    n_known_pack = sum(pack_mode.values())
    pack_consistency = (
        pack_mode.most_common(1)[0][1] / n_known_pack
        if n_known_pack
        else 1.0
    )

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
    parts = [brand_norm]
    if mode_flavor:
        parts.append(mode_flavor)
    if mode_type:
        parts.append(mode_type)
    # Explicit categorical fields are spoken before free-form n-grams. This
    # guarantees that polarity survives canonical generation even when its
    # source phrase is not among the top TF-IDF n-grams.
    parts.extend(sorted(carbonation_set))
    parts.extend(sorted(sweetener_set))
    parts.extend(sorted(pulp_set))
    parts.extend(sorted(flavor_set - ({mode_flavor} if mode_flavor else set())))
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
    for ng in salient_ngrams:
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
    if not kept_ngrams and salient_ngrams:
        for ng in salient_ngrams:
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
            kept_ngrams = [salient_ngrams[0]]
            spoken.update(
                w for w in salient_ngrams[0].split("_") if w
            )
    # raw list kept for the strip-audit visibility (what token-once removed)
    raw_salient_ngrams = list(salient_ngrams)
    salient_ngrams = kept_ngrams
    parts.extend(salient_ngrams)
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
    canonical = " ".join(final_tokens)

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
    for attr in attributes:
        parsed = parse_universe_cell(attr)
        parsed.pop("unclassified_keys", None)
        parsed.pop("volume", None)
        for key, values in parsed.items():
            if values:
                canonical_universe_evidence.setdefault(key, set()).update(
                    str(token) for token in values
                )
    universe_evidence_json = json.dumps(
        {key: sorted(values) for key, values in sorted(canonical_universe_evidence.items())}
    )

    # BOUNDARY CONTRACT (lib.schemas): one validated record per canonical.
    # brand NaN-guard: a group whose brand column is all-NaN would carry a
    # float NaN into mode_brand (pandas would write ""), which pydantic's
    # str field would coerce to "nan" — the exact title-poisoning bug class
    # the lane fixed for titles. Clean it here so the RECORD is honest.
    brand_clean = brand if isinstance(brand, str) else ""
    rec = CanonicalRecord(
        gtin=gtin,
        canonical=canonical,
        mode_brand=brand_clean,
        mode_flavor=mode_flavor,
        mode_type=mode_type,
        salient_ngrams=salient_ngrams,
        dropped_redundant_ngrams=[
            ng for ng in raw_salient_ngrams if ng not in set(kept_ngrams)
        ],
        # NOTE: kept as SETS here — gate logic intersects them (pack_set &
        # pack_set). data_prep sorts them AT THE CSV WRITE so the display
        # is deterministic (PYTHONHASHSEED-proof) without touching logic.
        volume_set=volume_set,
        pack_set=pack_set,
        packaging_level_set=packaging_level_set,
        package_type_set=package_type_set,
        package_material_set=package_material_set,
        flavor_set=flavor_set,
        carbonation_set=carbonation_set,
        sweetener_set=sweetener_set,
        sweetener_type_set=sweetener_type_set,
        sweetening_set=sweetening_set,
        attribute_consistency_flags=attribute_consistency_flags,
        pulp_set=pulp_set,
        organic_set=organic_set,
        volume_confidence=round(vol_conf, 3),
        pack_confidence=round(pack_conf, 3),
        volume_consistency=round(volume_consistency, 3),
        pack_consistency=round(pack_consistency, 3),
        n_titles=n,
    )
    # Additive persistence key (post-dump, like description_evidence before
    # it was a model field): the canonical record model stays extra='forbid'
    # for its gate-facing fields; the universe evidence rides the CSV
    # contract next to them as the one rendered JSON string.
    out = rec.model_dump()
    out["universe_evidence"] = universe_evidence_json
    # GTIN CARD (evidence ledger, one per listing): every claim any of the
    # gtin's listing cards recorded, with listing origin kept so the surface
    # can walk a card listing by listing. Ordered PER ATTRIBUTE — sorted by
    # (field, source, value) via whole-entry JSON — deterministic across
    # hash seeds; exact repeats (two listings extracting the identical
    # claim) collapse via the same serialization.
    json_entries = [
        json.dumps({"listing": listing_index, **entry}, sort_keys=True)
        for listing_index, per_listing in enumerate(extracted)
        for entry in (per_listing.get("evidence_ledger") or [])
    ]
    out["evidence_ledger"] = json.dumps(
        [json.loads(e) for e in sorted(dict.fromkeys(json_entries))]
    )
    return out


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
    would silently capture every barcode-less row in the payload space.
    """
    df = pd.read_csv(
        RESULTS / F["canonical_records"],
        dtype=str,
        keep_default_na=False,
    )
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
# ── decade/short decade tails: 50s, 3h, 6m, 4er, 1c, 10liters, 12shots ──────
_NUM_WORD_RE = re.compile(r"^\d+(?:\.\d+)?[a-z]{0,4}$", re.IGNORECASE)


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


def census_texts(df: pd.DataFrame) -> list[str]:
    """Pre-number-strip sku texts (the census input): the official cleaning
    WITHOUT the final number-token strip, so every digit token in the corpus
    appears in the reference."""
    out = []
    for t, a in zip(df["title"].fillna(""), df["attributes"].fillna(""), strict=True):
        text = normalize_text(t) + " " + normalize_text(a or "")
        text = _VOLUME_PACK_RE.sub(" ", text)
        toks = [x for x in text.split() if x not in MINIMAL_STOPWORDS and len(x) > 1]
        out.append(" ".join(toks))
    return out


def build_reference(texts: list[str], brand_vocab: set[str]) -> pd.DataFrame:
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


def reference_path() -> Path:
    """DATA_DIR / F["number_reference"] — the token-verdict CSV (SSOT)."""
    return DATA_DIR / F["number_reference"]


_VERDICTS_CACHE: dict[str, str] | None = None
_VERDICTS_LOADED = False
# AUDIT 2026-09-09: process-lifetime count of digit tokens that fell through
# to the regex fallback because the reference CSV did not carry them —
# printed at data-prep exit so the degradation is visible, not silent.
_UNSEEN_TOKEN_TOTAL = 0


def load_verdicts() -> dict[str, str] | None:
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
    """Remove number tokens from an already-clean sku text.

    `brand` is the ROW's brand string: numeric brand tokens ("28" in
    "28 Black") survive only when that number appears in this row's own
    brand; the same number in another row's title (e.g. a 24-pack of a
    different brand) is stripped. A digit token that IS this row's spelled
    numeric brand (1724 → seventeen) is replaced by the spelled form.
    Everything else resolves via the reference CSV (SSOT) with the
    regex-rule fallback.
    """
    if not re.search(r"\d", text):
        return text
    verdicts = load_verdicts()
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
        global _UNSEEN_TOKEN_TOTAL
        _UNSEEN_TOKEN_TOTAL += n_unseen
    return " ".join(out)


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


def run_within_brand_pipeline(
    df_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # (gate results, canonical records)
    # Preserve the evidence-bearing source fields as canonical-level inputs.
    #  They remain OUTSIDE the frozen canonical
    # text until a component-safe ablation establishes their value.
    for column in (
        "description_short_eng", "breadcrumbs_eng",
        "sku_url", "image_url", "category", "country", "retailer",
    ):
        if column not in df_full:
            df_full = df_full.assign(**{column: ""})

    def _source_evidence(values: pd.Series) -> list[str]:
        """Deterministic, non-empty raw strings for review/feature ablation."""
        return sorted(
            {
                str(value).strip()
                for value in values
                if pd.notna(value) and str(value).strip()
            }
        )

    # ── CONSOLIDATED TRACE: stage 1 writer ────────────────────────────────
    # ONE writer for the whole stage, committed once at the end of the
    # function. The first row records the COLUMN CONTRACT this frame arrived
    # with — the raw export's names — because the stage that follows consumes a
    # different contract (see build_training_data) and that handoff used to be
    # invisible until a KeyError fired somewhere downstream.
    from core.tracing import TraceRun

    trace = TraceRun("data_prep")
    trace.add_column_contract(
        df_full,
        contract="raw_export (core.common.load_raw_export)",
        required=RAW_EXPORT_REQUIRED_COLUMNS,
        note=(
            "stage 2 (build_training_data) does NOT consume this frame: it "
            "reloads the deduped dataset through load_dataset_deduped(), whose "
            "columns are the canonical ones (barcode/title/attributes) — the "
            "two stages meet at canonical_records.csv + gate_results.csv"
        ),
    )

    # NaN/empty GTINs must NOT form a group: 41,545 rows (58% of the corpus)
    # share gtin=NaN and used to collapse into ONE canonical record with an
    # arbitrary mode-brand — poisoning canonical_records.csv AND the global
    # IDF every other GTIN was scored against. Drop them explicitly.
    from core.identity_policy import apply_identity_links
    df_full = apply_identity_links(df_full)
    n_before = len(df_full)
    gtin_valid = (
        df_full["gtin"].notna()
        & (df_full["gtin"].astype(str).str.strip() != "")
        & (df_full["gtin"].astype(str).str.lower() != "nan")
    )
    # Checksum enforcement (owner ruling): 1,747 of 14,997 distinct barcodes
    # (3,715 rows) FAIL the GS1 check digit — retailer-export noise. An
    # invalid barcode must not assert product identity: no canonical forms
    # on it, so no (sku, canonical) positive pairs and no false labels leak
    # into training/eval. The ROWS survive (corpus unchanged); only the
    # identity claim dies. Loud per lane doctrine — never silent.
    from core.gtin import barcode_validity

    bc_valid = barcode_validity(df_full["gtin"].fillna("").astype(str).str.strip())
    from core.identity_policy import reviewed_row_mask
    reviewed = reviewed_row_mask(df_full)
    bc_valid &= ~reviewed
    checksum_bad = gtin_valid & ~bc_valid & ~reviewed
    n_checksum_dropped = int(checksum_bad.sum())
    df_full = df_full[gtin_valid & bc_valid]
    if reviewed.any():
        print(f"[gtin-guard] excluded {int(reviewed.sum()):,} identity-review rows (GLN or unresolved formulation)", flush=True)
    if n_checksum_dropped:
        print(
            f"[gtin-guard] dropped {n_checksum_dropped:,} rows whose gtin "
            f"FAILS the GS1 check digit (no canonical/labels form on a "
            f"barcode that cannot be trusted as identity)",
            flush=True,
        )
    if n_before != len(df_full):
        print(
            f"[gtin-guard] total dropped {n_before - len(df_full):,} rows "
            f"(missing/NaN gtin or failed checksum) — they cannot be "
            f"grouped by product",
            flush=True,
        )
    # CONSOLIDATED TRACE (§gtin-guard): the guard is where identity dies, so
    # both populations are recorded with the reason that removed them.
    trace.add(
        "gtin_guard",
        "identity_claims_evaluated",
        in_count=n_before,
        out_count=len(df_full),
        reason="rows keep identity only with a present, GS1-valid barcode",
        detail={
            "gtin_missing_or_nan": int((~gtin_valid).sum()),
            "gs1_checksum_failed": n_checksum_dropped,
            "identity_review_quarantined": int(reviewed.sum()),
            "rows_retained": int(len(df_full)),
        },
        source="raw export",
    )

    # Group by GTIN
    grouped = (
        df_full.groupby("gtin")
        .agg(
            rows=(
                "sku_name_eng",
                lambda x: list(zip(x, df_full.loc[x.index, "attribute"], strict=True)),
            ),
            descriptions=("description_short_eng", list),
            urls=("sku_url", list),
            image_urls=("image_url", list),
            category_paths=("breadcrumbs_eng", list),
            categories=("category", list),
            countries=("country", list),
            retailers=("retailer", list),
            brand=("brand", lambda x: Counter(x).most_common(1)[0][0]),
            description_evidence=("description_short_eng", _source_evidence),
            breadcrumb_evidence=("breadcrumbs_eng", _source_evidence),
            source_rows=("sku_name_eng", lambda x: _source_rows_for(
                df_full.loc[x.index]
            )),
        )
        .reset_index()
    )

    # Build global n‑gram IDF from all GTINs
    rows_by_gtin = {row["gtin"]: row["rows"] for _, row in grouped.iterrows()}
    global_idf = NgramIDF(rows_by_gtin)

    # Precompute within‑brand IDF per brand
    brand_to_gtins = defaultdict(list)
    for gtin, brand in zip(grouped["gtin"], grouped["brand"], strict=True):
        brand_to_gtins[brand.lower().strip()].append(gtin)

    # For each brand, build an IDF from that brand's GTINs
    brand_idf_map = {}
    for brand, gtins in brand_to_gtins.items():
        brand_rows = {gtin: rows_by_gtin[gtin] for gtin in gtins}
        brand_idf_map[brand] = NgramIDF(brand_rows)

    # Generate canonical records
    canonical_records = []
    from tqdm import tqdm
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), unit="gtin", desc="cards", disable=None):
        brand_key = row["brand"].lower().strip()
        record = generate_canonical(
            row["gtin"],
            row["brand"],
            row["rows"],
            global_idf,
            brand_idf_map[brand_key],
            descriptions=row["descriptions"],
            urls=row["urls"],
            image_urls=row["image_urls"],
            category_paths=row["category_paths"],
            categories=row["categories"],
        )
        record["description_evidence"] = row["description_evidence"]
        record["breadcrumb_evidence"] = row["breadcrumb_evidence"]
        # Per-title original evidence, carried so the engine's stage-7
        # clarification can reach the real columns (see _source_rows_for).
        record["source_rows"] = row["source_rows"]
        canonical_records.append(record)
    df_canon = pd.DataFrame(canonical_records)

    # ── CONSOLIDATED TRACE: the row identity closes here ──────────────────
    # Every GS1-valid row is either promoted to its gtin's canonical record or
    # collapsed into it (kept and aggregated — a distinct destiny from the
    # guard's two drop populations). With this row the trace alone closes
    #   rows_in == canonical_records + collapsed_same_gtin
    #              + gtin_missing_or_nan + gs1_checksum_failed
    # which is what core.tracing.accounting() recomputes from the file.
    trace.add(
        "canonical",
        "records_built",
        in_count=len(df_full),
        out_count=len(df_canon),
        reason=(
            "one canonical record per distinct GS1-valid gtin; the other rows "
            "collapse into their own gtin's record (kept and aggregated, not "
            "dropped)"
        ),
        detail={
            "distinct_gtins": int(len(df_canon)),
            "collapsed_same_gtin": int(len(df_full) - len(df_canon)),
            "brands": int(grouped["brand"].nunique()) if len(grouped) else 0,
            "brands_with_pairs": int(
                sum(1 for gtins in brand_to_gtins.values() if len(gtins) > 1)
            ),
        },
        source="raw export",
    )

    # ── CONSOLIDATED TRACE: canonical-side universe-evidence census ──────
    # Closure evidence for the wiring gap this column closes: how many
    # canonicals carry ANY universe evidence, per registered key. Audit
    # readback only — no decision reads this row.
    if len(df_canon):
        from core.attribute_conflicts import _universe_evidence_of

        _evid = [ _universe_evidence_of(row) for row in df_canon.to_dict("records") ]
        _per_key: Counter[str] = Counter()
        for evidence in _evid:
            for key in evidence:
                _per_key[key] += 1
        trace.add(
            "canonical",
            "universe_evidence_census",
            in_count=int(len(df_canon)),
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

    # ── CONSOLIDATED TRACE: attribute-gate evidence sections (owner ruling
    # 2026-10-01, "ALL ATTRIBUTES are used to make ALL DECISIONS") ──────────
    # The registry census (results/attribute_universe_census.json) measured
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

    trace.add(
        "attribute_gate",
        "universe_decision_scope",
        reason="every AttributeUniverse-registered dimension enters pair-level evaluation; absence never vetoes",
        detail=attribute_gate_universe_scope_detail(),
        source="core.attribute_universe census artifact",
    )
    trace.add(
        "attribute_gate",
        "veto_eligibility",
        reason="per-dimension veto-eligibility ledger: evidence class + CURRENT config state + the exact owner delta",
        detail={"ledger": veto_eligibility_ledger()},
        source="core.attribute_universe census + config/training.yaml (both read-only)",
    )

    # Brand blocking
    candidate_pairs = set()
    for brand, gtins in brand_to_gtins.items():
        if len(gtins) < 2:
            continue
        for i in range(len(gtins)):
            for j in range(i + 1, len(gtins)):
                candidate_pairs.add((gtins[i], gtins[j]))

    # Gate and similarity
    gtin_to_canon = {row["gtin"]: row for _, row in df_canon.iterrows()}
    results = []
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

    gate_vis = []
    from tqdm import tqdm as _tqdm_pairs
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
    for g1, g2 in _tqdm_pairs(sorted(candidate_pairs), unit="pair", desc="gate", disable=None):
        a1 = gtin_to_canon[g1]
        a2 = gtin_to_canon[g2]
        gate = three_way_gate(a1, a2)
        # Jaccard on the SHORT (non-compound) tokens of the canonical —
        # brand + flavour + type semantics. On the FULL string the
        # discriminative-ngram compounds (kr_white_grape_flavored...) almost
        # never match across titles, crushing the distribution (14/44,530
        # proceed pairs >= 0.8 vs 7,929 on the short form — measured
        # 2026-09-06). The compounds stay in the canonical for other uses.
        sim = jaccard_similarity(
            " ".join(t for t in str(a1["canonical"]).split() if "_" not in t),
            " ".join(t for t in str(a2["canonical"]).split() if "_" not in t),
        )
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
        results.append(
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
        gate_vis.append(
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
    results_df = pd.DataFrame(results)

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
        "carbonation_set",
        "sweetener_set",
        "sweetener_type_set",
        "sweetening_set",
        "attribute_consistency_flags",
        "pulp_set",
    ):
        df_canon[_col] = df_canon[_col].map(lambda s: sorted(s))
    # Same for the pair ROW ORDER: candidate_pairs is a SET, so iteration
    # order is process-random. Gate decisions themselves are order-free —
    # only the CSV row sequence drifted. Sort on the identity columns.
    results_df = results_df.sort_values(
        ["gtin1", "gtin2"], kind="stable"
    ).reset_index(drop=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # FRAME CONTRACTS (lib.schemas): column sets, decision domain, similarity
    # bounds, GTIN endpoints — asserted at the WRITE boundary so a corrupted
    # transform can never land in the CSVs every downstream step reads.
    require_populated_source_rows(check_canonical_records_frame(df_canon))
    check_gate_results_frame(results_df)
    # SILENT_DROPS task 6: every CSV write goes through the atomic
    # mechanism (tmp sibling + fsync + rename) so an interrupt can never
    # leave a truncated artifact for downstream steps to read.
    from core.manifest import atomic_write_csv

    atomic_write_csv(df_canon, RESULTS / F["canonical_records"], index=False)
    atomic_write_csv(results_df, RESULTS / F["gate_results"], index=False)
    # ── CONSOLIDATED TRACE: gate stage ─────────────────────────────────────
    # One CSV carries the whole story: run-scope funnels (candidate census →
    # decision census → complete reason census), one exact group row per
    # (decision, reason) bucket, then a bounded stratified SAMPLE of pairs with
    # the literal readback. Every pair's decision and reason is counted exactly
    # in the census rows; the sample exists so the evidence can be eyeballed
    # without opening gate_results.csv. Replaces the former
    # results/logs/gate_visibility.csv.
    gate_frame = (
        pd.DataFrame(gate_vis).sort_values(["gtin1", "gtin2"], kind="stable")
        if gate_vis
        else pd.DataFrame()
    )
    vis_counts = (
        gate_frame["decision"].value_counts().to_dict() if len(gate_frame) else {}
    )
    vis_reasons = (
        gate_frame["reason"].value_counts().to_dict() if len(gate_frame) else {}
    )
    trace.add(
        "gate",
        "candidates_gated",
        in_count=len(candidate_pairs),
        out_count=len(results_df),
        reason="every same-brand pair receives exactly one decision; none is dropped",
        detail={"decisions": {str(k): int(v) for k, v in vis_counts.items()}},
        source="canonical_records.csv (in-memory frame)",
    )
    # One group row per decision: its EXACT population plus the complete reason
    # distribution inside it (count_rows with no limit — the label set is small
    # and bounded, so "which pairs got which decision and why" is answered here
    # rather than by opening gate_results.csv).
    for decision in ("hard_no", "fallback", "proceed"):
        subset = gate_frame[gate_frame["decision"] == decision] if len(gate_frame) else gate_frame
        trace.add(
            "gate",
            f"decision_{decision}",
            scope="group",
            in_count=len(candidate_pairs),
            out_count=int(len(subset)),
            reason=f"gate_decision == {decision}",
            detail={
                "reasons": count_rows(subset["reason"]) if len(subset) else [],
                "reason_census": (
                    count_rows(subset["reason"], limit=None) if len(subset) else []
                ),
            },
            source="gate_results.csv",
        )
    # FULL-ATTRIBUTES pair census rollup (owner ruling 2026-10-01): a
    # run-scope row over the whole gated population states exactly how many
    # pairs carried at least one recorded dimension conflict and which
    # dimensions are the loud ones — per-pair detail rides the sampled
    # pair_decision rows (bounded sample, see core.tracing) and this row
    # carries the exact counts.
    if len(gate_frame):
        conflicts = gate_frame["dimension_conflicts"].astype(str)
        trace.add(
            "attribute_gate",
            "pair_dimension_census",
            scope="group",
            in_count=int(len(gate_frame)),
            out_count=int((conflicts != "").sum()),
            reason="pairs with at least one recorded dimension conflict (absence stays unknown, never minted)",
            detail={
                "pairs": int(len(gate_frame)),
                "conflict_paired": count_rows(conflicts[conflicts != ""], limit=None),
                "no_conflict": int((conflicts == "").sum()),
            },
            source="gate_stage in-memory readback",
        )
    if len(gate_frame):
        # Named `reason_census`, NOT `decision_reasons`: every group step
        # starting with "gate.decision_" is a decision bucket and is summed by
        # core.tracing.accounting(), so this cross-decision row must not share
        # that prefix.
        trace.add(
            "gate",
            "reason_census",
            scope="group",
            in_count=int(len(gate_frame)),
            out_count=int(len(vis_reasons)),
            reason="complete reason census over every decision, not just hard_no",
            detail={
                "reasons": count_rows(gate_frame["reason"], limit=None),
                "pairs": int(len(gate_frame)),
            },
            source="gate_results.csv",
        )
        # Full per-pair readback: exactly what the gate SAW on both sides
        # (volume/pack/package sets + their confidences and consistency) next
        # to what it DECIDED and the similarity downstream mining bands on.
        # The inputs are JSON so one cell stays machine-readable. Bucketed by
        # (decision :: reason) so the census rows above and the sampled rows
        # below join on the same label.
        trace.add_entities(
            "pair_decision",
            list(gate_frame.itertuples(index=False)),
            key_of=lambda r: f"{r.gtin1}|{r.gtin2}",
            reason_of=lambda r: f"{r.decision} :: {r.reason}",
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
    trace.write()
    print(
        f"[trace] data_prep steps written -> {trace_path()} | "
        f"gate decisions: {vis_counts}",
        flush=True,
    )

    return results_df, df_canon


# ============================================================================
# PAIRS
# ============================================================================
def build_training_data(
    df: pd.DataFrame,
    *,
    payload_variant: str = "full",
) -> dict:
    """Build payload + pos/neg pairs from the deduped dataset + gate results.

    Returns dict with:
        payload : list[str]  — clean sku text per row + one canonical per GTIN
        structured_features : list[list[float]] — normalized numeric features
        row_bc  : np.ndarray — barcode per payload entry (gtin for canonicals)
        pos     : np.ndarray (N,2) — (sku_row, canon_idx) for every row whose
                  barcode has a canonical
        neg     : np.ndarray (M,2) — (rep_row(g1), canon(g2)) and mirror, for
                  every gate hard-no pair with similarity >= threshold
        stats   : dict — counts (nothing dropped silently)
    """
    from core.identity_policy import exclude_reviewed_rows
    df = exclude_reviewed_rows(df).reset_index(drop=True)
    print(
        f"[payload-stage] building variant={payload_variant} rows={len(df):,}",
        flush=True,
    )
    # ── CONSOLIDATED TRACE: stage 2 writer ────────────────────────────────
    # Created here so the column contract of the frame THIS stage received is
    # the first row of the stage — see run_within_brand_pipeline for the other
    # half of the two-stage handoff.
    from core.tracing import TraceRun

    trace = TraceRun("pairs")
    trace.add_column_contract(
        df,
        contract="canonical dataset (core.common.load_dataset_deduped)",
        required=CANONICAL_DATASET_REQUIRED_COLUMNS,
        note=(
            "stage 1 (run_within_brand_pipeline) consumes the RAW export "
            "contract (gtin/sku_name_eng/attribute) — the two stages are joined "
            "by canonical_records.csv + gate_results.csv, never by passing this "
            "frame between them"
        ),
    )
    cfg = load_config()
    structured_cfg = cfg["training"]["structured_features"]
    structured_enabled = bool(structured_cfg["enabled"])
    from core.model_input import (
        build_canonical_text,
        build_sku_texts,
        model_input_info,
        model_input_composition,
        token_budget_report,
    )
    from core.structured_features import (
        canonical_info as canonical_structured_info,
        vector as structured_vector,
    )

    # The encoder text this stage materializes is an INPUT CONTRACT for every
    # downstream artifact (embeddings, ANN index, checkpoints, reports), so the
    # active composition is recorded on the run before any text is built — a
    # reader can then tell which composition produced what, after the fact.
    trace.add(
        "payload",
        "model_input_composition",
        detail=model_input_composition().model_dump(),
    )
    thr_pos = float(cfg["pairs"]["proceed_sim_threshold"])
    thr_neg = float(cfg["pairs"]["hardneg_sim_threshold"])

    canon_map = load_canonical_map()
    gates = pd.read_csv(
        RESULTS / F["gate_results"],
        dtype={"gtin1": str, "gtin2": str},
        keep_default_na=False,
    )

    bc = df["barcode"].fillna("").astype(str).str.strip()
    title = df["title"].fillna("")
    attrs = df["attributes"].fillna("")

    # ── clean sku text per row (variant: full = title+attr, title_only) ──
    # schema words (type/content/material/...) die on the MODEL side only —
    # the gate's inputs are untouched (owner 2026-09-07: stage-2 strip).
    # Both variants go through core.model_input, the shared builder.
    # The per-row composition loop is the SSOT core.model_input.build_sku_texts
    # (was inlined here and in predict_items / rand_matching / record_linkage).
    if payload_variant == "full":
        model_frame = df
    elif payload_variant == "title_only":
        model_frame = df.copy()
        for column in ("attributes", "attr", "description", "description_short_eng"):
            if column in model_frame.columns:
                model_frame[column] = ""
    else:
        raise SystemExit(f"unknown payload variant: {payload_variant}")
    sku_texts, sku_structured = build_sku_texts(
        model_frame, structured_enabled=structured_enabled
    )

    # ── payload: sku rows + canonical entries (in sorted-gtin order) ──
    payload = list(sku_texts)
    row_bc = [str(x) for x in bc]
    canon_gtins = sorted(canon_map)
    canon_start = len(payload)
    gtin_to_canon_idx = {g: canon_start + i for i, g in enumerate(canon_gtins)}
    # MODEL payload: schema-free canonical variant plus normalized structured
    # volume/pack/package-type tokens. The gate's CSV keeps the original values and schema
    # labels for decisions; the model receives the stable normalized tokens
    # explicitly so those attributes are no longer discarded.
    canonical_records = pd.read_csv(
        RESULTS / F["canonical_records"], dtype={"gtin": str}, keep_default_na=False
    )
    canonical_record_map = {
        str(row["gtin"]): row.to_dict()
        for _, row in canonical_records.iterrows()
    }
    canon_structured = [
        model_input_info(canonical_structured_info(canonical_record_map.get(g, {})))
        if structured_enabled
        else {"volume": set(), "pack": set(), "package_type": set()}
        for g in canon_gtins
    ]
    canon_texts = [
        build_canonical_text(canonical_record_map.get(g, {}), info)
        for g, info in zip(canon_gtins, canon_structured, strict=True)
    ]
    payload.extend(canon_texts)

    # The structured tail is appended LAST, so at max_seq_length it is the
    # first thing truncated. Measure the assembled payload and record it, so a
    # dropped field group is a named number in the run trace rather than an
    # invisible shortening. Reuses the tracing SSOT and the config SSOT.
    from transformers import AutoTokenizer

    from core.common import resolve_model, runtime

    budget = token_budget_report(
        payload,
        tokenizer=AutoTokenizer.from_pretrained(
            str(resolve_model(str(runtime("base_model"))))
        ),
        max_seq_length=int(runtime("max_seq_length")),
    )
    trace.add("payload", "token_budget", detail=budget.model_dump())
    if budget.n_field_groups_dropped:
        print(
            f"    [token-budget] WARNING: {budget.n_over_budget:,}/"
            f"{budget.n_records:,} payload records exceed max_seq_length="
            f"{budget.max_seq_length}; dropped field groups: "
            f"{budget.dropped_groups}",
            flush=True,
        )
    row_bc.extend(canon_gtins)
    print(
        f"[payload-stage] materialized sku_payload={len(sku_texts):,} "
        f"canonical_payload={len(canon_texts):,}",
        flush=True,
    )
    structured_infos = sku_structured + canon_structured
    structured_features = [
        structured_vector(
            info,
            volume_scale_ml=float(structured_cfg["volume_scale_ml"]),
            pack_scale=float(structured_cfg["pack_scale"]),
            max_set_size=int(structured_cfg["max_set_size"]),
        )
        for info in structured_infos
    ]

    # ── empty-text guard (stage-3 soft stop) ──────────────────────────
    # Low-signal rows ("Single 2 Liter Bottle", "water 1.5 lt pack of 6")
    # strip to "". An empty string must not train as a positive — it pulls
    # a garbage vector onto its canonical. Counted in stats (lane doctrine:
    # nothing drops silently). Negatives keep empty texts: a weak
    # in-batch negative is harmless, a positive is not.
    empty_sku = {i for i, s in enumerate(sku_texts) if not s}
    empty_canon_idx = {
        gtin_to_canon_idx[g] for g, s in zip(canon_gtins, canon_texts, strict=True) if not s
    }

    # ── positives: every row whose barcode has a canonical ──
    cand_pos = [
        (i, gtin_to_canon_idx[g]) for i, g in enumerate(bc) if g in gtin_to_canon_idx
    ]
    pos_pairs = [
        (i, j)
        for i, j in cand_pos
        if i not in empty_sku and j not in empty_canon_idx
    ]
    pos = np.array(pos_pairs, dtype=int).reshape(-1, 2)

    # ── representative row per GTIN (longest title — most signal) ──
    # UNEXPECTED-BEHAVIOR FIX: the old code sorted titles
    # lexicographically DESCENDING and called it "longest" — a short
    # z-titled row won over a long a-titled one. Rank by title LENGTH;
    # ties break by row index (stable, reproducible).
    t_len = title.astype(str).str.len().to_numpy()
    order = np.lexsort((np.arange(len(t_len)), -t_len))
    seen: set[str] = set()
    gtin_to_row: dict[str, int] = {}
    bc_arr = bc.to_numpy() if hasattr(bc, "to_numpy") else list(bc)
    for i in order:
        g = bc_arr[i]
        if g and g not in seen:
            seen.add(g)
            gtin_to_row[g] = i

    # ── negatives: gate hard-no pairs (both directions) ──
    # A hard_no gate decision is not sufficient for training: separate GTINs
    # can still resolve to the same canonical item.  Those rows are true
    # matches and must never be emitted as label-0 pairs.
    gate_canon1 = gates["gtin1"].map(canon_map)
    gate_canon2 = gates["gtin2"].map(canon_map)
    same_canonical = (
        gate_canon1.notna()
        & gate_canon2.notna()
        & gate_canon1.eq(gate_canon2)
    )
    hard_no_band = (gates["gate_decision"] == "hard_no") & (
        gates["similarity"] >= thr_neg
    )
    neg_mask = hard_no_band & ~same_canonical
    neg_gates = gates[neg_mask]
    # The other two gate outcomes, kept as masks so the label-destiny census
    # below accounts for EVERY candidate pair rather than only the negatives.
    proceeded = gates["gate_decision"] == "proceed"
    fell_back = gates["gate_decision"] == "fallback"
    a = neg_gates["gtin1"].map(gtin_to_row)
    b = neg_gates["gtin2"].map(gtin_to_row)
    ca = neg_gates["gtin2"].map(gtin_to_canon_idx)
    cb = neg_gates["gtin1"].map(gtin_to_canon_idx)
    ok1 = a.notna() & ca.notna()
    ok2 = b.notna() & cb.notna()
    fwd = np.stack([a[ok1].astype(int), ca[ok1].astype(int)], axis=1)
    rev = np.stack([b[ok2].astype(int), cb[ok2].astype(int)], axis=1)
    neg = np.vstack([fwd, rev]) if len(fwd) or len(rev) else np.empty((0, 2), dtype=int)

    # Targeted critical-attribute candidates lower the similarity floor from
    # the generic gate-negative threshold while retaining the same brand/name
    # and explicit-conflict requirements. They remain a separate population
    # so training can enable/disable them through the mining profile and keep
    # source provenance intact.
    # The funnel is the miner's OWN attrition accounting, so the trace records
    # why each candidate died rather than restating a similarity threshold. The
    # miner stays the single source of truth for every filter it applies.
    from core.hard_negatives import (
        MiningFunnel,
        mine_targeted_attribute_negatives,
    )

    targeted_cfg = cfg["mining"]["attribute_conflict"]
    mining_funnel = (
        MiningFunnel() if bool(targeted_cfg["same_product_name"]) else None
    )
    targeted_attribute_neg, targeted_attribute_scores = (
        mine_targeted_attribute_negatives(
            df,
            gates,
            canonical_records,
            gtin_to_row,
            gtin_to_canon_idx,
            existing=neg,
            n_target=int(targeted_cfg["target"]),
            min_similarity=float(targeted_cfg["min_similarity"]),
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
            canonical_map=canon_map,
            funnel=mining_funnel,
        )
        if bool(targeted_cfg["same_product_name"])
        else (np.empty((0, 2), dtype=int), np.empty((0,), dtype=float))
    )

    # ── CROSS-BRAND HARD NEGATIVES (the brand-separation lever, §15) ──────
    # The gate's candidate space is brand-blocked upstream ("Brand blocking"),
    # so brand is constant across every labelled pair and the encoder can only
    # learn that brand is noise (measured separation 0.000 vs volume +0.837).
    # This lane mines the mirror population — brands DIFFER, every other
    # critical attribute agrees — and stays a separate population so training
    # can enable/disable it through the mining profile while provenance
    # survives. The funnel is the miner's OWN accounting: it reports candidate
    # GENERATION (blocking census) and then every filter's attrition.
    from core.hard_negatives import (
        CrossBrandMiningFunnel,
        mine_cross_brand_negatives,
    )

    cross_cfg = cfg["mining"]["cross_brand"]
    cross_brand_funnel = (
        CrossBrandMiningFunnel() if bool(cross_cfg["enabled"]) else None
    )
    cross_brand_neg, cross_brand_scores = (
        mine_cross_brand_negatives(
            df,
            canonical_records,
            gtin_to_row,
            gtin_to_canon_idx,
            existing=neg,
            n_target=int(cross_cfg["target"]),
            require_agreement=tuple(str(d) for d in cross_cfg["require_agreement"]),
            min_similarity=float(cross_cfg["min_similarity"]),
            max_per_canonical=int(cross_cfg["max_per_canonical"]),
            max_per_brand=int(cross_cfg["max_per_brand"]),
            # Same volume tolerance the training-label gate uses, so a pair the
            # gate calls compatible can never be mined here as a conflict.
            # BOTH cuts (see the targeted lane above).
            volume_relative_tolerance=float(training_cfg().gate.vol_tolerance),
            volume_absolute_tolerance_ml=float(
                training_cfg().gate.vol_abs_tolerance
            ),
            funnel=cross_brand_funnel,
        )
        if bool(cross_cfg["enabled"])
        else (np.empty((0, 2), dtype=int), np.empty((0,), dtype=float))
    )

    n_forward_source_unresolved = int(a.isna().sum())
    n_forward_target_unresolved = int(ca.isna().sum())
    n_reverse_source_unresolved = int(b.isna().sum())
    n_reverse_target_unresolved = int(cb.isna().sum())
    n_resolution_dropped = int(len(neg_gates) * 2 - len(neg))
    stats = {
        "n_rows": len(df),
        "n_sku_with_canonical": len(cand_pos),
        "n_pos_empty_dropped": len(cand_pos) - len(pos_pairs),
        "n_empty_sku_texts": len(empty_sku),
        "n_empty_canon_texts": len(empty_canon_idx),
        "n_canonicals": len(canon_gtins),
        "n_pos_gate_rows": int(
            (
                (gates["gate_decision"] == "proceed") & (gates["similarity"] >= thr_pos)
            ).sum()
        ),
        "n_neg_same_canonical_dropped": int((hard_no_band & same_canonical).sum()),
        "n_neg_hard_no_band": int(hard_no_band.sum()),
        "n_neg_gate_rows": int(neg_mask.sum()),
        "n_neg_resolved": len(neg),
        "n_neg_forward_resolved": int(len(fwd)),
        "n_neg_reverse_resolved": int(len(rev)),
        "n_neg_forward_source_unresolved": n_forward_source_unresolved,
        "n_neg_forward_target_unresolved": n_forward_target_unresolved,
        "n_neg_reverse_source_unresolved": n_reverse_source_unresolved,
        "n_neg_reverse_target_unresolved": n_reverse_target_unresolved,
        "n_neg_resolution_dropped": n_resolution_dropped,
        "n_neg_dropped": n_resolution_dropped,
        "n_targeted_attribute_candidates": int(len(targeted_attribute_scores)),
        "n_targeted_attribute_resolved": int(len(targeted_attribute_neg)),
        # Candidates ENTERING the cross-brand funnel (the pairs its
        # require_agreement blocking generated) and the pair rows it emitted.
        # The two differ by the funnel's own attrition, which the trace
        # records step by step.
        "n_cross_brand_candidates": int(
            cross_brand_funnel.candidates_in_blocks
            if cross_brand_funnel is not None
            else 0
        ),
        "n_cross_brand_resolved": int(len(cross_brand_neg)),
    }
    print(
        f"[payload-stage] pairs resolved positives={len(pos):,} "
        f"hard_negatives={len(neg):,} unresolved_or_dropped={n_resolution_dropped:,}",
        flush=True,
    )
    print(
        f"[targeted-attribute-negatives] {len(targeted_attribute_neg):,} "
        f"same-brand/name explicit-conflict pairs with gate similarity "
        f"> {float(targeted_cfg['min_similarity']):.2f}",
        flush=True,
    )
    if cross_brand_funnel is not None:
        _cb = cross_brand_funnel
        print(
            f"[cross-brand-negatives] {len(cross_brand_neg):,} label-0 pair rows "
            f"from {_cb.accepted_candidates:,} candidates "
            f"({_cb.candidates_in_blocks:,} generated -> "
            f"{_cb.passed_candidates:,} survived every filter; "
            f"target {int(cross_cfg['target']):,}, "
            f"reached={_cb.emitted_pairs >= int(cross_cfg['target']) > 0})",
            flush=True,
        )
    else:
        print(
            "[cross-brand-negatives] disabled by mining.cross_brand.enabled",
            flush=True,
        )
    # ── EXACT MODEL PAYLOAD DUMP (owner directive 2026-09-07) ──────────
    # Every pair the model trains on, with the LITERAL texts it ingests.
    # The rows are recorded in the ONE consolidated trace (core.tracing) below,
    # replacing the former per-stage payload_pairs.csv.
    _rows = []
    for i, j in pos:
        _rows.append(
            {
                "kind": "pos",
                "payload_idx_a": int(i),
                "payload_idx_b": int(j),
                "barcode_a": row_bc[i],
                "barcode_b": row_bc[j],
                "text_a": payload[i],
                "text_b": payload[j],
            }
        )
    for i, j in neg:
        _rows.append(
            {
                "kind": "neg_hard",
                "payload_idx_a": int(i),
                "payload_idx_b": int(j),
                "barcode_a": row_bc[i],
                "barcode_b": row_bc[j],
                "text_a": payload[i],
                "text_b": payload[j],
            }
        )
    for i, j in targeted_attribute_neg:
        _rows.append(
            {
                "kind": "neg_targeted_attribute",
                "payload_idx_a": int(i),
                "payload_idx_b": int(j),
                "barcode_a": row_bc[i],
                "barcode_b": row_bc[j],
                "text_a": payload[i],
                "text_b": payload[j],
            }
        )
    for i, j in cross_brand_neg:
        _rows.append(
            {
                "kind": "neg_cross_brand",
                "payload_idx_a": int(i),
                "payload_idx_b": int(j),
                "barcode_a": row_bc[i],
                "barcode_b": row_bc[j],
                "text_a": payload[i],
                "text_b": payload[j],
            }
        )
    _kinds = {
        "pos": int(len(pos)),
        "neg_hard": int(len(neg)),
        "neg_targeted_attribute": int(len(targeted_attribute_neg)),
        "neg_cross_brand": int(len(cross_brand_neg)),
    }
    # BOUNDARY CONTRACT (lib.schemas.TrainingData): payload/row_bc locked,
    # every pos/neg index in range, gtin_to_row targets valid — the bundle
    # crosses into src/training/train + src/training/training; a shape break must die
    # HERE with a named field, not as an IndexError in a fold.
    from core.schemas import TrainingData as _TrainingData

    _bundle = _TrainingData(
        payload=payload,
        structured_features=structured_features,
        row_bc=np.array(row_bc),
        pos=pos,
        neg=neg,
        targeted_attribute_neg=targeted_attribute_neg,
        cross_brand_neg=cross_brand_neg,
        gtin_to_row=gtin_to_row,
        stats=stats,
    )
    # ── CONSOLIDATED TRACE: pairs + mining funnel ──────────────────────────
    # The former per-stage files (negative_resolution_manifest.csv,
    # payload_pairs.csv) folded into the ONE trace (core.tracing). Stage 2
    # commits onto stage 1's rows for the same run, so the file reads as one
    # continuous flow. This block used to sit AFTER `return _bundle.model_dump()`
    # and was therefore dead: stage 2 ran, printed its counts, and wrote nothing
    # to the trace. The bundle is validated first (fail fast on a shape break),
    # then every step is recorded, then the bundle is returned.
    _kinds_row = _bundle.model_dump()

    # EVERY candidate pair gets exactly one label destiny. Summing these group
    # rows reproduces len(gates) exactly, which is the pair-side accounting
    # identity: no gate pair is unaccounted for, and each row states in words
    # why that population did or did not become a training label.
    n_gates = int(len(gates))
    label_buckets = [
        (
            "proceed_not_a_training_pair",
            int(proceeded.sum()),
            "gate says same product: a PROCEED pair yields no label here — "
            "positives come from the row→canonical relation, not the pair",
            {"gate_decision": "proceed"},
        ),
        (
            "fallback_unresolved",
            int(fell_back.sum()),
            "gate could not resolve the pair: neither a verified match nor a "
            "hard no, so neither mining lane may use it",
            {"gate_decision": "fallback"},
        ),
        (
            "negative_hard",
            int(neg_mask.sum()),
            "hard_no inside the mining similarity band and NOT same-canonical: "
            "emitted as a label-0 pair in both directions",
            {
                "gate_decision": "hard_no",
                "similarity_threshold": float(thr_neg),
                "directions_per_pair": 2,
            },
        ),
        (
            "true_match_same_canonical_excluded",
            int((hard_no_band & same_canonical).sum()),
            "hard_no in band but both gtins share one canonical record: a true "
            "match, so label 0 would be wrong",
            {"gate_decision": "hard_no"},
        ),
        (
            "hard_no_below_similarity_floor",
            int(((gates["gate_decision"] == "hard_no") & ~hard_no_band).sum()),
            "hard_no below the mining similarity floor: a valid hard no that "
            "this lane's negative mining does not reach",
            {"gate_decision": "hard_no", "similarity_threshold": float(thr_neg)},
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
        in_count=int(len(cand_pos)),
        out_count=int(len(pos)),
        reason="a row with a resolvable canonical and non-empty model text is a positive",
        detail={
            "rows": int(len(df)),
            "sku_with_canonical": int(len(cand_pos)),
            "dropped_empty_sku_text": int(len(empty_sku)),
            "dropped_empty_canon_text": int(len(empty_canon_idx)),
            "canonicals": int(len(canon_gtins)),
            "gate_proceed_rows": int(
                (
                    (gates["gate_decision"] == "proceed")
                    & (gates["similarity"] >= thr_pos)
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
        out_count=int(len(neg_gates)),
        reason=(
            "hard_no with similarity >= the mining threshold; same-canonical "
            "pairs are true matches and are excluded here"
        ),
        detail={
            "hard_no_and_in_band": int(hard_no_band.sum()),
            "dropped_same_canonical": int((hard_no_band & same_canonical).sum()),
            "similarity_threshold": float(thr_neg),
            "both_directions": int(len(neg_gates) * 2),
        },
        source="gate_results.csv",
    )
    trace.add(
        "negatives",
        "index_resolution",
        scope="group",
        in_count=int(len(neg_gates) * 2),
        out_count=int(len(neg)),
        reason="both endpoints must resolve to a payload row index",
        detail={
            "forward_resolved": int(len(fwd)),
            "reverse_resolved": int(len(rev)),
            "forward_source_unresolved": n_forward_source_unresolved,
            "forward_target_unresolved": n_forward_target_unresolved,
            "reverse_source_unresolved": n_reverse_source_unresolved,
            "reverse_target_unresolved": n_reverse_target_unresolved,
        },
        source="model payload index maps (rows + canonicals)",
    )
    # ONE row per real filter, taken from the miner's OWN funnel accounting.
    # The previous single row restated "gate rows above the similarity floor"
    # as the input and claimed the filters generically, which hid that ~39,896
    # candidates die at the name filter and made the lane's ceiling
    # unanswerable from the trace. The miner stays the label authority; this
    # only records what it did.
    if mining_funnel is not None:
        for _step, _in, _out, _why in mining_funnel.stages():
            trace.add(
                "mining",
                f"targeted_attribute_funnel.{_step}",
                in_count=int(_in),
                out_count=int(_out),
                reason=_why,
                detail={
                    "gate_similarity_floor": float(targeted_cfg["min_similarity"]),
                    "volume_tolerance": float(training_cfg().gate.vol_tolerance),
                    "target": int(targeted_cfg["target"]),
                    "funnel": mining_funnel.to_dict(),
                },
                source="gate_results.csv",
            )
    else:
        trace.add(
            "mining",
            "targeted_attribute_funnel.disabled",
            in_count=0,
            out_count=0,
            reason="mining.attribute_conflict.same_product_name is false",
            detail={"target": int(targeted_cfg["target"])},
            source="config/training.yaml",
        )
    # The cross-brand lane's own funnel, in the same shape as the targeted one:
    # its generation step (blocking census) then every filter's attrition. A
    # lane whose population is generated rather than handed in cannot be
    # audited from its output count alone, so the census is the trace's job.
    if cross_brand_funnel is not None:
        for _step, _in, _out, _why in cross_brand_funnel.stages():
            trace.add(
                "mining",
                f"cross_brand_funnel.{_step}",
                in_count=int(_in),
                out_count=int(_out),
                reason=_why,
                detail={
                    "target": int(cross_cfg["target"]),
                    "min_similarity": float(cross_cfg["min_similarity"]),
                    "require_agreement": list(cross_cfg["require_agreement"]),
                    "volume_tolerance": float(training_cfg().gate.vol_tolerance),
                    "funnel": cross_brand_funnel.to_dict(),
                },
                source="canonical_records.csv + gate_results.csv",
            )
    else:
        trace.add(
            "mining",
            "cross_brand_funnel.disabled",
            in_count=0,
            out_count=0,
            reason="mining.cross_brand.enabled is false",
            detail={"target": int(cross_cfg["target"])},
            source="config/training.yaml",
        )
    trace.add(
        "payload",
        "materialized",
        in_count=int(len(df)),
        out_count=int(len(payload)),
        reason="every source row plus one canonical per GTIN",
        detail={
            "sku_payload": int(len(df)),
            "canonical_payload": int(len(canon_gtins)),
            "structured_feature_dim": int(len(structured_features[0])),
            "structured_encode": bool(structured_cfg.get("enabled", True)),
        },
        source="canonical_records.csv",
    )
    trace.add(
        "payload",
        "pair_census",
        scope="group",
        in_count=int(
            len(pos) + len(neg) + len(targeted_attribute_neg) + len(cross_brand_neg)
        ),
        out_count=int(len(_rows)),
        reason="final label populations handed to training",
        detail={
            "pos": int(len(pos)),
            "neg_hard": int(len(neg)),
            "neg_targeted_attribute": int(len(targeted_attribute_neg)),
            "neg_cross_brand": int(len(cross_brand_neg)),
            "text_columns": ["text_a", "text_b"],
        },
        source="model payload",
    )
    # Bounded per-pair readback with the LITERAL model texts, so the trace is
    # a sample of the training input rather than only a count of it. Bucketed
    # by label kind, so the three label populations above and the sampled rows
    # below join on the same label.
    trace.add_entities(
        "pair_payload",
        _rows,
        key_of=lambda r: f"{r['kind']}|{r['barcode_a']}|{r['barcode_b']}",
        reason_of=lambda r: r["kind"],
        detail_of=lambda r: json.dumps(
            {
                "payload_idx_a": r["payload_idx_a"],
                "payload_idx_b": r["payload_idx_b"],
                "barcode_a": r["barcode_a"],
                "barcode_b": r["barcode_b"],
                "text_a": r["text_a"],
                "text_b": r["text_b"],
            },
            sort_keys=True,
        ),
        source="model payload",
        per_reason=ENTITY_PER_REASON,
        total_cap=ENTITY_TOTAL_CAP,
    )
    trace.write()
    print(
        f"[trace] pairs steps written -> {trace_path()} | {_kinds}",
        flush=True,
    )
    # Contract preserved: the caller receives the TrainingData bundle. The
    # trace stage runs BEFORE this return (it was previously unreachable dead
    # code placed after it), and the dump is built once and reused.
    return _kinds_row
