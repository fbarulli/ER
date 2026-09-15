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

from core.common import (
    DATA_DIR,
    RESULTS,
    F,
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
)
from ner.ner_product_attributes import extract_title_attributes
from core.critical_attributes import (
    categorical_conflict,
    extract_critical_claims,
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
RAW_EXPORT_REQUIRED_COLUMNS = (
    "gtin",
    "sku_name_eng",
    "attribute",
    "brand",
    "description_short_eng",
    "breadcrumbs_eng",
)
CANONICAL_DATASET_REQUIRED_COLUMNS = ("barcode", "title", "attributes")
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

# ── regex patterns (owner's second_extraction.py verbatim) ──────────────────
VOLUME_PATTERN_METRIC_EXT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ml|milliliters?|cc|cl|centiliters?|l|lt|ltr|liters?|litres?)\b",
    re.IGNORECASE,
)
VOLUME_PATTERN_US_EXT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|fluid\s+ounces?|oz\.?|ounces?|qt|quarts?|pt|pints?|gal|gallons?)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    # NaN-guard (bug fix): missing titles arrive as float NaN; str(NaN) is
    # the string "nan", which leaked into 6 canonicals as a "nan_volume_946"
    # token. Coerce missing input to ""; other non-strings stringify.
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN without pandas  # noqa: PLR0124
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.replace("×", "x")
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_volume_from_title(title: str) -> dict:
    from core.unit_canonicalization import canonical_volume_ml

    t = normalize_text(title)
    m = VOLUME_PATTERN_US_EXT.search(t)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        raw = m.group(0)
        if value <= 0:
            ml = 0.0
            conf = 0.0
        elif "oz" in unit or "ounce" in unit:
            ml = canonical_volume_ml(value, unit)
            conf = 0.85 if ("fl" in unit or "fluid" in unit) else 0.75
        elif "qt" in unit or "quart" in unit:
            ml = canonical_volume_ml(value, unit)
            conf = 0.95
        elif "pt" in unit or "pint" in unit:
            ml = canonical_volume_ml(value, unit)
            conf = 0.95
        elif "gal" in unit or "gallon" in unit:
            ml = canonical_volume_ml(value, unit)
            conf = 0.95
        else:
            ml = 0.0
            conf = 0.0
        if ml > 0:
            return {
                "volume_ml": round(ml, 2),
                "confidence": conf,
                "raw_match": raw,
                "parse_status": "us_volume",
            }
    m = VOLUME_PATTERN_METRIC_EXT.search(t)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        raw = m.group(0)
        if value <= 0:
            ml = 0.0
            conf = 0.0
        elif unit == "ml" or "milliliter" in unit or unit == "cc":
            ml = canonical_volume_ml(value, unit)
            conf = 0.98 if "." in m.group(1) else 0.95
        elif unit == "cl" or "centiliter" in unit:
            ml = canonical_volume_ml(value, unit)
            conf = 0.95
        elif unit in ("l", "lt", "ltr") or "liter" in unit or "litre" in unit:
            ml = canonical_volume_ml(value, unit)
            conf = 0.98 if "." in m.group(1) else 0.95
        else:
            ml = 0.0
            conf = 0.0
        if ml > 0:
            return {
                "volume_ml": ml,
                "confidence": conf,
                "raw_match": raw,
                "parse_status": "metric_volume",
            }
    return {
        "volume_ml": 0.0,
        "confidence": 0.0,
        "raw_match": "",
        "parse_status": "no_volume_mention",
    }


def extract_pack_from_title(title: str) -> tuple:
    t = normalize_text(title)
    # Nested: "2 x 12 x 330ml"
    m = re.search(r"(\d+)\s*x\s*(\d+)\s*x\s*\d+", t)
    if m:
        return int(m.group(1)) * int(m.group(2)), 0.95
    # Simple: "24 x 330ml"
    m = re.search(r"(\d+)\s*x\s*\d+", t)
    if m:
        return int(m.group(1)), 0.90
    # "Pack of N" / "Case of N" (including parentheses)
    m = re.search(
        r"\(\s*(?:packs?|packages?|cases?)\s+of\s+(\d+)\s*\)",
        t,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), 0.90
    m = re.search(
        r"\b(?:packs?|packages?|cases?)\s+of\s+(\d+)\b", t, re.IGNORECASE
    )
    if m:
        return int(m.group(1)), 0.90
    # "N pack" / "N pk" / "N ct" / "N count". ZERO-GUARD (found by the
    # ExtractedAttributes schema, 2026-09-08): a captured 0 is never a
    # pack COUNT — it is a percent-zero ("0% sugar ... pack") or a
    # decimal-volume fragment ("pack 0.5 l" -> "0 5"). Those rows poisoned
    # pack_set with an impossible 0 (nothing can overlap it except another
    # 0). Skip zero captures and keep scanning for the real count.
    m = re.search(
        r"\b(\d+)\s*(?:pcs?|pieces?|packs?|packages?|pk|cases?|units?|ct|count)\b",
        t,
        re.IGNORECASE,
    )
    if m and int(m.group(1)) > 0:
        return int(m.group(1)), 0.85
    # Concatenated "pack23"
    m = re.search(r"\bpack\s*(\d+)\b", t)
    if m and int(m.group(1)) > 0:
        return int(m.group(1)), 0.75
    # Number followed by container words: "24 Glass Bottles", "12 cans", "6 bottles"
    m = re.search(
        r"(\d+)\s*(?:glass\s*)?(?:bottles?|cans?|tins?|cartons?|boxes?|packets?|sachets?|bags?)\b",
        t,
        re.IGNORECASE,
    )
    if m and int(m.group(1)) > 0:
        return int(m.group(1)), 0.90
    # Number followed by "count" or "ct"
    m = re.search(r"(\d+)\s*(?:count|ct)\b", t, re.IGNORECASE)
    if m and int(m.group(1)) > 0:
        return int(m.group(1)), 0.85
    # No explicit pack evidence: keep the schema-safe quantity of one, but
    # mark it unknown.  Downstream attribute/conflict code must not turn this
    # parser default into a fabricated ``pack_set={1}`` observation.
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
    m_vol = re.search(
        r"Volume:\s*(\d+(?:[.,]\d+)?)\s*"
        r"(ml|milliliters?|millilitres?|cc|cl|centiliters?|centilitres?|"
        r"l|lt|ltr|liters?|litres?|fl\.?\s*oz|fluid\s+ounces?|oz\.?|"
        r"ounces?|qt|quarts?|pt|pints?|gal|gallons?)?",
        attr_str,
        re.IGNORECASE,
    )
    if m_vol:
        vol_ml = canonical_volume_ml(m_vol.group(1), m_vol.group(2) or "ml")
        vol_conf = 0.9
    m_pack = re.search(r"Count per Unit:\s*(\d+)", attr_str, re.IGNORECASE)
    if m_pack and int(m_pack.group(1)) > 0:
        # zero-guard: same contract as extract_pack_from_title — a 0 here is
        # export noise, not a pack count (default 1 with conf 0 below)
        pack_qty = canonical_pack_count(m_pack.group(1))
        pack_conf = 0.9
    return vol_ml, vol_conf, pack_qty, pack_conf


# extract_salient_tokens REMOVED (audit 2026-09-09): zero callers across
# the repo (verified by grep). Its "salient token" job is done by the
# NgramIDF discriminative extractor; this legacy variant duplicated a
# volume/pack regex inline (a second declaration the config cannot steer).


def extract_all(sku_name: str, attribute: str) -> dict:
    """Extract structured fields plus salient tokens from a single SKU row."""
    t = normalize_text(sku_name)
    # Critical categorical evidence is parsed once for the canonical, model,
    # mining, and inference lanes.  Keep the historical scalar flavor as a
    # deterministic first value for compatibility with existing CSV readers.
    critical = extract_critical_claims(sku_name, attribute)
    flavor_set = set(critical["flavor"])
    flavor = sorted(flavor_set)[0] if flavor_set else ""
    if re.search(r"\bcoconut\s+water\b", t):
        ptype = "coconut water"
    elif re.search(r"\bmineral\s+water\b", t) or re.search(r"\bwater\b", t):
        ptype = "water"
    elif re.search(r"\bjuice\b", t):
        ptype = "juice"
    elif re.search(r"\b(?:ice\s+)?tea\b", t):
        ptype = "tea"
    elif re.search(r"\benergy\s+(?:drink|water)\b", t):
        ptype = "energy"
    elif re.search(r"\b(?:soda|soft\s+drink)\b", t):
        ptype = "soda"
    elif re.search(r"\btonic\b", t):
        ptype = "tonic"
    else:
        ptype = ""

    # Volume and pack from title
    vol_title = extract_volume_from_title(sku_name)
    pack_title, pack_conf_title = extract_pack_from_title(sku_name)

    # Attribute parsing
    attr_vol, attr_vol_conf, attr_pack, attr_pack_conf = parse_attribute_volume_pack(
        attribute
    )

    # Combine: prefer attribute if present
    if attr_vol > 0:
        volume_ml = attr_vol
        volume_conf = attr_vol_conf
        volume_raw = f"attribute: {attr_vol}"
        volume_status = "attribute_volume"
    else:
        volume_ml = vol_title["volume_ml"]
        volume_conf = vol_title["confidence"]
        volume_raw = vol_title["raw_match"]
        volume_status = vol_title["parse_status"]

    if attr_pack > 1 or attr_pack_conf > 0:
        pack_qty = attr_pack
        pack_conf = attr_pack_conf
    else:
        pack_qty = pack_title
        pack_conf = pack_conf_title

    # BOUNDARY CONTRACT (lib.schemas): the extracted-attribute dict is the
    # input to BOTH the canonical build and the gate — validate the shape
    # once here so a confidence out of [0,1] or a pack_qty < 1 crashes at
    # the transform, not downstream in the gate's comparisons.
    title_attributes = extract_title_attributes(sku_name)
    return ExtractedAttributes(
        flavor=flavor,
        type=ptype,
        volume_ml=volume_ml,
        volume_confidence=volume_conf,
        volume_raw=volume_raw,
        volume_status=volume_status,
        pack_qty=pack_qty,
        pack_confidence=pack_conf,
        package_types=title_attributes["package_types"],
        package_materials=title_attributes["package_materials"],
        flavor_set=flavor_set,
        carbonation_set=set(critical["carbonation"]),
        sweetener_set=set(critical["sweetener"]),
        pulp_set=set(critical["pulp"]),
    ).model_dump()


# ============================================================================
# GATING
# ============================================================================
def pack_gate(
    score: float,
    sku_a: object,
    sku_b: object,
    *,
    volume_relative_tolerance: float = 0.0,
    volume_absolute_tolerance_ml: float = 0.0,
    trust_threshold: float | None = None,
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
        raw = _value(obj, *names)
        if raw is None or raw == "":
            return True
        if trust_threshold is None:
            return True
        try:
            return float(raw) >= float(trust_threshold)
        except (TypeError, ValueError):
            return False

    # PACK COUNT: shared evidence agrees; disjoint counts conflict.
    left_pack = _set(_value(sku_a, "pack_size", "pack_set", "pack_qty"))
    right_pack = _set(_value(sku_b, "pack_size", "pack_set", "pack_qty"))
    if (
        left_pack
        and right_pack
        and not (left_pack & right_pack)
        and _trusted(sku_a, "pack_confidence")
        and _trusted(sku_b, "pack_confidence")
    ):
        return False

    # PACKAGE TYPE: disjoint categorical evidence conflicts.
    left_type = _set(_value(sku_a, "package_type", "package_type_set"))
    right_type = _set(_value(sku_b, "package_type", "package_type_set"))
    if left_type and right_type and not (left_type & right_type):
        return False

    left_volume = _set(_value(sku_a, "volume", "volume_set", "volume_ml"))
    right_volume = _set(_value(sku_b, "volume", "volume_set", "volume_ml"))
    if (
        left_volume
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

    for dimension in ("carbonation", "sweetener", "pulp"):
        left = _claim_set(sku_a, dimension)
        right = _claim_set(sku_b, dimension)
        if left and right and categorical_conflict(
            dimension, {dimension: left}, {dimension: right}
        ):
            return False
    return True


def three_way_gate(
    attrs1: dict,
    attrs2: dict,
    vol_tolerance: float | None = None,
    raw_conf_threshold: float | None = None,
    consistency_fallback_threshold: float | None = None,
) -> dict:
    """Deterministic volume/pack/flavor gate.

    NO-FALLBACK SSOT (audit round 2, F01): the three decision thresholds
    live in config/training.yaml `gate:` and are read through training_cfg()
    — the old signature defaults (0.05/0.85/0.3) were a second declaration
    the config could not steer. Passing a value explicitly still wins
    (selftest pins known-good gate behavior with explicit values).
    """
    if (
        vol_tolerance is None
        or raw_conf_threshold is None
        or consistency_fallback_threshold is None
    ):
        _g = training_cfg().gate
        if vol_tolerance is None:
            vol_tolerance = float(_g.vol_tolerance)
        if raw_conf_threshold is None:
            raw_conf_threshold = float(_g.raw_conf_threshold)
        if consistency_fallback_threshold is None:
            consistency_fallback_threshold = float(
                _g.consistency_fallback_threshold
            )
    if not pack_gate(
        0.0,
        attrs1,
        attrs2,
        volume_relative_tolerance=float(vol_tolerance),
        trust_threshold=float(raw_conf_threshold),
    ):
        return GateResult(
            decision="hard_no",
            reason="Pack blocker: pack size, package type, or volume mismatch",
        ).model_dump()
    # raw confidence check
    if (
        attrs1["volume_confidence"] < raw_conf_threshold
        or attrs2["volume_confidence"] < raw_conf_threshold
    ):
        return GateResult(
            decision="fallback", reason="Low raw volume confidence"
        ).model_dump()
    if (
        attrs1["pack_confidence"] < raw_conf_threshold
        or attrs2["pack_confidence"] < raw_conf_threshold
    ):
        return GateResult(
            decision="fallback", reason="Low raw pack confidence"
        ).model_dump()

    # volume overlap
    vol_overlap = False
    for v1 in attrs1["volume_set"]:
        for v2 in attrs2["volume_set"]:
            if v1 == 0 or v2 == 0:
                continue
            if abs(v1 - v2) / max(v1, v2) <= vol_tolerance:
                vol_overlap = True
                break
        if vol_overlap:
            break
    if not vol_overlap:
        return GateResult(decision="hard_no", reason="No volume overlap").model_dump()

    # pack overlap
    pack_overlap = attrs1["pack_set"] & attrs2["pack_set"]
    if not pack_overlap:
        return GateResult(decision="hard_no", reason="No pack overlap").model_dump()

    for field, reason in (("package_type_set", "Package type mismatch"), ("package_material_set", "Package material mismatch")):
        left, right = set(attrs1.get(field, set())), set(attrs2.get(field, set()))
        if left and right and not (left & right):
            return GateResult(decision="hard_no", reason=reason).model_dump()

    # Every explicit categorical conflict uses the same dimension/evidence
    # definition as targeted mining and final inference. Unknown stays
    # unknown here; it is not fabricated into a conflict or an agreement.
    from core.attribute_conflicts import (
        canonical_attribute_info,
        critical_attribute_evaluation,
    )

    critical = critical_attribute_evaluation(
        canonical_attribute_info(attrs1),
        canonical_attribute_info(attrs2),
        volume_relative_tolerance=float(vol_tolerance),
    )
    categorical_conflicts = [
        name
        for name in critical["conflicts"]
        if name in {"flavor", "carbonation", "sweetener", "pulp"}
    ]
    if categorical_conflicts:
        return GateResult(
            decision="hard_no",
            reason="Critical attribute mismatch: " + ",".join(categorical_conflicts),
        ).model_dump()

    # consistency check
    if (
        attrs1["volume_consistency"] < consistency_fallback_threshold
        or attrs2["volume_consistency"] < consistency_fallback_threshold
        or attrs1["pack_consistency"] < consistency_fallback_threshold
        or attrs2["pack_consistency"] < consistency_fallback_threshold
    ):
        return GateResult(
            decision="fallback", reason="Overlap but low consistency"
        ).model_dump()

    return GateResult(
        decision="proceed", reason="Known critical attributes compatible"
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
) -> dict:  # CanonicalRecord.model_dump() — validated shape, plain dict
    titles = [sku for sku, attr in rows]
    attributes = [attr for sku, attr in rows]
    extracted = [extract_all(sku, attr) for sku, attr in rows]

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
    package_material_set = {value for x in extracted for value in x["package_materials"]}
    flavor_set = {value for x in extracted for value in x["flavor_set"]}
    carbonation_set = {value for x in extracted for value in x["carbonation_set"]}
    sweetener_set = {value for x in extracted for value in x["sweetener_set"]}
    pulp_set = {value for x in extracted for value in x["pulp_set"]}

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
        package_type_set=package_type_set,
        package_material_set=package_material_set,
        flavor_set=flavor_set,
        carbonation_set=carbonation_set,
        sweetener_set=sweetener_set,
        pulp_set=pulp_set,
        volume_confidence=round(vol_conf, 3),
        pack_confidence=round(pack_conf, 3),
        volume_consistency=round(volume_consistency, 3),
        pack_consistency=round(pack_consistency, 3),
        n_titles=n,
    )
    return rec.model_dump()


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


def run_within_brand_pipeline(
    df_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # (gate results, canonical records)
    # Preserve the two high-coverage, previously-unused source fields as
    # canonical-level evidence.  They remain OUTSIDE the frozen canonical
    # text until a component-safe ablation establishes their value.
    for column in ("description_short_eng", "breadcrumbs_eng"):
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
    checksum_bad = gtin_valid & ~bc_valid
    n_checksum_dropped = int(checksum_bad.sum())
    df_full = df_full[gtin_valid & bc_valid]
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
            brand=("brand", lambda x: Counter(x).most_common(1)[0][0]),
            description_evidence=("description_short_eng", _source_evidence),
            breadcrumb_evidence=("breadcrumbs_eng", _source_evidence),
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
    for _, row in grouped.iterrows():
        brand_key = row["brand"].lower().strip()
        record = generate_canonical(
                row["gtin"],
                row["brand"],
                row["rows"],
                global_idf,
                brand_idf_map[brand_key],
            )
        record["description_evidence"] = row["description_evidence"]
        record["breadcrumb_evidence"] = row["breadcrumb_evidence"]
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
    gate_vis = []
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
    for g1, g2 in candidate_pairs:
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
        "package_material_set",
        "flavor_set",
        "carbonation_set",
        "sweetener_set",
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
    check_canonical_records_frame(df_canon)
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
        build_sku_text,
        model_input_info,
        model_input_composition,
        token_budget_report,
    )
    from core.structured_features import (
        canonical_info as canonical_structured_info,
        sku_info as sku_structured_info,
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

    # Keep the structured source of truth alongside every payload endpoint.
    # The old text lane deliberately removed these tokens; that made the
    # volume/pack work useful for labels but invisible to the embedding.
    sku_structured = [
        model_input_info(sku_structured_info(t, a))
        if structured_enabled
        else {"volume": set(), "pack": set(), "package_type": set()}
        for t, a in zip(title, attrs, strict=True)
    ]

    # ── clean sku text per row (variant: full = title+attr, title_only) ──
    # schema words (type/content/material/...) die on the MODEL side only —
    # the gate's inputs are untouched (owner 2026-09-07: stage-2 strip).
    # Both variants go through core.model_input, the shared builder.
    if payload_variant == "full":
        model_frame = df
    elif payload_variant == "title_only":
        model_frame = df.copy()
        for column in ("attributes", "attr", "description", "description_short_eng"):
            if column in model_frame.columns:
                model_frame[column] = ""
    else:
        raise SystemExit(f"unknown payload variant: {payload_variant}")
    sku_texts = [
        build_sku_text(row, info)
        for (_, row), info in zip(model_frame.iterrows(), sku_structured, strict=True)
    ]

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
            volume_relative_tolerance=float(training_cfg().gate.vol_tolerance),
            # Same canonical-identity rule the baseline negative lane already
            # applies: a same-canonical pair is a true match, not a label-0 row.
            canonical_map=canon_map,
            funnel=mining_funnel,
        )
        if bool(targeted_cfg["same_product_name"])
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
    _kinds = {
        "pos": int(len(pos)),
        "neg_hard": int(len(neg)),
        "neg_targeted_attribute": int(len(targeted_attribute_neg)),
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
        in_count=int(len(pos) + len(neg) + len(targeted_attribute_neg)),
        out_count=int(len(_rows)),
        reason="final label populations handed to training",
        detail={
            "pos": int(len(pos)),
            "neg_hard": int(len(neg)),
            "neg_targeted_attribute": int(len(targeted_attribute_neg)),
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
