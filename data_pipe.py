"""data_pipe.py — THE official data pipeline, all transformations in ONE
module (owner directive: smash the DATA_PIPE folder into one file).

Sections (in dependency order):
  1. extraction    — normalize_text, volume/pack/flavor extractors, extract_all
  2. gating        — three_way_gate (owner's second_gating.py, verbatim)
  3. similarity    — jaccard / embedding similarity
  4. canonical     — per-GTIN canonical generation + clean_sku_text +
                     load_canonical_map (owner's second_canonical.py)
  5. numbers       — number-token reference + strip (95.2% coverage)
  6. pipeline      — run_within_brand_pipeline (canonical + gate CSVs)
  7. pairs         — build_training_data (the OFFICIAL training pairs)

Public surface (old DATA_PIPE imports keep working):
  normalize_text, extract_all, three_way_gate, jaccard_similarity,
  embedding_similarity, generate_canonical, clean_sku_text,
  load_canonical_map, run_within_brand_pipeline, build_training_data,
  strip_number_tokens, build_reference, census_texts, token_verdict
"""

from __future__ import annotations

import json as _json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from lib.common import DATA_DIR, RESULTS, TRAIN_ROOT, F, load_config

# ============================================================================
# EXTRACTION
# ============================================================================


def _load_stopwords(key: str) -> set:
    """STOPWORDS / MINIMAL_STOPWORDS from stopwords.json (SSOT via 00_config)."""
    path = TRAIN_ROOT / F["stopwords"]
    if not path.exists():
        raise SystemExit(f"stopwords file missing: {path}")
    return set(_json.loads(path.read_text(encoding="utf-8"))[key])


STOPWORDS = _load_stopwords("STOPWORDS")

# ── regex patterns (owner's second_extraction.py verbatim) ──────────────────
FLAVOR_PATTERN = re.compile(
    r"\b(lemon|lime|orange|strawberry|raspberry|peach|apple|cherry|"
    r"mango|pineapple|coconut|watermelon|berry|berries|cola|coffee|"
    r"ginger|mint|vanilla|chocolate|tonic|citrus|aloe|rose)\b",
    re.IGNORECASE,
)
VOLUME_PATTERN_METRIC_EXT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ml|milliliters?|cc|cl|centiliters?|l|lt|ltr|liters?|litres?)\b",
    re.IGNORECASE,
)
VOLUME_PATTERN_US_EXT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|fluid\s+ounces?|oz\.?|ounces?|qt|quarts?|pt|pints?|gal|gallons?)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("×", "x")
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_volume_from_title(title: str) -> dict:
    t = normalize_text(title)
    m = VOLUME_PATTERN_US_EXT.search(t)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        raw = m.group(0)
        if "oz" in unit or "ounce" in unit:
            ml = value * 29.5735
            conf = 0.85 if ("fl" in unit or "fluid" in unit) else 0.75
        elif "qt" in unit or "quart" in unit:
            ml = value * 946.353
            conf = 0.95
        elif "pt" in unit or "pint" in unit:
            ml = value * 473.176
            conf = 0.95
        elif "gal" in unit or "gallon" in unit:
            ml = value * 3785.41
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
        if unit == "ml" or "milliliter" in unit or unit == "cc":
            ml = value
            conf = 0.98 if "." in m.group(1) else 0.95
        elif unit == "cl" or "centiliter" in unit:
            ml = value * 10
            conf = 0.95
        elif unit in ("l", "lt", "ltr") or "liter" in unit or "litre" in unit:
            ml = value * 1000
            conf = 0.98 if "." in m.group(1) else 0.95
        else:
            ml = 0.0
            conf = 0.0
        if ml > 0:
            return {
                "volume_ml": round(ml, 2),
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
    m = re.search(r"\(\s*(?:pack|case)\s+of\s+(\d+)\s*\)", t, re.IGNORECASE)
    if m:
        return int(m.group(1)), 0.90
    m = re.search(r"\b(?:pack|case)\s+of\s+(\d+)\b", t, re.IGNORECASE)
    if m:
        return int(m.group(1)), 0.90
    # "N pack" / "N pk" / "N ct" / "N count"
    m = re.search(
        r"\b(\d+)\s*(?:pcs?|pieces?|pack|packs|pk|case|cases|units?|ct|count)\b",
        t,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), 0.85
    # Concatenated "pack23"
    m = re.search(r"\bpack\s*(\d+)\b", t)
    if m:
        return int(m.group(1)), 0.75
    # Number followed by container words: "24 Glass Bottles", "12 cans", "6 bottles"
    m = re.search(
        r"(\d+)\s*(?:glass\s*)?(?:bottles?|cans?|tins?|cartons?|boxes?|packets?|sachets?|bags?)\b",
        t,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), 0.90
    # Number followed by "count" or "ct"
    m = re.search(r"(\d+)\s*(?:count|ct)\b", t, re.IGNORECASE)
    if m:
        return int(m.group(1)), 0.85
    # Default single
    return 1, 0.95


def parse_attribute_volume_pack(attr_str: str):
    vol_ml = 0.0
    vol_conf = 0.0
    pack_qty = 1
    pack_conf = 0.0
    if not attr_str or attr_str == "nan":
        return vol_ml, vol_conf, pack_qty, pack_conf
    m_vol = re.search(r"Volume:\s*(\d+(?:\.\d+)?)", attr_str, re.IGNORECASE)
    if m_vol:
        vol_ml = float(m_vol.group(1))
        vol_conf = 0.9
    m_pack = re.search(r"Count per Unit:\s*(\d+)", attr_str, re.IGNORECASE)
    if m_pack:
        pack_qty = int(m_pack.group(1))
        pack_conf = 0.9
    return vol_ml, vol_conf, pack_qty, pack_conf


def extract_salient_tokens(titles: list) -> list:
    """
    Extract tokens that are consistent across titles and are not generic.
    We count tokens after removing stopwords, brand tokens, volume/pack patterns,
    and then select tokens that appear in at least 2 titles or have high frequency.
    """
    token_counter = Counter()
    title_count = len(titles)
    for title in titles:
        t = normalize_text(title)
        # Remove volume/pack patterns
        t = re.sub(
            r"\b\d+(\.\d+)?\s*(ml|l|lt|ltr|liter|litre|cl|centiliter|oz|fl oz|qt|gal|ounce|fluid ounce|pack|case|pcs?|pieces?|units?|x)\b",
            " ",
            t,
            flags=re.IGNORECASE,
        )
        # Remove brand? We'll handle brand separately outside; here we just split
        tokens = t.split()
        # Filter stopwords and short/meaningless tokens
        tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
        token_counter.update(tokens)
    # Keep tokens that appear in at least 2 titles, or frequency >= 2
    salient = [tok for tok, cnt in token_counter.items() if cnt >= 2]
    # Also include tokens that appear in all titles if title_count > 1
    if title_count > 1:
        all_titles_tokens = set()
        for i, title in enumerate(titles):
            t = normalize_text(title)
            t = re.sub(
                r"\b\d+(\.\d+)?\s*(ml|l|lt|ltr|liter|litre|cl|centiliter|oz|fl oz|qt|gal|ounce|fluid ounce|pack|case|pcs?|pieces?|units?|x)\b",
                " ",
                t,
                flags=re.IGNORECASE,
            )
            tokens = set(t.split())
            tokens = {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1}
            if i == 0:
                all_titles_tokens = tokens
            else:
                all_titles_tokens &= tokens
        salient = list(set(salient) | all_titles_tokens)
    return sorted(salient)


def extract_all(sku_name: str, attribute: str) -> dict:
    """Extract structured fields plus salient tokens from a single SKU row."""
    t = normalize_text(sku_name)
    # Flavor and type
    flavor = (
        FLAVOR_PATTERN.search(t).group(1).lower() if FLAVOR_PATTERN.search(t) else ""
    )
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

    return {
        "flavor": flavor,
        "type": ptype,
        "volume_ml": volume_ml,
        "volume_confidence": volume_conf,
        "volume_raw": volume_raw,
        "volume_status": volume_status,
        "pack_qty": pack_qty,
        "pack_confidence": pack_conf,
    }


# ============================================================================
# GATING
# ============================================================================
def three_way_gate(
    attrs1,
    attrs2,
    vol_tolerance=0.05,
    raw_conf_threshold=0.85,
    consistency_fallback_threshold=0.3,
):
    # raw confidence check
    if (
        attrs1["volume_confidence"] < raw_conf_threshold
        or attrs2["volume_confidence"] < raw_conf_threshold
    ):
        return {"decision": "fallback", "reason": "Low raw volume confidence"}
    if (
        attrs1["pack_confidence"] < raw_conf_threshold
        or attrs2["pack_confidence"] < raw_conf_threshold
    ):
        return {"decision": "fallback", "reason": "Low raw pack confidence"}

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
        return {"decision": "hard_no", "reason": "No volume overlap"}

    # pack overlap
    pack_overlap = attrs1["pack_set"] & attrs2["pack_set"]
    if not pack_overlap:
        return {"decision": "hard_no", "reason": "No pack overlap"}

    # flavor check (only if both have a non-empty flavor): kills the
    # flavor-blind proceed tail (ZUMOSOL apple vs orange nectar at the
    # same size/pack — measured 258 proceed pairs below sim 0.40)
    flavor1 = attrs1.get("mode_flavor", "")
    flavor2 = attrs2.get("mode_flavor", "")
    if flavor1 and flavor2 and flavor1 != flavor2:
        return {
            "decision": "hard_no",
            "reason": f"Flavor mismatch: {flavor1} vs {flavor2}",
        }

    # consistency check
    if (
        attrs1["volume_consistency"] < consistency_fallback_threshold
        or attrs2["volume_consistency"] < consistency_fallback_threshold
        or attrs1["pack_consistency"] < consistency_fallback_threshold
        or attrs2["pack_consistency"] < consistency_fallback_threshold
    ):
        return {"decision": "fallback", "reason": "Overlap but low consistency"}

    return {"decision": "proceed", "reason": "Volume, pack, flavor all compatible"}


# ============================================================================
# SIMILARITY
# ============================================================================
def jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


# Placeholder for embeddings
def embedding_similarity(text1, text2):
    # to be implemented with sentence-transformers later
    pass


# ============================================================================
# CANONICAL
# ============================================================================


MINIMAL_STOPWORDS = _load_stopwords("MINIMAL_STOPWORDS")


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
    "no_sugar",
    "sugar_free",
    "added_sugar",
    "with_pulp",
    "no_pulp",
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
def generate_ngrams(tokens, n):
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# -----------------------------------------------------------------------------
# Discriminative n‑gram extraction
# -----------------------------------------------------------------------------
def extract_discriminative_ngrams(
    titles, attributes, brand_tokens, global_idf, brand_idf, top_k=5
):
    """
    Select n‑grams (1‑4) with highest TF‑IDF, considering global and within‑brand IDF.
    """
    # Combine all text into token list
    tokens = []
    for title, attr in zip(titles, attributes):
        text = normalize_text(title) + " " + normalize_text(attr)
        text = re.sub(
            r"\b\d+(\.\d+)?\s*(ml|l|lt|ltr|liter|litre|cl|centiliter|oz|fl oz|qt|gal|ounce|fluid ounce|pack|case|pcs?|pieces?|units?|x)\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        toks = text.split()
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
    for keep in KEEP_TOKENS:
        hit = (keep in doc_tokens) if "_" not in keep else (keep in bigrams)
        if hit and keep not in selected:
            selected.append(keep)
            if len(selected) >= top_k + 3:
                break

    return selected[: top_k + 3]


# -----------------------------------------------------------------------------
# Canonical generation (now uses n‑grams)
# -----------------------------------------------------------------------------
def generate_canonical(gtin, brand, rows, global_idf, brand_idf):
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
    pack_set = {x["pack_qty"] for x in extracted}

    # Confidence / consistency
    vol_confs = [x["volume_confidence"] for x in extracted if x["volume_ml"] > 0]
    pack_confs = [x["pack_confidence"] for x in extracted]
    vol_conf = sum(vol_confs) / len(vol_confs) if vol_confs else 0.0
    pack_conf = sum(pack_confs) / len(pack_confs) if pack_confs else 0.0
    n = len(extracted)
    # consistency = share of rows agreeing with the MOST COMMON value.
    # The old formula divided conflicts by ROW COUNT n, so a 41k-row group
    # with 2,000 distinct volumes scored 0.95 "consistent" — more rows made
    # contradiction look BETTER. Mode-share is scale-free and monotone.
    vol_mode = Counter(x["volume_ml"] for x in extracted if x["volume_ml"] > 0)
    pack_mode = Counter(x["pack_qty"] for x in extracted)
    # mode share over rows that HAVE a volume (unknown-volume rows don't vote)
    volume_consistency = (
        (vol_mode.most_common(1)[0][1] / sum(vol_mode.values())) if vol_mode else 1.0
    )
    pack_consistency = (pack_mode.most_common(1)[0][1] / n) if n else 1.0

    # Build canonical string
    parts = [brand_norm]
    if mode_flavor:
        parts.append(mode_flavor)
    if mode_type:
        parts.append(mode_type)
    parts.extend(salient_ngrams)
    canonical = " ".join(parts)

    return {
        "gtin": gtin,
        "canonical": canonical,
        "mode_brand": brand,
        "mode_flavor": mode_flavor,
        "mode_type": mode_type,
        "salient_ngrams": salient_ngrams,
        "volume_set": volume_set,
        "pack_set": pack_set,
        "volume_confidence": round(vol_conf, 3),
        "pack_confidence": round(pack_conf, 3),
        "volume_consistency": round(volume_consistency, 3),
        "pack_consistency": round(pack_consistency, 3),
        "n_titles": n,
    }


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


def clean_sku_text(title: str, attribute: str = "", brand: str = "") -> str:
    """The OFFICIAL cleaned sku text the model sees.

    normalize(title) + ' ' + normalize(attribute), volume/pack tokens
    stripped, MINIMAL_STOPWORDS + single-char tokens removed, then the
    number-token reference strip (bare volumes/counts/multipliers/codes
    removed; name-embedded digits b12/o2/alkaline88 and this row's numeric
    brand tokens survive — data/number_tokens_reference.csv, 95.2% coverage).
    """
    text = normalize_text(title) + " " + normalize_text(attribute or "")
    text = _VOLUME_PACK_RE.sub(" ", text)
    toks = [t for t in text.split() if t not in MINIMAL_STOPWORDS and len(t) > 1]
    return strip_number_tokens(" ".join(toks), spell_numeric_brand(brand or ""))


def load_canonical_map() -> dict[str, str]:
    """gtin -> canonical string, from the pipeline's canonical_records.csv
    (file name via 00_config SSOT)."""
    df = pd.read_csv(RESULTS / F["canonical_records"], dtype={"gtin": str})
    return dict(zip(df["gtin"], df["canonical"]))


# tokens that are SEMANTIC despite carrying digits (same whitelist the
# number-reference build measured: b12/o2/alkaline88 brands etc.) survive;
# everything else with a digit is stripped from the MODEL-side canonical.
_CANON_KEEP_DIGIT = re.compile(
    r"^(?:b\d+|o2|co2|h2o?|ph\d*(?:\.\d+)?|\d+(?:\.\d+)?ph\.?)$", re.IGNORECASE
)


def canonical_model_text(canonical: str) -> str:
    """Number-free canonical for the MODEL payload.

    The gate's canonical (canonical_records.csv) keeps volumes/packs/metric
    mentions — that file drives hard_no decisions. The MODEL must never see
    numbers (owner spec): strip every digit token that is not a semantic
    nutrient/brand whitelist member. Underscore n-gram compounds keep their
    alpha part when it survives (kr_white_grape_flavored → the words stay,
    the 12x500 volume dies with the compound).
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
    for t, a in zip(df["title"].fillna(""), df["attributes"].fillna("")):
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


def reference_path():
    return DATA_DIR / F["number_reference"]


def load_verdicts() -> dict[str, str] | None:
    """token -> verdict map from the reference CSV (None if not built yet)."""
    p = reference_path()
    if not p.exists():
        # SSOT missing: SAY IT — the caller falls back to regex rules only
        # (95.2% coverage comes from the CSV; regex-only is a degradation)
        print(
            f"[numbers] reference CSV missing ({p}) — regex-rule fallback only",
            flush=True,
        )
        return None
    df = pd.read_csv(p, dtype={"token": str})
    return dict(zip(df["token"], df["verdict"]))


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
    for t in text.split():
        if not re.search(r"\d", t):
            out.append(t)
            continue
        v = (verdicts or {}).get(t)
        if v is None:
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
    return " ".join(out)


# ============================================================================
# PIPELINE
# ============================================================================
from collections import defaultdict


def run_within_brand_pipeline(df_full):
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
    df_full = df_full[gtin_valid]
    if n_before != len(df_full):
        print(
            f"[gtin-guard] dropped {n_before - len(df_full):,} rows with "
            f"missing/NaN gtin (they cannot be grouped by product)",
            flush=True,
        )

    # Group by GTIN
    grouped = (
        df_full.groupby("gtin")
        .agg(
            rows=(
                "sku_name_eng",
                lambda x: list(zip(x, df_full.loc[x.index, "attribute"])),
            ),
            brand=("brand", lambda x: Counter(x).most_common(1)[0][0]),
        )
        .reset_index()
    )

    # Build global n‑gram IDF from all GTINs
    rows_by_gtin = {row["gtin"]: row["rows"] for _, row in grouped.iterrows()}
    global_idf = NgramIDF(rows_by_gtin)

    # Precompute within‑brand IDF per brand
    brand_to_gtins = defaultdict(list)
    for gtin, brand in zip(grouped["gtin"], grouped["brand"]):
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
        canonical_records.append(
            generate_canonical(
                row["gtin"],
                row["brand"],
                row["rows"],
                global_idf,
                brand_idf_map[brand_key],
            )
        )
    df_canon = pd.DataFrame(canonical_records)

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
    results_df = pd.DataFrame(results)

    # TRAIN_GPU writes ONLY inside its own tree (lib.common RESULTS —
    # the repo's results dir must never be touched by the standalone lane).
    RESULTS.mkdir(parents=True, exist_ok=True)
    df_canon.to_csv(RESULTS / F["canonical_records"], index=False)
    results_df.to_csv(RESULTS / F["gate_results"], index=False)

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
        row_bc  : np.ndarray — barcode per payload entry (gtin for canonicals)
        pos     : np.ndarray (N,2) — (sku_row, canon_idx) for every row whose
                  barcode has a canonical
        neg     : np.ndarray (M,2) — (rep_row(g1), canon(g2)) and mirror, for
                  every gate hard-no pair with similarity >= threshold
        stats   : dict — counts (nothing dropped silently)
    """
    cfg = load_config()
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
    if payload_variant == "full":
        sku_texts = [clean_sku_text(t, a) for t, a in zip(title, attrs)]
    elif payload_variant == "title_only":
        sku_texts = [clean_sku_text(t) for t in title]
    else:
        raise SystemExit(f"unknown payload variant: {payload_variant}")

    # ── payload: sku rows + canonical entries (in sorted-gtin order) ──
    payload = list(sku_texts)
    row_bc = [str(x) for x in bc]
    canon_gtins = sorted(canon_map)
    canon_start = len(payload)
    gtin_to_canon_idx = {g: canon_start + i for i, g in enumerate(canon_gtins)}
    # MODEL payload: number-free canonical variant — the gate's CSV keeps
    # numbers (hard_no volume/pack decisions), the model never sees them
    payload.extend(canonical_model_text(canon_map[g]) for g in canon_gtins)
    row_bc.extend(canon_gtins)

    # ── positives: every row whose barcode has a canonical ──
    pos = np.array(
        [(i, gtin_to_canon_idx[g]) for i, g in enumerate(bc) if g in gtin_to_canon_idx],
        dtype=int,
    ).reshape(-1, 2)

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
    neg_mask = (gates["gate_decision"] == "hard_no") & (gates["similarity"] >= thr_neg)
    neg_gates = gates[neg_mask]
    a = neg_gates["gtin1"].map(gtin_to_row)
    b = neg_gates["gtin2"].map(gtin_to_row)
    ca = neg_gates["gtin2"].map(gtin_to_canon_idx)
    cb = neg_gates["gtin1"].map(gtin_to_canon_idx)
    ok1 = a.notna() & ca.notna()
    ok2 = b.notna() & cb.notna()
    fwd = np.stack([a[ok1].astype(int), ca[ok1].astype(int)], axis=1)
    rev = np.stack([b[ok2].astype(int), cb[ok2].astype(int)], axis=1)
    neg = np.vstack([fwd, rev]) if len(fwd) or len(rev) else np.empty((0, 2), dtype=int)

    stats = {
        "n_rows": len(df),
        "n_sku_with_canonical": len(pos),
        "n_canonicals": len(canon_gtins),
        "n_pos_gate_rows": int(
            (
                (gates["gate_decision"] == "proceed") & (gates["similarity"] >= thr_pos)
            ).sum()
        ),
        "n_neg_gate_rows": int(neg_mask.sum()),
        "n_neg_resolved": len(neg),
        "n_neg_dropped": int(neg_mask.sum() * 2 - len(neg)),
    }
    return {
        "payload": payload,
        "row_bc": np.array(row_bc),
        "pos": pos,
        "neg": neg,
        "gtin_to_row": gtin_to_row,
        "stats": stats,
    }
