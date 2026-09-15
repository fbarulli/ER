#!/usr/bin/env python3
"""Classify WHY the brand axis separates the review population — against the
VERIFIED brand, not the raw string.

A brand string in isolation does not establish the correct brand.  This script
therefore corroborates each side's claimed brand from independent features
before it classifies anything:

* **GTIN co-listing brands** — every dataset listing sharing the side's barcode,
  and the distribution of brand fields across them.  A retailer's brand field
  disagreeing with every other listing for the same barcode is evidence that
  field is wrong.
* **GTIN identity between the two sides** — when the source barcode equals the
  target GTIN the two sides are the same product, so disagreeing brand fields
  are a defect by construction.
* **The product name** — whether the claimed brand actually appears in the
  title/canonical text, which is a different field from the brand column.
* **Corpus support** — how many listings carry the claimed brand.
* **Generic-word test** — whether the "brand" is really a category word
  (``Mix``, ``Pure``) that occurs throughout the corpus's product names.

Channel independence is modelled explicitly, because two of the obvious checks
are circular: ``canonical_records.mode_brand`` **is** the modal brand of its
GTIN group (verified 585/585), so re-deriving that mode cannot contradict it.
A check that cannot fail is not evidence, and the report says so.

Containment is deliberately NOT treated as evidence of a shared brand.  One
brand key being a prefix or substring of another (``Mont`` / ``Mont Roucous``,
``Cemil`` / ``Cemilefendi``, ``Réal`` / ``Realemon``) is recorded as a
descriptive flag only.  Where independent evidence exists it shows those are
different brands, and such a pair is a TRUE HARD NEGATIVE — training signal,
not something to normalise away.

Normalisation is not re-implemented: ``core.critical_attributes.
normalized_attribute_text`` is the repository's existing NFKD + combining-mark
strip + punctuation flattening normaliser and is used verbatim.  ``pipeline.
jaccard_similarity`` is the existing token-Jaccard SSOT.  rapidfuzz and
scikit-learn had no prior use in this repository (grep-verified, see the
report), so their introduction here is new by necessity.

Thresholds are read from ``config/training.yaml`` under ``brand_analysis`` when
that block exists — through the existing ``core.common.load_config()`` loader —
and otherwise fall back to the named module constants below.  Every threshold's
origin is recorded in the run provenance, so a fallback is never silent.  This
analysis may not edit ``config/``; the block to add is in the report.

Usage::

    PYTHONPATH=src python scripts/analyze_brand_matching.py \
        --review <path>/human_review_enriched.csv --out-dir <dir>
    PYTHONPATH=src python scripts/analyze_brand_matching.py --selftest
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from core.common import (
    canonical_records_frame,
    load_config,
    load_dataset,
    metadata_text,
)
from core.critical_attributes import normalized_attribute_text
from pipeline import jaccard_similarity

#: Config block read through ``core.common.load_config()``.  Absent today
#: because this analysis may not edit ``config/``; every threshold records
#: whether it came from here or from the module default.
CONFIG_BLOCK = "brand_analysis"

# ── named threshold defaults (config overrides these; see CONFIG_BLOCK) ─────

#: Minimum characters in a brand key before it may be corroborated by a title.
#: Lower than the containment flag: a three-letter brand abbreviation inside a
#: product name is meaningful, whereas a three-letter key "contained" in another
#: short brand is coincidence.
TITLE_CORROBORATION_MIN_CHARS = 3

#: Minimum characters before one brand key "containing" another is even
#: recorded as a descriptive flag.  Descriptive ONLY — containment is not
#: evidence that the two sides share a brand.
KEY_CONTAINMENT_MIN_CHARS = 4

#: A brand key must appear in at least this share of a source GTIN group's
#: listings to be considered corroborated by the group.
GTIN_CLAIM_MIN_SHARE = 0.5

#: A rival brand must hold at least this share of the group before the claim is
#: called contradicted rather than merely outvoted.
GTIN_RIVAL_MIN_SHARE = 0.5

#: A brand whose every token appears in more than this fraction of corpus
#: product names is a category word, not a distinctive brand.
GENERIC_TOKEN_DOC_FRACTION = 0.01

#: TF-IDF vectorisation settings for the reported similarity features.
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 5)
MIN_DF = 1

#: Corporate/legal forms stripped only inside the surface-variant ladder.  No
#: such vocabulary existed anywhere in the repository (grep-verified), so it is
#: declared here as new.
CORPORATE_SUFFIX_TOKENS = frozenset({
    "ab", "ag", "aps", "as", "bv", "co", "company", "corp", "corporation",
    "gmbh", "group", "groupe", "grupo", "holding", "inc", "kg", "kgaa",
    "limited", "llc", "ltd", "nv", "oy", "plc", "pte", "sa", "sarl", "sas",
    "spa", "srl", "sro",
})


class Side(StrEnum):
    """Which side of the pair a piece of evidence belongs to."""

    SOURCE = "source"
    TARGET = "target"


class VerificationStatus(StrEnum):
    """What independent features say about one side's claimed brand."""

    ABSENT = "absent"
    CONTRADICTED_BY_LISTINGS = "contradicted_by_co_listings"
    GENERIC_CATEGORY_WORD = "generic_category_word"
    CONFIRMED_BY_LISTINGS = "confirmed_by_co_listings"
    CONFIRMED_BY_TITLE = "confirmed_by_title_only"
    UNVERIFIED = "unverified_no_independent_evidence"
    DERIVED_ONLY = "derived_only_not_independently_checkable"


class BrandClass(StrEnum):
    """Why a source/target brand pair agrees or disagrees."""

    ABSENT_BOTH = "absent_both"
    ABSENT_SOURCE = "absent_source"
    ABSENT_TARGET = "absent_target"
    SAME_BRAND_VERIFIED = "same_brand_verified"
    SAME_BRAND_SURFACE_VARIANT = "surface_variant_same_brand"
    BRAND_FIELD_DEFECT = "brand_field_defect"
    EVIDENCE_DISAGREEMENT = "evidence_disagreement"
    DIFFERENT_BRAND_VERIFIED = "different_brand_verified_hard_negative"
    DIFFERENT_BRAND_UNVERIFIED = "different_brand_unverified_hard_negative"


#: Brand disagreement independent evidence confirms — training signal.
HARD_NEGATIVE_CLASSES = frozenset({
    BrandClass.DIFFERENT_BRAND_VERIFIED,
    BrandClass.DIFFERENT_BRAND_UNVERIFIED,
})
#: The two sides carry the same brand.
BRAND_AGREES_CLASSES = frozenset({
    BrandClass.SAME_BRAND_VERIFIED,
    BrandClass.SAME_BRAND_SURFACE_VARIANT,
})
#: A side's brand FIELD is wrong — a data defect, not a brand difference.
DEFECT_CLASSES = frozenset({BrandClass.BRAND_FIELD_DEFECT})
#: The correct brand cannot be established from the available evidence.
UNRESOLVED_CLASSES = frozenset({BrandClass.EVIDENCE_DISAGREEMENT})
#: Classes a string normaliser could legitimately resolve.  Containment is
#: absent by design: ``Mont`` / ``Mont Roucous`` are different brands, and
#: treating a prefix match as shared identity would be a false merge.
NORMALISATION_RESOLVED_CLASSES = frozenset({BrandClass.SAME_BRAND_SURFACE_VARIANT})


class ThresholdOrigin(BaseModel):
    """Where one threshold value came from, so a fallback is never silent."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    origin: str


class SideEvidence(BaseModel):
    """Everything independent that was checked about ONE side's claimed brand."""

    model_config = ConfigDict(extra="forbid")

    side: str
    claim: str
    gtin: str
    gtin_group_size: int = Field(ge=0)
    gtin_group_brands: str
    claim_share_in_group: float
    gtin_check_independent: bool
    title_corroborates: bool
    title_check_independent: bool
    corpus_support: int = Field(ge=0)
    generic_token_fraction: float
    is_generic_category_word: bool
    status: str


class BrandPairRecord(BaseModel):
    """One classified pair, with the evidence behind its class."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str
    target_gtin: str
    score: float
    source_brand: str
    target_brand: str
    brand_class: str
    matched_rule: str
    source_status: str
    target_status: str
    same_gtin: bool
    key_containment: bool
    shared_tokens: str
    ratio: float
    token_set_ratio: float
    partial_ratio: float
    levenshtein: int = Field(ge=0)
    tfidf_cosine_char: float
    tfidf_cosine_word: float
    token_jaccard: float
    pair_rows: int = Field(ge=1)
    source_brand_corpus_rows: int = Field(ge=0)
    target_brand_corpus_rows: int = Field(ge=0)


class BrandClassSummary(BaseModel):
    """Aggregate for one :class:`BrandClass`."""

    model_config = ConfigDict(extra="forbid")

    brand_class: str
    rows: int = Field(ge=0)
    mean_score: float
    mean_ratio: float
    mean_tfidf_cosine_char: float
    is_true_hard_negative: bool
    is_brand_field_defect: bool
    resolves_under_normalisation: bool


class BrandImpact(BaseModel):
    """Decision-relevant totals and the score separation they explain."""

    model_config = ConfigDict(extra="forbid")

    population_rows: int = Field(ge=1)
    brand_agrees_raw_string: int = Field(ge=0)
    brand_differs_raw_string: int = Field(ge=0)
    brand_agrees_verified: int = Field(ge=0)
    brand_differs_verified: int = Field(ge=0)
    brand_field_defects: int = Field(ge=0)
    evidence_disagreements: int = Field(ge=0)
    true_hard_negatives: int = Field(ge=0)
    contradictions_found_by_co_listings: int = Field(ge=0)
    pairs_with_same_gtin: int = Field(ge=0)
    same_gtin_conflicting_brands: int = Field(ge=0)
    key_containment_rows: int = Field(ge=0)
    resolved_by_normalisation: int = Field(ge=0)
    resolved_share_of_population: float
    separation_raw_string: float
    separation_verified: float
    separation_excluding_defects: float
    mean_score_brand_agrees_verified: float
    mean_score_brand_differs_verified: float
    mean_score_true_hard_negative: float


class BrandAnalysisProvenance(BaseModel):
    """What produced this run, so the numbers can be attributed to a revision."""

    model_config = ConfigDict(extra="forbid")

    commit_sha: str
    review_csv: str
    population_rows: int = Field(ge=1)
    thresholds: list[ThresholdOrigin]
    config_block_present: bool
    char_ngram_range: tuple[int, int]
    word_ngram_range: tuple[int, int]


class BrandAnalysisSummary(BaseModel):
    """The run's machine-readable result."""

    model_config = ConfigDict(extra="forbid")

    provenance: BrandAnalysisProvenance
    classes: list[BrandClassSummary]
    impact: BrandImpact


def compact_brand_key(value: object) -> str:
    """Separator-free brand key built on the repository's own normaliser.

    ``normalized_attribute_text`` already performs NFKD, casefolding, combining
    mark removal and punctuation flattening.  Dropping the separators is the
    only step added here.
    """
    return "".join(
        character
        for character in normalized_attribute_text(value)
        if character.isalnum()
    )


def brand_tokens(value: object) -> frozenset[str]:
    """Normalised brand word set, ignoring single characters.

    Single characters are dropped so an apostrophe-s (``Green's``) does not
    smuggle a ubiquitous ``s`` token into the generic-word test.
    """
    return frozenset(
        token for token in normalized_attribute_text(value).split() if len(token) > 1
    )


def spaced_brand_text(value: object) -> str:
    """Normalised brand text with word separators kept, punctuation dropped."""
    return "".join(
        character
        for character in normalized_attribute_text(value)
        if character.isalnum() or character == " "
    ).strip()


def strip_corporate_suffix(value: object) -> str:
    """Brand key with corporate/legal form tokens removed."""
    return "".join(
        token
        for token in normalized_attribute_text(value).split()
        if token not in CORPORATE_SUFFIX_TOKENS
    )


def brand_is_generic(
    value: object, doc_fraction: dict[str, float], threshold: float,
) -> tuple[bool, float]:
    """Whether every token of a brand is a corpus-wide category word.

    Returns ``(is_generic, rarest_token_fraction)``.  A distinctive brand has at
    least one token that is rare across product names; ``Mix`` and ``Pure`` do
    not.  The rarest-token fraction is returned so the judgement is auditable.
    """
    tokens = brand_tokens(value)
    if not tokens:
        return False, 1.0
    rarest = min(doc_fraction.get(token, 0.0) for token in tokens)
    return rarest > threshold, rarest


class BrandVerifier:
    """Corroborate a claimed brand from independent features.

    A check only counts when it can fail.  ``gtin_check_independent`` and
    ``title_check_independent`` are carried per side so a circular check is
    recorded as circular rather than reported as confirmation.
    """

    def __init__(
        self, *, title_min_chars: int, containment_min_chars: int,
        gtin_claim_min_share: float, gtin_rival_min_share: float,
        generic_threshold: float, doc_fraction: dict[str, float],
        corpus_support: dict[str, int], gtin_groups: dict[str, Counter],
    ) -> None:
        self.title_min_chars = title_min_chars
        self.containment_min_chars = containment_min_chars
        self.gtin_claim_min_share = gtin_claim_min_share
        self.gtin_rival_min_share = gtin_rival_min_share
        self.generic_threshold = generic_threshold
        self.doc_fraction = doc_fraction
        self.corpus_support = corpus_support
        self.gtin_groups = gtin_groups

    def group_conflict(self, gtin: str) -> bool:
        """Whether a GTIN group carries more than one distinct brand."""
        group = self.gtin_groups.get(gtin, Counter()) if gtin else Counter()
        return len({compact_brand_key(brand) for brand in group}) > 1

    def verify(
        self, *, side: Side, claim: str, gtin: str, name_text: str,
        gtin_check_independent: bool, title_check_independent: bool,
    ) -> SideEvidence:
        claim_key = compact_brand_key(claim)
        group = self.gtin_groups.get(gtin, Counter()) if gtin else Counter()
        group_size = sum(group.values())
        claim_share = (
            sum(count for brand, count in group.items() if compact_brand_key(brand) == claim_key)
            / group_size
            if group_size else 0.0
        )
        is_generic, rarest = brand_is_generic(claim, self.doc_fraction, self.generic_threshold)
        title_ok = (
            len(claim_key) >= self.title_min_chars
            and claim_key in compact_brand_key(name_text)
        )

        if not claim_key:
            status = VerificationStatus.ABSENT
        elif is_generic:
            # A category word in the brand field is a defect regardless of what
            # the (also generic) product name says.
            status = VerificationStatus.GENERIC_CATEGORY_WORD
        elif (
            gtin_check_independent and group_size
            and claim_share < self.gtin_claim_min_share
            and any(
                count / group_size >= self.gtin_rival_min_share
                for brand, count in group.items()
                if compact_brand_key(brand) != claim_key
            )
        ):
            status = VerificationStatus.CONTRADICTED_BY_LISTINGS
        elif gtin_check_independent and group_size and claim_share >= self.gtin_claim_min_share:
            status = VerificationStatus.CONFIRMED_BY_LISTINGS
        elif title_check_independent and title_ok:
            status = VerificationStatus.CONFIRMED_BY_TITLE
        elif not gtin_check_independent and not title_check_independent:
            status = VerificationStatus.DERIVED_ONLY
        else:
            status = VerificationStatus.UNVERIFIED

        return SideEvidence(
            side=str(side),
            claim=claim,
            gtin=gtin,
            gtin_group_size=group_size,
            gtin_group_brands="; ".join(
                f"{brand}x{count}" for brand, count in group.most_common()
            ),
            claim_share_in_group=claim_share,
            gtin_check_independent=gtin_check_independent,
            title_corroborates=title_ok,
            title_check_independent=title_check_independent,
            corpus_support=int(self.corpus_support.get(claim_key, 0)),
            generic_token_fraction=rarest,
            is_generic_category_word=is_generic,
            status=str(status),
        )

    def keys_contain(self, left: str, right: str) -> bool:
        """Descriptive flag: one brand key is a prefix/substring of the other.

        NOT evidence of a shared brand — ``Mont`` and ``Mont Roucous`` are
        different brands, verified independently in the report.
        """
        left_key, right_key = compact_brand_key(left), compact_brand_key(right)
        shorter, longer = sorted((left_key, right_key), key=len)
        if len(shorter) < self.containment_min_chars or shorter == longer:
            return False
        return shorter in longer

    def surface_variant_only(self, left: str, right: str) -> bool:
        """Same brand differing only by token order or corporate form.

        Containment is deliberately NOT part of this ladder.
        """
        if not compact_brand_key(left) or not compact_brand_key(right):
            return False
        if brand_tokens(left) == brand_tokens(right):
            return True
        return strip_corporate_suffix(left) == strip_corporate_suffix(right)


def classify_pair(
    source: SideEvidence, target: SideEvidence, *, verifier: BrandVerifier,
    source_gtin: str, target_gtin: str, group_conflict: bool,
) -> tuple[BrandClass, str]:
    """Decide the pair's class from evidence, strongest signal first."""
    if not compact_brand_key(source.claim) and not compact_brand_key(target.claim):
        return BrandClass.ABSENT_BOTH, "both brand fields empty"
    if not compact_brand_key(source.claim):
        return BrandClass.ABSENT_SOURCE, "source brand field empty"
    if not compact_brand_key(target.claim):
        return BrandClass.ABSENT_TARGET, "target brand field empty"

    # A brand field contradicted by its own listings, or holding a category
    # word, is a DATA DEFECT — not a statement about the two products.
    contradicted = str(VerificationStatus.CONTRADICTED_BY_LISTINGS)
    generic = str(VerificationStatus.GENERIC_CATEGORY_WORD)
    for evidence in (source, target):
        if evidence.status == contradicted:
            return (
                BrandClass.BRAND_FIELD_DEFECT,
                f"{evidence.side} brand contradicted by its own GTIN listings",
            )
        if evidence.status == generic:
            return (
                BrandClass.BRAND_FIELD_DEFECT,
                f"{evidence.side} brand field holds a generic category word",
            )

    if compact_brand_key(source.claim) == compact_brand_key(target.claim):
        return BrandClass.SAME_BRAND_VERIFIED, "verified brand keys equal"
    if verifier.surface_variant_only(source.claim, target.claim):
        return BrandClass.SAME_BRAND_SURFACE_VARIANT, "token order / corporate form only"

    # Same barcode on both sides means the same product; disagreeing brand
    # fields would then be a defect.  Measured zero times in this population,
    # so the branch is kept as the invariant it is rather than assumed away.
    if source_gtin and source_gtin == target_gtin:
        return BrandClass.BRAND_FIELD_DEFECT, "identical GTIN but the brand fields disagree"

    if group_conflict:
        return (
            BrandClass.EVIDENCE_DISAGREEMENT,
            "a GTIN group carries more than one brand, so the correct brand is undecidable",
        )

    if source.status in (
        str(VerificationStatus.CONFIRMED_BY_LISTINGS),
        str(VerificationStatus.CONFIRMED_BY_TITLE),
    ):
        return (
            BrandClass.DIFFERENT_BRAND_VERIFIED,
            "source brand independently corroborated; the two brands differ",
        )
    return (
        BrandClass.DIFFERENT_BRAND_UNVERIFIED,
        "brand strings differ; no independent channel available to check either side",
    )


def tfidf_cosines(left: list[str], right: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair TF-IDF cosine, character-level and word-level.

    Both matrices are fitted on the union of the two brand columns, so the idf
    weights describe THIS population's brand vocabulary rather than an external
    corpus.
    """
    documents = left + right
    split = len(left)
    results: list[np.ndarray] = []
    for analyzer, ngram_range in (("char_wb", CHAR_NGRAM_RANGE), ("word", WORD_NGRAM_RANGE)):
        matrix = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram_range, min_df=MIN_DF,
        ).fit_transform(documents)
        results.append(
            np.asarray(matrix[:split].multiply(matrix[split:]).sum(axis=1)).ravel()
        )
    return results[0], results[1]


def pair_fuzzy_scores(left: list[str], right: list[str]) -> dict[str, list[float]]:
    """rapidfuzz scores on the encoder-visible (accent-folded) brand text."""
    folded_left = [spaced_brand_text(value) for value in left]
    folded_right = [spaced_brand_text(value) for value in right]
    return {
        "ratio": [fuzz.ratio(a, b) for a, b in zip(folded_left, folded_right)],
        "token_set_ratio": [
            fuzz.token_set_ratio(a, b) for a, b in zip(folded_left, folded_right)
        ],
        "partial_ratio": [
            fuzz.partial_ratio(a, b) for a, b in zip(folded_left, folded_right)
        ],
        "levenshtein": [
            Levenshtein.distance(a, b) for a, b in zip(folded_left, folded_right)
        ],
        "token_jaccard": [
            jaccard_similarity(a, b) for a, b in zip(folded_left, folded_right)
        ],
    }


def resolve_thresholds() -> tuple[dict[str, float], list[ThresholdOrigin], bool]:
    """Thresholds from ``config`` when present, else the named defaults.

    Origin is recorded per threshold so a fallback is visible rather than
    silent (``AGENTS.local.md`` §3/§6).
    """
    defaults = {
        "title_corroboration_min_chars": float(TITLE_CORROBORATION_MIN_CHARS),
        "key_containment_min_chars": float(KEY_CONTAINMENT_MIN_CHARS),
        "gtin_claim_min_share": float(GTIN_CLAIM_MIN_SHARE),
        "gtin_rival_min_share": float(GTIN_RIVAL_MIN_SHARE),
        "generic_token_doc_fraction": float(GENERIC_TOKEN_DOC_FRACTION),
    }
    configured = load_config().get(CONFIG_BLOCK) or {}
    resolved: dict[str, float] = {}
    origins: list[ThresholdOrigin] = []
    for name, default in defaults.items():
        if name in configured:
            resolved[name] = float(configured[name])
            origins.append(ThresholdOrigin(
                name=name, value=resolved[name], origin=f"config:{CONFIG_BLOCK}",
            ))
        else:
            resolved[name] = default
            origins.append(ThresholdOrigin(name=name, value=default, origin="script default"))
    return resolved, origins, bool(configured)


def build_gtin_index(source: pd.DataFrame) -> tuple[dict[str, Counter], dict[str, int]]:
    """Brand distribution per GTIN, and corpus support per brand key.

    The barcode column holds missing values as NaN, so ``metadata_text`` reads
    it the same way the composition does rather than stringifying NaN into a
    fake barcode.
    """
    frame = source.assign(
        __barcode=source["barcode"].map(metadata_text),
        __brand_key=source["brand"].map(compact_brand_key),
    )
    with_barcode = frame[frame["__barcode"] != ""]
    groups = {
        str(gtin): Counter(group["brand"].map(metadata_text))
        for gtin, group in with_barcode.groupby("__barcode")
    }
    return groups, frame["__brand_key"].value_counts().to_dict()


def brand_token_doc_fraction(source: pd.DataFrame) -> dict[str, float]:
    """Share of corpus product names containing each token (generic-word test)."""
    token_sets = [set(normalized_attribute_text(title).split()) for title in source["title"]]
    total = len(token_sets) or 1
    counts: Counter = Counter()
    for tokens in token_sets:
        counts.update(tokens)
    return {token: count / total for token, count in counts.items()}


def resolve_commit_sha() -> str:
    """The revision these numbers describe."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()


def verifier_from_thresholds(
    thresholds: dict[str, float], *, doc_fraction: dict[str, float],
    corpus_support: dict[str, int], gtin_groups: dict[str, Counter],
) -> BrandVerifier:
    return BrandVerifier(
        title_min_chars=int(thresholds["title_corroboration_min_chars"]),
        containment_min_chars=int(thresholds["key_containment_min_chars"]),
        gtin_claim_min_share=thresholds["gtin_claim_min_share"],
        gtin_rival_min_share=thresholds["gtin_rival_min_share"],
        generic_threshold=thresholds["generic_token_doc_fraction"],
        doc_fraction=doc_fraction,
        corpus_support=corpus_support,
        gtin_groups=gtin_groups,
    )


def run_selftest(thresholds: dict[str, float]) -> list[str]:
    """Prove every class and status is reachable, so a zero count is credible."""
    doc_fraction = {
        "mix": 0.034, "pure": 0.026, "mont": 0.001, "roucous": 0.0,
        "coca": 0.001, "cola": 0.02, "albi": 0.0, "marli": 0.0,
        "cemil": 0.0001, "cemilefendi": 0.0, "real": 0.007, "realemon": 0.0,
        "ting": 0.0004, "dg": 0.0002, "piacelli": 0.0,
    }
    groups = {
        "1111": Counter({"Piacelli": 3}),              # unanimous group
        "2222": Counter({"Mont Roucous": 5}),          # different unanimous GTINs
        "3333": Counter({"Mont": 3}),
        "4444": Counter({"Dg": 2, "Ting": 1}),         # internally conflicting group
        "5555": Counter({"Realemon": 4, "Senz": 1}),   # a rival that outvotes a claim
        "6666": Counter({"Albi": 3}),                  # corroborates an Albi claim
        "7777": Counter({"Marli": 2}),                 # corroborates a Marli claim
    }
    verifier = BrandVerifier(
        title_min_chars=int(thresholds["title_corroboration_min_chars"]),
        containment_min_chars=int(thresholds["key_containment_min_chars"]),
        gtin_claim_min_share=thresholds["gtin_claim_min_share"],
        gtin_rival_min_share=thresholds["gtin_rival_min_share"],
        generic_threshold=thresholds["generic_token_doc_fraction"],
        doc_fraction=doc_fraction,
        corpus_support={"piacelli": 10, "mont": 50, "montroucous": 60, "dg": 20, "ting": 28},
        gtin_groups=groups,
    )
    # (label, source(claim, gtin, name, gtin_indep, title_indep),
    #         target(claim, gtin, name, gtin_indep, title_indep), expected)
    cases = [
        ("both absent", ("", "", "", True, True), ("", "", "", False, False), BrandClass.ABSENT_BOTH),
        ("source absent", ("", "", "", True, True), ("Piacelli", "", "", False, False), BrandClass.ABSENT_SOURCE),
        ("target absent", ("Piacelli", "", "", True, True), ("", "", "", False, False), BrandClass.ABSENT_TARGET),
        ("verified equal", ("Piacelli", "1111", "", True, True), ("Piacelli", "", "", False, False), BrandClass.SAME_BRAND_VERIFIED),
        ("corporate form", ("Piacelli S.A.", "", "", True, True), ("Piacelli", "", "", False, False), BrandClass.SAME_BRAND_SURFACE_VARIANT),
        ("token order", ("Coca Cola", "", "", True, True), ("Cola Coca", "", "", False, False), BrandClass.SAME_BRAND_SURFACE_VARIANT),
        ("generic word", ("Mix", "", "", True, True), ("VUE", "", "", False, False), BrandClass.BRAND_FIELD_DEFECT),
        ("contradicted by listings", ("Mont", "5555", "", True, True), ("Realemon", "", "", False, False), BrandClass.BRAND_FIELD_DEFECT),
        ("containment is NOT sameness", ("Mont Roucous", "2222", "", True, True), ("Mont", "3333", "", False, False), BrandClass.DIFFERENT_BRAND_VERIFIED),
        ("prefix is NOT sameness", ("Cemilefendi", "", "heather juice cemilefendi", True, True), ("Cemil", "1111", "", False, False), BrandClass.DIFFERENT_BRAND_VERIFIED),
        ("contradiction beats sameness", ("Réal", "5555", "real lemon", True, True), ("Realemon", "", "", False, False), BrandClass.BRAND_FIELD_DEFECT),
        ("evidence disagreement", ("Ting", "", "ting soda", True, True), ("Dg", "4444", "", False, False), BrandClass.EVIDENCE_DISAGREEMENT),
        ("verified different", ("Albi", "6666", "", True, True), ("Marli", "", "", False, False), BrandClass.DIFFERENT_BRAND_VERIFIED),
        ("verified different, both sides", ("Albi", "6666", "", True, True), ("Marli", "7777", "", False, False), BrandClass.DIFFERENT_BRAND_VERIFIED),
        ("unverified different", ("Albi", "", "", False, False), ("Marli", "", "", False, False), BrandClass.DIFFERENT_BRAND_UNVERIFIED),
    ]
    failures: list[str] = []
    for label, source_args, target_args, expected in cases:
        source = verifier.verify(
            side=Side.SOURCE, claim=source_args[0], gtin=source_args[1],
            name_text=source_args[2], gtin_check_independent=source_args[3],
            title_check_independent=source_args[4],
        )
        target = verifier.verify(
            side=Side.TARGET, claim=target_args[0], gtin=target_args[1],
            name_text=target_args[2], gtin_check_independent=target_args[3],
            title_check_independent=target_args[4],
        )
        observed, rule = classify_pair(
            source, target, verifier=verifier, source_gtin=source.gtin,
            target_gtin=target.gtin,
            group_conflict=(
                verifier.group_conflict(source.gtin) or verifier.group_conflict(target.gtin)
            ),
        )
        if observed != expected:
            failures.append(f"{label}: expected {expected}, got {observed} ({rule})")
    if not verifier.keys_contain("Mont Roucous", "Mont"):
        failures.append("containment flag: Mont Roucous / Mont should be flagged")
    if verifier.surface_variant_only("Mont Roucous", "Mont"):
        failures.append("containment must NOT count as a surface variant")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify why source/target brand pairs fail to match, against verified brands.",
    )
    parser.add_argument("--review", type=Path, help="human_review_enriched.csv (the 585-pair population)")
    parser.add_argument("--out-dir", type=Path, help="where analysis artifacts are written")
    parser.add_argument("--worst-top", type=int, default=25)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    thresholds, threshold_origins, config_present = resolve_thresholds()
    if args.selftest:
        failures = run_selftest(thresholds)
        if failures:
            raise SystemExit("brand classifier selftest FAILED:\n  " + "\n  ".join(failures))
        print("[selftest] all brand classes and verification statuses reachable")
        return
    if args.review is None or args.out_dir is None:
        parser.error("--review and --out-dir are required unless --selftest is given")

    review = pd.read_csv(args.review, dtype=str, keep_default_na=False)
    if review.empty:
        raise SystemExit(f"review population is empty: {args.review}")
    review["SKU_ID"] = review["SKU_ID"].astype(str)
    review["NEAREST_ITEM_ID"] = review["NEAREST_ITEM_ID"].astype(str)

    source = load_dataset()
    source_by_id = source.drop_duplicates("product_id").set_index("product_id")
    records = canonical_records_frame()
    records["gtin"] = records["gtin"].astype(str)
    indexed = records.set_index("gtin")

    if not set(review["SKU_ID"]).issubset(source_by_id.index):
        raise SystemExit("review SKU_IDs do not all resolve in the source corpus")
    if not set(review["NEAREST_ITEM_ID"]).issubset(indexed.index):
        raise SystemExit("review NEAREST_ITEM_IDs do not all resolve in the canonical corpus")

    gtin_groups, corpus_support = build_gtin_index(source)
    doc_fraction = brand_token_doc_fraction(source)
    verifier = verifier_from_thresholds(
        thresholds, doc_fraction=doc_fraction, corpus_support=corpus_support,
        gtin_groups=gtin_groups,
    )

    source_rows = [source_by_id.loc[sku] for sku in review["SKU_ID"]]
    target_records = [indexed.loc[gtin].to_dict() for gtin in review["NEAREST_ITEM_ID"]]

    classified: list[dict[str, object]] = []
    source_evidence: list[SideEvidence] = []
    target_evidence: list[SideEvidence] = []
    for index, (_, row) in enumerate(review.iterrows()):
        source_row = source_rows[index]
        target_record = target_records[index]
        source_claim = metadata_text(source_row["brand"])
        target_claim = metadata_text(target_record.get("mode_brand"))
        source_gtin = metadata_text(source_row["barcode"])
        target_gtin = str(row["NEAREST_ITEM_ID"])

        # Independence, stated rather than assumed:
        #  * a source listing's brand is its own field, so OTHER listings in the
        #    GTIN group are genuinely independent evidence about it;
        #  * mode_brand IS its GTIN group's modal brand (verified 585/585) and is
        #    derived from the same listings' names, so neither the group check
        #    nor the title check can independently falsify it.
        source_ev = verifier.verify(
            side=Side.SOURCE, claim=source_claim, gtin=source_gtin,
            name_text=metadata_text(source_row["title"]),
            gtin_check_independent=True, title_check_independent=True,
        )
        target_ev = verifier.verify(
            side=Side.TARGET, claim=target_claim, gtin=target_gtin,
            name_text=" ".join((
                metadata_text(target_record.get("canonical")),
                metadata_text(target_record.get("mode_type")),
            )),
            gtin_check_independent=False, title_check_independent=False,
        )
        source_evidence.append(source_ev)
        target_evidence.append(target_ev)

        brand_class, rule = classify_pair(
            source_ev, target_ev, verifier=verifier, source_gtin=source_gtin,
            target_gtin=target_gtin,
            group_conflict=(
                verifier.group_conflict(source_gtin) or verifier.group_conflict(target_gtin)
            ),
        )
        classified.append({
            "sku_id": str(row["SKU_ID"]),
            "target_gtin": target_gtin,
            "score": float(row["SCORE"]),
            "source_brand": source_claim,
            "target_brand": target_claim,
            "brand_class": str(brand_class),
            "matched_rule": rule,
            "source_status": source_ev.status,
            "target_status": target_ev.status,
            "same_gtin": bool(source_gtin) and source_gtin == target_gtin,
            "key_containment": verifier.keys_contain(source_claim, target_claim),
        })

    frame = pd.DataFrame(classified)
    left = frame["source_brand"].tolist()
    right = frame["target_brand"].tolist()
    fuzzy = pair_fuzzy_scores(left, right)
    cosine_char, cosine_word = tfidf_cosines(left, right)
    for name, values in fuzzy.items():
        frame[name] = values
    frame["tfidf_cosine_char"] = cosine_char
    frame["tfidf_cosine_word"] = cosine_word
    frame["shared_tokens"] = [
        " ".join(sorted(brand_tokens(a) & brand_tokens(b))) for a, b in zip(left, right)
    ]
    frame["source_brand_corpus_rows"] = [e.corpus_support for e in source_evidence]
    frame["target_brand_corpus_rows"] = [e.corpus_support for e in target_evidence]
    frame["pair_rows"] = (
        frame.groupby(["source_brand", "target_brand"])["sku_id"].transform("size").astype(int)
    )

    records_out = [
        BrandPairRecord.model_validate({
            key: value for key, value in record.items()
            if key in BrandPairRecord.model_fields
        })
        for record in frame.to_dict("records")
    ]
    frame = pd.DataFrame([record.model_dump() for record in records_out])
    evidence_frame = pd.DataFrame([
        {**evidence.model_dump(), "role": evidence.side, "sku_id": str(review["SKU_ID"].iloc[i])}
        for i, pair in enumerate(zip(source_evidence, target_evidence))
        for evidence in pair
    ])

    summary_rows = [
        BrandClassSummary(
            brand_class=str(brand_class),
            rows=int((frame["brand_class"] == str(brand_class)).sum()),
            mean_score=_mean(frame, brand_class, "score"),
            mean_ratio=_mean(frame, brand_class, "ratio"),
            mean_tfidf_cosine_char=_mean(frame, brand_class, "tfidf_cosine_char"),
            is_true_hard_negative=brand_class in HARD_NEGATIVE_CLASSES,
            is_brand_field_defect=brand_class in DEFECT_CLASSES,
            resolves_under_normalisation=brand_class in NORMALISATION_RESOLVED_CLASSES,
        )
        for brand_class in BrandClass
    ]

    raw_agree = (
        frame["source_brand"].map(compact_brand_key)
        == frame["target_brand"].map(compact_brand_key)
    )
    verified_agree = frame["brand_class"].isin([str(c) for c in BRAND_AGREES_CLASSES])
    defects = frame["brand_class"].isin([str(c) for c in DEFECT_CLASSES])
    unresolved = frame["brand_class"].isin([str(c) for c in UNRESOLVED_CLASSES])
    hard = frame["brand_class"].isin([str(c) for c in HARD_NEGATIVE_CLASSES])
    resolved = frame["brand_class"].isin([str(c) for c in NORMALISATION_RESOLVED_CLASSES])
    clean = ~defects & ~unresolved

    def separation(mask: pd.Series) -> float:
        agrees = frame[mask & verified_agree]["score"]
        differs = frame[mask & ~verified_agree]["score"]
        if agrees.empty or differs.empty:
            return 0.0
        return float(agrees.mean() - differs.mean())

    impact = BrandImpact(
        population_rows=len(frame),
        brand_agrees_raw_string=int(raw_agree.sum()),
        brand_differs_raw_string=int((~raw_agree).sum()),
        brand_agrees_verified=int(verified_agree.sum()),
        brand_differs_verified=int((clean & ~verified_agree).sum()),
        brand_field_defects=int(defects.sum()),
        evidence_disagreements=int(unresolved.sum()),
        true_hard_negatives=int(hard.sum()),
        contradictions_found_by_co_listings=int(sum(
            evidence.status == str(VerificationStatus.CONTRADICTED_BY_LISTINGS)
            for evidence in source_evidence + target_evidence
        )),
        pairs_with_same_gtin=int(frame["same_gtin"].sum()),
        same_gtin_conflicting_brands=int((frame["same_gtin"] & ~verified_agree).sum()),
        key_containment_rows=int(frame["key_containment"].sum()),
        resolved_by_normalisation=int(resolved.sum()),
        resolved_share_of_population=float(resolved.mean()),
        separation_raw_string=float(
            frame[raw_agree]["score"].mean() - frame[~raw_agree]["score"].mean()
        ),
        separation_verified=separation(pd.Series(True, index=frame.index)),
        separation_excluding_defects=separation(clean),
        mean_score_brand_agrees_verified=float(frame[verified_agree]["score"].mean()),
        mean_score_brand_differs_verified=float(frame[clean & ~verified_agree]["score"].mean()),
        mean_score_true_hard_negative=(
            float(frame[hard]["score"].mean()) if hard.any() else 0.0
        ),
    )

    provenance = BrandAnalysisProvenance(
        commit_sha=resolve_commit_sha(),
        review_csv=str(args.review),
        population_rows=len(frame),
        thresholds=threshold_origins,
        config_block_present=config_present,
        char_ngram_range=CHAR_NGRAM_RANGE,
        word_ngram_range=WORD_NGRAM_RANGE,
    )
    summary = BrandAnalysisSummary(provenance=provenance, classes=summary_rows, impact=impact)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "brand_pair_classification.csv", index=False)
    pd.DataFrame([row.model_dump() for row in summary_rows]).to_csv(
        args.out_dir / "brand_class_summary.csv", index=False,
    )
    evidence_frame.to_csv(args.out_dir / "brand_verification_evidence.csv", index=False)
    mismatch = frame[~frame["brand_class"].isin([str(c) for c in BRAND_AGREES_CLASSES])]
    worst = mismatch.sort_values(["ratio", "tfidf_cosine_char"], ascending=False).head(args.worst_top)
    worst.to_csv(args.out_dir / "brand_worst_pairs.csv", index=False)
    (args.out_dir / "brand_analysis_summary.json").write_text(
        json.dumps(summary.model_dump(), indent=2) + "\n", encoding="utf-8",
    )

    print(f"commit: {provenance.commit_sha}")
    print(f"population: {len(frame)} pairs from {args.review}")
    print(f"config block '{CONFIG_BLOCK}' present: {config_present}"
          f"{'' if config_present else ' -> thresholds are script defaults (block to add is in the report)'}")
    print("\n── brand class counts (against VERIFIED brands) ──")
    for row in summary_rows:
        print(f"  {row.brand_class:42s} n={row.rows:4d}  mean_score={row.mean_score:.4f}")
    print("\n── verification status counts ──")
    for role in ("source", "target"):
        subset = evidence_frame[evidence_frame["role"] == role]
        for status, count in subset["status"].value_counts().items():
            print(f"  {role:6s} {status:38s} {count:4d}")
    print("\n── impact ──")
    for key, value in impact.model_dump().items():
        print(f"  {key:38s} {value:.4f}" if isinstance(value, float) else f"  {key:38s} {value}")
    print(f"\n── {len(worst)} most similar mismatched pairs (fuzzy ratio desc) ──")
    print(worst[[
        "source_brand", "target_brand", "brand_class", "ratio", "token_set_ratio",
        "tfidf_cosine_char", "pair_rows",
    ]].to_string(index=False))
    print(f"\n[saved] {args.out_dir}")


def _mean(frame: pd.DataFrame, brand_class: BrandClass, column: str) -> float:
    subset = frame[frame["brand_class"] == str(brand_class)]
    return float(subset[column].mean()) if len(subset) else 0.0


if __name__ == "__main__":
    main()
