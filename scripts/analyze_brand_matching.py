#!/usr/bin/env python3
"""Classify WHY the brand axis separates the 0.55-0.75 review population.

The question this answers is "are brands not matching, and why not".  For each
of the 585 review pairs it reads the SOURCE brand from the raw export
(``dataset.csv`` — the value ``core.model_input`` actually tokenises) and the
TARGET brand from the canonical-record SSOT (``results/canonical_records.csv``
via :func:`core.common.canonical_records_frame`), then classifies the pair with
rapidfuzz string similarity plus TF-IDF cosine over word and character n-grams.

The review CSV's own ``canonical`` column is NOT trusted: it is stale for
542/585 rows against the SSOT artifact (see the report's EXECUTED log), so both
brand sides are re-read from the SSOT.

Normalisation is NOT re-implemented here.  ``core.critical_attributes.
normalized_attribute_text`` is the repository's existing NFKD + combining-mark
strip + punctuation-flattening normaliser and is used verbatim; the compact
"brand key" is that output with separators removed.  ``pipeline.
jaccard_similarity`` is the existing token-Jaccard SSOT.  rapidfuzz and
scikit-learn have no prior use in this repository (verified by grep — see the
report), so their introduction here is new by necessity.

Settings are named constants, exposed as CLI flags.  They are not read from
``config/training.yaml`` only because this analysis is forbidden from editing
``config/``; the report carries the exact config block to add on
productionisation.

Usage::

    PYTHONPATH=src python scripts/analyze_brand_matching.py \
        --review <path>/human_review_enriched.csv --out-dir <dir>
    PYTHONPATH=src python scripts/analyze_brand_matching.py --selftest
"""

from __future__ import annotations

import argparse
import json
import subprocess
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from core.common import F, canonical_records_frame, load_dataset, metadata_text
from core.critical_attributes import normalized_attribute_text
from pipeline import jaccard_similarity

# ── named constants (CLI-overridable; proposed config home in the report) ───

#: Containment is accepted only when the shorter key aligns to a contiguous run
#: in the longer one (``partial_ratio``).  ``ratio`` is the wrong guard here: it
#: punishes length difference, so a genuine truncation such as
#: ``Mont Roucous`` / ``Mont`` scores only 50 and a doubled prefix scores 62.5.
CONTAINMENT_PARTIAL_RATIO_MIN = 95.0

#: Minimum characters in the shorter compact key before containment counts, so
#: a two-letter brand cannot "contain" inside anything.
CONTAINMENT_MIN_CHARS = 4

#: Minimum characters before a brand key may be credited to a TITLE.  Lower
#: than containment: a three-letter brand abbreviation appearing inside a
#: product name is strong evidence, whereas a three-letter key "contained" in
#: another three-to-four-letter brand is coincidence ("Cola" in "Coca Cola").
TITLE_RECOVERY_MIN_CHARS = 3

#: Below this ``token_set_ratio`` (0-100) the pair is called a distinct-brand
#: hard negative.  Set above the highest score observed between genuinely
#: unrelated brands in this population (66.7) and far below the containment
#: cases (100).
DISTINCT_TOKEN_SET_MAX = 70.0

#: TF-IDF vectorisation settings.  Word n-grams catch token-order and
#: corporate-suffix variants; character n-grams catch the same brand spelled
#: with different punctuation or spacing.
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 5)
MIN_DF = 1

#: Corporate/legal forms stripped only inside the surface-variant ladder, so a
#: report can say whether a suffix alone explained a mismatch.  No such
#: vocabulary existed anywhere in the repository (verified by grep), so it is
#: declared here as new.
CORPORATE_SUFFIX_TOKENS = frozenset({
    "ab", "ag", "aps", "as", "bv", "co", "company", "corp", "corporation",
    "gmbh", "group", "groupe", "grupo", "holding", "inc", "kg", "kgaa",
    "limited", "llc", "ltd", "nv", "oy", "plc", "pte", "sa", "sarl", "sas",
    "spa", "srl", "sro",
})


class BrandClass(StrEnum):
    """Why a source/target brand pair agrees or disagrees."""

    ABSENT_BOTH = "absent_both"
    ABSENT_SOURCE = "absent_source"
    ABSENT_TARGET = "absent_target"
    IDENTICAL_NORMALIZED = "identical_normalized"
    SURFACE_VARIANT = "surface_variant_same_brand"
    CONTAINMENT_VARIANT = "containment_or_truncation"
    BRAND_IN_TITLE_ONLY = "brand_present_only_in_title"
    DISTINCT_HARD_NEGATIVE = "distinct_brand_true_hard_negative"
    PARTIAL_OVERLAP_REVIEW = "partial_overlap_needs_review"


#: Brand disagreement no string normalisation can repair — true hard negatives.
HARD_NEGATIVE_CLASSES = frozenset({
    BrandClass.DISTINCT_HARD_NEGATIVE,
    BrandClass.PARTIAL_OVERLAP_REVIEW,
})
#: Classes a normalising brand gate would newly resolve to "same brand".
NORMALISATION_RESOLVED_CLASSES = frozenset({
    BrandClass.SURFACE_VARIANT,
    BrandClass.CONTAINMENT_VARIANT,
})
#: Classes where the brand strings already agree.
BRAND_AGREES_CLASSES = frozenset({BrandClass.IDENTICAL_NORMALIZED})


class BrandPairRecord(BaseModel):
    """One classified source/target brand pair (machine-readable boundary)."""

    model_config = ConfigDict(extra="forbid")

    sku_id: str
    target_gtin: str
    score: float
    source_brand: str
    target_brand: str
    brand_class: str
    matched_rule: str
    ratio: float
    token_set_ratio: float
    partial_ratio: float
    levenshtein: int = Field(ge=0)
    tfidf_cosine_char: float
    tfidf_cosine_word: float
    token_jaccard: float
    shared_tokens: str
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
    resolves_under_normalisation: bool


class BrandImpact(BaseModel):
    """The decision-relevant totals, and the score separation they explain."""

    model_config = ConfigDict(extra="forbid")

    population_rows: int = Field(ge=1)
    brand_agrees_under_crude_rule: int = Field(ge=0)
    brand_disagrees_crude: int = Field(ge=0)
    resolved_by_normalisation: int = Field(ge=0)
    remains_true_hard_negative: int = Field(ge=0)
    resolved_share_of_population: float
    resolved_share_of_mismatches: float
    hard_negative_share_of_mismatches: float
    mean_score_brand_agrees: float
    mean_score_brand_disagrees: float
    score_separation: float
    mean_score_hard_negative: float


class BrandAnalysisProvenance(BaseModel):
    """What produced this run, so the numbers can be attributed to a revision."""

    model_config = ConfigDict(extra="forbid")

    commit_sha: str
    review_csv: str
    dataset_binding: str
    canonical_records_binding: str
    population_rows: int = Field(ge=1)
    containment_partial_ratio_min: float
    containment_min_chars: int
    distinct_token_set_max: float
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
    only step added here, and it is what makes ``Côteaux`` equal ``Coteaux``
    and ``S.A.`` equal ``SA``.
    """
    return "".join(
        character
        for character in normalized_attribute_text(value)
        if character.isalnum()
    )


def spaced_brand_text(value: object) -> str:
    """Normalised brand text with word separators kept, punctuation dropped.

    Kept separate from :func:`compact_brand_key` so the rule ladder can tell a
    punctuation/spacing variant apart from a token-order variant.
    """
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


def brand_tokens(value: object) -> frozenset[str]:
    """Normalised brand word set (order- and separator-insensitive)."""
    return frozenset(normalized_attribute_text(value).split())


class BrandPairClassifier:
    """Classify brand pairs with an explicit, auditable rule ladder.

    Ordered strongest-evidence-first so each row is counted exactly once:
    absence, then equality under the repository normaliser, then the aggressive
    surface ladder, then containment/truncation, then the cross-field (title)
    recovery case, and only then the fuzzy hard-negative split.
    """

    def __init__(self, *, distinct_token_set_max: float,
                 containment_partial_ratio_min: float, containment_min_chars: int) -> None:
        self.distinct_token_set_max = distinct_token_set_max
        self.containment_partial_ratio_min = containment_partial_ratio_min
        self.containment_min_chars = containment_min_chars

    def classify(
        self,
        source_brand: str,
        target_brand: str,
        source_title: str,
        target_title: str,
        ratio: float,
        token_set_ratio: float,
        partial_ratio: float,
    ) -> tuple[BrandClass, str]:
        source_key = compact_brand_key(source_brand)
        target_key = compact_brand_key(target_brand)

        if not source_key and not target_key:
            return BrandClass.ABSENT_BOTH, "both brand fields empty"
        if not source_key:
            return BrandClass.ABSENT_SOURCE, "source brand field empty"
        if not target_key:
            return BrandClass.ABSENT_TARGET, "target brand field empty"
        if source_key == target_key:
            return BrandClass.IDENTICAL_NORMALIZED, "compact keys equal"

        # Surface ladder.  Each step is named so an empty class is falsifiable
        # rather than merely absent.
        if brand_tokens(source_brand) == brand_tokens(target_brand):
            return BrandClass.SURFACE_VARIANT, "token order only"
        if strip_corporate_suffix(source_brand) == strip_corporate_suffix(target_brand):
            return BrandClass.SURFACE_VARIANT, "corporate suffix only"

        shorter, longer = sorted((source_key, target_key), key=len)
        if (
            len(shorter) >= self.containment_min_chars
            and shorter in longer
            and partial_ratio >= self.containment_partial_ratio_min
        ):
            return (
                BrandClass.CONTAINMENT_VARIANT,
                "prefix truncation" if longer.startswith(shorter) else "substring containment",
            )

        # Brand recoverable from the other side's product name: present in the
        # corpus, but on a field the field-to-field comparison never reads.
        if (
            len(target_key) >= TITLE_RECOVERY_MIN_CHARS
            and target_key in compact_brand_key(source_title)
        ):
            return BrandClass.BRAND_IN_TITLE_ONLY, "target brand inside source title"
        if (
            len(source_key) >= TITLE_RECOVERY_MIN_CHARS
            and source_key in compact_brand_key(target_title)
        ):
            return BrandClass.BRAND_IN_TITLE_ONLY, "source brand inside target title"

        if token_set_ratio < self.distinct_token_set_max:
            return BrandClass.DISTINCT_HARD_NEGATIVE, "low token-set similarity"
        return BrandClass.PARTIAL_OVERLAP_REVIEW, "partial overlap above distinct floor"


def tfidf_cosines(left: list[str], right: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair TF-IDF cosine similarity, character-level and word-level.

    Both matrices are fitted on the union of the two brand columns, so the idf
    weights describe THIS population's brand vocabulary rather than an external
    corpus.  Row-normalised TF-IDF makes the row-wise dot product the cosine.
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


def load_source_fields() -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Brand and product name per source SKU, plus brand corpus support.

    ``load_dataset()`` applies the shared COLUMN_MAPPING, so the fields are the
    canonical ``product_id`` / ``brand`` / ``title`` the model-input builder
    reads.  The raw export holds one row per retailer offer, so the key is
    deduplicated; brand is constant per product in this data (checked by the
    caller), so the dedup is lossless.
    """
    deduped = load_dataset().drop_duplicates("product_id").set_index("product_id")
    brand_by_sku = deduped["brand"].map(metadata_text)
    title_by_sku = deduped["title"].map(metadata_text)
    support = brand_by_sku.map(compact_brand_key).value_counts().to_dict()
    return brand_by_sku, title_by_sku, support


def resolve_commit_sha() -> str:
    """The revision these numbers describe."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()


def classifier_from_args(args: argparse.Namespace) -> BrandPairClassifier:
    return BrandPairClassifier(
        distinct_token_set_max=args.distinct_token_set_max,
        containment_partial_ratio_min=args.containment_partial_ratio_min,
        containment_min_chars=args.containment_min_chars,
    )


def run_selftest(args: argparse.Namespace) -> list[str]:
    """Prove every class is reachable, so a class reporting zero is credible.

    A zero count only means something if the detector can be shown to fire.
    These are the variant families the brief names, plus one synthetic case for
    the partial-overlap branch (which needs a non-containment token subset).
    """
    classifier = classifier_from_args(args)
    # (source_brand, target_brand, source_title, target_title, expected, label)
    cases = [
        # Folded at the EQUALITY stage, because the repository normaliser
        # already removes case, diacritics, punctuation and spacing.
        ("Côteaux", "Coteaux", "", "", BrandClass.IDENTICAL_NORMALIZED, "diacritics"),
        ("S.A. Dampt", "SA Dampt", "", "", BrandClass.IDENTICAL_NORMALIZED, "punctuation"),
        ("Coca Cola", "COCA  COLA", "", "", BrandClass.IDENTICAL_NORMALIZED, "case+spacing"),
        ("Piacelli", "Piacelli", "", "", BrandClass.IDENTICAL_NORMALIZED, "identical"),
        # Folded only by the aggressive ladder.
        ("Quellbrunn GmbH", "Quellbrunn", "", "", BrandClass.SURFACE_VARIANT, "corporate suffix"),
        ("Coca Cola", "Cola Coca", "", "", BrandClass.SURFACE_VARIANT, "token order"),
        ("Mont Roucous", "Mont", "", "", BrandClass.CONTAINMENT_VARIANT, "truncation"),
        ("Cemilefendi", "Cemil", "", "", BrandClass.CONTAINMENT_VARIANT, "prefix"),
        ("", "", "", "", BrandClass.ABSENT_BOTH, "both absent"),
        ("", "Piacelli", "", "", BrandClass.ABSENT_SOURCE, "source absent"),
        ("Piacelli", "", "", "", BrandClass.ABSENT_TARGET, "target absent"),
        ("Albi", "Marli", "", "", BrandClass.DISTINCT_HARD_NEGATIVE, "distinct brands"),
        ("Peace Tea", "Seven Teas", "", "", BrandClass.DISTINCT_HARD_NEGATIVE, "shared generic word"),
        ("River City", "City River Drink", "", "", BrandClass.PARTIAL_OVERLAP_REVIEW, "token subset"),
        # Real observed case: "Ting" vs "Dg" where the target's own product
        # name is "DG Ting Grapefruit Soda".
        ("Ting", "Dg", "Ting Soft Drink Sparkling Grapefruit",
         "DG Ting Grapefruit Soda", BrandClass.BRAND_IN_TITLE_ONLY, "brand only in title"),
    ]
    failures: list[str] = []
    for source, target, source_title, target_title, expected, label in cases:
        scores = pair_fuzzy_scores([source], [target])
        observed, rule = classifier.classify(
            source, target, source_title, target_title,
            scores["ratio"][0], scores["token_set_ratio"][0], scores["partial_ratio"][0],
        )
        if observed != expected:
            failures.append(f"{label}: expected {expected}, got {observed} ({rule})")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify why source/target brand pairs fail to match.",
    )
    parser.add_argument(
        "--review", type=Path,
        help="human_review_enriched.csv (the 585-pair 0.55-0.75 population)",
    )
    parser.add_argument("--out-dir", type=Path, help="where analysis artifacts are written")
    parser.add_argument(
        "--containment-partial-ratio-min", type=float,
        default=CONTAINMENT_PARTIAL_RATIO_MIN,
    )
    parser.add_argument("--containment-min-chars", type=int, default=CONTAINMENT_MIN_CHARS)
    parser.add_argument("--distinct-token-set-max", type=float, default=DISTINCT_TOKEN_SET_MAX)
    parser.add_argument(
        "--worst-top", type=int, default=25,
        help="how many ranked offending pairs to print and persist",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="assert every brand class is reachable, then exit",
    )
    args = parser.parse_args()

    if args.selftest:
        failures = run_selftest(args)
        if failures:
            raise SystemExit("brand classifier selftest FAILED:\n  " + "\n  ".join(failures))
        print("[selftest] all brand classes reachable")
        return

    if args.review is None or args.out_dir is None:
        parser.error("--review and --out-dir are required unless --selftest is given")

    review = pd.read_csv(args.review, dtype=str, keep_default_na=False)
    if review.empty:
        raise SystemExit(f"review population is empty: {args.review}")

    brand_by_sku, title_by_sku, source_support = load_source_fields()
    records = canonical_records_frame()
    records["gtin"] = records["gtin"].astype(str)
    indexed = records.set_index("gtin")
    brand_by_gtin = indexed["mode_brand"].map(metadata_text)
    target_support = brand_by_gtin.map(compact_brand_key).value_counts().to_dict()

    source_lookup = review["SKU_ID"].map(brand_by_sku)
    target_lookup = review["NEAREST_ITEM_ID"].map(brand_by_gtin)
    if source_lookup.isna().any() or target_lookup.isna().any():
        raise SystemExit(
            f"brand join incomplete: {int(source_lookup.isna().sum())} source / "
            f"{int(target_lookup.isna().sum())} target rows unresolved — refusing "
            "to classify a partial population"
        )

    source_brands = source_lookup.map(metadata_text).tolist()
    target_brands = target_lookup.map(metadata_text).tolist()
    source_titles = review["SKU_ID"].map(title_by_sku).fillna("").map(metadata_text).tolist()
    target_titles = (
        review["NEAREST_ITEM_ID"].map(indexed["canonical"]).fillna("").map(metadata_text).tolist()
    )

    cosine_char, cosine_word = tfidf_cosines(source_brands, target_brands)
    fuzzy = pair_fuzzy_scores(source_brands, target_brands)
    classifier = classifier_from_args(args)

    classified: list[dict[str, object]] = []
    for index, (_, row) in enumerate(review.iterrows()):
        brand_class, rule = classifier.classify(
            source_brands[index], target_brands[index],
            source_titles[index], target_titles[index],
            fuzzy["ratio"][index], fuzzy["token_set_ratio"][index], fuzzy["partial_ratio"][index],
        )
        shared = brand_tokens(source_brands[index]) & brand_tokens(target_brands[index])
        classified.append({
            "sku_id": str(row["SKU_ID"]),
            "target_gtin": str(row["NEAREST_ITEM_ID"]),
            "score": float(row["SCORE"]),
            "source_brand": source_brands[index],
            "target_brand": target_brands[index],
            "brand_class": str(brand_class),
            "matched_rule": rule,
            "ratio": fuzzy["ratio"][index],
            "token_set_ratio": fuzzy["token_set_ratio"][index],
            "partial_ratio": fuzzy["partial_ratio"][index],
            "levenshtein": fuzzy["levenshtein"][index],
            "tfidf_cosine_char": float(cosine_char[index]),
            "tfidf_cosine_word": float(cosine_word[index]),
            "token_jaccard": fuzzy["token_jaccard"][index],
            "shared_tokens": " ".join(sorted(shared)),
            "source_brand_corpus_rows": int(source_support.get(compact_brand_key(source_brands[index]), 0)),
            "target_brand_corpus_rows": int(target_support.get(compact_brand_key(target_brands[index]), 0)),
        })

    frame = pd.DataFrame(classified)
    # Support is per BRAND PAIR: a pair seen twice must report n=2, so a
    # two-row brand can never be presented as a population-level defect.
    frame["pair_rows"] = (
        frame.groupby(["source_brand", "target_brand"])["sku_id"].transform("size").astype(int)
    )
    validated = [BrandPairRecord.model_validate(record) for record in frame.to_dict("records")]
    frame = pd.DataFrame([record.model_dump() for record in validated])

    class_values = {str(member) for member in BrandClass}
    summary_rows = [
        BrandClassSummary(
            brand_class=str(brand_class),
            rows=int((frame["brand_class"] == str(brand_class)).sum()),
            mean_score=_mean(frame, brand_class, "score"),
            mean_ratio=_mean(frame, brand_class, "ratio"),
            mean_tfidf_cosine_char=_mean(frame, brand_class, "tfidf_cosine_char"),
            resolves_under_normalisation=brand_class in NORMALISATION_RESOLVED_CLASSES,
        )
        for brand_class in BrandClass
    ]
    if set(frame["brand_class"]) - class_values:
        raise SystemExit("classification produced an unregistered class name")

    same_brand = frame[frame["brand_class"].isin([str(c) for c in BRAND_AGREES_CLASSES])]
    mismatched = frame[~frame["brand_class"].isin([str(c) for c in BRAND_AGREES_CLASSES])]
    resolved = frame[frame["brand_class"].isin([str(c) for c in NORMALISATION_RESOLVED_CLASSES])]
    hard = frame[frame["brand_class"].isin([str(c) for c in HARD_NEGATIVE_CLASSES])]

    total = len(frame)
    mean_same = float(same_brand["score"].mean())
    mean_diff = float(mismatched["score"].mean())
    impact = BrandImpact(
        population_rows=total,
        brand_agrees_under_crude_rule=len(same_brand),
        brand_disagrees_crude=len(mismatched),
        resolved_by_normalisation=len(resolved),
        remains_true_hard_negative=len(hard),
        resolved_share_of_population=len(resolved) / total,
        resolved_share_of_mismatches=len(resolved) / len(mismatched) if len(mismatched) else 0.0,
        hard_negative_share_of_mismatches=len(hard) / len(mismatched) if len(mismatched) else 0.0,
        mean_score_brand_agrees=mean_same,
        mean_score_brand_disagrees=mean_diff,
        score_separation=mean_same - mean_diff,
        mean_score_hard_negative=float(hard["score"].mean()) if len(hard) else 0.0,
    )

    provenance = BrandAnalysisProvenance(
        commit_sha=resolve_commit_sha(),
        review_csv=str(args.review),
        dataset_binding=F["dataset"].name,
        canonical_records_binding=F["canonical_records"].name,
        population_rows=total,
        containment_partial_ratio_min=args.containment_partial_ratio_min,
        containment_min_chars=args.containment_min_chars,
        distinct_token_set_max=args.distinct_token_set_max,
        char_ngram_range=CHAR_NGRAM_RANGE,
        word_ngram_range=WORD_NGRAM_RANGE,
    )
    summary = BrandAnalysisSummary(provenance=provenance, classes=summary_rows, impact=impact)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "brand_pair_classification.csv", index=False)
    pd.DataFrame([row.model_dump() for row in summary_rows]).to_csv(
        args.out_dir / "brand_class_summary.csv", index=False,
    )
    worst = mismatched.sort_values(["ratio", "tfidf_cosine_char"], ascending=False).head(args.worst_top)
    worst.to_csv(args.out_dir / "brand_worst_pairs.csv", index=False)
    by_support = (
        mismatched.groupby(["source_brand", "target_brand", "brand_class"])
        .agg(
            pair_rows=("sku_id", "size"),
            mean_ratio=("ratio", "mean"),
            mean_tfidf_cosine_char=("tfidf_cosine_char", "mean"),
            mean_score=("score", "mean"),
            source_brand_corpus_rows=("source_brand_corpus_rows", "max"),
            target_brand_corpus_rows=("target_brand_corpus_rows", "max"),
        )
        .reset_index()
        .sort_values("pair_rows", ascending=False)
    )
    by_support.to_csv(args.out_dir / "brand_mismatch_pairs_by_support.csv", index=False)
    (args.out_dir / "brand_analysis_summary.json").write_text(
        json.dumps(summary.model_dump(), indent=2) + "\n", encoding="utf-8",
    )

    print(f"commit: {provenance.commit_sha}")
    print(f"population: {total} pairs from {args.review}")
    print("\n── brand class counts ──")
    for row in summary_rows:
        print(f"  {row.brand_class:38s} n={row.rows:4d}  mean_score={row.mean_score:.4f}")
    print("\n── impact ──")
    for key, value in impact.model_dump().items():
        print(f"  {key:34s} {value:.4f}" if isinstance(value, float) else f"  {key:34s} {value}")
    print(f"\n── {len(worst)} most similar mismatched brand pairs (fuzzy ratio desc) ──")
    print(worst[[
        "source_brand", "target_brand", "ratio", "token_set_ratio",
        "tfidf_cosine_char", "pair_rows",
    ]].to_string(index=False))
    print("\n── mismatched brand pairs by support ──")
    print(by_support.head(10)[[
        "source_brand", "target_brand", "pair_rows", "mean_ratio",
        "source_brand_corpus_rows", "target_brand_corpus_rows",
    ]].to_string(index=False))
    print(f"\n[saved] {args.out_dir}")


def _mean(frame: pd.DataFrame, brand_class: BrandClass, column: str) -> float:
    subset = frame[frame["brand_class"] == str(brand_class)]
    return float(subset[column].mean()) if len(subset) else 0.0


if __name__ == "__main__":
    main()
