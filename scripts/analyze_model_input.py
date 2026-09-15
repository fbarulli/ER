#!/usr/bin/env python3
"""Corpus-level analysis of the text the encoder actually receives.

Offline measurement — this is not the productionised run-report metric.  It
answers five questions about the composed encoder input built by
``core.model_input``:

1. token budget against ``training.max_seq_length``, and what truncation costs;
2. composition per field group, and how much of it can discriminate at all;
3. field presence / coverage across the corpus (input completeness);
4. per-field symmetry on the pairs where the same product feeds both sides;
5. what the ``cleaned`` profile changes against ``legacy`` — what is gained and
   what is lost, including the reported identical-row Jaccard regression.

The composition is NEVER re-implemented.  Every string measured here comes from
``core.model_input.build_sku_text`` / ``build_canonical_text``.  Group
attribution calls the composition's OWN normaliser
(``core.model_input._normalized_tokens``) rather than re-deriving it, because
writing a second tokeniser is the exact duplication this project forbids; the
result is then asserted to reproduce the SSOT string byte for byte, so a
drifting attribution fails loudly instead of silently mislabelling tokens.

Thresholds are named constants exposed as CLI flags (see the report's proposed
``config/training.yaml`` block for the productionised home).

Usage::

    PYTHONPATH=src python scripts/analyze_model_input.py \
        --review <path>/human_review_enriched.csv --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from core.common import (
    canonical_records_frame,
    load_dataset,
    resolve_model,
    row_metadata_text,
    runtime,
)
from core.model_input import (
    _normalized_tokens,
    _structured_text_enabled,
    build_canonical_text,
    build_sku_text,
)
from core.schemas import TrainingSpec
from core.structured_features import canonical_info, sku_info, text_tokens
from pipeline import jaccard_similarity

# ── named constants (CLI-overridable; proposed config home in the report) ───

#: A token present in more than this fraction of a corpus cannot separate two
#: records of that corpus, however informative it looks in isolation.  This is
#: the "filler" test used for the discriminative/filler split.
UBIQUITOUS_DOC_FRACTION = 0.5

#: Whitespace tokenisation for the idf reference fit, so the vectoriser seeing
#: the composed text agrees with ``_normalized_tokens``'s own ``split()``.  The
#: sklearn default word pattern would silently split ``[FIELD_VOLUME]``.
TFIDF_TOKEN_PATTERN = r"\S+"

#: How many over-cap rows to attribute token-by-token.  Attribution re-runs the
#: tokeniser per row, so it is bounded; the artifact records how many truncated
#: rows were actually attributed rather than implying the whole set.
TRUNCATION_ATTRIBUTION_CAP = 4096

#: Structured channel markers emitted by core.structured_features.text_tokens.
#: A marker present on one side of a pair and absent on the other is a field the
#: comparison cannot make, however good the encoder is.
STRUCTURED_MARKERS = (
    "VOLUME", "PACK_SIZE", "PACKAGE_TYPE", "FLAVOR",
    "CARBONATION", "SWEETENER_DIET", "PULP",
)

#: Field groups of the cleaned composition, in emission order, so the
#: symmetry and truncation walks use the same order the builder writes.
CLEANED_SOURCE_GROUPS = ("brand", "title", "attributes")
STRUCTURED_GROUP = "structured"

PROFILE_CLEANED = "cleaned"
PROFILE_LEGACY = "legacy"
#: The shipped pre-consolidation composition: the legacy builder WITH the
#: description/breadcrumb evidence channel.  Named because it, not the config's
#: current ``legacy`` (evidence off), is the "BEFORE" every reported delta and
#: the 0.5129 -> 0.4646 regression are measured against.
PROFILE_LEGACY_BEFORE = "legacy+evidence"


class TokenBudgetRow(BaseModel):
    """Token-count distribution for one (population, side, profile)."""

    model_config = ConfigDict(extra="forbid")

    population: str
    side: str
    profile: str
    records: int = Field(ge=0)
    max_seq_length: int = Field(ge=1)
    mean_tokens: float
    median_tokens: float
    p95_tokens: float
    max_tokens: int = Field(ge=0)
    truncated_records: int = Field(ge=0)
    truncated_share: float
    tokens_lost_total: int = Field(ge=0)
    mean_headroom_when_fits: float


class GroupCompositionRow(BaseModel):
    """Token mass and discriminating power of one field group."""

    model_config = ConfigDict(extra="forbid")

    population: str
    side: str
    profile: str
    group: str
    tokens: int = Field(ge=0)
    token_share: float
    distinct_tokens: int = Field(ge=0)
    ubiquitous_token_share: float
    mean_idf: float


class FieldCoverageRow(BaseModel):
    """How often a record carries a populated field."""

    model_config = ConfigDict(extra="forbid")

    population: str
    field: str
    records: int = Field(ge=0)
    populated: int = Field(ge=0)
    populated_share: float


class ProfileComparisonRow(BaseModel):
    """One metric under the three compositions that matter.

    ``before`` is :data:`PROFILE_LEGACY_BEFORE` (the shipped pre-consolidation
    composition), so ``delta_vs_before`` is the honest change a reader cares
    about.  ``legacy_no_evidence`` is carried separately because it isolates how
    much of ``before`` was the evidence channel alone.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    population: str
    side: str
    cleaned: float
    before: float
    legacy_no_evidence: float
    delta_vs_before: float


class TruncationLossRow(BaseModel):
    """Tokens past max_seq_length, attributed to the group that overflowed."""

    model_config = ConfigDict(extra="forbid")

    population: str
    side: str
    group: str
    tokens_lost: int = Field(ge=0)
    truncated_records_attributed: int = Field(ge=0)


class StructuredAsymmetryRow(BaseModel):
    """Per-marker presence across the two sides of the review pairs."""

    model_config = ConfigDict(extra="forbid")

    marker: str
    source_present: int = Field(ge=0)
    target_present: int = Field(ge=0)
    both: int = Field(ge=0)
    source_only: int = Field(ge=0)
    target_only: int = Field(ge=0)
    neither: int = Field(ge=0)


class ModelInputProvenance(BaseModel):
    """What produced these numbers, so they can be attributed to a revision."""

    model_config = ConfigDict(extra="forbid")

    commit_sha: str
    review_csv: str
    profile_default: str
    include_evidence_default: bool
    structured_text_enabled: bool
    max_seq_length: int = Field(ge=1)
    tokenizer_path: str
    ubiquitous_doc_fraction: float
    canonical_corpus_rows: int = Field(ge=0)
    source_corpus_rows: int = Field(ge=0)
    review_rows: int = Field(ge=0)


class ModelInputSummary(BaseModel):
    """The run's machine-readable result."""

    model_config = ConfigDict(extra="forbid")

    provenance: ModelInputProvenance
    token_budget: list[TokenBudgetRow]
    group_composition: list[GroupCompositionRow]
    field_coverage: list[FieldCoverageRow]
    truncation_loss: list[TruncationLossRow]
    structured_asymmetry: list[StructuredAsymmetryRow]
    profile_comparison: list[ProfileComparisonRow]


def token_counts(tokenizer, texts: list[str]) -> np.ndarray:
    """Encoder token counts including the two special tokens."""
    encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    return np.array([len(ids) for ids in encoded["input_ids"]], dtype=int)


def group_tokens_for_source(row, info) -> dict[str, list[str]]:
    """Per-group token lists for one cleaned source row (composition order)."""
    return {
        "brand": _normalized_tokens(row_metadata_text(row, "brand"), drop_schema_words=False),
        "title": _normalized_tokens(row_metadata_text(row, "title"), drop_schema_words=False),
        "attributes": _normalized_tokens(
            row_metadata_text(row, "attributes", "attr"), drop_schema_words=True
        ),
        STRUCTURED_GROUP: text_tokens(info) if _structured_text_enabled() else [],
    }


def group_tokens_for_target(record, info) -> dict[str, list[str]]:
    """Per-group token lists for one cleaned canonical record.

    ``canonical`` is measured after the builder's own brand de-duplication, so
    the group counts sum to the emitted text rather than to the raw record.
    """
    brand = _normalized_tokens(record.get("mode_brand", ""), drop_schema_words=False)
    brand_words = {word.casefold() for word in brand}
    return {
        "brand": brand,
        "canonical": [
            word
            for word in _normalized_tokens(record.get("canonical", ""), drop_schema_words=True)
            if word.casefold() not in brand_words
        ],
        "mode_type": _normalized_tokens(record.get("mode_type", ""), drop_schema_words=True),
        STRUCTURED_GROUP: text_tokens(info) if _structured_text_enabled() else [],
    }


def cursor_attribution(tokenizer, groups: dict[str, list[str]], full_text: str) -> tuple[list[int], str]:
    """Per-group token counts from cumulative prefixes, with a loud check.

    Tokenising cumulative prefixes (rather than each group alone) keeps
    sub-word merges at group boundaries accounted for, and the final prefix
    must reproduce the SSOT string's own token count — otherwise the
    attribution is wrong and this raises instead of mislabelling tokens.
    """
    ordered = [name for name in groups if groups[name]]
    counts: list[int] = []
    previous = 0
    parts: list[str] = []
    for name in ordered:
        parts.append(" ".join(groups[name]))
        prefix = " ".join(parts)
        current = len(tokenizer(prefix, add_special_tokens=True)["input_ids"])
        counts.append(current - previous)
        previous = current
    if len(tokenizer(full_text, add_special_tokens=True)["input_ids"]) != previous:
        raise SystemExit(
            "group attribution does not reproduce the SSOT text: "
            f"{previous} attributed tokens vs "
            f"{len(tokenizer(full_text, add_special_tokens=True)['input_ids'])} observed"
        )
    return counts, ordered


def fit_idf_reference(texts: list[str]):
    """idf lookup over a corpus, tokenised the same way the composition splits.

    Returns ``(idf_by_token, document_frequency_by_token, document_count)``.
    A token absent from the fitted vocabulary has document frequency zero.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(token_pattern=TFIDF_TOKEN_PATTERN, lowercase=False)
    matrix = vectorizer.fit_transform(texts)
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = vectorizer.idf_
    return (
        dict(zip(vectorizer.get_feature_names_out(), idf)),
        dict(zip(vectorizer.get_feature_names_out(), document_frequency)),
        matrix.shape[0],
    )


def accumulate_groups(
    records: list[tuple[dict[str, list[str]], str]],
) -> dict[str, Counter]:
    """Token counters per field group over every attributed row."""
    counters: dict[str, Counter] = {}
    for groups, _ in records:
        for name, tokens in groups.items():
            counters.setdefault(name, Counter()).update(tokens)
    return counters


def composition_rows(
    *, population: str, side: str, profile: str,
    counters: dict[str, Counter],
    idf_by_token: dict[str, float], df_by_token: dict[str, int], documents: int,
) -> list[GroupCompositionRow]:
    """Turn per-group token counters into token-mass and idf statistics."""
    total_tokens = sum(sum(counter.values()) for counter in counters.values())
    output: list[GroupCompositionRow] = []
    for name, counter in counters.items():
        tokens = sum(counter.values())
        ubiquitous = sum(
            count
            for token, count in counter.items()
            if documents and df_by_token.get(token, 0) / documents > UBIQUITOUS_DOC_FRACTION
        )
        weighted = sum(
            idf_by_token.get(token, 0.0) * count for token, count in counter.items()
        )
        output.append(GroupCompositionRow(
            population=population, side=side, profile=profile, group=name,
            tokens=tokens,
            token_share=tokens / total_tokens if total_tokens else 0.0,
            distinct_tokens=len(counter),
            ubiquitous_token_share=ubiquitous / tokens if tokens else 0.0,
            # Averaged over token INSTANCES, so a group is not flattered by
            # having many distinct-but-rare tokens.
            mean_idf=weighted / tokens if tokens else 0.0,
        ))
    return output


def budget_row(
    *, population: str, side: str, profile: str, counts: np.ndarray, max_seq_length: int,
) -> TokenBudgetRow:
    """Summarise a token-count vector against the encoder's own cap."""
    truncated = counts > max_seq_length
    fits = counts[~truncated]
    return TokenBudgetRow(
        population=population, side=side, profile=profile, records=len(counts),
        max_seq_length=max_seq_length,
        mean_tokens=float(counts.mean()) if len(counts) else 0.0,
        median_tokens=float(np.median(counts)) if len(counts) else 0.0,
        p95_tokens=float(np.percentile(counts, 95)) if len(counts) else 0.0,
        max_tokens=int(counts.max()) if len(counts) else 0,
        truncated_records=int(truncated.sum()),
        truncated_share=float(truncated.mean()) if len(counts) else 0.0,
        tokens_lost_total=int((counts[truncated] - max_seq_length).sum()),
        mean_headroom_when_fits=float((max_seq_length - fits).mean()) if len(fits) else 0.0,
    )


def text_presence(frame: pd.DataFrame, fields: tuple[str, ...]) -> dict[str, list[bool]]:
    """Populated-share flags for free-text columns (non-empty string)."""
    return {
        field: frame[field].astype(str).str.strip().ne("").tolist()
        for field in fields
    }


def parsed_presence(infos: list[dict], keys: tuple[str, ...]) -> dict[str, list[bool]]:
    """Populated-share flags for parsed attribute groups.

    Uses the composition's own parsers (``canonical_info`` / ``sku_info``)
    rather than the raw column, because the canonical artifact stores an absent
    set as the literal text ``[]`` — a non-empty string that would otherwise be
    counted as 100% populated.
    """
    return {
        key: [bool(info.get(key)) for info in infos]
        for key in keys
    }


def truncation_loss(
    tokenizer, grouped: list[tuple[dict[str, list[str]], str]],
    counts: np.ndarray, max_seq_length: int, *, cap: int,
) -> tuple[Counter, int]:
    """Attribute over-cap tokens to the field group that overflowed.

    Walks the groups in the builder's own emission order, so "lost to
    truncation" names the group the encoder never sees.  ``cap`` bounds the
    per-row tokeniser work; the returned record count says how many truncated
    rows were actually attributed, so the number is never silently partial.
    """
    lost: Counter = Counter()
    attributed = 0
    for index in np.flatnonzero(counts > max_seq_length)[:cap]:
        group_counts, ordered = cursor_attribution(
            tokenizer, grouped[index][0], grouped[index][1]
        )
        position = 0
        for name, group_count in zip(ordered, group_counts):
            overflow = max(0, position + group_count - max_seq_length)
            if overflow:
                lost[name] += overflow
            position += group_count
        attributed += 1
    return lost, attributed


def coverage_rows(
    *, population: str, presence: dict[str, list[bool]],
) -> list[FieldCoverageRow]:
    """Populated-share per field from explicit presence flags."""
    output: list[FieldCoverageRow] = []
    for field, flags in presence.items():
        records = len(flags)
        populated = int(sum(flags))
        output.append(FieldCoverageRow(
            population=population, field=field, records=records,
            populated=populated,
            populated_share=populated / records if records else 0.0,
        ))
    return output


def resolve_commit_sha() -> str:
    """The revision these numbers describe."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--ubiquitous-doc-fraction", type=float, default=UBIQUITOUS_DOC_FRACTION,
    )
    parser.add_argument(
        "--skip-source-corpus", action="store_true",
        help="analysis only the canonical corpus and the review population",
    )
    args = parser.parse_args()

    max_seq_length = int(runtime("max_seq_length"))
    spec = TrainingSpec.ModelInputSpec.model_validate(
        {"profile": PROFILE_CLEANED, "include_evidence": False}
    )
    legacy_spec = TrainingSpec.ModelInputSpec.model_validate(
        {"profile": PROFILE_LEGACY, "include_evidence": False}
    )
    legacy_before_spec = TrainingSpec.ModelInputSpec.model_validate(
        {"profile": PROFILE_LEGACY, "include_evidence": True}
    )
    tokenizer_path = resolve_model("minilm_l6")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    canonical = canonical_records_frame()
    canonical["gtin"] = canonical["gtin"].astype(str)
    # load_dataset() applies COLUMN_MAPPING (config/paths.yaml) and validates
    # the export, so title/attributes/description/category_path are the
    # canonical names the model-input builder reads.  Reading the raw file
    # directly leaves those fields empty and silently measures brand-only text.
    source = load_dataset().drop_duplicates("product_id").reset_index(drop=True)

    review = pd.read_csv(args.review, dtype=str, keep_default_na=False)
    review["SKU_ID"] = review["SKU_ID"].astype(str)
    review["NEAREST_ITEM_ID"] = review["NEAREST_ITEM_ID"].astype(str)
    source_by_sku = source.set_index("product_id")
    canonical_by_gtin = canonical.set_index("gtin")
    if not set(review["SKU_ID"]).issubset(source_by_sku.index):
        raise SystemExit("review SKU_IDs do not all resolve in the source corpus")
    if not set(review["NEAREST_ITEM_ID"]).issubset(canonical_by_gtin.index):
        raise SystemExit("review NEAREST_ITEM_IDs do not all resolve in the canonical corpus")

    budget: list[TokenBudgetRow] = []
    composition: list[GroupCompositionRow] = []
    coverage: list[FieldCoverageRow] = []
    comparison: list[ProfileComparisonRow] = []
    truncation_losses: list[TruncationLossRow] = []

    # ── 3. field presence / coverage across the corpus ─────────────────────
    # Parsed attribute presence comes from the composition's own parsers, so a
    # set column stored as "[]" counts as ABSENT rather than as populated.
    canonical_records = canonical.to_dict("records")
    canonical_infos = [canonical_info(record) for record in canonical_records]
    coverage += coverage_rows(
        population="canonical_corpus",
        presence={
            **text_presence(canonical, (
                "mode_brand", "mode_flavor", "mode_type", "salient_ngrams",
                "description_evidence", "breadcrumb_evidence",
            )),
            **parsed_presence(canonical_infos, (
                "volume", "pack", "package_type", "flavor",
                "carbonation", "sweetener", "pulp",
            )),
        },
    )
    source_infos = [
        sku_info(
            row_metadata_text(row, "title"), row_metadata_text(row, "attributes", "attr")
        )
        for _, row in source.iterrows()
    ]
    coverage += coverage_rows(
        population="source_corpus",
        presence={
            **text_presence(source, (
                "brand", "title", "attributes", "description",
                "category_path", "barcode", "category",
            )),
            **parsed_presence(source_infos, (
                "volume", "pack", "package_type", "flavor",
                "carbonation", "sweetener", "pulp",
            )),
        },
    )

    # ── 1+2. canonical corpus: budget, composition ─────────────────────────
    canonical_texts = [
        build_canonical_text(record, info, spec=spec)
        for record, info in zip(canonical_records, canonical_infos)
    ]
    counts = token_counts(tokenizer, canonical_texts)
    budget.append(budget_row(
        population="canonical_corpus", side="target", profile=PROFILE_CLEANED,
        counts=counts, max_seq_length=max_seq_length,
    ))
    idf, df, documents = fit_idf_reference(canonical_texts)
    grouped = [
        (group_tokens_for_target(record, info), text)
        for record, info, text in zip(canonical_records, canonical_infos, canonical_texts)
    ]
    # The attribution is asserted on a bounded probe, then applied in bulk.
    for groups, text in grouped[:2048]:
        cursor_attribution(tokenizer, groups, text)
    counters = accumulate_groups(grouped)
    composition += composition_rows(
        population="canonical_corpus", side="target", profile=PROFILE_CLEANED,
        counters=counters, idf_by_token=idf, df_by_token=df, documents=documents,
    )
    lost_by_group, attributed_targets = truncation_loss(
        tokenizer, grouped, counts, max_seq_length, cap=TRUNCATION_ATTRIBUTION_CAP
    )
    for group, tokens in lost_by_group.most_common():
        truncation_losses.append(TruncationLossRow(
            population="canonical_corpus", side="target", group=group,
            tokens_lost=tokens, truncated_records_attributed=attributed_targets,
        ))

    # ── 1+2. source corpus: budget, composition ────────────────────────────
    source_texts: list[str] = []
    source_grouped: list[tuple[dict[str, list[str]], str]] = []
    if not args.skip_source_corpus:
        for (_, row), info in zip(source.iterrows(), source_infos):
            text = build_sku_text(row, info, spec=spec)
            source_texts.append(text)
            source_grouped.append((group_tokens_for_source(row, info), text))
        source_counts = token_counts(tokenizer, source_texts)
        budget.append(budget_row(
            population="source_corpus", side="source", profile=PROFILE_CLEANED,
            counts=source_counts, max_seq_length=max_seq_length,
        ))
        source_idf, source_df, source_documents = fit_idf_reference(source_texts)
        for groups, text in source_grouped[:2048]:
            cursor_attribution(tokenizer, groups, text)
        source_lost, attributed_source = truncation_loss(
            tokenizer, source_grouped, source_counts, max_seq_length,
            cap=TRUNCATION_ATTRIBUTION_CAP,
        )
        for group, tokens in source_lost.most_common():
            truncation_losses.append(TruncationLossRow(
                population="source_corpus", side="source", group=group,
                tokens_lost=tokens, truncated_records_attributed=attributed_source,
            ))
        source_counters = accumulate_groups(source_grouped)
        composition += composition_rows(
            population="source_corpus", side="source", profile=PROFILE_CLEANED,
            counters=source_counters,
            idf_by_token=source_idf, df_by_token=source_df, documents=source_documents,
        )

    # ── review population: cleaned vs legacy, both sides ───────────────────
    review_source_rows = [source_by_sku.loc[sku] for sku in review["SKU_ID"]]
    review_target_records = [
        canonical_by_gtin.loc[gtin].to_dict() for gtin in review["NEAREST_ITEM_ID"]
    ]
    review_source_infos = [
        sku_info(row_metadata_text(row, "title"), row_metadata_text(row, "attributes", "attr"))
        for row in review_source_rows
    ]
    review_target_infos = [canonical_info(record) for record in review_target_records]

    texts = {
        ("source", PROFILE_CLEANED): [
            build_sku_text(row, info, spec=spec)
            for row, info in zip(review_source_rows, review_source_infos)
        ],
        ("target", PROFILE_CLEANED): [
            build_canonical_text(record, info, spec=spec)
            for record, info in zip(review_target_records, review_target_infos)
        ],
        ("source", PROFILE_LEGACY): [
            build_sku_text(row, info, spec=legacy_spec)
            for row, info in zip(review_source_rows, review_source_infos)
        ],
        ("target", PROFILE_LEGACY): [
            build_canonical_text(record, info, spec=legacy_spec)
            for record, info in zip(review_target_records, review_target_infos)
        ],
        ("target", PROFILE_LEGACY_BEFORE): [
            build_canonical_text(record, info, spec=legacy_before_spec)
            for record, info in zip(review_target_records, review_target_infos)
        ],
        ("source", PROFILE_LEGACY_BEFORE): [
            build_sku_text(row, info, spec=legacy_before_spec)
            for row, info in zip(review_source_rows, review_source_infos)
        ],
    }
    review_counts = {
        key: token_counts(tokenizer, value) for key, value in texts.items()
    }
    for (side, profile), counts in review_counts.items():
        budget.append(budget_row(
            population="review_pairs", side=side, profile=profile,
            counts=counts, max_seq_length=max_seq_length,
        ))

    # Composition on the review population uses the FULL corpus idf reference,
    # so "ubiquitous" means ubiquitous in the corpus, not in these 585 rows.
    review_source_grouped = [
        (group_tokens_for_source(row, info), text)
        for row, info, text in zip(
            review_source_rows, review_source_infos, texts[("source", PROFILE_CLEANED)]
        )
    ]
    review_target_grouped = [
        (group_tokens_for_target(record, info), text)
        for record, info, text in zip(
            review_target_records, review_target_infos, texts[("target", PROFILE_CLEANED)]
        )
    ]
    if not args.skip_source_corpus:
        counters = accumulate_groups(review_source_grouped)
        composition += composition_rows(
            population="review_pairs", side="source", profile=PROFILE_CLEANED,
            counters=counters,
            idf_by_token=source_idf, df_by_token=source_df, documents=source_documents,
        )
    counters = accumulate_groups(review_target_grouped)
    composition += composition_rows(
        population="review_pairs", side="target", profile=PROFILE_CLEANED,
        counters=counters, idf_by_token=idf, df_by_token=df, documents=documents,
    )

    # ── 4. per-field symmetry on the pairs sharing one product ─────────────
    same_product = review["SKU_ID"] == review["target_original_product_id"]
    symmetry_rows = []
    for index in np.flatnonzero(same_product.to_numpy()):
        source_groups = group_tokens_for_source(
            review_source_rows[index], review_source_infos[index]
        )
        target_groups = group_tokens_for_target(
            review_target_records[index], review_target_infos[index]
        )
        row = {
            "sku_id": review["SKU_ID"].iloc[index],
            "score": float(review["SCORE"].iloc[index]),
            "source_brand": row_metadata_text(review_source_rows[index], "brand"),
            "target_brand": canonical_by_gtin.loc[
                review["NEAREST_ITEM_ID"].iloc[index], "mode_brand"
            ],
        }
        for name in CLEANED_SOURCE_GROUPS:
            row[f"source_{name}_tokens"] = len(source_groups[name])
            row[f"target_{name}_tokens"] = len(target_groups.get(name, []))
        row["source_structured_tokens"] = len(source_groups[STRUCTURED_GROUP])
        row["target_structured_tokens"] = len(target_groups[STRUCTURED_GROUP])
        row["source_has_pack_group"] = "[FIELD_PACK_SIZE]" in source_groups[STRUCTURED_GROUP]
        row["target_has_pack_group"] = "[FIELD_PACK_SIZE]" in target_groups[STRUCTURED_GROUP]
        # Three compositions, plus the shared/union split, so the reported
        # regression can be attributed to tokens rather than asserted.
        for label, profile in (
            ("before", PROFILE_LEGACY_BEFORE),
            ("legacy_no_evidence", PROFILE_LEGACY),
            ("cleaned", PROFILE_CLEANED),
        ):
            source_tokens = set(texts[("source", profile)][index].split())
            target_tokens = set(texts[("target", profile)][index].split())
            union = source_tokens | target_tokens
            row[f"{label}_shared_tokens"] = len(source_tokens & target_tokens)
            row[f"{label}_union_tokens"] = len(union)
            row[f"{label}_jaccard"] = jaccard_similarity(
                texts[("source", profile)][index], texts[("target", profile)][index]
            )
        symmetry_rows.append(row)
    symmetry = pd.DataFrame(symmetry_rows)

    # ── 4b. structured-channel presence across the WHOLE population ────────
    # A structured group only one side can emit is not a comparison.  The
    # canonical side reads pack_set directly, so an unobserved pack is absent
    # there, while sku_info() uses 1.0 as its explicit "not observed" sentinel
    # and the source side still emits [FIELD_PACK_SIZE] pack_qty_1.
    asymmetry: list[StructuredAsymmetryRow] = []
    for marker in STRUCTURED_MARKERS:
        name = f"[FIELD_{marker}]"
        source_flags = [name in groups[STRUCTURED_GROUP] for groups, _ in review_source_grouped]
        target_flags = [name in groups[STRUCTURED_GROUP] for groups, _ in review_target_grouped]
        both = sum(a and b for a, b in zip(source_flags, target_flags))
        source_only = sum(a and not b for a, b in zip(source_flags, target_flags))
        target_only = sum(b and not a for a, b in zip(source_flags, target_flags))
        asymmetry.append(StructuredAsymmetryRow(
            marker=marker,
            source_present=int(sum(source_flags)),
            target_present=int(sum(target_flags)),
            both=both, source_only=source_only, target_only=target_only,
            neither=len(source_flags) - both - source_only - target_only,
        ))

    # ── 5. cleaned vs legacy at input level ────────────────────────────────
    def true_cross_jaccard(sources: list[str], targets: list[str], per_row: int = 12):
        true = float(np.mean([jaccard_similarity(a, b) for a, b in zip(sources, targets)]))
        step = max(1, len(targets) // per_row)
        values = [
            jaccard_similarity(sources[i], targets[(i + k + 1) % len(targets)])
            for i in range(len(sources))
            for k in range(0, len(targets), step)
        ]
        cross = float(np.mean(values))
        return true, cross

    curve: dict[str, tuple[float, float]] = {}
    for label, profile in (
        ("cleaned", PROFILE_CLEANED),
        ("before", PROFILE_LEGACY_BEFORE),
        ("legacy_no_evidence", PROFILE_LEGACY),
    ):
        curve[label] = true_cross_jaccard(
            texts[("source", profile)], texts[("target", profile)]
        )
    identical_24: dict[str, float] = {}
    for label in curve:
        values = symmetry[f"{label}_jaccard"].to_numpy() if not symmetry.empty else np.array([])
        identical_24[label] = float(values.mean()) if len(values) else 0.0
    for metric, values in (
        ("true_pair_jaccard", {k: v[0] for k, v in curve.items()}),
        ("cross_pair_jaccard", {k: v[1] for k, v in curve.items()}),
        ("retrieval_margin", {k: v[0] - v[1] for k, v in curve.items()}),
        ("identical_product_24_jaccard", identical_24),
        ("mean_source_tokens", {
            label: float(review_counts[("source", profile)].mean())
            for label, profile in (
                ("cleaned", PROFILE_CLEANED), ("before", PROFILE_LEGACY_BEFORE),
                ("legacy_no_evidence", PROFILE_LEGACY),
            )
        }),
        ("mean_target_tokens", {
            label: float(review_counts[("target", profile)].mean())
            for label, profile in (
                ("cleaned", PROFILE_CLEANED), ("before", PROFILE_LEGACY_BEFORE),
                ("legacy_no_evidence", PROFILE_LEGACY),
            )
        }),
        ("target_truncated_records", {
            label: float((review_counts[("target", profile)] > max_seq_length).sum())
            for label, profile in (
                ("cleaned", PROFILE_CLEANED), ("before", PROFILE_LEGACY_BEFORE),
                ("legacy_no_evidence", PROFILE_LEGACY),
            )
        }),
        # What the evidence channel alone contributes, per side: the difference
        # between the pre-consolidation composition and legacy with it switched
        # off.  This is the quantity the identical-row regression turns on.
        ("evidence_channel_target_tokens", {
            "cleaned": 0.0,
            "before": float((review_counts[("target", PROFILE_LEGACY_BEFORE)]
                             - review_counts[("target", PROFILE_LEGACY)]).mean()),
            "legacy_no_evidence": 0.0,
        }),
        ("evidence_channel_source_tokens", {
            "cleaned": 0.0,
            "before": float((review_counts[("source", PROFILE_LEGACY_BEFORE)]
                             - review_counts[("source", PROFILE_LEGACY)]).mean()),
            "legacy_no_evidence": 0.0,
        }),
    ):
        comparison.append(ProfileComparisonRow(
            metric=metric, population="review_pairs", side="both",
            cleaned=values["cleaned"], before=values["before"],
            legacy_no_evidence=values["legacy_no_evidence"],
            delta_vs_before=values["cleaned"] - values["before"],
        ))

    provenance = ModelInputProvenance(
        commit_sha=resolve_commit_sha(),
        review_csv=str(args.review),
        profile_default=spec.profile,
        include_evidence_default=spec.include_evidence,
        structured_text_enabled=_structured_text_enabled(),
        max_seq_length=max_seq_length,
        tokenizer_path=str(tokenizer_path),
        ubiquitous_doc_fraction=args.ubiquitous_doc_fraction,
        canonical_corpus_rows=len(canonical),
        source_corpus_rows=len(source),
        review_rows=len(review),
    )
    summary = ModelInputSummary(
        provenance=provenance, token_budget=budget, group_composition=composition,
        field_coverage=coverage, truncation_loss=truncation_losses,
        structured_asymmetry=asymmetry,
        profile_comparison=comparison,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "model_input_summary.json").write_text(
        json.dumps(summary.model_dump(), indent=2) + "\n", encoding="utf-8",
    )
    budget_frame = pd.DataFrame([row.model_dump() for row in budget])
    composition_frame = pd.DataFrame([row.model_dump() for row in composition])
    coverage_frame = pd.DataFrame([row.model_dump() for row in coverage])
    comparison_frame = pd.DataFrame([row.model_dump() for row in comparison])
    budget_frame.to_csv(args.out_dir / "token_budget.csv", index=False)
    composition_frame.to_csv(args.out_dir / "group_composition.csv", index=False)
    coverage_frame.to_csv(args.out_dir / "field_coverage.csv", index=False)
    comparison_frame.to_csv(args.out_dir / "profile_comparison.csv", index=False)
    symmetry.to_csv(args.out_dir / "pair_symmetry_same_product.csv", index=False)
    pd.DataFrame([row.model_dump() for row in truncation_losses]).to_csv(
        args.out_dir / "truncation_loss_by_group.csv", index=False,
    )
    asymmetry_frame = pd.DataFrame([row.model_dump() for row in asymmetry])
    asymmetry_frame.to_csv(args.out_dir / "structured_channel_asymmetry.csv", index=False)

    print(f"commit: {provenance.commit_sha}")
    print(f"max_seq_length: {max_seq_length}  |  cleaned default: "
          f"{spec.profile}/evidence={spec.include_evidence}  |  "
          f"structured text: {provenance.structured_text_enabled}")
    print("\n── token budget ──")
    print(budget_frame[[
        "population", "side", "profile", "records", "mean_tokens", "median_tokens",
        "p95_tokens", "max_tokens", "truncated_records", "truncated_share",
        "tokens_lost_total", "mean_headroom_when_fits",
    ]].to_string(index=False))
    if truncation_losses:
        print("\n── tokens past max_seq_length, by field group ──")
        print(pd.DataFrame([row.model_dump() for row in truncation_losses]).to_string(index=False))
    print("\n── group composition (token share / ubiquitous share / mean idf) ──")
    for row in composition:
        print(f"  {row.population:17s} {row.side:7s} {row.group:11s} "
              f"share={row.token_share:6.1%} filler={row.ubiquitous_token_share:6.1%} "
              f"mean_idf={row.mean_idf:5.2f} distinct={row.distinct_tokens}")
    print("\n── field coverage (lowest 12 populated shares per population) ──")
    for population, group in coverage_frame.groupby("population"):
        print(f"  {population}:")
        for row in group.sort_values("populated_share").head(12).itertuples():
            print(f"    {row.field:24s} {row.populated_share:6.1%}  ({row.populated}/{row.records})")
    print(f"\n── structured channel presence across the {len(review)} review pairs ──")
    print(asymmetry_frame.to_string(index=False))
    print("\n── cleaned vs BEFORE (legacy+evidence) and legacy-without-evidence ──")
    print(comparison_frame.to_string(index=False))
    if not symmetry.empty:
        print(f"\n── symmetry on the {len(symmetry)} rows sharing one product ──")
        for label in ("before", "legacy_no_evidence", "cleaned"):
            print(f"  {label:20s} mean Jaccard={symmetry[f'{label}_jaccard'].mean():.4f} "
                  f"shared={symmetry[f'{label}_shared_tokens'].mean():5.2f} "
                  f"union={symmetry[f'{label}_union_tokens'].mean():6.2f}")
        print(f"  rows where the SOURCE emits a pack group but the TARGET does not: "
              f"{int((symmetry.source_has_pack_group & ~symmetry.target_has_pack_group).sum())}"
              f"/{len(symmetry)}")
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
