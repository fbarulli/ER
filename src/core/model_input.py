"""The single source of truth for the text the encoder sees.

Both lanes — training candidate retrieval (``training.rand_matching``) and
scoring (``predict_items``) — plus the payload/audit stage in ``pipeline``
build their model input HERE.  Before this module the same composition was
copy-pasted in all three places, which is precisely how the two lanes drifted
apart: the source side read the raw ``description``/``category``/``breadcrumbs``
columns while the target side read ``description_evidence``/``breadcrumb_evidence``
from the canonical record, so one product produced two unrelated strings.

Profile selection lives in ``config/training.yaml`` (``training.model_input``)
and is validated by ``core.schemas.TrainingSpec.ModelInputSpec``:

``cleaned``
    The shipped default.  ``[Brand] [Title] [Attributes]`` composition — those
    three field groups, in that order, and nothing else.  Underscore compounds
    are split so they can lexically match source text, discriminative numbers
    survive into the model input, and the description/breadcrumb evidence
    channel is excluded.
``legacy``
    The pre-change composition, reproduced byte for byte.  Kept selectable
    from config so going back is a config edit rather than a code revert;
    ``tests/test_model_input_contract.py`` pins it against fixtures captured
    from the unmodified code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from core.common import load_config, row_metadata_text
from core.schemas import TrainingSpec

__all__ = [
    "build_canonical_text",
    "build_sku_text",
    "implicit_pack_qty",
    "model_input_composition",
    "model_input_info",
    "model_input_spec",
    "token_budget_report",
]

# Percentage evidence, captured before normalize_text removes the sign.
# "0-2%" is a range (juice content bands), "100%" a single value.
_PERCENT_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# the [FIELD_*] groups the structured channel can emit
_FIELD_GROUP_RE = re.compile(r"\[FIELD_([A-Z_]+)\]")


def model_input_spec() -> TrainingSpec.ModelInputSpec:
    """The validated model-input composition settings (config SSOT)."""
    return TrainingSpec.ModelInputSpec.model_validate(
        load_config()["training"]["model_input"]
    )


def model_input_composition() -> TrainingSpec.ModelInputComposition:
    """The ACTIVE composition, for fingerprints, traces, manifests, bundles.

    Any artifact whose contents depend on the encoder text — a persisted
    embedding index, a checkpoint, a frozen payload bundle above all — must be
    able to name the exact input contract that produced it.  Changing the
    profile changes the text but NOT the catalog, the checkpoint or the code
    path, so without this the artifact looks reusable when it is not.
    """
    return TrainingSpec.ModelInputComposition.from_spec(model_input_spec())


def implicit_pack_qty() -> float:
    """The configured implicit pack count for an unobserved pack (SSOT)."""
    return float(load_config()["training"]["structured_features"]["implicit_pack_qty"])


def token_budget_report(
    texts: Sequence[str], *, tokenizer, max_seq_length: int
):
    """Count what the encoder window KEEPS, and name every dropped field group.

    The ``[FIELD_*]`` groups are appended last, so at ``max_seq_length`` they
    are truncated first. Measured before this guard existed: 11.9 % of target
    texts lost the whole structured tail, 27.7 % exceeded the window, and none
    of it was visible anywhere except a shorter string.

    Returns ``core.schemas.TokenBudgetReport``: one record per payload, plus a
    count per dropped field group, so a lost group is a named number rather
    than an absence. Never raises on a long payload — the caller decides
    whether to reorder fields or raise the budget.
    """
    from core.schemas import TokenBudgetReport

    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")

    n_over = 0
    dropped: dict[str, int] = {}
    for text in texts:
        encoded = tokenizer(
            str(text), add_special_tokens=True, return_offsets_mapping=True
        )
        offsets = encoded["offset_mapping"]
        if len(offsets) <= max_seq_length:
            continue
        n_over += 1
        kept_end = offsets[max_seq_length - 1][1]
        for group in _FIELD_GROUP_RE.findall(str(text)):
            marker = f"[FIELD_{group}]"
            if str(text).find(marker) >= kept_end:
                dropped[group] = dropped.get(group, 0) + 1
    return TokenBudgetReport(
        max_seq_length=int(max_seq_length),
        n_records=len(texts),
        n_over_budget=n_over,
        n_field_groups_dropped=sum(dropped.values()),
        dropped_groups=dict(sorted(dropped.items())),
    )


def _structured_text_enabled() -> bool:
    """Whether the structured token channel is appended to the model text.

    Owned here so the three call sites stop recomputing the same
    ``enabled and append_to_text`` pair from their own config reads.
    """
    cfg = load_config()["training"]["structured_features"]
    return bool(cfg["enabled"]) and bool(cfg["append_to_text"])


def _normalized_tokens(text: object, *, drop_schema_words: bool) -> list[str]:
    """One normalizer for BOTH lanes.

    Underscore compounds are split into their parts so a canonical token such
    as ``bcaa_pear`` can match the source words ``bcaa`` and ``pear``.  Digits
    are deliberately preserved: ``6000mg`` is discriminative, and no
    number-stripping runs on this path.

    Percentage values become ``pct*`` tokens BEFORE ``normalize_text`` deletes
    the ``%`` sign.  92.6% of the review-band source rows carry a juice-content
    percentage (``100%``, ``0-2%``, ...), and once the sign is gone the bare
    number collides with the ``MINIMAL_STOPWORDS`` volume entries (``100``,
    ``2``), so the attribute was silently discarded on the legacy path.

    Diacritics are NOT folded here (owner ruling): brand-string normalisation
    is owned by the dedicated brand-analysis work, and the measured evidence
    for it is recorded in MODEL_INPUT_FIX_REPORT.md section 24 as input to that
    deliberate decision rather than being applied ahead of it.
    """
    from pipeline import MINIMAL_STOPWORDS, normalize_text, strip_schema_words

    # normalize_text preserves [a-z0-9.], so a decimal such as 5.5% survives as
    # ONE token. Never use "_" here: the compound splitter below would cut the
    # value in half.
    protected = _PERCENT_RANGE_RE.sub(
        lambda m: f" pct{m.group(1)}to{m.group(2)} ", str(text)
    )
    protected = _PERCENT_RE.sub(lambda m: f" pct{m.group(1)} ", protected)
    # Fold accents BEFORE normalize_text. Without this every non-ASCII letter
    # becomes a word break, which does not merely leave a brand unnormalised —
    # it CORRUPTS it: "Brämhults" -> "br mhults", "Côteaux Nantais" ->
    # "teaux nantais", and "Reál"/"Réal" -> "re"/"al", so two spellings of one
    # brand can never match. 47 distinct canonical brands carry non-ASCII.
    split = " ".join(
        part
        for token in normalize_text(protected).split()
        for part in token.split("_")
        if part
    )
    if drop_schema_words:
        split = strip_schema_words(split)
    return [t for t in split.split() if len(t) > 1 and t not in MINIMAL_STOPWORDS]


def _cleaned_sku_text(row, info: Mapping[str, object]) -> str:
    """[Brand] [Title] [Attributes], in that order, with no other field.

    The three blocks are emitted as plain tokens in a fixed order rather than
    behind literal ``[BRAND]``-style markers: measured on both fixture
    populations, the markers are constant mass shared by every pair, so they
    lifted cross-pair token overlap (0.084 -> 0.164 on the singleton-GTIN
    group) further than true-pair overlap and lowered the true-vs-cross
    margin.  Field SELECTION and ORDER carry the user's requested structure;
    the markers cost discrimination.
    """
    from core.structured_features import append_text

    tokens: list[str] = []
    tokens += _normalized_tokens(row_metadata_text(row, "brand"), drop_schema_words=False)
    tokens += _normalized_tokens(row_metadata_text(row, "title"), drop_schema_words=False)
    tokens += _normalized_tokens(
        row_metadata_text(row, "attributes", "attr"), drop_schema_words=True
    )
    symmetric = model_input_info(info)
    return append_text(" ".join(tokens), symmetric, enabled=_structured_text_enabled())


def _cleaned_canonical_text(
    record: Mapping[str, object], info: Mapping[str, object]
) -> str:
    """[Brand] [Title] [Attributes] for the canonical side, same normalizer."""
    from core.structured_features import append_text

    brand = _normalized_tokens(record.get("mode_brand", ""), drop_schema_words=False)
    brand_words = {word.casefold() for word in brand}
    canonical = [
        word
        for word in _normalized_tokens(
            record.get("canonical", ""), drop_schema_words=True
        )
        if word.casefold() not in brand_words
    ]
    tokens = [
        *brand,
        *canonical,
        *_normalized_tokens(record.get("mode_type", ""), drop_schema_words=True),
    ]
    symmetric = model_input_info(info)
    return append_text(" ".join(tokens), symmetric, enabled=_structured_text_enabled())


def _legacy_sku_text(row, info: Mapping[str, object]) -> str:
    from core.structured_features import append_text
    from pipeline import clean_sku_text, strip_schema_words

    base = strip_schema_words(clean_sku_text(
        row_metadata_text(row, "title"),
        row_metadata_text(row, "attributes", "attr"),
        row_metadata_text(row, "brand"),
        row_metadata_text(row, "description", "description_short_eng"),
        row_metadata_text(row, "category", "category_path"),
        row_metadata_text(row, "category_path", "breadcrumbs_eng"),
    ))
    return append_text(base, info, enabled=_structured_text_enabled())


def _legacy_canonical_text(
    record: Mapping[str, object], info: Mapping[str, object], *, evidence: bool
) -> str:
    from core.structured_features import append_text
    from pipeline import (
        canonical_evidence_text,
        canonical_model_text,
        strip_schema_words,
    )

    parts = [
        str(record.get("canonical", "")),
        str(record.get("mode_brand", "")),
        str(record.get("mode_type", "")),
    ]
    if evidence:
        parts.append(canonical_evidence_text(record.get("description_evidence", "")))
        parts.append(canonical_evidence_text(record.get("breadcrumb_evidence", "")))
    base = strip_schema_words(canonical_model_text(" ".join(parts)))
    return append_text(base, info, enabled=_structured_text_enabled())


def model_input_info(
    info: Mapping[str, object],
    *,
    spec: TrainingSpec.ModelInputSpec | None = None,
) -> dict[str, set[float] | set[str]]:
    """The structured info as the ACTIVE composition sees it.

    Callers use this once per row and feed the result to BOTH the text builder
    and the numeric vector, so an attribute's unobserved treatment can never
    differ between the two channels.

    ``legacy`` returns the info unchanged (its historical, one-sided treatment
    is what the golden fixtures pin). ``cleaned`` applies the universal
    implicit-default rule, which is the shipped default.
    """
    resolved = _resolve(spec)
    if resolved.profile != "cleaned":
        return dict(info)
    from core.structured_features import symmetric_info

    return symmetric_info(info, implicit_pack_qty=implicit_pack_qty())


def _resolve(
    spec: TrainingSpec.ModelInputSpec | None,
) -> TrainingSpec.ModelInputSpec:
    return model_input_spec() if spec is None else spec


def build_sku_text(
    row,
    info: Mapping[str, object],
    *,
    spec: TrainingSpec.ModelInputSpec | None = None,
) -> str:
    """Model text for one SOURCE sku row.

    ``row`` is a pandas row/Series so field fallbacks go through the shared
    ``core.common.row_metadata_text`` reader; ``info`` is the structured
    attribute mapping the caller already built for the numeric channel.
    """
    resolved = _resolve(spec)
    if resolved.profile == "cleaned":
        return _cleaned_sku_text(row, info)
    return _legacy_sku_text(row, info)


def build_canonical_text(
    record: Mapping[str, object],
    info: Mapping[str, object],
    *,
    spec: TrainingSpec.ModelInputSpec | None = None,
) -> str:
    """Model text for one canonical (target) record."""
    resolved = _resolve(spec)
    if resolved.profile == "cleaned":
        return _cleaned_canonical_text(record, info)
    return _legacy_canonical_text(record, info, evidence=resolved.include_evidence)
