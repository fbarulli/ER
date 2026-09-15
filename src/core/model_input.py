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

``legacy``
    The committed composition, reproduced byte for byte, so an untouched
    config keeps exactly today's behaviour and a rollback is a config edit
    rather than a code revert.
``cleaned``
    ``[Brand] [Title] [Attributes]`` composition — those three field groups,
    in that order, and nothing else.  Underscore compounds are split so they
    can lexically match source text, discriminative numbers survive into the
    model input, and the description/breadcrumb evidence channel is excluded.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from core.common import load_config, row_metadata_text
from core.schemas import TrainingSpec

__all__ = ["build_canonical_text", "build_sku_text", "model_input_spec"]

# Percentage evidence, captured before normalize_text removes the sign.
# "0-2%" is a range (juice content bands), "100%" a single value.
_PERCENT_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def model_input_spec() -> TrainingSpec.ModelInputSpec:
    """The validated model-input composition settings (config SSOT)."""
    return TrainingSpec.ModelInputSpec.model_validate(
        load_config()["training"]["model_input"]
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
    """
    from pipeline import MINIMAL_STOPWORDS, normalize_text, strip_schema_words

    # Keep the dot: normalize_text preserves [a-z0-9.], so 5.5% stays one
    # token.  Never use "_" here — the compound splitter below would cut the
    # value in half.
    protected = _PERCENT_RANGE_RE.sub(
        lambda m: f" pct{m.group(1)}to{m.group(2)} ", str(text)
    )
    protected = _PERCENT_RE.sub(lambda m: f" pct{m.group(1)} ", protected)
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
    return append_text(" ".join(tokens), info, enabled=_structured_text_enabled())


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
    return append_text(" ".join(tokens), info, enabled=_structured_text_enabled())


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
