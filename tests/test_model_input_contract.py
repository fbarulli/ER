"""Model-input composition contract: one builder, two config-selectable profiles.

``core.model_input`` is the single source of truth for the encoder text in both
lanes.  These tests pin the two things that make the config switch safe:

* the ``legacy`` profile still reproduces the committed byte stream exactly
  (the rollback contract, checked against fixtures captured from the
  unmodified code before any change), and
* the ``cleaned`` profile actually delivers what it claims — compound
  splitting, number preservation, boilerplate exclusion and source/target
  symmetry on real data, without collapsing different products together.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from core.model_input import (
    _normalized_tokens,
    build_canonical_text,
    build_sku_text,
    model_input_spec,
)
from core.schemas import TrainingSpec
from core.structured_features import canonical_info, sku_info

FIXTURE = Path(__file__).parent / "fixtures" / "model_input_golden.json"
GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))
RECORDS = GOLDEN["records"]

LEGACY = TrainingSpec.ModelInputSpec(profile="legacy", include_evidence=True)
LEGACY_NO_EVIDENCE = TrainingSpec.ModelInputSpec(profile="legacy", include_evidence=False)
CLEANED = TrainingSpec.ModelInputSpec(profile="cleaned", include_evidence=False)


def _group(name: str) -> list[dict]:
    rows = [r for r in RECORDS if r["group"] == name]
    assert rows, f"fixture group {name!r} is empty"
    return rows


def _pair(record: dict, spec: TrainingSpec.ModelInputSpec) -> tuple[str, str]:
    row = pd.Series(record["sku"])
    return (
        build_sku_text(row, sku_info(row["title"], row["attributes"]), spec=spec),
        build_canonical_text(
            record["canonical_record"], canonical_info(record["canonical_record"]), spec=spec
        ),
    )


def _jaccard(left: str | set[str], right: str | set[str]) -> float:
    a = left.split() if isinstance(left, str) else left
    b = right.split() if isinstance(right, str) else right
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _cross_jaccard(sources: list[str], targets: list[str], per_row: int = 12) -> float:
    """Mean Jaccard of each source against OTHER rows' targets."""
    step = max(1, len(targets) // per_row)
    values = [
        _jaccard(sources[i], targets[(i + k + 1) % len(targets)])
        for i in range(len(sources))
        for k in range(0, len(targets), step)
    ]
    return sum(values) / len(values)


# ── the switch itself ──────────────────────────────────────────────────────


def test_shipped_config_defaults_to_the_legacy_profile() -> None:
    """An untouched config must keep today's behaviour (rollback by config)."""
    spec = model_input_spec()
    assert spec.profile == "legacy"
    assert spec.include_evidence is True


def test_cleaned_profile_refuses_to_also_request_the_evidence_channel() -> None:
    """One profile plus one granular flag must not admit a contradiction."""
    with pytest.raises(ValidationError, match="excludes the"):
        TrainingSpec.ModelInputSpec(profile="cleaned", include_evidence=True)


def test_model_input_block_is_declared_in_the_config_contract() -> None:
    """No silent default: the block is part of the validated config shape."""
    assert "model_input" in TrainingSpec.model_fields


# ── legacy profile: the rollback contract ──────────────────────────────────


def test_legacy_profile_reproduces_golden_bytes() -> None:
    """Every fixture row must round-trip byte for byte on BOTH sides.

    The fixtures were captured from the committed composition before this
    module existed, so a mismatch here means the rollback path changed.
    """
    checked = 0
    for record in RECORDS:
        row = pd.Series(record["sku"])
        got_sku = build_sku_text(
            row, sku_info(row["title"], row["attributes"]), spec=LEGACY
        )
        got_canonical = build_canonical_text(
            record["canonical_record"],
            canonical_info(record["canonical_record"]),
            spec=LEGACY,
        )
        assert got_sku == record["legacy_sku_text"], f"sku text changed: {record['sku_id']}"
        assert (
            got_canonical == record["legacy_canonical_text"]
        ), f"canonical text changed: {record['nearest_item_id']}"
        checked += 1
    assert checked == len(RECORDS) > 800


def test_legacy_no_evidence_profile_removes_only_the_evidence_channel() -> None:
    """The granular flag is a real ablation, not a no-op."""
    record = next(
        r
        for r in _group("singleton_gtin")
        if set(r["legacy_canonical_text"].split()) - set(
            _pair(r, LEGACY_NO_EVIDENCE)[1].split()
        )
    )
    kept = set(_pair(record, LEGACY)[1].split())
    dropped = set(_pair(record, LEGACY_NO_EVIDENCE)[1].split())

    removed = kept - dropped
    assert removed, "the evidence channel must actually be removable"
    assert not dropped - kept, "disabling evidence must not ADD tokens"

    from pipeline import normalize_text

    normalized_evidence = normalize_text(" ".join(
        record["canonical_record"][field].strip("[]'\"")
        for field in ("description_evidence", "breadcrumb_evidence")
    ))
    for token in removed:
        assert token in normalized_evidence, f"{token!r} is not evidence text"


# ── cleaned profile: the claimed behaviour ─────────────────────────────────


def test_cleaned_profile_splits_underscore_compounds() -> None:
    """A canonical compound must become lexically matchable source words."""
    record = next(
        r for r in _group("review_band_585") if "_" in r["canonical_record"]["canonical"]
    )
    compound = next(
        t for t in record["canonical_record"]["canonical"].split() if "_" in t
    )
    _, legacy = _pair(record, LEGACY)
    _, cleaned = _pair(record, CLEANED)

    assert compound in legacy.split(), "precondition: legacy keeps the compound"
    assert compound not in cleaned.split()
    for part in compound.split("_"):
        assert part in cleaned.split()


def test_cleaned_profile_preserves_discriminative_numbers() -> None:
    """Numbers destroyed by the legacy canonical cleaner must survive."""
    record = next(
        r for r in _group("review_band_585") if "6000mg" in r["canonical_record"]["canonical"]
    )
    _, legacy = _pair(record, LEGACY)
    _, cleaned = _pair(record, CLEANED)

    assert "6000mg" not in legacy.split(), "precondition: legacy strips the number"
    assert "6000mg" in cleaned.split()


def test_cleaned_profile_preserves_percentage_evidence() -> None:
    """Juice-content percentages must survive instead of colliding with stops.

    Bare ``100`` and ``2`` are volume entries in ``MINIMAL_STOPWORDS``, so
    before this the whole ``Juice Content`` attribute was discarded.
    """
    assert _normalized_tokens("Juice Content: 100%", drop_schema_words=True) == ["pct100"]
    assert _normalized_tokens("Juice Content: 0-2%", drop_schema_words=True) == ["pct0to2"]
    assert "pct5.5" in _normalized_tokens("Alcohol: 5.5%", drop_schema_words=True)

    carried = sum(
        1
        for record in _group("review_band_585")
        if any(t.startswith("pct") for t in _pair(record, CLEANED)[0].split())
    )
    assert carried > 400, f"only {carried} rows carry percentage evidence"


def test_cleaned_profile_excludes_the_evidence_channel() -> None:
    """No description/breadcrumb evidence token may reach the cleaned text."""
    checked = 0
    for record in _group("singleton_gtin"):
        canonical_record = record["canonical_record"]
        sku_source, canonical_target = _pair(record, CLEANED)
        allowed_target = set(
            _normalized_tokens(canonical_record["canonical"], drop_schema_words=True)
            + _normalized_tokens(canonical_record["mode_brand"], drop_schema_words=False)
            + _normalized_tokens(canonical_record["mode_type"], drop_schema_words=True)
        )
        allowed_source = set(
            _normalized_tokens(record["sku"]["brand"], drop_schema_words=False)
            + _normalized_tokens(record["sku"]["title"], drop_schema_words=False)
            + _normalized_tokens(record["sku"]["attributes"], drop_schema_words=True)
        )
        evidence_only = {
            token
            for field in ("description_evidence", "breadcrumb_evidence")
            for token in _normalized_tokens(canonical_record[field], drop_schema_words=True)
        } - allowed_target
        assert not (evidence_only & set(canonical_target.split())), record["nearest_item_id"]

        raw_only = {
            token
            for field in ("description", "category", "category_path")
            for token in _normalized_tokens(record["sku"][field], drop_schema_words=True)
        } - allowed_source
        assert not (raw_only & set(sku_source.split())), record["sku_id"]
        checked += 1
    assert checked == len(_group("singleton_gtin"))


def test_cleaned_profile_emits_no_literal_block_markers() -> None:
    """Field blocks are ordered, not marked: markers measured as constant mass."""
    record = _group("review_band_585")[0]
    for text in _pair(record, CLEANED):
        assert "[BRAND]" not in text
        assert "[TITLE]" not in text
        assert "[ATTRIBUTES]" not in text


def test_cleaned_profile_uses_one_normalizer_for_both_lanes() -> None:
    """Identical input text must yield identical tokens whichever lane builds it."""
    text = "bcaa_6000mg Pear CAN 100%"
    sku_row = pd.Series(
        {"title": text, "attributes": "", "brand": "", "description": "",
         "category": "", "category_path": ""}
    )
    sku_tokens = build_sku_text(sku_row, {}, spec=CLEANED).split()
    canonical_tokens = build_canonical_text(
        {"canonical": text, "mode_brand": "", "mode_type": ""}, {}, spec=CLEANED
    ).split()

    assert sku_tokens == canonical_tokens
    assert "6000mg" in sku_tokens and "pct100" in sku_tokens
    assert "bcaa_6000mg" not in sku_tokens


def test_cleaned_profile_improves_the_true_vs_cross_margin() -> None:
    """On the review band, true pairs must be closer to their target than to others."""
    records = _group("review_band_585")
    margins = {}
    for name, spec in (("legacy", LEGACY), ("cleaned", CLEANED)):
        pairs = [_pair(r, spec) for r in records]
        sources = [p[0] for p in pairs]
        targets = [p[1] for p in pairs]
        true = sum(_jaccard(s, t) for s, t in zip(sources, targets)) / len(records)
        margins[name] = true - _cross_jaccard(sources, targets)

    assert margins["cleaned"] > margins["legacy"], margins


def test_cleaned_profile_keeps_true_pairs_far_above_cross_pairs() -> None:
    """Symmetry must not be bought by making every product look alike.

    ``singleton_gtin`` rows are one-row GTINs: the canonical is a pure function
    of the source row, so a true pair is genuinely the same product while the
    cross pairs are genuinely different products.
    """
    records = _group("singleton_gtin")
    pairs = [_pair(r, CLEANED) for r in records]
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    true = sum(_jaccard(s, t) for s, t in zip(sources, targets)) / len(records)
    cross = _cross_jaccard(sources, targets)

    assert true > 0.5, true
    assert cross < 0.2, cross
    assert true - cross > 0.35, (true, cross)


def test_different_brands_do_not_collapse_to_one_string() -> None:
    """Different products must stay distinguishable in the cleaned profile."""
    records = _group("singleton_gtin")
    left, right = records[0], records[1]
    assert left["sku"]["brand"] != right["sku"]["brand"]

    left_text = _pair(left, CLEANED)[0]
    right_text = _pair(right, CLEANED)[0]
    assert left_text != right_text
    left_brand = _normalized_tokens(left["sku"]["brand"], drop_schema_words=False)
    right_brand = _normalized_tokens(right["sku"]["brand"], drop_schema_words=False)
    assert left_brand and right_brand
    assert set(left_brand) <= set(left_text.split())
    assert set(right_brand) <= set(right_text.split())
    assert _jaccard(left_text, right_text) < 1.0


def test_structured_token_channel_is_identical_across_profiles() -> None:
    """The structured channel is orthogonal to the text-composition profile."""
    record = _group("review_band_585")[0]
    info = canonical_info(record["canonical_record"])
    legacy = build_canonical_text(record["canonical_record"], info, spec=LEGACY)
    cleaned = build_canonical_text(record["canonical_record"], info, spec=CLEANED)

    def structured_tail(text: str) -> list[str]:
        return [t for t in text.split() if t.startswith("[FIELD_")]

    assert structured_tail(legacy) == structured_tail(cleaned)


def test_title_only_payload_variant_still_blanks_attributes_and_description() -> None:
    """The payload ablation variant must keep its documented field selection.

    ``title_only`` was the payload stage's own composition before the
    consolidation; blanking the attribute/description columns must reproduce it.
    """
    from core.structured_features import append_text
    from pipeline import clean_sku_text, strip_schema_words

    record = _group("singleton_gtin")[0]
    row = pd.Series(record["sku"])
    blanked = row.copy()
    for column in ("attributes", "attr", "description", "description_short_eng"):
        if column in blanked.index:
            blanked[column] = ""
    info = sku_info(row["title"], row["attributes"])

    expected = append_text(
        strip_schema_words(clean_sku_text(
            row["title"], "", row["brand"], "", row["category"], row["category_path"]
        )),
        info,
    )
    assert build_sku_text(blanked, info, spec=LEGACY) == expected


def test_both_lanes_call_the_shared_builder() -> None:
    """Guard against the composition being re-duplicated at a call site."""
    import inspect

    import predict_items
    from training import rand_matching

    for module, expected in (
        (predict_items, 2),
        (rand_matching, 2),
    ):
        source = inspect.getsource(module)
        calls = source.count("build_sku_text(") + source.count("build_canonical_text(")
        assert calls == expected, f"{module.__name__}: {calls} shared-builder calls"
