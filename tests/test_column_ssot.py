"""tests/test_column_ssot.py — ONE declaration per column, no retyping.

Defect this pins (owner ruling 2026-10-01). The project's raw export and
canonical dataset name the SAME thirteen columns differently (`sku_name_eng`
vs `title`, `gtin` vs `barcode`, `attribute` vs `attributes`, …), and
config/paths.yaml already declared every one of those names as a verified
bijection in `column_mapping`. Three other places nonetheless retyped column
lists by hand:

    * `RAW_EXPORT_REQUIRED_COLUMNS` in src/pipeline.py
    * `CANONICAL_DATASET_REQUIRED_COLUMNS` in src/pipeline.py (which held
      `("barcode", "title", "attributes")` — a THIRD, different answer to
      "what does the canonical lane require", silently narrower than the raw
      requirement it is supposed to be the rename of)
    * the per-title evidence-capture field list in `_source_rows_for`

A hand-typed list is steered by nothing. It drifts from the mapping, it
disagrees with the validator that enforces it, and the disagreement is silent
until an artifact is written. This is the same shape as the volume-tolerance
defect: one concept, several declarations, no single source to steer them.

These tests pin the column SSOT:

(a) both vocabularies and both required-column contracts are DERIVED from
    `column_mapping` + the declared roles, so a rename in the config moves
    every lane at once;
(b) the mapping stays a bijection and every declared role names a real column
    — the three ways the SSOT can rot, checked;
(c) every evidence-capture field carries a non-empty reason, and the fields
    the owner ruled out (`url`, `image_url`, `price`) are provably absent —
    exclusions auditable, not re-argued by taste;
(d) the capture preserves PER-TITLE multiplicity (a canonical spans ~4.77
    source titles, and pick-one loses evidence);
(e) the canonical frame validator rejects an empty, unparseable, or
    undeclared-keyed capture, because the stage-7 clarification failing open
    is exactly the defect that motivated the column;
(f) stage 7 actually fires off the capture — it was wired and INERT.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from core import columns as colssot
from core.columns import (
    CANONICAL_COLUMNS,
    COLUMN_ALIASES,
    EXCLUDED_SOURCE_ROW_FIELDS,
    CANONICAL_DATASET_REQUIRED_COLUMNS,
    COLUMN_MAPPING,
    DATA_PREP_REQUIRED_COLUMNS,
    RAW_EXPORT_COLUMNS,
    SOURCE_ROW_FIELD_NAMES,
    alias_names,
    canonical_of,
    raw_of,
    read_column,
    require_canonical_columns,
    require_raw_columns,
)
from core.schemas import (
    CANONICAL_RECORDS_COLUMNS,
    check_canonical_records_frame,
    require_populated_source_rows,
    upgrade_canonical_records_frame,
)

# All 13 mapped columns are captured (owner ruling 2026-10-01, reversed on
# evidence). URL and image_url were previously ruled OUT as "listing
# identifiers with no product semantics" — a reason never checked against a
# value. Reading dataset.csv shows walmart's
# /ip/Concord-Foods-Smoothie-Banana-Drink-Mixes-2-oz-Shelf-Stable carries the
# product name verbatim, so they are captured. This list is the SPOT CHECK
# that the reversal is actually declared, not merely implied.
WAS_WRONGLY_EXCLUDED = ("url", "image_url", "price", "product_id", "barcode")
EXCLUDED = frozenset(EXCLUDED_SOURCE_ROW_FIELDS)


# ── (a) both vocabularies derived from one mapping ───────────────────────────


def test_both_vocabularies_come_from_the_mapping() -> None:
    assert RAW_EXPORT_COLUMNS == tuple(sorted(COLUMN_MAPPING))
    assert CANONICAL_COLUMNS == tuple(sorted(COLUMN_MAPPING.values()))


def test_canonical_requirement_is_the_renamed_raw_requirement() -> None:
    """The canonical lane requires the raw requirement, renamed.

    Not a hand-written answer: it must be EXACTLY the image of the raw list
    across the mapping. This is what the old `("barcode", "title",
    "attributes")` literal failed.
    """
    assert set(CANONICAL_DATASET_REQUIRED_COLUMNS) == {
        COLUMN_MAPPING[raw] for raw in DATA_PREP_REQUIRED_COLUMNS
    }
    # and it must not be narrower than the raw requirement it renames
    assert set(DATA_PREP_REQUIRED_COLUMNS) <= set(RAW_EXPORT_COLUMNS)
    assert {"barcode", "title", "attributes"} <= set(
        CANONICAL_DATASET_REQUIRED_COLUMNS
    )


def test_name_resolvers_round_trip() -> None:
    for raw, canonical in COLUMN_MAPPING.items():
        assert canonical_of(raw) == canonical
        assert raw_of(canonical) == raw
    assert canonical_of("not_a_column") is None
    assert raw_of("not_a_column") is None


def test_pipeline_consumes_the_ssot_not_a_literal() -> None:
    import pipeline

    assert tuple(pipeline.RAW_EXPORT_REQUIRED_COLUMNS) == DATA_PREP_REQUIRED_COLUMNS
    assert (
        tuple(pipeline.CANONICAL_DATASET_REQUIRED_COLUMNS)
        == CANONICAL_DATASET_REQUIRED_COLUMNS
    )


# ── (b) the mapping stays a bijection; roles name real columns ───────────────


def test_mapping_is_a_bijection() -> None:
    assert len(COLUMN_MAPPING) == len(set(COLUMN_MAPPING.values()))
    assert len(COLUMN_MAPPING) == 13


def test_every_raw_column_is_mapped() -> None:
    """No raw column may be unmapped: unmapped means dropped silently."""
    raw_export = {
        "attribute",
        "brand",
        "breadcrumbs_eng",
        "category",
        "country",
        "description_short_eng",
        "gtin",
        "image_url",
        "retailer",
        "sku_id",
        "sku_last_price",
        "sku_name_eng",
        "sku_url",
    }
    assert set(COLUMN_MAPPING) == raw_export


def test_source_row_fields_are_real_canonical_columns() -> None:
    canonical = set(COLUMN_MAPPING.values())
    for field in SOURCE_ROW_FIELD_NAMES:
        assert field in canonical, f"{field} is not a canonical column"


def test_required_columns_are_real_raw_columns() -> None:
    for raw in DATA_PREP_REQUIRED_COLUMNS:
        assert raw in COLUMN_MAPPING


def test_guards_name_the_missing_column() -> None:
    incomplete = pd.DataFrame(columns=[c for c in RAW_EXPORT_COLUMNS if c != "gtin"])
    with pytest.raises(ValueError, match="gtin"):
        require_raw_columns(incomplete)
    canonical_incomplete = pd.DataFrame(
        columns=[c for c in CANONICAL_COLUMNS if c != "title"]
    )
    with pytest.raises(ValueError, match="title"):
        require_canonical_columns(canonical_incomplete)


# ── (c) exclusions are auditable ─────────────────────────────────────────────


def test_every_column_states_a_reason_either_way() -> None:
    """Captured OR excluded: a column with no stated basis is a bug.

    A column merely absent from the config is indistinguishable from one
    somebody forgot, which is why the declaration covers all thirteen.
    """
    from core.common import data_cfg

    cfg = data_cfg()
    ruled = cfg.column_evidence
    assert set(ruled) == set(COLUMN_MAPPING.values()), (
        "column_evidence must be a total partition of the mapped columns"
    )
    for name, spec in ruled.items():
        assert spec.reason.strip(), f"{name} has no stated reason"


def test_every_mapped_column_is_captured() -> None:
    """The reversal is total: all 13 columns reach source_rows."""
    assert set(SOURCE_ROW_FIELD_NAMES) == set(CANONICAL_COLUMNS)
    assert not EXCLUDED


def test_the_previously_excluded_columns_state_why_they_were_restored() -> None:
    from core.common import data_cfg

    ruled = data_cfg().column_evidence
    for column in WAS_WRONGLY_EXCLUDED:
        assert ruled[column].capture is True, f"{column} must be captured"
        assert ruled[column].reason.strip(), f"{column} needs a stated basis"


def test_price_is_captured_but_not_claimed_as_attribute_evidence() -> None:
    """Honesty check: capture must not be read as "this is evidence".

    price is a bare decimal; it corroborates no attribute claim. It is
    captured for completeness of the row and its config reason says so, so
    nobody later mistakes presence in the record for evidentiary weight.
    """
    from core.common import data_cfg

    reason = data_cfg().column_evidence["price"].reason.lower()
    assert "not read as attribute evidence" in reason


def test_excluded_columns_are_absent_from_the_capture() -> None:
    for excluded in EXCLUDED:
        assert excluded not in SOURCE_ROW_FIELD_NAMES
        # present in the mapping (they are real columns) but deliberately
        # uncaptured, with the reason on record
        assert excluded in CANONICAL_COLUMNS
        assert EXCLUDED_SOURCE_ROW_FIELDS[excluded].strip()


def test_captured_and_excluded_partition_the_mapped_columns() -> None:
    assert set(SOURCE_ROW_FIELD_NAMES) | EXCLUDED == set(CANONICAL_COLUMNS)
    assert not (set(SOURCE_ROW_FIELD_NAMES) & EXCLUDED)


def test_capture_keeps_evidence_bearing_columns() -> None:
    for kept in ("title", "attributes", "description", "brand", "category_path"):
        assert kept in SOURCE_ROW_FIELD_NAMES


# ── (d)/(e) the capture preserves multiplicity and validates loudly ──────────


def _raw_row(**overrides: object) -> dict:
    row = {
        "sku_id": "1",
        "retailer": "shop",
        "country": "IT",
        "sku_name_eng": "Cola 330ml",
        "description_short_eng": "cola drink",
        "breadcrumbs_eng": "beverages>cola",
        "sku_last_price": "1.20",
        "gtin": "8000000000001",
        "brand": "Acme",
        "category": "cola",
        "attribute": "type sparkling water",
        "sku_url": "http://x/1",
        "image_url": "http://x/1.jpg",
    }
    row.update(overrides)
    return row


def test_capture_preserves_every_source_title_and_every_column() -> None:
    frame = pd.DataFrame(
        [
            _raw_row(sku_name_eng="Cola 330ml", attribute="type cola"),
            _raw_row(sku_name_eng="Cola Zero 330ml", attribute="type cola zero"),
        ]
    )
    capture = json.loads(pipeline_source_rows(frame))
    assert len(capture) == 2, "per-title multiplicity must survive"
    assert {entry["title"] for entry in capture} == {
        "Cola 330ml",
        "Cola Zero 330ml",
    }
    for entry in capture:
        assert set(entry) == set(SOURCE_ROW_FIELD_NAMES)


def test_capture_keys_are_canonical_never_raw() -> None:
    """A raw key would be unreadable downstream; the mapping owns the rename."""
    capture = json.loads(pipeline_source_rows(pd.DataFrame([_raw_row()])))[0]
    assert capture["title"] == "Cola 330ml"
    assert capture["attributes"] == "type sparkling water"
    assert capture["url"] == "http://x/1"
    assert capture["price"] == "1.20"
    # the raw names that differ from canonical must not appear as keys
    for raw_only in ("sku_name_eng", "attribute", "gtin", "sku_id", "sku_url",
                     "sku_last_price", "breadcrumbs_eng", "description_short_eng"):
        assert raw_only not in capture


def pipeline_source_rows(frame: pd.DataFrame) -> str:
    """The pipeline's own capture helper, so the test exercises the writer."""
    import pipeline

    return pipeline._source_rows_for(frame)


def _canonical_record(**overrides: object) -> dict:
    """A contract-complete canonical record.

    Every contract column is defaulted to a placeholder and only the ones the
    checker actually inspects are given real values. The fixture therefore
    needs no knowledge of which columns are lists, maps or scalars — a
    shape-aware fixture is a shape list that will drift.
    """
    row: dict[str, object] = {column: "[]" for column in CANONICAL_RECORDS_COLUMNS}
    row["universe_evidence"] = "{}"
    row["gtin"] = "8000000000001"
    row["canonical"] = "Cola 330ml"
    for column in ("volume_confidence", "pack_confidence",
                   "volume_consistency", "pack_consistency"):
        row[column] = 1.0
    row["n_titles"] = 1
    row["source_rows"] = '[{"title": "Cola 330ml", "attributes": "type cola"}]'
    row.update(overrides)
    # The contract is order-exact, so build in contract order rather than
    # trusting dict insertion order.
    return {column: row[column] for column in CANONICAL_RECORDS_COLUMNS}


@pytest.mark.parametrize("bad", ["", "   ", "[]", "{}", "not json"])
def test_write_time_check_rejects_an_empty_capture(bad: str) -> None:
    """Empty is legal on READ, forbidden on WRITE.

    check_* must tolerate "[]" because upgrade_canonical_records_frame emits
    exactly that for a pre-column artifact. The writer may not: a record built
    now always has a source title behind it, so an empty capture means the
    evidence was lost.
    """
    frame = pd.DataFrame([_canonical_record(source_rows=bad)])
    with pytest.raises(ValueError):
        require_populated_source_rows(frame)


def test_read_tolerates_an_empty_capture_because_the_adapter_emits_it() -> None:
    frame = pd.DataFrame([_canonical_record(source_rows="[]")])
    check_canonical_records_frame(frame)
    require_populated_source_rows(pd.DataFrame([_canonical_record()]))


def test_source_rows_is_part_of_the_canonical_contract() -> None:
    assert "source_rows" in CANONICAL_RECORDS_COLUMNS


def test_frame_validator_accepts_a_well_formed_capture() -> None:
    check_canonical_records_frame(pd.DataFrame([_canonical_record()]))


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "not json",
        '{"title": "x"}',
        '[{"title": "x", "not_a_column": "y"}]',
        "[null]",
        '["string entry"]',
    ],
)
def test_frame_validator_rejects_a_broken_capture(bad: str) -> None:
    """Failing LOUD is the point: stage 7 failing open is the defect."""
    with pytest.raises(ValueError):
        check_canonical_records_frame(pd.DataFrame([_canonical_record(source_rows=bad)]))


def test_upgrade_marks_absent_capture_empty_not_invented() -> None:
    """A pre-column artifact upgrades to an EMPTY capture, never a fake one.

    The legacy shape here is the REAL one: data/canonical_records.csv carries
    universe_evidence but not the ingredient set columns, which the previous
    exact-match shim could not load at all.
    """
    legacy = {c: "x" for c in CANONICAL_RECORDS_COLUMNS if c != "source_rows"}
    upgraded = upgrade_canonical_records_frame(pd.DataFrame([legacy]))
    assert list(upgraded.columns) == list(CANONICAL_RECORDS_COLUMNS)
    assert upgraded["source_rows"].tolist() == ["[]"]


def test_upgrade_preserves_columns_that_already_exist() -> None:
    """A missing column is filled; a present one is never overwritten."""
    row = _canonical_record()
    row.pop("source_rows")
    row.pop("sweetener_type_set")
    row["universe_evidence"] = '{"volume": "0.9"}'
    upgraded = upgrade_canonical_records_frame(pd.DataFrame([row]))
    assert list(upgraded.columns) == list(CANONICAL_RECORDS_COLUMNS)
    assert upgraded["universe_evidence"].tolist() == ['{"volume": "0.9"}']
    assert upgraded["sweetener_type_set"].tolist() == ["[]"]
    assert upgraded["source_rows"].tolist() == ["[]"]


def test_the_shipped_artifact_loads_through_the_upgrade() -> None:
    """data/canonical_records.csv must remain readable after adding a column.

    This is the failure the exact-match shim had: adding source_rows made the
    artifact on disk an unrecognised shape and the read blew up at load.
    """
    from core.common import DATA_DIR

    path = DATA_DIR / "canonical_records.csv"
    if not path.is_file():
        pytest.skip("canonical_records.csv not built in this environment")
    raw = pd.read_csv(path)
    upgraded = upgrade_canonical_records_frame(raw)
    assert list(upgraded.columns) == list(CANONICAL_RECORDS_COLUMNS)
    assert set(upgraded["source_rows"]) == {"[]"}


# ── (f) stage 7 actually fires off the capture ──────────────────────────────


def _canonical_with_titles(*titles: str) -> dict:
    return {
        "volume_set": {355.0},
        "flavor_set": {"cola"},
        "source_rows": json.dumps(
            [{"title": title, "attributes": ""} for title in titles]
        ),
    }


def _engine():
    from core.attribute_decision import AttributeDecisionEngine

    return AttributeDecisionEngine(
        volume_relative_tolerance=0.05, volume_absolute_tolerance_ml=5.0
    )


def test_stage7_resolves_from_the_capture() -> None:
    """The clarification was wired and INERT; this is it firing."""
    left = _canonical_with_titles("Cola Zero Sugar 330ml")
    right = _canonical_with_titles("Cola No Added Sugar 330ml")
    engine = _engine()
    evidence = engine.evaluate(left, right, left_raw=left, right_raw=right)
    decision = evidence.dimensions["sweetener"]
    assert decision.fallback_from == "original_columns", (
        "stage 7 did not read the capture — the columns it needs are present "
        "in source_rows, so an empty fallback_from means it never looked"
    )


def test_stage7_unions_across_every_captured_title() -> None:
    """A canonical spans ~4.77 titles; pick-one would drop the evidence."""
    left = _canonical_with_titles("Cola 330ml bottle", "Cola Zero Sugar 330ml bottle")
    right = _canonical_with_titles("Cola 330ml can", "Cola No Added Sugar 330ml can")
    engine = _engine()
    evidence = engine.evaluate(left, right, left_raw=left, right_raw=right)
    assert evidence.dimensions["sweetener"].fallback_from == "original_columns"


def test_stage7_stays_inert_and_quiet_on_an_empty_capture() -> None:
    """An upgraded artifact must not crash, and must not invent a verdict."""
    empty = {
        "volume_set": {355.0},
        "flavor_set": {"cola"},
        "source_rows": "[]",
    }
    engine = _engine()
    evidence = engine.evaluate(empty, empty, left_raw=empty, right_raw=empty)
    for decision in evidence.dimensions.values():
        assert decision.fallback_from == ""


def test_stage7_will_not_verdict_on_one_sided_evidence() -> None:
    """One side declaring is not a conflict; no claim, no verdict."""
    left = _canonical_with_titles("Cola Zero Sugar 330ml")
    right = _canonical_with_titles("Cola 330ml")
    engine = _engine()
    evidence = engine.evaluate(left, right, left_raw=left, right_raw=right)
    decision = evidence.dimensions["sweetener"]
    assert decision.fallback_from == ""
    assert decision.result.name == "INCONCLUSIVE"


def test_columns_module_does_not_restate_the_descriptor_split() -> None:
    """That concept belongs to core.product_identity; a second copy is the defect.

    This test exists to FAIL if anyone re-adds a descriptor list here.
    """
    from core.product_identity import DESCRIPTOR_COLUMNS

    assert not hasattr(colssot, "DESCRIPTOR_COLUMNS")
    assert not hasattr(colssot, "NON_DESCRIPTOR_COLUMNS")
    # the split itself is declared once, in its owning module
    assert DESCRIPTOR_COLUMNS


def test_aliases_are_unique_and_resolve_to_real_columns() -> None:
    seen: dict[str, str] = {}
    for column, aliases in COLUMN_ALIASES.items():
        assert column in CANONICAL_COLUMNS
        for alias in aliases:
            assert alias not in COLUMN_MAPPING.values(), alias
            assert alias not in seen, f"{alias} claimed twice"
            seen[alias] = column
    assert alias_names("attributes")[:2] == ("attributes", "attribute")
    assert "attr" in alias_names("attributes")


def test_read_column_prefers_canonical_then_raw_then_alias() -> None:
    assert read_column({"title": "A", "sku_name_eng": "B"}, "title") == "A"
    assert read_column({"sku_name_eng": "B"}, "title") == "B"
    assert read_column({"attr": "type cola"}, "attributes") == "type cola"
    assert read_column({}, "title", default="fallback") == "fallback"
    # a blank value must not shadow a later name that has content
    assert read_column({"title": "   ", "sku_name_eng": "B"}, "title") == "B"


def test_pipeline_uses_the_shared_alias_resolver() -> None:
    """No read path may re-declare an alias pair the config already owns."""
    import inspect

    from core import attribute_decision, model_input, record_linkage

    for module in (attribute_decision, model_input, record_linkage):
        source = inspect.getsource(module)
        for pair in ('"attributes", "attr"', '"description", "description_short_eng"',
                     '"title": "sku_name_eng"', '"attributes": "attribute"'):
            assert pair not in source, f"{module.__name__} re-declares {pair}"