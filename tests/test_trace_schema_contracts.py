"""tests/test_trace_schema_contracts.py — the consolidated-trace row contract
(core.schemas.TraceRow / check_trace_frame) and REACHABILITY of every frame
contract core.schemas declares.

Two jobs, both evidence-first:

  1. TRACE CONTRACT. ``core.tracing.TRACE_COLUMNS`` is the single declaration
     of the trace's columns; ``core.schemas`` imports it (never copies it) and
     publishes the row model + the frame checker. These tests prove, by
     executing pydantic, that:
       - what the real producers emit (``tracing.record`` / ``TraceRun`` /
         ``add_entities`` including its ``_truncated`` marker) validates;
       - what the doctrine forbids does NOT: a negative count, a
         ``dropped_count`` that contradicts in/out, an unknown scope, an extra
         column, an empty stage, a bad ``at``, a non-text ``detail``.
     Both shapes of frame a trace actually takes are exercised: the in-memory
     frame (None / NaN counts) and the CSV read-back frame (``read_trace``
     uses dtype=str + keep_default_na=False, so an absent count is ``""``).

  2. FRAME CONTRACT REACHABILITY. Every ``check_*_frame`` in core.schemas is
     defined to sit at a CSV boundary; a defined-but-unreachable checker is a
     gap. Each is driven here with one valid frame (must pass) and one
     corrupted frame (must raise), so no checker can rot unnoticed, and
     ``FRAME_CHECKERS`` is proven to name them all.

Nothing here writes artifacts or runs the pipeline: frames are built in
memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import core.tracing as tracing
from core.schemas import (
    CANONICAL_RECORDS_COLUMNS,
    CROSS_COUNTRY_PAIR_COLUMNS,
    EVAL_SUMMARY_COLUMNS,
    FRAME_CHECKERS,
    GATE_RESULTS_COLUMNS,
    LABELED_PAIRS_COLUMNS,
    TRACE_FRAME_COLUMNS,
    ZERO_SHOT_TRACE_COLUMNS,
    PairArrays,
    TraceRow,
    check_canonical_records_frame,
    check_cross_country_pair_frame,
    check_eval_summary_frame,
    check_gate_results_frame,
    check_labeled_pairs_frame,
    check_trace_frame,
    check_verdict_map,
    check_zero_shot_similarity_frame,
)

VALID_GTIN = "4006381333931"  # GS1 checksum valid
OTHER_GTIN = "4006381333932"


# ── trace row fixture ──────────────────────────────────────────────────────


def trace_row(**over: object) -> dict[str, object]:
    """One row that satisfies every clause of the contract."""
    row: dict[str, object] = {
        "stage": "data_prep.gtin_guard",
        "step": "identity_claims_evaluated",
        "scope": "run",
        "key": "",
        "in_count": 100,
        "out_count": 90,
        "dropped_count": 10,
        "reason": "rows keep identity only with a present, GS1-valid barcode",
        "detail": '{"rows_retained": 90}',
        "source": "raw export",
        "producer": "core.tracing",
        "at": "2026-09-15T08:58:06.480410+00:00",
    }
    row.update(over)
    return row


def trace_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(TRACE_FRAME_COLUMNS))


# ══════════════════════════════════════════════════════════════════════════
# 1. THE TRACE ROW CONTRACT
# ══════════════════════════════════════════════════════════════════════════


def test_trace_frame_columns_are_the_tracing_contract_not_a_copy():
    """The row contract has ONE declaration: core.tracing's own tuple.

    A second literal tuple in schemas.py would be exactly the "second
    declaration" this tree removed; the import is proven, not assumed.
    """
    assert TRACE_FRAME_COLUMNS is tracing.TRACE_COLUMNS
    assert tuple(TraceRow.model_fields) == tuple(tracing.TRACE_COLUMNS)
    assert TraceRow.model_config["extra"] == "forbid"


def test_valid_run_row_passes():
    row = TraceRow.model_validate(trace_row())
    assert (row.scope, row.in_count, row.out_count, row.dropped_count) == (
        "run",
        100,
        90,
        10,
    )


def test_valid_group_row_passes():
    row = TraceRow.model_validate(
        trace_row(scope="group", key="", in_count=7, out_count=3, dropped_count=4)
    )
    assert row.scope == "group"
    assert row.dropped_count == 4


def test_entity_row_without_counts_passes():
    """Entity scope is per-object: it has no funnel, so no counts at all."""
    row = TraceRow.model_validate(
        trace_row(
            scope="entity",
            key=VALID_GTIN,
            in_count=None,
            out_count=None,
            dropped_count=None,
        )
    )
    assert (row.in_count, row.out_count, row.dropped_count) == (None, None, None)


@pytest.mark.parametrize("field", ["in_count", "out_count", "dropped_count"])
def test_negative_count_fails(field: str):
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        TraceRow.model_validate(trace_row(**{field: -1}))


def test_dropped_count_that_contradicts_in_out_fails():
    """The arithmetic (in - out) is declared ONCE, in the validator."""
    with pytest.raises(
        ValidationError,
        match=r"trace dropped_count 4 contradicts in_count - out_count \(100 - 90 = 10\)",
    ):
        TraceRow.model_validate(trace_row(dropped_count=4))


def test_dropped_count_without_both_ends_fails():
    """dropped_count is DERIVED — it may not stand alone."""
    with pytest.raises(ValidationError, match="derived from in_count/out_count"):
        TraceRow.model_validate(
            trace_row(in_count=None, out_count=None, dropped_count=3)
        )


def test_lost_derivation_with_both_ends_present_fails():
    """in/out stated but dropped_count dropped is a producer bug, not None."""
    with pytest.raises(ValidationError, match="but no dropped_count"):
        TraceRow.model_validate(trace_row(dropped_count=None))


def test_out_count_exceeding_in_count_fails():
    with pytest.raises(ValidationError, match="exceeds in_count"):
        TraceRow.model_validate(trace_row(out_count=120, dropped_count=0))


def test_unknown_scope_fails():
    with pytest.raises(
        ValidationError, match="Input should be 'run', 'entity' or 'group'"
    ):
        TraceRow.model_validate(trace_row(scope="batch"))


def test_extra_column_fails():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TraceRow.model_validate(trace_row(stage_extra="nope"))


def test_missing_column_fails():
    row = trace_row()
    del row["producer"]
    with pytest.raises(ValidationError, match="Field required"):
        TraceRow.model_validate(row)


def test_empty_stage_fails():
    with pytest.raises(ValidationError, match="at least 1 character"):
        TraceRow.model_validate(trace_row(stage=""))


def test_whitespace_only_step_fails():
    with pytest.raises(ValidationError, match="trace step must be non-empty"):
        TraceRow.model_validate(trace_row(step="   "))


@pytest.mark.parametrize(
    "bad_at",
    [
        "yesterday",
        "2026-13-45T99:00:00+00:00",
        "2026-09-15T08:58:06",  # naive: the trace is stamped UTC
        "",
    ],
)
def test_bad_at_fails(bad_at: str):
    with pytest.raises(ValidationError, match="trace at="):
        TraceRow.model_validate(trace_row(at=bad_at))


def test_detail_must_be_json_text_not_a_mapping():
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        TraceRow.model_validate(trace_row(detail={"rows": 90}))


def test_detail_may_be_empty_text():
    """An entity row with no readback is legal; a dict is not."""
    assert TraceRow.model_validate(trace_row(detail="")).detail == ""


def test_boolean_count_is_not_an_integer():
    """bool is an int subclass — the CSV cell must not smuggle True as 1."""
    with pytest.raises(ValidationError, match="got bool True"):
        TraceRow.model_validate(trace_row(in_count=True, dropped_count=9))


@pytest.mark.parametrize("cell", ["lots", "9.5", "1e3"])
def test_non_numeric_and_non_integral_counts_fail(cell: str):
    with pytest.raises(ValidationError):
        TraceRow.model_validate(trace_row(in_count=cell, dropped_count=10))


def test_blank_count_cell_reads_as_none():
    """""/NaN are the ONLY non-int spellings that mean "no count"."""
    row = TraceRow.model_validate(
        trace_row(scope="entity", in_count="", out_count=np.nan, dropped_count=None)
    )
    assert (row.in_count, row.out_count, row.dropped_count) == (None, None, None)


# ── the real producers must satisfy the contract they publish ──────────────


def test_rows_emitted_by_the_real_producer_validate():
    """core.tracing.record / TraceRun / add_entities are the live writers."""
    run_row = tracing.record(
        "data_prep.gtin_guard",
        "identity_claims_evaluated",
        scope="run",
        in_count=71_623,
        out_count=71_000,
        reason="checksum",
        detail={"rows_retained": 71_000},
        source="raw export",
    )
    group_row = tracing.record(
        "pairs.negatives",
        "gate_hard_no_band",
        scope="group",
        in_count=10,
        out_count=4,
    )
    entity_row = tracing.record(
        "gate.pairs", "scored", scope="entity", key=VALID_GTIN
    )
    for row in (run_row, group_row, entity_row):
        TraceRow.model_validate(row)

    stage = tracing.TraceRun("stage")
    stage.add("step", "substep", in_count=5, out_count=5)
    stage.add_entities("entities", list(range(4)), limit=1)  # emits _truncated
    rows = stage.rows().to_dict("records")
    assert rows[-1]["step"].endswith("_truncated")
    assert check_trace_frame(stage.rows()) is not None


def test_trace_run_frame_passes_the_checker():
    stage = tracing.TraceRun("data_prep")
    stage.add("gtin_guard", "identity_claims_evaluated", in_count=9, out_count=8)
    stage.add("gtin_guard", "checksum", scope="group", in_count=8, out_count=6)
    frame = stage.rows()
    assert check_trace_frame(frame) is frame


# ── the frame boundary ─────────────────────────────────────────────────────


def test_check_trace_frame_accepts_a_valid_frame():
    frame = trace_frame(
        trace_row(),
        trace_row(
            scope="entity",
            key=OTHER_GTIN,
            in_count=None,
            out_count=None,
            dropped_count=None,
        ),
    )
    assert check_trace_frame(frame) is frame


def test_check_trace_frame_accepts_an_empty_typed_frame():
    """read_trace returns an empty-but-typed frame when nothing was written."""
    assert check_trace_frame(pd.DataFrame(columns=list(TRACE_FRAME_COLUMNS))).empty


def test_check_trace_frame_accepts_the_csv_read_back_shape():
    """read_trace(dtype=str, keep_default_na=False) turns None into ""."""
    frame = trace_frame(trace_row(), trace_row(scope="entity", in_count=None,
                                                out_count=None, dropped_count=None))
    csv_like = frame.fillna("").astype(str)
    assert csv_like.loc[1, "in_count"] == ""
    check_trace_frame(csv_like)


def test_check_trace_frame_rejects_a_corrupt_count_cell():
    frame = trace_frame(trace_row(), trace_row(scope="entity", in_count=None,
                                               out_count=None, dropped_count=None))
    csv_like = frame.fillna("").astype(str)
    csv_like.loc[1, "in_count"] = "lots"
    with pytest.raises(ValueError, match=r"trace frame row 1 violates TraceRow"):
        check_trace_frame(csv_like)


def test_check_trace_frame_rejects_an_extra_column():
    frame = trace_frame(trace_row()).assign(payload_idx=0)
    with pytest.raises(ValueError, match=r"trace frame columns .* != contract"):
        check_trace_frame(frame)


def test_check_trace_frame_rejects_reordered_columns():
    """Column ORDER is contractual: the CSV is a documented artifact."""
    frame = trace_frame(trace_row())[list(reversed(TRACE_FRAME_COLUMNS))]
    with pytest.raises(ValueError, match=r"trace frame columns .* != contract"):
        check_trace_frame(frame)


def test_check_trace_frame_rejects_a_contradicting_dropped_count_by_row_index():
    frame = trace_frame(trace_row(), trace_row(step="second"))
    frame.loc[1, "dropped_count"] = 7
    with pytest.raises(
        ValueError,
        match=r"(?s)trace frame row 1 violates TraceRow.*contradicts in_count - out_count",
    ):
        check_trace_frame(frame)


def test_check_trace_frame_rejects_an_unknown_scope_by_row_index():
    frame = trace_frame(trace_row(), trace_row(scope="entity", in_count=None,
                                               out_count=None, dropped_count=None))
    frame.loc[1, "scope"] = "batch"
    with pytest.raises(ValueError, match=r"trace frame row 1 violates TraceRow"):
        check_trace_frame(frame)


def test_trace_checker_is_discoverable_by_name():
    """The tracing owner wires assert_trace_frame through this lookup."""
    assert FRAME_CHECKERS["trace"] is check_trace_frame
    assert set(FRAME_CHECKERS) == {
        "canonical_records",
        "gate_results",
        "labeled_pairs",
        "cross_country_pairs",
        "zero_shot_similarity",
        "eval_summary",
        "trace",
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. REACHABILITY OF EVERY FRAME CONTRACT
# ══════════════════════════════════════════════════════════════════════════


def canonical_frame(**over: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "gtin": VALID_GTIN,
        "canonical": "brand orange water",
        "mode_brand": "brand",
        "mode_flavor": "orange",
        "mode_type": "water",
        "salient_ngrams": "['orange']",
        "dropped_redundant_ngrams": "[]",
        "volume_set": "[500.0]",
        "pack_set": "[1]",
        "package_type_set": "[]",
        "package_material_set": "[]",
        "flavor_set": "['orange']",
        "carbonation_set": "['sparkling']",
        "sweetener_set": "[]",
        "pulp_set": "[]",
        "volume_confidence": 0.9,
        "pack_confidence": 0.9,
        "volume_consistency": 1.0,
        "pack_consistency": 1.0,
        "n_titles": 2,
        "description_evidence": "[]",
        "breadcrumb_evidence": "[]",
    }
    row.update(over)
    return pd.DataFrame([row], columns=list(CANONICAL_RECORDS_COLUMNS))


def test_check_canonical_records_frame_is_reachable():
    assert check_canonical_records_frame(canonical_frame()).shape == (1, 22)


@pytest.mark.parametrize(
    "corruption",
    [
        {"n_titles": 0},
        {"volume_confidence": 1.5},
        {"canonical": "   "},
    ],
)
def test_check_canonical_records_frame_rejects_corruption(corruption):
    with pytest.raises(ValueError):
        check_canonical_records_frame(canonical_frame(**corruption))


def test_check_canonical_records_frame_rejects_duplicate_gtins():
    frame = pd.concat([canonical_frame(), canonical_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate GTINs"):
        check_canonical_records_frame(frame)


def gate_frame(**over: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "gtin1": VALID_GTIN,
        "gtin2": OTHER_GTIN,
        "canon1": "a",
        "canon2": "b",
        "gate_decision": "proceed",
        "gate_reason": "volume overlap",
        "similarity": 0.81,
    }
    row.update(over)
    return pd.DataFrame([row], columns=list(GATE_RESULTS_COLUMNS))


def test_check_gate_results_frame_is_reachable():
    assert check_gate_results_frame(gate_frame()).shape == (1, 7)


@pytest.mark.parametrize(
    "corruption",
    [
        {"gate_decision": "maybe"},
        {"similarity": 1.4},
        {"gtin2": VALID_GTIN},
        {"gate_reason": ""},
    ],
)
def test_check_gate_results_frame_rejects_corruption(corruption):
    with pytest.raises(ValueError):
        check_gate_results_frame(gate_frame(**corruption))


def labeled_frame(**over: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "gtin1": VALID_GTIN,
        "gtin2": OTHER_GTIN,
        "true_label": 1,
    }
    row.update(over)
    return pd.DataFrame([row], columns=list(LABELED_PAIRS_COLUMNS))


def test_check_labeled_pairs_frame_is_reachable():
    assert check_labeled_pairs_frame(labeled_frame()).shape == (1, 3)


@pytest.mark.parametrize(
    "corruption",
    [{"true_label": 2}, {"true_label": "yes"}, {"gtin1": ""}],
)
def test_check_labeled_pairs_frame_rejects_corruption(corruption):
    with pytest.raises(ValueError):
        check_labeled_pairs_frame(labeled_frame(**corruption))


def test_check_labeled_pairs_frame_rejects_duplicate_pairs():
    frame = pd.concat([labeled_frame(), labeled_frame()], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        check_labeled_pairs_frame(frame)


def cross_country_frame(**over: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "sku_id_a": "sku-a",
        "sku_id_b": "sku-b",
        "cross_country": True,
        "gtin": VALID_GTIN,
        "country_a": "DE",
        "country_b": "FR",
    }
    row.update(over)
    return pd.DataFrame([row], columns=list(CROSS_COUNTRY_PAIR_COLUMNS))


def test_check_cross_country_pair_frame_is_reachable():
    assert check_cross_country_pair_frame(cross_country_frame()).shape == (1, 6)


@pytest.mark.parametrize(
    "corruption",
    [
        {"gtin": "4006381333932"},  # bad check digit
        {"cross_country": False},
        {"country_b": "DE"},
        {"sku_id_b": "sku-a"},
    ],
)
def test_check_cross_country_pair_frame_rejects_corruption(corruption):
    with pytest.raises(ValueError):
        check_cross_country_pair_frame(cross_country_frame(**corruption))


def zero_shot_frame(**over: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "gtin1": VALID_GTIN,
        "gtin2": OTHER_GTIN,
        "gate_decision": "proceed",
        "gate_reason": "ok",
        "canonical1": "a",
        "canonical2": "b",
        "canonical_model_text1": "a",
        "canonical_model_text2": "b",
        "model_input_text1": "a",
        "model_input_text2": "b",
        "source_row_ids1": "[]",
        "source_row_ids2": "[]",
        "source_sku_ids1": "[]",
        "source_sku_ids2": "[]",
        "source_metadata1": "{}",
        "source_metadata2": "{}",
        "canonical_metadata1": "{}",
        "canonical_metadata2": "{}",
        "mask_status1": "none",
        "mask_status2": "none",
        "mask_applied1": False,
        "mask_applied2": False,
        "mask_realized_extent1": 0.0,
        "mask_realized_extent2": 0.0,
        "mask_config_fingerprint": "a" * 64,
        "model_keys": "mini",
        "lineage_id": "0123456789abcdef",
        "sim_mini": 0.75,
    }
    row.update(over)
    return pd.DataFrame([row], columns=[*ZERO_SHOT_TRACE_COLUMNS, "sim_mini"])


def test_check_zero_shot_similarity_frame_is_reachable():
    frame = zero_shot_frame()
    assert check_zero_shot_similarity_frame(frame) is frame


@pytest.mark.parametrize(
    "corruption",
    [
        {"mask_realized_extent1": 2.0},
        {"canonical1": ""},
        {"lineage_id": "short"},
        {"sim_mini": "high"},
    ],
)
def test_check_zero_shot_similarity_frame_rejects_corruption(corruption):
    with pytest.raises(ValueError):
        check_zero_shot_similarity_frame(zero_shot_frame(**corruption))


def test_check_zero_shot_similarity_frame_requires_a_similarity_column():
    frame = zero_shot_frame().drop(columns=["sim_mini"])
    with pytest.raises(ValueError, match="no similarity columns"):
        check_zero_shot_similarity_frame(frame)


def eval_summary_frame(**over: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "model": "mini",
        "eval_half": "test",
        "threshold_source": "dev_youden",
        "youden_thr_dev": 0.5,
        "pr_auc": 0.8,
        "precision_at_1": 0.9,
        "recall_at_1": 0.9,
        "precision_at_5": 0.9,
        "recall_at_5": 0.9,
        "precision_at_10": 0.9,
        "recall_at_10": 0.9,
        "hits_at_1": 0.9,
        "roc_auc": 0.9,
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1": 0.9,
        "tp": 5,
        "tn": 3,
        "fp": 1,
        "fn": 1,
        "n_dev": 10,
        "n_test": 10,
        "youden_thr_test_descriptive": 0.5,
    }
    row.update(over)
    return pd.DataFrame([row], columns=list(EVAL_SUMMARY_COLUMNS))


def test_check_eval_summary_frame_is_reachable():
    frame = eval_summary_frame()
    assert check_eval_summary_frame(frame) is frame


@pytest.mark.parametrize(
    "corruption",
    [
        {"tp": 99},  # confusion counts stop matching n_test
        {"eval_half": "dev"},  # the leak the contract exists to stop
        {"threshold_source": "test_youden"},
        {"pr_auc": 1.5},
    ],
)
def test_check_eval_summary_frame_rejects_corruption(corruption):
    with pytest.raises(ValueError):
        check_eval_summary_frame(eval_summary_frame(**corruption))


def test_check_verdict_map_is_reachable():
    assert check_verdict_map({"b12": "keep_nutrient", "473": "strip"}) == {
        "b12": "keep_nutrient",
        "473": "strip",
    }
    with pytest.raises(ValueError):
        check_verdict_map({"x": "keep"})


def test_every_registered_frame_checker_is_reachable():
    """FRAME_CHECKERS is the discovery surface; none of its entries may rot."""
    frames = {
        "canonical_records": canonical_frame(),
        "gate_results": gate_frame(),
        "labeled_pairs": labeled_frame(),
        "cross_country_pairs": cross_country_frame(),
        "zero_shot_similarity": zero_shot_frame(),
        "eval_summary": eval_summary_frame(),
        "trace": trace_frame(trace_row()),
    }
    assert set(frames) == set(FRAME_CHECKERS)
    for name, frame in frames.items():
        assert FRAME_CHECKERS[name](frame) is frame, name


# ── PairArrays: the model itself is exercisable (see audit note) ───────────


def test_pair_arrays_contract_still_holds_when_constructed():
    """PairArrays has no producer call site in the tree (audit gap), so this
    test is the only thing keeping its contract honest until one lands."""
    arrays = PairArrays(
        pos=np.array([[0, 1]]), neg=np.empty((0, 2), dtype=int), n_payload=2
    )
    assert arrays.pos.tolist() == [[0, 1]]
    with pytest.raises(ValidationError, match="out of range"):
        PairArrays(
            pos=np.array([[0, 9]]), neg=np.empty((0, 2), dtype=int), n_payload=2
        )
    with pytest.raises(ValidationError, match=r"must be \(N,2\)"):
        PairArrays(pos=np.array([0, 1]), neg=np.empty((0, 2), dtype=int), n_payload=2)
