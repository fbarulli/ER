"""The streamed candidate/audit assembly must equal the in-memory reference.

The lane used to build one candidate frame for the whole population and one
audit frame per fold (21.3 GB peak on 8,657 SKUs, OOM-killed). It now streams
candidate parts and appends audit rows chunk by chunk; this pins that the
streamed path produces the same predictions and the same diagnostics CSV bytes
as the whole-population pass it replaced.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from core.workspace import Workspace
from training import matching_parts
from training.matching_parts import CandidatePartStore, DiagnosticsCsvWriter
from training.rand_matching import (
    _DIAGNOSTICS_COLUMNS_SPEC,
    AUDIT_ANNOTATION_COLUMNS,
    _assignments_with_trace,
    _audit_trace,
    _candidate_counts,
    _run_streamed_assignment,
    candidate_gate_fields,
)

#: Columns the audit phase appends itself (never part of a stored candidate row).
_AUDIT_ADDED = {
    *AUDIT_ANNOTATION_COLUMNS,
    "ITEM_ID",
    "n_candidates",
    "source_file",
    "evaluation_partition",
    "evaluation_fold",
    "evaluation_threshold",
    "predicted_ITEM_ID",
    "true_candidate_retrieved",
    "true_candidate_accepted",
    "prediction_correct",
    "error_type",
}


def _candidate_frame() -> pd.DataFrame:
    """Four SKUs x two candidates over the full diagnostics column contract."""
    base_columns = [
        column
        for column in _DIAGNOSTICS_COLUMNS_SPEC.expected_columns
        if column not in _AUDIT_ADDED
    ]
    rows: list[dict[str, object]] = []
    for sku, true_item in (
        ("sku-a", "gtin-1"),
        ("sku-b", "gtin-2"),
        ("sku-c", "gtin-3"),
        ("sku-d", "gtin-4"),
    ):
        for rank, (candidate, score) in enumerate(
            ((true_item, 0.95), (f"gtin-x-{sku}", 0.90)), start=1
        ):
            row = {column: "" for column in base_columns}
            row.update(
                {
                    "SKU_ID": sku,
                    "candidate_gtin": candidate,
                    "score": score,
                    "gtin_status": "different",
                    "exact_gtin": 0,
                    "rule_ok": 1,
                    "brand_conflict": 0,
                    "targeted_gate_route": "auto_merge",
                    "attribute_matches": 3,
                    "candidate_rank": rank,
                    "retrieval_source": "hnsw_semantic_top_k",
                    "true_item_id": true_item,
                    "calibration_fold": "0",
                    "sku_title": f"title {sku}",
                    "candidate_text": f"canonical {candidate}",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=base_columns)


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_streamed_audit_matches_the_in_memory_reference(tmp_path, chunk_size):
    frame = _candidate_frame()
    truth = frame[["SKU_ID", "true_item_id"]].drop_duplicates("SKU_ID")
    threshold = 0.80
    partition = "fit"
    fold = "0"

    predictions, trace = _assignments_with_trace(frame, threshold)
    reference = _audit_trace(
        trace,
        predictions,
        truth,
        _candidate_counts(frame),
        partition=partition,
        fold=fold,
        threshold=threshold,
    )

    store = CandidatePartStore(tmp_path / "parts")
    for index, start in enumerate(range(0, len(frame), chunk_size * 2)):
        store.write(index, frame.iloc[start : start + chunk_size * 2])
    writer = DiagnosticsCsvWriter(
        store.path("diagnostics.csv"),
        _DIAGNOSTICS_COLUMNS_SPEC.expected_columns,
    )
    streamed_predictions = _run_streamed_assignment(
        store,
        threshold,
        threshold_by_gtin_status=None,
        writer=writer,
        partition=partition,
        fold=fold,
        truth=truth,
        label="test",
    )
    streamed_path = writer.parts().write_csv(tmp_path / "streamed.csv")
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    assert streamed_predictions.equals(predictions)
    assert streamed_path.read_bytes() == reference_path.read_bytes()
    assert streamed_path.read_text().splitlines()[0].split(",") == list(
        reference.columns
    )
    store.close()


def test_build_reclaims_a_killed_predecessors_scratch_slot(tmp_path, monkeypatch):
    """A SIGKILLed run leaves its slot behind; the next build must reclaim it.

    The sweep branch is on the first call of every run after a killed run, so a
    defect there takes the whole lane down before a single SKU is scored.
    """
    monkeypatch.setattr(
        matching_parts.Workspace,
        "from_config",
        classmethod(lambda cls: Workspace(tmp_path)),
    )
    stale = tmp_path / f"rand_matching_parts_{os.getpid()}"
    stale.mkdir()
    (stale / "part-00000.parquet").write_bytes(b"stale predecessor")

    with CandidatePartStore.build("rand_matching_parts") as store:
        assert store.root == stale
        assert store.parts == ()
    assert not stale.exists()


def test_diagnostics_contract_declares_every_column_the_gate_emits(tmp_path):
    """The report contract must describe the row the producer actually emits.

    The gate grew ``targeted_attribute_policy``/``targeted_vetoed_conflicts``
    after the diagnostics spec was written, so the lane's own final
    ``validate_frame`` raised on its own output — the last step of the graded
    run could never complete.
    """
    info = {
        "pack": {6},
        "volume": {355},
        "package_type": {"can"},
        "flavor": "orange",
        "flavor_set": {"orange"},
        "carbonation": set(),
        "carbonation_set": set(),
        "sweetener": set(),
        "sweetener_set": set(),
        "pulp": set(),
        "pulp_set": set(),
    }
    row = pd.Series(
        {
            "SKU_ID": "sku-a",
            "gtin": "4006381333931",
            "sku_name_eng": "Acme soda 6 pack 355 ml cans",
            "brand": "Acme",
        }
    )
    candidate = candidate_gate_fields(
        row,
        info,
        "4006381333931",
        {
            "canonical": "Acme soda 6 pack 355 ml cans",
            "mode_brand": "Acme",
            "mode_flavor": "orange",
            "volume_set": "[355]",
            "pack_set": "[6]",
            "package_type_set": "['can']",
        },
        0.95,
        sku_id="sku-a",
        source_row_index="0",
    )
    # Per-candidate provenance the scorer adds to every row (rand_matching._candidate_row).
    candidate["true_item_id"] = ""
    candidate["calibration_fold"] = ""
    frame = pd.DataFrame([candidate])
    predictions, trace = _assignments_with_trace(frame, 0.80)
    diagnostics = _audit_trace(
        trace,
        predictions,
        None,
        _candidate_counts(frame),
        partition="final",
        fold=None,
        threshold=0.80,
    )
    _DIAGNOSTICS_COLUMNS_SPEC.validate_frame(diagnostics, "diagnostics")
