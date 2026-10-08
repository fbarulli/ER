"""Corpus deltas must enforce source alignment and separate additive fields."""
import gzip
import json

import pytest

from scripts.measure_gate_regex_fixes import compare


def write_snapshot(path, records, *, digest="same-source"):
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"raw_dataset_size": digest}) + "\n")
        for record in records:
            stream.write(json.dumps(record) + "\n")


def test_delta_reports_assignments_provenance_and_resolved_errors(tmp_path):
    before, after, report = (tmp_path / name for name in ("before.gz", "after.gz", "delta.json"))
    write_snapshot(before, [{"row_index": 0, "sku_id": "001", "extraction": {"volume_ml": 100, "pack_qty": 1}},
                            {"row_index": 1, "sku_id": "002", "error": {"type": "ValueError", "message": "bad"}}])
    write_snapshot(after, [{"row_index": 0, "sku_id": "001", "extraction": {"volume_ml": 330, "pack_qty": 1, "measurement_evidence": []}},
                           {"row_index": 1, "sku_id": "002", "extraction": {"volume_ml": 0, "pack_qty": 1}}])
    result = compare(before, after, report)
    assert result["rows_checked"] == result["changed_rows"] == 2
    assert result["existing_extraction_field_changed_rows"] == 1
    assert result["volume_or_pack_assignment_changed_rows"] == 2
    assert result["error_transitions"] == {"baseline_error_resolved": 1}
    assert result["newly_added_field_counts"]["measurement_evidence"] == 1
    with gzip.open(report.with_suffix(".changed.jsonl.gz"), "rt") as stream:
        changed = [json.loads(line) for line in stream]
    assert [row["sku_id"] for row in changed] == ["001", "002"]


@pytest.mark.parametrize("mismatch", ["hash", "row", "length"])
def test_delta_rejects_different_sources_or_alignment(tmp_path, mismatch):
    before, after = tmp_path / "before.gz", tmp_path / "after.gz"
    record = {"row_index": 0, "sku_id": "001", "extraction": {}}
    write_snapshot(before, [record])
    changed = dict(record, sku_id="different") if mismatch == "row" else record
    write_snapshot(after, [changed, record] if mismatch == "length" else [changed],
                   digest="other" if mismatch == "hash" else "same-source")
    with pytest.raises(ValueError):
        compare(before, after, tmp_path / "report.json")
