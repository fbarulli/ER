"""Frozen real listings with independent source-evidence correction rulings."""
import json
from pathlib import Path

import pytest

from pipeline import extract_all, three_way_gate
from training.gate_replay import fired_stage

FIXTURE = json.loads((Path(__file__).parent / "fixtures/gate_regex_regressions.json").read_text())


@pytest.mark.parametrize("sample", FIXTURE["listings"], ids=lambda sample: sample["product_id"])
def test_corrected_source_listing(sample):
    row, expected = sample["source_row"], sample["expected_corrected"]
    actual = extract_all(row["title"], row["attributes"], row["description"], row["url"],
                         row["image_url"], row["category_path"], row["category"])
    if "volume_ml" in expected:
        assert actual["volume_ml"] == pytest.approx(expected["volume_ml"], abs=expected["volume_absolute_tolerance_ml"])
    if "pack_qty" in expected:
        assert actual["pack_qty"] == expected["pack_qty"]
    assert set(expected.get("required_flags", [])).issubset(actual["attribute_consistency_flags"])
    if "volume_confidence_below" in expected:
        assert actual["volume_confidence"] < expected["volume_confidence_below"]
    if "measurement_roles" in expected:
        assert set(expected["measurement_roles"]).issubset({entry["role"] for entry in actual["measurement_evidence"]})
    if "negated_sweetener_type_includes" in expected:
        assert set(expected["negated_sweetener_type_includes"]).issubset(actual["negated_sweetener_type_set"])
    for entry in expected.get("required_pack_evidence", []):
        assert any(all(claim.get(key) == value for key, value in entry.items())
                   for claim in actual["pack_evidence"])


def test_unknown_pack_is_reviewed_without_fabricating_single_unit():
    sample = FIXTURE["pairs"][0]
    records = []
    for side in ("baseline_left", "baseline_right"):
        record = dict(sample[side])
        for key in record:
            if key.endswith("_set") or key.endswith("_flags"):
                record[key] = set(record[key])
        records.append(record)
    actual = three_way_gate(*records)
    assert actual["decision"] == sample["expected_corrected"]["decision"]
    assert fired_stage(actual["reason"]) == sample["expected_corrected"]["reason_stage"]
