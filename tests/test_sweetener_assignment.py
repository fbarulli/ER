"""Ingredient declarations must survive extraction, serialization, and both encoder lanes."""

import pandas as pd
import pytest

from core.attribute_conflicts import canonical_attribute_info
from core.model_input import build_canonical_text, build_sku_text
from core.schemas import CANONICAL_RECORDS_COLUMNS, TrainingSpec, upgrade_canonical_records_frame
from core.structured_features import canonical_info, sku_info
from core.sweetener_values import declared_sweeteners
from pipeline import NgramIDF, extract_all, extract_volume_from_title, generate_canonical
from scripts.regex_miss_review import title_signals
from training.masking import field_of


def test_sweetener_values_reach_both_model_lanes_after_csv_roundtrip(tmp_path):
    title = "Example tea 330ml"
    attributes = "Sweetener: stevia, erythritol; Flavour: latte, tea"
    rows = [(title, attributes)]
    record = generate_canonical("1234567890123", "Example", rows,
                                NgramIDF({"1234567890123": rows}), None)
    for key in ("sweetener_type_set", "sweetening_set", "attribute_consistency_flags"):
        record[key] = sorted(record[key])
    path = tmp_path / "canonical.csv"
    pd.DataFrame([record]).to_csv(path, index=False)
    loaded = pd.read_csv(path, keep_default_na=False).iloc[0]
    source_info = sku_info(title, attributes)
    target_info = canonical_info(loaded)
    assert source_info["sweetener_type"] == target_info["sweetener_type"] == {"stevia", "erythritol"}
    assert source_info["sweetener"] == set()  # Ingredients do not assert no-sugar.
    assert canonical_attribute_info(loaded)["sweetener_type"] == {"stevia", "erythritol"}
    spec = TrainingSpec.ModelInputSpec(profile="cleaned", include_evidence=False)
    source_text = build_sku_text(pd.Series({"title": title, "attributes": attributes, "brand": "Example"}), source_info, spec=spec)
    target_text = build_canonical_text(loaded, target_info, spec=spec)
    for token in ("sweetener_type_stevia", "sweetener_type_erythritol", "flavor_latte", "flavor_tea"):
        assert token in source_text.split()
        assert token in target_text.split()
    assert field_of("sweetener_type_stevia") == "sweetener_type"
    assert field_of("sweetener_diet_no_sugar") == "sweetener"


def test_unsweetened_unknown_and_contradictory_declarations_stay_distinct():
    assert declared_sweeteners("Sweetener: unsweetened")["sweetening"] == {"unsweetened"}
    parsed = extract_all("Tea", "Sweetener: unsweetened, cane sugar")
    assert parsed["sweetening_set"] == {"unsweetened"}
    assert parsed["sweetener_type_set"] == {"cane_sugar"}
    assert parsed["attribute_consistency_flags"] == {"unsweetened_with_declared_sweetener"}
    assert declared_sweeteners("Sweetener: stevia, mystery")["unmapped"] == {"mystery"}
    assert declared_sweeteners("Health Claims: stevia")["sweetener_type"] == set()


def test_previous_canonical_schema_reads_as_unknown_without_guessing():
    added = {"sweetener_type_set", "sweetening_set", "attribute_consistency_flags"}
    previous = [key for key in CANONICAL_RECORDS_COLUMNS if key not in added]
    old = pd.DataFrame([{key: "" for key in previous}], columns=previous)
    upgraded = upgrade_canonical_records_frame(old)
    assert tuple(upgraded.columns) == CANONICAL_RECORDS_COLUMNS
    assert all(upgraded.iloc[0][key] == "[]" for key in added)
    assert tuple(old.columns) == tuple(previous)
    malformed = old.drop(columns="gtin")
    assert upgrade_canonical_records_frame(malformed) is malformed


def test_audit_never_invents_a_volume_from_deleted_text_or_decimal_pack():
    assert ("volume", "12 l") not in title_signals("12 tube, 1000 mg / l")
    assert not any(dimension == "pack" for dimension, _ in title_signals("water pack 0.5 l"))


@pytest.mark.parametrize("title, expected", [("Drink 87.5 Millilitre", 88), ("Drink 25 centilitres", 250), ("Drink 1 000 ml", 1000), ("Drink 24, 500ml", 500), ("Drink 24 500ml", 500)])
def test_observed_volume_notation_gaps(title, expected):
    # Canonical volume normalization rounds to integer ml, half up.
    assert extract_volume_from_title(title)["volume_ml"] == expected
