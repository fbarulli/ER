"""Regressions for independently reproduced quantity-role errors."""
import pytest

from core.text import extract_volume_evidence, extract_volume_match


@pytest.mark.parametrize("text,value", [("1 1/2 l", 1.5), ("3/2 l", 1.5),
                                        ("3 / 2 l", 1.5), ("2 1 / 4 l", 2.25)])
def test_fraction_volume_retains_whole_and_improper_parts(text, value):
    assert extract_volume_match(text)[:2] == (value, "l")


@pytest.mark.parametrize("text,roles,selected", [
    ("330 ml per serving; bottle 1 l", ["nutrition", "package_volume"], (1., "l")),
    ("contains 100 ml juice in 330 ml bottle", ["ingredient_volume", "package_volume"], (330., "ml")),
    ("1980 ml total; 6 × 330 ml", ["total_volume", "package_volume"], (330., "ml")),
    ("6 × 330 ml (Total1980ml)", ["package_volume", "total_volume"], (330., "ml")),
])
def test_roles_preserve_exact_spans_and_select_package(text, roles, selected):
    evidence = extract_volume_evidence(text)
    assert [item["role"] for item in evidence] == roles
    assert extract_volume_match(text)[:2] == selected
    assert all(text[item["start"]:item["end"]] == item["raw_match"] for item in evidence)


def test_total_only_is_not_per_unit_volume():
    assert extract_volume_match("Total 1980 ml")[0] is None
    assert extract_volume_match("Total of 72 Oz")[0] is None


@pytest.mark.parametrize("text,value,unit", [("LT.1.5 X 6BT", 1.5, "lt"),
                                            ("CL.17.5 x 24 Pieces", 17.5, "cl")])
def test_explicit_dotted_unit_prefix_keeps_source_measurement(text, value, unit):
    assert extract_volume_match(text)[:2] == (value, unit)
    entry, = extract_volume_evidence(text)
    assert text[entry["start"]:entry["end"]] == entry["raw_match"]


def test_fraction_repair_preserves_count_size_and_zero_denominator_controls():
    assert extract_volume_match("24 / 2oz")[:2] == (2., "oz")
    assert extract_volume_match("6X4 / 12 Oz")[:2] == (12., "oz")
    assert extract_volume_match("1/0 l; bottle 330 ml")[:2] == (330., "ml")
