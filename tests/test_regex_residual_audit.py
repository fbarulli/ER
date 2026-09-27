"""Focused checks for the diagnostic residual review."""

from scripts.regex_residual_audit import drop_nearby_repeats, residual
from scripts.regex_capture_review import NUMBER_VALUE_RE, attribute_number_captures, captures


def test_unique_tokens_keep_first_stem_across_fields() -> None:
    seen: set[str] = set()
    title, _ = residual("electrolytes water", field="title", round_name="unique_tokens", seen_tokens=seen)
    attributes, _ = residual(
        "electrolyte waters potassium", field="attributes", round_name="unique_tokens", seen_tokens=seen
    )
    assert title == "electrolytes water"
    assert attributes == "potassium"


def test_repeat_pass_preserves_text_without_a_repeat() -> None:
    assert drop_nearby_repeats("Brand-Water Drink", window=64, min_words=4) == (
        "Brand-Water Drink", 0, 0
    )


def test_numeric_attribute_capture_preserves_range_and_percent() -> None:
    raw = "Caffeine: 0-15 mg; Volume: 355; Juice Content: 0-2%"
    assert [(item["attribute_field"], item["matched_text"]) for item in attribute_number_captures(raw)] == [
        ("Caffeine", "0-15 mg"), ("Volume", "355"), ("Juice Content", "0-2%"),
    ]


def test_title_natural_claim_and_pack_numbers_are_captured() -> None:
    assert any(label == "natural_claim_lexical" and phrase == "100 natural"
               for _, _, label, phrase in captures("100 natural hydration", "title"))
    assert [match.group().strip() for match in NUMBER_VALUE_RE.finditer("100% Natural, Pack of 12x355ML")] == [
        "100%", "12x355ML",
    ]
