"""Title+attribute "Made From" extraction, config-owned vocabulary."""

from core.critical_attributes import (
    MADE_FROM_LEXICON,
    MADE_FROM_PHRASES,
    extract_made_from_tokens,
)


def test_attribute_field_only():
    assert extract_made_from_tokens("", "Made From: lemon, ginger") == frozenset(
        {"lemon", "ginger"}
    )


def test_title_contributes_ingredients():
    # A title that names an ingredient is captured even when the declared
    # `Made From:` field omits it — the measured gap this closes.
    assert extract_made_from_tokens("coconut water", "Made From: lemon") == frozenset(
        {"coconut", "lemon"}
    )


def test_multiword_phrase_matches():
    assert extract_made_from_tokens("passion fruit juice", "") == frozenset(
        {"passion fruit"}
    )


def test_unknown_words_never_invent_ingredients():
    assert extract_made_from_tokens("mystery elixir", "") == frozenset()


def test_vocabulary_is_config_owned_and_nonempty():
    # Loaded from config/vocabulary.json (not hardcoded in the module).
    assert len(MADE_FROM_LEXICON) >= 90
    assert len(MADE_FROM_PHRASES) >= 10
    assert "passion fruit" in MADE_FROM_PHRASES


def test_multiword_phrase_requires_complete_words():
    assert "passion fruit" not in extract_made_from_tokens("passion fruitful")
    assert "passion fruit" not in extract_made_from_tokens("compassion fruit")
    assert "passion fruit" in extract_made_from_tokens("(passion fruit) juice")


def test_review_preserves_declared_ingredients_outside_title_lexicon():
    from core.attribute_conflicts import _universe_value

    record = {
        "made_from_set": {"lemon"},
        "universe_evidence": {"made from": frozenset({"novel ingredient"})},
    }
    assert _universe_value(record, "made from", None) == frozenset(
        {"lemon", "novel ingredient"}
    )


def test_review_uses_declared_ingredients_for_legacy_records():
    from core.attribute_conflicts import _universe_value

    assert _universe_value(
        {"universe_evidence": {"made from": frozenset({"novel ingredient"})}},
        "made from", None,
    ) == frozenset({"novel ingredient"})
    assert _universe_value({}, "made from", None) == frozenset()


def test_shared_attribute_fields_preserves_declared_blank_fields():
    from core.text import attribute_fields, attribute_field_value

    cell = "Caffeine: ; Made From: lemon; malformed"
    assert attribute_fields(cell, include_empty=True) == [
        ("caffeine", ""), ("made from", "lemon")
    ]
    assert attribute_fields(cell) == [("made from", "lemon")]
    assert attribute_field_value("Caffeine: ", "caffeine") == []
