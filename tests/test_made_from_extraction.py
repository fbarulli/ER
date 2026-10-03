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
