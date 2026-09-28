"""Flavor alias normalization (EXP-02): aliases mint, false pairs do not."""

from core.critical_attributes import (
    FLAVOR_ALIASES,
    FLAVOR_LEXICON,
    extract_flavor_tokens,
)


def test_new_aliases_map_and_mint():
    for variant, canonical in [
        ("apples", "apple"), ("fruits", "fruit"), ("fruity", "fruit"),
        ("tonica", "tonic"), ("lemoni", "lemon"),
        ("strawberr", "strawberry"), ("grapefru", "grapefruit"),
        ("chery", "cherry"), ("rhubar", "rhubarb"), ("fruite", "fruit"),
    ]:
        assert extract_flavor_tokens(variant) == frozenset({canonical})


def test_tamarindo_mints_tamarind():
    assert "tamarind" in FLAVOR_LEXICON
    assert extract_flavor_tokens("tamarindo") == frozenset({"tamarind"})
    assert extract_flavor_tokens("tamarind") == frozenset({"tamarind"})


def test_pearl_never_maps_to_pear():
    assert extract_flavor_tokens("pearl") == frozenset()


def test_alias_that_is_lexicon_word_is_noop():
    assert FLAVOR_ALIASES.get("apple", "apple") in FLAVOR_LEXICON
    assert extract_flavor_tokens("apple juice") == frozenset({"apple"})


def test_multiword_value_normalizes_via_alias():
    from core.attribute_conflicts import normalized_flavor_tokens

    assert normalized_flavor_tokens_supports_alias()


def normalized_flavor_tokens_supports_alias() -> bool:
    from core.attribute_conflicts import flavor_overlap_metrics

    jaccard, overlap = flavor_overlap_metrics(
        {"tamarindo"}, {"tamarind"}
    )
    return overlap == 1.0
