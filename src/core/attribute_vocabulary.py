"""Leaf validation shared by configuration loading and attribute extraction.

This module must stay independent of core.common and parser modules so direct
extractor imports receive the same fail-closed contract without import cycles.
"""


def validated_attribute_vocabulary(vocabulary: dict) -> dict:
    """Return extraction vocabulary, rejecting missing or unusable lexicons."""
    attributes = vocabulary.get("attribute_vocabulary")
    if not isinstance(attributes, dict) or not attributes:
        raise SystemExit("vocabulary.attribute_vocabulary must be a non-empty mapping")
    _ATTR_VOCAB_LIST_KEYS = (
        "flavor_lexicon", "declared_flavor_lexicon", "made_from_lexicon",
        "made_from_phrases", "caffeine_sources", "sugar_ingredients",
    )
    for key in _ATTR_VOCAB_LIST_KEYS:
        values = attributes.get(key)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise SystemExit(
                f"vocabulary.attribute_vocabulary.{key} must be a non-empty list of strings"
            )
    aliases = attributes.get("flavor_aliases")
    if not isinstance(aliases, dict) or not aliases or not all(
        isinstance(alias, str) and alias.strip()
        and isinstance(target, str) and target.strip()
        for alias, target in aliases.items()
    ):
        raise SystemExit(
            "vocabulary.attribute_vocabulary.flavor_aliases must be a non-empty "
            "string-to-string mapping"
        )
    targets = {target.strip().casefold() for target in aliases.values()}
    missing = sorted(targets - {value.strip().casefold() for value in attributes["flavor_lexicon"]})
    if missing:
        raise SystemExit(
            "vocabulary.attribute_vocabulary.flavor_aliases targets missing from "
            f"flavor_lexicon: {missing}"
        )
    return attributes
