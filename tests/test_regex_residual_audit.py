"""Focused checks for the diagnostic residual review."""

from scripts.regex_residual_audit import drop_nearby_repeats, residual
from scripts.regex_capture_review import (
    NUMBER_VALUE_RE,
    attribute_number_captures,
    captures,
    declared_flavor_evidence,
    model_payload_review,
    semantic_profile,
    sweetener_type_evidence,
)
from core.critical_attributes import extract_critical_claims
from core.attribute_conflicts import sku_attribute_info
from scripts.regex_miss_review import flavor_suggestion
from scripts.regex_miss_evidence import capture_class, description_support, source_span


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


def test_attribute_pack_type_requires_declared_field() -> None:
    text = "sustainable packaging can be recycled pack material type carton pack type bottle"
    package_hits = [phrase for _, _, label, phrase in captures(text, "attributes")
                    if label == "package_type_lexical"]
    assert package_hits == ["bottle"]


def test_negative_sugar_variants_share_one_trusted_value() -> None:
    for phrase in ("sugar free", "Sugar-Free", "0 sugar", "0g sugar", "no dugar"):
        assert extract_critical_claims(phrase)["sweetener"] == frozenset({"no_sugar"})
        assert any("claim" in label.split("+") for _, _, label, _ in captures(phrase.lower().replace("-", " "), "title"))
    assert extract_critical_claims("0 sugar added")["sweetener"] == frozenset({"no_added_sugar"})
    assert extract_critical_claims("Made with Sugar")["sweetener"] == frozenset({"sugar"})
    for phrase in ("Real Sugar", "pure sugar syrup"):
        assert extract_critical_claims(phrase)["sweetener"] == frozenset({"sugar"})
    # Low/reduced phrasing is NOT a positive sugar claim: it has dedicated
    # sweetening states, and mapping it to `sugar` manufactures a false
    # positive-vs-diet conflict on genuinely low-sugar products.
    for phrase in ("low in sugar", "reduced in sugar", "reduced in calories and sugar"):
        assert extract_critical_claims(phrase)["sweetener"] == frozenset()
    assert extract_critical_claims("Made with Sugar Free Sweeteners")["sweetener"] == frozenset({"no_sugar"})


def test_soda_pop_and_abbreviated_pulp_are_resolved_from_explicit_product_wording() -> None:
    # Bare "soda" is ambiguous (still juices and syrups carry the word), so
    # only the unambiguous "soda pop" is a carbonation claim.
    assert extract_critical_claims("Lemon Lime Soda Pop")["carbonation"] == frozenset({"carbonated"})
    assert extract_critical_claims("Bubble Up Lemon Lime Soda")["carbonation"] == frozenset()
    assert extract_critical_claims("Snow Cone Syrup Shaved Ice soda")["carbonation"] == frozenset()
    assert extract_critical_claims("Ingredients include baking soda")["carbonation"] == frozenset()
    assert extract_critical_claims("Coconut juice w/pulp")["pulp"] == frozenset({"with_pulp"})
    assert extract_critical_claims("juice based on concentrates and pulps")["pulp"] == frozenset({"with_pulp"})


def test_semantic_profile_does_not_promote_raw_numbers_to_swap_values() -> None:
    numeric = attribute_number_captures("Caffeine: 0-15 mg; Juice Content: 0-2%; Volume: 355")
    title = [{"capture_type": "natural_claim_lexical", "matched_text": "100 natural",
              "start": 0, "end": 11}]
    profile = semantic_profile("100% Natural Maple Water 355ml", "Caffeine: 0-15 mg; Juice Content: 0-2%; Volume: 355", numeric, title)
    assert profile["schema_version"] == "er.attribute_evidence.v1"
    assert profile["trusted_structured_values"]["volume"] == [355.0]
    assert profile["numeric_observations"][0]["numbers"] == [0.0, 15.0]
    assert profile["numeric_observations"][1]["unit"] == "%"
    assert all(not item["swap_eligible"] for item in profile["numeric_observations"])
    assert profile["lexical_only_claims"][0]["status"] == "lexical_only"


def test_fuzzy_flavor_suggestions_exclude_nearby_but_different_words() -> None:
    assert flavor_suggestion("toffee") == ""
    assert flavor_suggestion("pea") == ""
    assert flavor_suggestion("strawbery") == "strawberry"


def test_sweetener_type_is_separate_from_claim_and_not_yet_swap_ready() -> None:
    attributes = "Sweetener: cane sugar, stevia; Health Claims: no sugar"
    evidence = sweetener_type_evidence(attributes)
    assert [item["canonical_value"] for item in evidence] == ["cane_sugar", "stevia"]
    assert [attributes[s:e] for s, e in (item["raw_span"] for item in evidence)] == [
        "cane sugar", "stevia",
    ]
    profile = semantic_profile("Drink", attributes, [], [])
    assert profile["trusted_structured_values"]["sweetener"] == ["no_sugar"]
    assert profile["candidate_typed_values"]["sweetener_type"] == []
    assert profile["consistency_flags"] == ["no_sugar_claim_conflicts_with_sugar_ingredient"]
    assert "sweetener" not in profile["swap_compatible_fields"]
    assert all(not item["swap_eligible"] for item in profile["typed_evidence"])
    assert sweetener_type_evidence("Sweetener: unsweetened")[0]["canonical_value"] == "unsweetened"
    assert profile["trusted_structured_values"]["sweetener_type"] == ["cane_sugar", "stevia"]


def test_declared_flavor_keeps_out_of_lexicon_values_with_source_spans() -> None:
    attributes = "Flavour: lime, maple; Water Type: maple"
    evidence = declared_flavor_evidence(attributes)
    assert [item["canonical_value"] for item in evidence] == ["lime", "maple"]
    assert [attributes[s:e] for s, e in (item["raw_span"] for item in evidence)] == [
        "lime", "maple",
    ]
    profile = semantic_profile("Maple Water", attributes, [], [])
    assert profile["candidate_typed_values"]["declared_flavor"] == []
    assert profile["trusted_structured_values"]["flavor"] == ["lime", "maple"]


def test_reviewed_flavors_require_an_explicit_flavor_field() -> None:
    parsed = sku_attribute_info(
        "Blueberry drink",
        "Flavour: blueberry, banana; Sweetener: caramel",
    )
    assert parsed["flavor_set"] == {"blueberry", "banana"}
    assert sku_attribute_info("Blueberry drink", "")["flavor_set"] == set()
    assert sku_attribute_info("Caramel latte", "Sweetener: caramel")["flavor_set"] == set()
    assert sku_attribute_info("Tea latte", "Flavour: tea, latte")["flavor_set"] == {"tea", "latte"}
    assert sku_attribute_info("Tea latte", "")["flavor_set"] == set()
    assert semantic_profile("Drink", "Flavor: guava", [], [])["trusted_structured_values"]["flavor"] == ["guava"]
    assert sku_attribute_info("Orange and carrot juice", "Flavour: orange, carrot")["flavor_set"] == {"orange", "carrot"}
    assert sku_attribute_info("Honey tea", "Sweetener: honey")["flavor_set"] == set()
    assert sku_attribute_info("Bubble Gum Drink", "Flavour: bubble gum, sea salt")["flavor_set"] == {"bubble gum", "sea salt"}


def test_remaining_misses_keep_original_spans_and_distinct_meanings() -> None:
    title = {"product_id": "1", "source": "title", "title": "Water 12 tube 1000 mg / l",
             "candidate": "12 l"}
    column, _, start, end, surface, relation = source_span(title)
    assert (column, surface, relation) == ("title", "12 tube 1000 mg / l", "synthetic_residual_phrase")
    assert title["title"][start:end] == surface
    declared = {"product_id": "2", "source": "attributes", "candidate": "sweetener: cane sugar, stevia",
                "attributes": "Volume: 500; Sweetener: cane sugar, stevia; Pack Type: Can"}
    column, field, start, end, surface, relation = source_span(declared)
    assert (column, field, surface, relation) == ("attributes", "Sweetener", "cane sugar, stevia", "exact_candidate")
    assert declared["attributes"][start:end] == surface
    assert capture_class({"reason": "ingredient_outside_claim_classes", "candidate": "sweetener: unsweetened, sugar"}) == (
        "unsweetened_declaration", ["unsweetened", "sugar"], "separate_unsweetened_claim"
    )


def test_description_support_distinguishes_claim_from_product_style() -> None:
    assert description_support("carbonation", "Bubble milk tea with boba")[0] == "context_only"
    assert description_support("carbonation", "Carbonated water")[0] == "explicit_attribute_cue"
    assert description_support("sweetener", "Made with cane sugar")[0] == "explicit_attribute_cue"
    assert [item["label"] for item in description_support("sweetener", "Made with sugar-free sweeteners")[1]] == ["no_sugar"]
    assert description_support("sweetener", "No artificial sweeteners")[0] == "explicit_attribute_cue"
    assert description_support("pulp", "Pulp Press orange juice")[0] == "context_only"


def test_payload_review_keeps_actual_model_text_separate_from_audit_dedup() -> None:
    review = model_payload_review({
        "brand": "Maple 3",
        "title": "Maple 3 100% Natural Water",
        "attributes": "Juice Content: 0-2%; Caffeine: 0-15 mg; Flavour: maple",
    })
    assert review["composition"]["profile"] == "cleaned"
    assert "pct100" in review["payload"]
    assert "pct0to2" in review["payload"]
    assert review["repeated_plain_tokens"]["maple"] > 1
    assert review["audit_unique_description"].split().count("maple") == 1
