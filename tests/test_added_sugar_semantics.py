import pytest

from core.critical_attributes import extract_critical_claims
from core.sweetener_values import negated_sweetener_types
from pipeline import extract_all


@pytest.mark.parametrize("phrase", ["No Sugar Added", "Zero Sugar Added", "0 sugar added",
                                     "without sugar added", "no sugars added", "No Added Sugar"])
def test_added_sugar_does_not_assert_total_sugar_absence(phrase):
    assert extract_critical_claims(phrase)["sweetener"] == {"no_added_sugar"}
    assert negated_sweetener_types(phrase) == set()
    actual = extract_all("Juice " + phrase, "Sweetener: fructose", "", "", "", "", "")
    assert actual["sweetener_set"] == {"no_added_sugar"}
    assert "sugar" not in actual["negated_sweetener_type_set"]


def test_independent_total_sugar_claim_and_ingredient_negation_survive():
    assert extract_critical_claims("No Sugar Added; Sugar Free")["sweetener"] == {"no_added_sugar", "no_sugar"}
    assert negated_sweetener_types("No Sugar Added; without sugar; no stevia") == {"sugar", "stevia"}
