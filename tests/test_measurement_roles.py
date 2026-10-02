"""Audited original source notation retains roles before text normalization."""

import pytest

from core.text import extract_volume_evidence, extract_volume_match
from core.sweetener_values import negated_sweetener_types, title_sweetener_types


@pytest.mark.parametrize('title,value,unit', [
    ('Cool Breeze Beverages Ready to Use Slush Mix, Hurricane, 1 / 2 gal', .5, 'gal'),  # SKU79028934
    ('Govinda\'s Hawaiian Super C Juice 1 / 2 gal.', .5, 'gal'),  # SKU85550443
    ('Drink 1 ,25l', 1.25, 'l'),
    ('Drink 0, 33l', .33, 'l'),
    ('Drink 24, 500ml', 500, 'ml'),
    ('Drink 24 500ml', 500, 'ml'),
    ('Drink 1 000 ml', 1000, 'ml'),
    ('Drink 23.7-ounce', 23.7, 'ounce'),
    ('24 Tweaker Extreme Energy Drinks 24 / 2oz', 2, 'oz'),  # SKU311411; count is not a fraction
    ("Reed's Ginger Beer Raspberry Ginger Brew (6X4 / 12 Oz)", 12, 'oz'),  # SKU71073123
])
def test_fraction_decimal_and_count_notation(title, value, unit):
    assert extract_volume_match(title)[:2] == (value, unit)


def test_nutrition_denominator_keeps_provenance_without_claiming_package_size():
    # SKU1052557924
    title = 'share Organic Splash Cucumber Mint - 1 kcal per 100 ml - Flavoured Water without Sugar (24 x 330 ml)'
    evidence = extract_volume_evidence(title)
    assert [(item['value'], item['role']) for item in evidence] == [(100, 'nutrition'), (330, 'package_volume')]
    assert extract_volume_match(title)[:2] == (330, 'ml')
    assert all(title[item['start']:item['end']] == item['raw_match'] for item in evidence)


def test_prepared_yield_and_dry_weight_stay_distinct():
    # SKU74185738
    title = 'Tang Orange Powdered Drink Mix (Makes 22 Quarts), 72-Ounce Canister (Pack of 2)'
    assert [(item['value'], item['role']) for item in extract_volume_evidence(title)] == [(22, 'yield'), (72, 'net_weight')]
    assert extract_volume_match(title)[0] is None
    assert extract_volume_match('Liquid drink 12 fl oz')[0] == 12
    assert extract_volume_match('Concentrate makes 10 litres; bottle 500ml')[:2] == (500, 'ml')


def test_negated_ingredient_evidence_is_explicit_and_not_positive():
    # SKU371312669 declares stevia in attributes but explicitly excludes it in title.
    title = 'Culture Pop Soda, Low Sugar, No Stevia (12pk)'
    assert negated_sweetener_types(title) == {'stevia'}
    assert title_sweetener_types('Tea without cane sugar') == set()
    assert negated_sweetener_types('Tea without cane sugar') == {'cane_sugar'}
    assert negated_sweetener_types('Tea') == set()
    assert title_sweetener_types('Tea made with stevia') == {'stevia'}


def test_oversized_glued_code_is_skipped_without_losing_later_size():
    assert extract_volume_match('BG14980 L Drink 500ml')[:2] == (500, 'ml')
