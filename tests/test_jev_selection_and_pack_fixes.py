"""Selected SKU and retail-pack regressions from the saved JEV audit."""
import json
from pathlib import Path

import pytest

from core.product_selection import selected_product_title
from core.declared_identity import listing_identity
from pipeline import extract_all, extract_pack_evidence, extract_pack_from_title
from test_jev_identity_fixes import canonical
from pipeline import three_way_gate, generate_canonical, NgramIDF


@pytest.mark.parametrize('title,count', [
    ('6x light feeling lemon juicy mineral water 0.33 l', 6),
    ('24 x domestic mineral water 0.33l', 24),
    ('Syrup orange cola mix set of 6 ever 600 ml', 6),
    ('4 x 250ml (Pack of 2)', 8),
    ('4 Pack 4 x 250ml (Pack of 4)', 16),
    ('4 x 250ml (x2)', 8),
    ('Unit Count 12.00 Count', 12),
])
def test_explicit_retail_quantities_preserve_counts_and_source_spans(title, count):
    assert extract_pack_from_title(title)[0] == count
    assert all(title[e['start']:e['end']] == e['raw_match'] for e in extract_pack_evidence(title))


@pytest.mark.parametrize('title', [
    'Vitamin B6x daily supplement 50mg',
    '6x daily dose of vitamin 50mg',
    'Unit Count 202.80 Fl Oz',
    'Set of 6 flavors to choose from, 600ml bottle',
])
def test_doses_and_option_lists_do_not_invent_packs(title):
    assert extract_pack_from_title(title) == (1, 0.)


def test_description_only_pack_quantity_and_cross_surface_conflict():
    row = extract_all('Mineral water carbonated', 'Volume: 1500',
                      'Mineral water carbonated 4x1.5 ltr.')
    assert row['pack_qty'] == 4 and row['pack_confidence'] > 0
    row = extract_all('Mineral water 6 x 1500ml', 'Volume: 1500',
                      'Mineral water 4x1.5 ltr.')
    assert 'pack_sources_disagree' in row['attribute_consistency_flags']


def test_selected_option_overrules_title_and_attribute_choice_lists():
    prefix = 'Tea Concentrates: Your Choice of Sassafras Tea, Green Tea, Raspberry Tea or Peach Tea 12 oz. Bottles '
    a, b = prefix + '(Sassafras Tea, 2 Bottles)', prefix + '(Raspberry Tea, 2 Bottles)'
    assert selected_product_title(a)[1] == 'sassafras tea'
    attrs = 'Volume: 355; Flavour: peach, tea, raspberry'
    assert 'raspberry' not in extract_all(a, attrs)['flavor_set']
    assert 'raspberry' in extract_all(b, attrs)['flavor_set']
    assert three_way_gate(canonical(a, attrs), canonical(b, attrs))['decision'] != 'proceed'
    assert three_way_gate(canonical(b, attrs), canonical(b, attrs))['decision'] == 'proceed'


def test_size_suffix_is_not_a_selected_flavor():
    title = 'Your Choice of Lemon or Orange water (Pack of 6)'
    assert selected_product_title(title) == (title, '')


@pytest.mark.parametrize('a,b', [
    ('Fruit Punch sparkling water 500ml', 'Original sparkling water 500ml'),
    ('Root Beer soda 500ml', 'Original soda 500ml'),
    ('White grape juice 500ml', 'Black grape juice 500ml'),
    ('Apple juice Gala 500ml', 'Apple juice Braeburn 500ml'),
    ('Cool Blue energy drink 500ml', 'Glacier Freeze energy drink 500ml'),
    ('Protein Boost iced coffee 230ml', 'Double Espresso iced coffee 230ml'),
    ('Women Fit vitamin water 500ml', 'C Mix vitamin water 500ml'),
    ('Plain turnip juice 500ml', 'Spicy turnip juice 500ml'),
    ('Zero sugar cola 500ml', 'Zero sugar zero caffeine cola 500ml'),
    ('Organic apple juice 500ml', 'Apple juice 500ml'),
])
def test_declared_variant_distinctions_never_silently_approve(a, b):
    assert three_way_gate(canonical(a), canonical(b))['decision'] != 'proceed'
    assert three_way_gate(canonical(b), canonical(a))['decision'] != 'proceed'


def test_pure_juice_and_diluted_juice_are_reviewed():
    a = canonical('Apple juice 500ml', 'Flavour: apple; Juice Content: 100%')
    b = canonical('Apple juice in mineral water 500ml', 'Flavour: apple; Juice Content: 25-50%')
    assert three_way_gate(a, b)['decision'] == 'fallback'


def test_equal_flavor_sets_do_not_acquire_a_mode_contradiction():
    a = canonical('Organic Super Fruit 7 juice 1l', 'Flavour: cherry, grape, pomegranate')
    b = dict(a, mode_flavor='fruit')
    a['mode_flavor'] = 'cherry'
    assert three_way_gate(a, b)['decision'] == 'proceed'


@pytest.mark.parametrize('rows,flag', [
    ([('Apple juice 200ml', 'Volume: 200'), ('Apple juice 1l', 'Volume: 1000')], 'volume_sources_disagree'),
    ([('Water 6 x 500ml', 'Volume: 500'), ('Water 12 x 500ml', 'Volume: 500')], 'pack_sources_disagree'),
    ([('Water 500ml', 'Volume: 500; Pack Type: Bottle'), ('Water 500ml', 'Volume: 500; Pack Type: Pouch')], 'categorical_source_conflict:package_type'),
    ([('Water 500ml', 'Volume: 500; Pack Material Type: Plastic'), ('Water 500ml', 'Volume: 500; Pack Material Type: Glass')], 'categorical_source_conflict:pack_material'),
])
def test_conflicting_source_rows_are_reviewed_even_against_themselves(rows, flag):
    r = generate_canonical('123', 'Example', rows, NgramIDF({'123': rows}), None)
    assert flag in r['attribute_consistency_flags']
    assert three_way_gate(r, r)['decision'] == 'fallback'
    from core.attribute_conflicts import canonical_attribute_info
    from training.rand_matching import targeted_veto_gate
    info = canonical_attribute_info(r)
    assert targeted_veto_gate(info, info, sku_brand='Example', candidate_brand='Example', exact_gtin=False)['targeted_gate_route'] == 'human_review'


def test_volume_conversion_noise_is_not_a_source_conflict():
    rows = [('Vanilla water 12 fl oz', 'Volume: 355'), ('Vanilla water 355ml', 'Volume: 355')]
    r = generate_canonical('123', 'Example', rows, NgramIDF({'123': rows}), None)
    assert 'volume_sources_disagree' not in r['attribute_consistency_flags']
    assert three_way_gate(r, r)['decision'] == 'proceed'
