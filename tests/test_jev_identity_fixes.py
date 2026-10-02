"""Exact retail identity regressions from inspected round-7 source listings."""
import json
from pathlib import Path

import pytest

from core.declared_identity import listing_identity
from pipeline import extract_all, generate_canonical, NgramIDF, three_way_gate


def canonical(title, attributes='', description='', category='', category_path=''):
    rows = [(title, attributes)]
    return generate_canonical('1234567890123', 'Example', rows,
                              NgramIDF({'1234567890123': rows}), None,
                              descriptions=[description], categories=[category],
                              category_paths=[category_path])


def test_categories_do_not_invent_declared_flavors():
    lemon = extract_all('Lemon lemonade 750ml', 'Flavour: lemon',
                        category='Lemonade/Lime', category_path='fruit & berry drinks')
    assert lemon['flavor_set'] == {'lemon'}
    mate = extract_all('Fritz-mate 330ml', '', category='Other Non-Cola Carbonates')
    assert mate['flavor_set'] == set()
    assert not any(e['field'] == 'flavor' and e['column'] == 'category'
                   for e in mate['evidence_ledger'])


def test_syrup_declared_flavors_veto_even_when_other_fields_agree():
    a = canonical('Syrup pure cane Cassis 50cl', 'Pack Type: Bottle')
    b = canonical('Syrup pure cane anise 50cl', 'Pack Type: Bottle')
    assert a['flavor_set'] == {'blackcurrant'}
    assert b['flavor_set'] == {'anise'}
    assert three_way_gate(a, b)['decision'] == 'hard_no'


def test_juice_bits_and_smooth_are_explicit_texture_evidence():
    a = canonical('Pure pressed smooth orange juice 330ml', 'Flavour: orange')
    b = canonical('Pure pressed orange juice 330ml', 'Flavour: orange',
                  'Ingredients Orange Juice Not From Concentrate With Bits (100%).')
    assert a['pulp_set'] == {'no_pulp'}
    assert b['pulp_set'] == {'with_pulp'}
    assert three_way_gate(a, b)['decision'] == 'hard_no'
    assert extract_all('Smooth coffee 330ml', '')['pulp_set'] == set()
    assert extract_all('Snack with bits of chocolate', '')['pulp_set'] == set()


@pytest.mark.parametrize('title_a,title_b', [
    ('Sparkling water slight 6x500ml', 'Sparkling water strong 6x500ml'),
    ('Ayus arishta Musta 500ml', 'Ayus arishta Khadira 500ml'),
    ('Fritz-kola 330ml', 'Fritz-mate 330ml'),
    ('Lemon lime lemonade 750ml', 'Lemon lemonade 750ml'),
])
def test_declared_distinctions_require_review(title_a, title_b):
    a, b = canonical(title_a), canonical(title_b)
    assert three_way_gate(a, b)['decision'] != 'proceed'
    assert three_way_gate(b, a)['decision'] != 'proceed'


@pytest.mark.parametrize('a,b', [
    ('Ozarka 100% Natural Spring Water 3 Liter', 'Ozarka 100% Natural Spring Water 3 L'),
    ('Starbucks vanilla coffee 4 x 9.5 oz', 'Starbucks vanilla coffee 4 x 9.5 oz'),
    ('Bare Nature peach iced tea 12 x 20 oz', 'Bare Nature peach iced tea 12 x 20 oz'),
    ('Ayus arishta Musta 500ml', 'Ayus arishta Musta 500ml'),
])
def test_matching_identity_controls_remain_approved(a, b):
    assert three_way_gate(canonical(a, 'Volume: 3000' if 'Ozarka' in a else 'Volume: 281' if 'Starbucks' in a else 'Volume: 591' if 'Bare Nature' in a else 'Volume: 500'),
                          canonical(b, 'Volume: 3000' if 'Ozarka' in b else 'Volume: 281' if 'Starbucks' in b else 'Volume: 591' if 'Bare Nature' in b else 'Volume: 500'))['decision'] == 'proceed'


def test_missing_declared_identity_requires_review_without_inventing_negative():
    a, b = canonical('Orange juice with pulp 330ml'), canonical('Orange juice 330ml')
    assert three_way_gate(a, b)['decision'] == 'fallback'


def test_strength_is_not_inferred_from_unrelated_claims():
    assert 'carbonation_strength' not in listing_identity('Strong coffee 330ml')
    assert 'carbonation_strength' not in listing_identity('Mineral water 500ml', 'Health Claims: strong immunity')


def test_original_source_capture_reviews_stale_generic_fields():
    a, b = canonical('Water 500ml'), canonical('Water 500ml')
    a['source_rows'] = json.dumps([{'title': 'Sparkling water slight 500ml'}])
    b['source_rows'] = json.dumps([{'title': 'Sparkling water strong 500ml'}])
    assert three_way_gate(a, b)['decision'] == 'fallback'


def test_confident_veto_still_wins_over_declared_identity_review():
    a, b = canonical('Ayus arishta Musta orange 500ml'), canonical('Ayus arishta Khadira lemon 500ml')
    assert three_way_gate(a, b)['decision'] == 'hard_no'


def test_inference_gate_also_reviews_declared_variants_and_preserves_exact_gtin():
    from core.attribute_conflicts import sku_attribute_info, canonical_attribute_info
    from training.rand_matching import targeted_veto_gate
    left = sku_attribute_info('Sparkling water slight 6x500ml', 'Volume: 500; Count per Unit: 6')
    right = canonical_attribute_info(canonical('Sparkling water strong 6x500ml', 'Volume: 500; Count per Unit: 6'))
    decision = targeted_veto_gate(left, right, sku_brand='Example', candidate_brand='Example', exact_gtin=False)
    assert decision['targeted_gate_route'] == 'human_review'
    assert decision['targeted_gate_reason'] == 'declared_identity:carbonation_strength'
    assert targeted_veto_gate(left, right, sku_brand='Example', candidate_brand='Example', exact_gtin=True)['targeted_gate_route'] == 'auto_merge'
