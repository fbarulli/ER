"""Source contradictions review their affected dimension; trusted others veto."""
import pytest
from pipeline import generate_canonical,NgramIDF,three_way_gate


def canonical(rows,descriptions=None):
    return generate_canonical('1234567890123','Example',rows,NgramIDF({'1234567890123':rows}),None,descriptions=descriptions)

@pytest.mark.parametrize('claim,description,dimension',[('no sugar','Contains sugar','sweetener'),
                                                       ('with pulp','No pulp','pulp')])
def test_description_conflict_is_review(claim,description,dimension):
    record=canonical([(f'Vanilla water {claim} 330ml','')],[description])
    assert f'description_conflict:{dimension}' in record['attribute_consistency_flags']
    assert three_way_gate(record,record)['decision']=='fallback'
    other=dict(record,flavor_set={'orange'},attribute_consistency_flags=set())
    assert three_way_gate(record,other)['decision']=='hard_no'

@pytest.mark.parametrize('left,right,dimension',[('no sugar','contains sugar','sweetener'),
                                               ('still','carbonated','carbonation'),
                                               ('with pulp','no pulp','pulp')])
def test_cross_listing_critical_contradiction_is_review(left,right,dimension):
    record=canonical([(f'Vanilla water {left} 330ml',''),(f'Vanilla water {right} 330ml','')])
    assert f'categorical_source_conflict:{dimension}' in record['attribute_consistency_flags']
    assert three_way_gate(record,record)['decision']=='fallback'


def test_uncertain_sugar_claim_does_not_veto_other_sugar_value():
    record=canonical([('Vanilla water no sugar 330ml','')],['Contains sugar'])
    other=canonical([('Vanilla water contains sugar 330ml','')])
    assert three_way_gate(record,other)['decision']=='fallback'


@pytest.mark.parametrize('flag',['unsweetened_with_declared_sweetener','sweetening_status_conflict'])
def test_explicit_sweetening_contradiction_is_review(flag):
    record=canonical([('Vanilla water 330ml','Sweetener: unsweetened, cane sugar')])
    record['attribute_consistency_flags']={flag}
    assert three_way_gate(record,record)['decision']=='fallback'


def test_no_added_sugar_with_cane_sugar_is_roasted_to_review():
    """FRUISS 935465903 / RISE 49918733: one listing declares both cane sugar
    (ingredient) and no added sugar (claim). The contradiction flag must be
    emitted and route the listing's sweetener dimension to review (fallback),
    never hard_no, while a clean sibling listing stays a plain proceed."""
    from pipeline import extract_all
    extracted = extract_all('Fruiss lemon syrup with cane sugar 50cl',
                            'Sweetener: cane sugar; Health Claims: no added sugar',
                            '', '', '', '', '')
    assert 'no_added_sugar_with_cane_sugar' in extracted['attribute_consistency_flags']
    record = canonical([('Fruiss lemon syrup with cane sugar 50cl',
                         'Sweetener: cane sugar; Health Claims: no added sugar')])
    result = three_way_gate(record, record)
    assert result['decision'] == 'fallback'
    assert result['reason'] == 'Contradictory source attribute evidence'
    clean = canonical([('Fruiss lemon syrup 50cl', '')])
    assert three_way_gate(clean, clean)['decision'] == 'proceed'
