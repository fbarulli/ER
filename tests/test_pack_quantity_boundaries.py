"""Preserve whole count boundaries and prove omitted multipliers by totals."""
import pytest
from pipeline import extract_pack_evidence, extract_pack_from_title

@pytest.mark.parametrize('title,count', [
    ('19 Pallets of 84 cases each = 1.596 cases, 38.304 bottles',38304),
    ('19 Pallets of 84 cases each = 1,596 cases, 38,304 bottles',38304),
    ('2016 Bottles - $0.05 Bottle Bill State',2016),
    ('2 x 12 bottles x 330 ml',24),
    ('2 × 12 cans × 330 ml',24),
    ('2 x 12 x 330 ml',24),
    ('6 330 ml. ( Total 1980 ml.)',6),
    ('3 200 ml. ( Total 600 ml.)',3),
    ('6 0.33 l (Total 1980 ml)',6),
    ('Pack of12,',12),('12 Fl Oz,12 Pack',12),('6 Pack of12oz Cans',6),('3 x2x100g',6),('6x20 organic cl',6),('24 x8 fl ounce',24),('Combo Pack - 1) rose water & 2) orange water Total 2 Bottles',2),('(Pack -12)',12),('6 x pack 1.5l',6),('Pack -6',6),('LT.1.5 X 6BT',6),('Three Pack 16oz Bottles',3),('12 Oz Glass Bottle 2 Of Each Total of 72 Oz',6),('12 boxes x6 sticks per box',72),('Pack12',12),('pack of 6',6),('6 x 330 ml',6),('12 bottles',12),
])
def test_proven_unit_quantities(title,count):
    quantity,confidence=extract_pack_from_title(title)
    assert quantity == count
    assert confidence > 0
    for entry in extract_pack_evidence(title):
        assert title[entry['start']:entry['end']] == entry['raw_match']

@pytest.mark.parametrize('title', [
    '0 ml bottle; total 600ml','Pack -12 Fl. Oz.','$0.05 Bottle','$ 5 Bottle','0.5 bottle','1.5 bottles','0,5 bottles',
    '6 330 ml','6 330 ml (Total 1000 ml)','84 cases','sku123bottles',
])
def test_unproven_or_fractional_counts_stay_unknown(title):
    assert extract_pack_from_title(title) == (1,0.)


def test_outer_count_and_unit_count_remain_distinct():
    evidence=extract_pack_evidence('84 cases each = 1.596 cases, 38.304 bottles')
    assert {(e['count'],e['role']) for e in evidence} == {(84,'outer_count'),(1596,'outer_count'),(38304,'unit_count')}



def test_generic_pack_inner_hierarchy_retains_context_and_reviews():
    from pipeline import extract_all, NgramIDF, generate_canonical,three_way_gate
    title='Vanilla water 330ml 6 Sticks per Box (Pack-12)'
    evidence=extract_pack_evidence(title)
    assert {(e['count'],e['role']) for e in evidence}=={(6,'inner_count'),(12,'outer_count'),(72,'derived_inner_total')}
    result=extract_all(title,'')
    assert result['pack_qty']==12 and 'pack_hierarchy_ambiguous' in result['attribute_consistency_flags']
    rows=[(title,'')]
    record=generate_canonical('1234567890123','Example',rows,NgramIDF({'1234567890123':rows}),None)
    assert three_way_gate(record,record)['decision']=='fallback'
