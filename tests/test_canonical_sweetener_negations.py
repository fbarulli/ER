"""Ingredient polarity must survive GTIN aggregation across listings."""
import pytest
from pipeline import NgramIDF, generate_canonical, three_way_gate

@pytest.mark.parametrize('negative_column', ['sku_name_eng','description_short_eng'])
def test_cross_listing_ingredient_negation_is_review(negative_column):
    rows=[('Vanilla water 330 ml no sucralose' if negative_column == 'sku_name_eng' else 'Vanilla water 330 ml',''),
          ('Vanilla water 330 ml', 'Sweetener: sucralose')]
    descriptions=['No sucralose' if negative_column == 'description_short_eng' else '', '']
    record=generate_canonical('1234567890123','Example',rows,
                              NgramIDF({'1234567890123':rows}),None,
                              descriptions=descriptions)
    assert 'sucralose' in record['sweetener_type_set']
    assert 'sweetener_source_conflict:sucralose' in record['attribute_consistency_flags']
    assert three_way_gate(record,record)['decision'] == 'fallback'


def test_negative_only_ingredient_does_not_invent_conflict():
    rows=[('Vanilla water 330 ml no sucralose','')]
    record=generate_canonical('1234567890123','Example',rows,
                              NgramIDF({'1234567890123':rows}),None)
    assert 'sucralose' not in record['sweetener_type_set']
    assert 'sweetener_source_conflict:sucralose' not in record['attribute_consistency_flags']
