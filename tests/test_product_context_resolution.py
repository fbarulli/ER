import pandas as pd
import pytest
from core.product_dimensions import row_dimensions
from core.product_context import compare_context
from core.product_identity import row_identity
from core.product_identity import evaluate_product_identity
from core.gtin import barcode_validity, normalize_and_validate_gtin
from core.identity_policy import exclude_reviewed_rows, review_mask


def row(**kwargs):
    return {'product_id': 'x', 'title': 'Energy 16 fl oz can', 'description': '',
            'attributes': '', 'barcode': '', **kwargs}


def test_same_explicit_quantity_resolves_different_raw_caffeine_ranges():
    a = row_dimensions(row(description='Contains 200 mg of caffeine per 16 oz can.', attributes='Caffeine: 100-150 mg'))
    b = row_dimensions(row(description='200mg of Caffeine per 16 oz can.', attributes='Caffeine: 25-50 mg'))
    c = compare_context(a.context, b.context)['caffeine']
    assert c['status'] == 'equal'
    assert c['left'][0] == pytest.approx(42.2833, rel=1e-5)
    assert not c['raw_ranges_compared']
    assert a.context['caffeine']['raw_basis'] == 'unknown'


def test_unlinked_number_and_sizeless_serving_cannot_invent_basis():
    a = row_dimensions(row(description='200mg of caffeine and 16 oz of water.'))
    b = row_dimensions(row(description='200 mg caffeine per serving.'))
    assert not a.context['caffeine']['claims']
    assert b.context['caffeine']['claims'][0]['mg_per_100ml'] is None
    assert compare_context(a.context, b.context)['caffeine']['status'] == 'unknown_basis'


def test_pack_count_is_recovered_without_default_single():
    assert row_dimensions(row(title='Tea 12 x 330 ml')).context['pack_count']['value'] == 12
    assert row_dimensions(row(title='Tea 330 ml')).context['pack_count']['value'] is None


def test_reviewed_packaging_is_listing_and_gtin_scoped():
    resolved = row_dimensions(row(product_id='282677444', barcode='6419806053204', title='Water 12x0.33 l', attributes='Pack Type: Pouch; Pack Material Type: FlexiblePack')).context
    assert resolved['inner_packaging']['types'] == ['can']
    assert resolved['outer_packaging']['types'] == ['shrinkwrap']
    assert resolved['pack_count']['value'] == 12
    wrong = row_dimensions(row(product_id='282677444', barcode='868784000346')).context
    assert not wrong['outer_packaging']['types']


@pytest.mark.parametrize('gtin', ['851107003032', '0851107003032', '00851107003032', '8410408010808', '8480011000022'])
def test_checksum_valid_quarantine_blocks_trust_and_split_rows(gtin):
    values = pd.Series([gtin])
    assert normalize_and_validate_gtin(values).gtin_structurally_valid.iat[0]
    assert review_mask(values).iat[0]
    assert not barcode_validity(values).iat[0]
    result = evaluate_product_identity(row_identity(row(barcode=gtin)), row_identity(row(barcode=gtin)))
    assert result['decision'] == 'review'
    assert result['identity_review_reasons']
    assert exclude_reviewed_rows(pd.DataFrame([row(barcode=gtin)])).empty


def test_split_graph_rejects_stale_quarantined_entities():
    import numpy as np
    from training.folds import merged_component_graph
    with pytest.raises(ValueError, match='quarantined'):
        merged_component_graph(np.empty((0,2),dtype=int), np.array(['851107003032']),
                               labeled_pairs=pd.DataFrame(columns=['gtin1','gtin2','true_label']))


def test_resolved_caffeine_does_not_remain_an_unresolved_identity_dimension():
    a = row(description='200 mg caffeine per 16 oz can.', attributes='Caffeine: 100-150 mg')
    b = row(description='200 mg caffeine per 16 oz can.', attributes='Caffeine: 25-50 mg')
    result = evaluate_product_identity(row_identity(a), row_identity(b))
    assert result['attributes']['Caffeine']['status'] == 'different'
    assert result['context_comparison']['caffeine']['status'] == 'equal'
    assert 'Caffeine' not in result['review_dimensions']
    assert 'Caffeine' in result['resolved_review_dimensions']
    assert result['decision'] == 'compatible_unverified'


def test_zero_sized_explicit_basis_is_unknown_and_not_replaced_by_title():
    context = row_dimensions(row(description='200 mg caffeine per 0 ml can.')).context
    assert context['caffeine']['claims'][0]['basis_ml'] is None
    assert context['caffeine']['claims'][0]['mg_per_100ml'] is None
