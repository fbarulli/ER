"""Graph adapter tests cover shared multiplicity and minted representations."""
import pandas as pd
import pytest

from graph_tracks.train import load_pairs
from model_tracks.shared_graph_data import _copy_record


def test_repeated_shared_relationships_retain_order_and_conflicts_fail(tmp_path):
    records = [{'sku_id': key, 'split': split} for split, keys in
               [('train', ['a', 'b', 'c']), ('dev', ['d', 'e', 'f'])] for key in keys]
    rows = [dict(sku_id1='a', sku_id2='b', label='1', split='train', example_id='0:1'),
            dict(sku_id1='a', sku_id2='c', label='0', split='train', example_id='0:0'),
            dict(sku_id1='a', sku_id2='b', label='1', split='train', example_id='1:1'),
            dict(sku_id1='d', sku_id2='e', label='1', split='dev', example_id=''),
            dict(sku_id1='d', sku_id2='f', label='0', split='dev', example_id='')]
    path = tmp_path / 'pairs.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    indices, labels = load_pairs(path, records)['train']
    assert indices.tolist() == [[0, 1], [0, 2], [0, 1]]
    assert labels.tolist() == [1., 0., 1.]
    rows[2]['label'] = '0'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='conflicting pair'):
        load_pairs(path, records)


def test_swaps_change_only_frozen_graph_fields_and_masks_inherit():
    parent = dict(sku_id='original', split='train',
                  attribute={'brand': ['cola'], 'flavor': ['lemon'], 'sweetener': ['diet']},
                  numeric={'volume_ml': [250.], 'pack': [1.]})
    changed = _copy_record(parent, 'flavor_cherry volume_ml_500 pack_qty_6 sweetener_diet_regular',
                           {'target_mode': 'swap_values', 'fields_hit': ['flavor', 'volume', 'pack', 'sweetener']},
                           'augmentation:4')
    assert changed['attribute']['flavor'] == ['cherry']
    assert changed['attribute']['sweetener'] == ['regular']
    assert changed['numeric'] == {'volume_ml': [500.], 'pack': [6.]}
    assert changed['attribute']['brand'] == ['cola']
    masked = _copy_record(parent, '[MASK]', {'target_mode': 'targeted', 'fields_hit': ['volume']},
                          'augmentation:5')
    assert masked['numeric'] == parent['numeric']
    assert masked['attribute'] == parent['attribute']
    assert parent['attribute']['flavor'] == ['lemon']
