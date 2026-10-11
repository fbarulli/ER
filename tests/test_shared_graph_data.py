"""Graph adapter tests cover shared multiplicity and minted representations."""
import json

import pandas as pd
import pytest

from graph_tracks.data import load_records
from graph_tracks.train import load_pairs
from model_tracks.shared_graph_data import _copy_record, prepare_shared_graph
from model_tracks.training_data import SharedTrainingData


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


def _listing(sku_id: str, split: str) -> dict:
    return {'sku_id': sku_id, 'split': split, 'attribute': {'brand': ['acme']}, 'numeric': {}}


def _carve_shaped_setup(root):
    """A setup whose clean listings carry a bundle-local record from a DIFFERENT bundle."""
    from core.common import prepared_setup_layout
    layout = prepared_setup_layout()
    setup = root / 'setup'
    prepared = setup / layout.prepared_dir
    prepared.mkdir(parents=True)
    clean = [_listing('strain', 'train'), _listing('strain2', 'train'),
             _listing('sdev', 'dev'), _listing('sdev2', 'dev'), _listing('sdev3', 'dev'),
             _listing('stest', 'test'), _listing('stest2', 'test'), _listing('stest3', 'test'),
             _listing('augmentation:3', 'dev')]   # carried; the bundle re-mints this ID
    (prepared / layout.listings).write_text(
        json.dumps({'schema': 'er-graph-listings-v1', 'listings': clean}))
    (prepared / 'report_attributes.json').write_text(json.dumps(
        {'schema': 'er-report-attributes-v1', 'attribute': ['brand'],
         'listings': [{'sku_id': r['sku_id'], 'attribute': {'brand': ['acme']}} for r in clean]}))
    (prepared / layout.input_manifest).write_text('{}')
    (setup / layout.manifest).write_text('{}')
    (setup / layout.catalog).write_text(
        'sku_id,gtin,frozen_payload\n'
        + ''.join(f'{r["sku_id"]},gtin-{index},\n' for index, r in enumerate(clean)))
    pairs = pd.DataFrame([
        dict(example_id='t', sku_id1='strain', sku_id2='strain2', label='1', split='train'),
        dict(example_id='', sku_id1='sdev', sku_id2='sdev2', label='1', split='dev'),
        dict(example_id='', sku_id1='sdev', sku_id2='sdev3', label='0', split='dev'),
        # dev pair naming the carried record; the projection re-mints that ID
        dict(example_id='', sku_id1='sdev2', sku_id2='augmentation:3', label='0', split='dev'),
        dict(example_id='', sku_id1='stest', sku_id2='stest2', label='1', split='test'),
        dict(example_id='', sku_id1='stest', sku_id2='stest3', label='0', split='test'),
    ])
    pairs.to_csv(prepared / 'pairs.csv', index=False)
    return setup


def _shared_bundle(payload: list[str]):
    from core.portable_archive import ByteCount

    def size(value: str) -> int:
        return ByteCount(value.encode()).total

    shared = SharedTrainingData.model_validate({
        'source_rows': 2, 'canonical_rows': 0, 'payload_rows': len(payload),
        'endpoints': [
            dict(payload_index=0, kind='listing', entity='e0', source_id='strain', text_size=size(payload[0])),
            dict(payload_index=1, kind='listing', entity='e1', source_id='strain2', text_size=size(payload[1])),
            dict(payload_index=3, kind='augmentation', entity='e3', source_id='augmentation:3',
                 parent_index=0, text_size=size(payload[3])),
        ],
        'examples': [dict(example_id=0, anchor=0, positive=1, negative=3, population='twin')],
    })
    bundle = {
        'df': pd.DataFrame({'sku_id': ['strain', 'strain2']}),
        'payload': list(payload),
        'canonical_records_csv': b'gtin\n',
        'mask_audit': [dict(copy_payload_idx=3, anchor_payload_idx=0,
                            target_mode='targeted', fields_hit=[])],
        'hard_negative_mask_audit': [],
    }
    return bundle, shared


def test_projection_re_mints_carried_endpoints_and_load_pairs_passes(tmp_path):
    """A carried bundle-local endpoint must not shadow THIS bundle's payload."""
    from core.common import prepared_setup_layout
    setup = _carve_shaped_setup(tmp_path)
    bundle, shared = _shared_bundle(['alpha', 'beta', 'gamma', 'delta'])
    projection = prepare_shared_graph(setup, bundle, shared)
    layout = prepared_setup_layout()
    records = {r['sku_id']: r for r in load_records(setup / layout.prepared_dir / layout.listings)}
    pairs = pd.read_csv(setup / layout.prepared_dir / 'pairs.csv', dtype=str, keep_default_na=False)
    # the carried record is re-minted from this bundle, so the projected pair's
    # endpoint and the evaluation pair agree with the enforced split contract
    assert records['augmentation:3']['split'] == 'train'
    assert projection['virtual_counts']['superseded_evaluation_pairs'] == 1
    assert pairs.split.value_counts().to_dict() == {'train': 2, 'dev': 2, 'test': 2}
    assert not ((pairs.sku_id1 == 'sdev2') & (pairs.sku_id2 == 'augmentation:3')).any()
    # the enforcer must accept the producer's output (this was the launch blocker)
    load_pairs(setup / layout.prepared_dir / 'pairs.csv', list(records.values()))

