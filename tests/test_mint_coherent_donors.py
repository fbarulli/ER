import numpy as np
from training.masking import _coherent_splice_field, augment_counterfactual_twins, augment_value_swaps


def test_coherent_edits_rewrite_existing_claims_and_preserve_other_fields():
    assert _coherent_splice_field('apple juice apple flavor_apple volume_ml_250', 'flavor', ['flavor_cherry']) == 'cherry juice flavor_cherry volume_ml_250'
    assert _coherent_splice_field('sparkling apple carbonation_carbonated flavor_apple', 'carbonation', ['carbonation_still']) == 'still apple carbonation_still flavor_apple'
    assert _coherent_splice_field('water 1l volume_ml_1000', 'volume', ['volume_ml_250']) == 'water 250ml volume_ml_250'
    assert _coherent_splice_field('water 4x330ml pack_qty_4 volume_ml_330', 'pack', ['pack_qty_6']) == 'water 6x330ml pack_qty_6 volume_ml_330'


def test_ambiguous_numbers_and_mixed_claims_do_not_mint():
    assert _coherent_splice_field('15 tapped birch trees pack_qty_15', 'pack', ['pack_qty_12']) is None
    assert _coherent_splice_field('water 500ml and 1l volume_ml_1000', 'volume', ['volume_ml_250']) is None
    assert _coherent_splice_field('apple cherry flavor_apple', 'flavor', ['flavor_lime']) is None
    assert _coherent_splice_field('non carbonated carbonation_still', 'carbonation', ['carbonation_carbonated']) is None


def test_train_scope_applies_to_both_anchors_and_both_donor_endpoints():
    payload = ['listing apple flavor_apple','canonical apple flavor_apple','listing cherry flavor_cherry','canonical cherry flavor_cherry','listing lime flavor_lime','canonical lime flavor_lime']
    pairs = np.array([[0,1],[2,3],[4,5]])
    bc = np.array(['a','a','b','b','held','held'])
    for augment in [augment_counterfactual_twins,augment_value_swaps]:
        kwargs = {'symmetric':True} if augment is augment_value_swaps else {}
        _,texts,_,added,audit = augment(pairs,payload,bc,frac=1,seed=2,allowed_payload_indices={0,1,2,3},coherent_prose=True,**kwargs)
        assert added > 0
        assert all(r['anchor_payload_idx'] < 4 and r['pair_payload_idx'] < 4 and r['donor_anchor_payload_idx'] < 4 and r['donor_pair_payload_idx'] < 4 for r in audit)
        for r in audit:
            assert r['coherent_prose']
            assert texts[r['copy_payload_idx']] != r['anchor_text']


def test_canonical_matched_negative_has_explicit_feature_parent():
    from training.masking import extend_augmented_features
    payload = ['listing apple flavor_apple', 'canonical apple flavor_apple',
               'listing cherry flavor_cherry', 'canonical cherry flavor_cherry']
    pairs = np.array([[0,1],[2,3]])
    _,texts,_,added,audit = augment_counterfactual_twins(pairs,payload,np.array(['a','a','b','b']),frac=1,seed=2,allowed_payload_indices={0,1,2,3},coherent_prose=True,canonical_indices={1,3})
    assert added == 2
    features = np.repeat(np.arange(4,dtype=np.float32)[:,None],10,axis=1)
    extended = extend_augmented_features(features,texts,audit)
    for r in audit:
        assert r['copy_source_payload_idx'] == r['pair_payload_idx']
        assert texts[r['copy_payload_idx']].startswith('canonical ')
        assert np.array_equal(extended[r['copy_payload_idx']],features[r['copy_source_payload_idx']])
        assert r['donor_split'] == 'train'
        assert r['fields_before']['flavor'] != r['fields_after']['flavor']
