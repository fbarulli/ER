import pytest
from pydantic import ValidationError

from model_tracks.training_data import SharedTrainingData, TrackTrainingBinding, retrieval_indices


def shared():
    return SharedTrainingData.model_validate({
        'source_rows': 2, 'canonical_rows': 1, 'payload_rows': 4,
        'endpoints': [
            dict(payload_index=0, kind='listing', entity='entity', source_id='sku', text_size=64),
            dict(payload_index=2, kind='canonical', entity='entity', source_id='canonical', text_size=64),
            dict(payload_index=3, kind='augmentation', entity='entity', source_id='copy',
                 parent_index=0, text_size=64),
        ],
        'examples': [dict(example_id=i, anchor=0, positive=2, negative=3, population='twin') for i in range(2)],
    })


def test_projection_retains_canonical_copy_and_repeated_positive_relationship():
    data = shared()
    assert SharedTrainingData.model_validate_json(data.model_dump_json()) == data
    rows = data.pair_rows()
    assert len(rows) == 4
    assert [(row['payload_index2'], row['label']) for row in rows] == [(2, 1), (3, 0), (2, 1), (3, 0)]
    binding = TrackTrainingBinding(track='gnn_only',
                                  example_ids=[0, 1], endpoint_indices=[0, 2, 3])
    binding.validate_data(data)
    assert retrieval_indices([{'sku_id': value} for value in
                              ('sku', 'canonical:entity', 'augmentation:3')]) == [0]
    with pytest.raises(ValueError, match='differs from shared data'):
        binding.model_copy(update={'endpoint_indices': [0, 2]}).validate_data(data)


def test_reject_copy_as_listing_and_conflicting_relationship_labels():
    data = shared().model_dump()
    data['endpoints'][2]['kind'] = 'listing'
    with pytest.raises(ValidationError):
        SharedTrainingData.model_validate(data)
    data = shared().model_dump()
    data['examples'][1].update(positive=3, negative=2)
    with pytest.raises(ValidationError, match='conflicting relationship labels'):
        SharedTrainingData.model_validate(data)
