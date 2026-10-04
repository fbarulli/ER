import copy

import pytest

from training.run_plan import validate_epoch_batches


def plan():
    return {'inputs': {'folds': [{'objective': {
        'dataset': {'anchor': ['a', 'b', 'c'], 'positive': ['d', 'e', 'f']},
        'sampler': {'cuda': {'batch_size': 2, 'epochs': [[[0, 1], [2]], [[2, 0], [1]]]}}
    }}]}}


def test_complete_frozen_epochs():
    validate_epoch_batches(plan(), epochs=2, batch_sizes={'cuda': 2})


@pytest.mark.parametrize('change', ['missing_device', 'stale_batch', 'missing_epoch', 'duplicate', 'oversized'])
def test_reject_incomplete_or_stale_cpu_preparation(change):
    prepared = copy.deepcopy(plan())
    sampler = prepared['inputs']['folds'][0]['objective']['sampler']
    if change == 'missing_device':
        sampler.clear()
    elif change == 'stale_batch':
        sampler['cuda']['batch_size'] = 1
    elif change == 'missing_epoch':
        sampler['cuda']['epochs'].pop()
    elif change == 'duplicate':
        sampler['cuda']['epochs'][1] = [[0, 1], [1]]
    else:
        sampler['cuda']['epochs'][1] = [[0, 1, 2]]
    with pytest.raises(ValueError, match='rebuild locally'):
        validate_epoch_batches(prepared, epochs=2, batch_sizes={'cuda': 2})
