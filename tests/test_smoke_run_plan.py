import copy

import pytest

from training import run_plan
from training.sampler import FrozenBatchSampler


@pytest.fixture
def saved_plan(monkeypatch):
    identity = dict(loss='mnrl', train_frac=1., sample=True, seed=1729,
                    config_sha256='current-config', data_sha256='same-data')
    monkeypatch.setattr(run_plan, 'plan_identity', lambda *args, **kwargs: dict(identity, **{
        key: kwargs[key] for key in ('loss', 'train_frac', 'sample', 'seed')}))
    return dict(version=1, identity=dict(identity, config_sha256='old-config'),
                inputs=dict(skipped=[], folds=[{}]))


def validate(plan, *, sample=True):
    return run_plan.validate_run_plan({}, plan, loss='mnrl', train_frac=1.,
                                      sample=sample, seed=1729)


def test_smoke_accepts_config_drift_without_mutating_plan(saved_plan):
    before = copy.deepcopy(saved_plan)
    assert validate(saved_plan) is saved_plan
    assert saved_plan == before


@pytest.mark.parametrize('field,value', [('data_sha256', 'other-data'),
                                       ('loss', 'contrastive'), ('seed', 42),
                                       ('train_frac', .5), ('sample', False)])
def test_smoke_rejects_incompatible_plan(saved_plan, field, value):
    saved_plan['identity'][field] = value
    with pytest.raises(ValueError, match='row plan differs'):
        validate(saved_plan)


def test_full_training_still_rejects_config_drift(saved_plan):
    saved_plan['identity']['sample'] = False
    with pytest.raises(ValueError, match='row plan differs'):
        validate(saved_plan, sample=False)


@pytest.mark.parametrize('device,batch_size', [('cpu', 2), ('cuda', 4)])
def test_device_plans_preserve_complete_coverage(device, batch_size):
    batches = [[0, 1], [2]] if device == 'cpu' else [[0, 1, 2]]
    assert list(FrozenBatchSampler([batches], expected_rows=3,
                                 batch_size=batch_size)) == batches
    for invalid in ([[0, 1], [1, 2]], [[0, 1]], [[0, 1], [3]]):
        with pytest.raises(ValueError):
            FrozenBatchSampler([invalid], expected_rows=3, batch_size=batch_size)
