import copy

import pytest

from training import run_plan
from training.sampler import FrozenBatchSampler


@pytest.fixture
def saved_plan(monkeypatch):
    identity = dict(loss='mnrl', train_frac=1., sample=True, seed=1729,
                    data_sha256='same-data')
    monkeypatch.setattr(run_plan, 'plan_identity', lambda *args, **kwargs: dict(identity, **{
        key: kwargs[key] for key in ('loss', 'train_frac', 'sample', 'seed')}))
    # A frozen plan built before the whole-config hash was retired still
    # carries a legacy `config_sha256`; the run-plan path must ignore it.
    return dict(version=1, identity=dict(identity, config_sha256='old-config'),
                inputs=dict(skipped=[], folds=[{}]))


def validate(plan, *, sample=True):
    return run_plan.validate_run_plan({}, plan, loss='mnrl', train_frac=1.,
                                      sample=sample, seed=1729)


def _valid_collapse_config():
    return {
        'training': {
            'loss': 'contrastive',
            'uniformity_regularization': {
                'enabled': True, 'weight': 0.05,
                'temperature': 2.0, 'min_batch_size': 4},
        },
        'collapse_guardrail': {
            'enabled': True, 'profile': 'threshold_80', 'unrelated_pairs': 100,
            'seed': 42, 'max_token_frequency': 0.05, 'operating_threshold': 0.80,
            'crossing_rate_ceiling': 0.02, 'median_penalty_start': 0.50,
            'p90_penalty_start': 0.70, 'cosine_std_floor': 0.05,
            'reject_median': 0.80, 'penalty_weight': 1.0},
        'collapse_guardrail_profiles': {
            'threshold_80': {'operating_threshold': 0.80}},
    }


def test_plan_identity_drops_whole_config_hash(monkeypatch):
    monkeypatch.setattr(run_plan, 'data_digest', lambda bundle: 'same-data')
    identity = run_plan.plan_identity({}, loss='mnrl', train_frac=1., sample=False)
    assert 'config_sha256' not in identity
    assert identity == {'loss': 'mnrl', 'train_frac': 1.0, 'sample': False,
                        'seed': run_plan.SEED, 'data_sha256': 'same-data'}


def test_smoke_accepts_config_drift_without_mutating_plan(saved_plan):
    before = copy.deepcopy(saved_plan)
    assert validate(saved_plan) is saved_plan
    assert saved_plan == before


def test_full_training_accepts_unrelated_config_drift(saved_plan):
    saved_plan['identity']['sample'] = False
    assert validate(saved_plan, sample=False) is saved_plan


def test_full_training_still_rejects_data_drift(saved_plan):
    saved_plan['identity']['sample'] = False
    saved_plan['identity']['data_sha256'] = 'other-data'
    with pytest.raises(ValueError, match='row plan differs'):
        validate(saved_plan, sample=False)


@pytest.mark.parametrize('field,value', [('data_sha256', 'other-data'),
                                       ('loss', 'contrastive'), ('seed', 42),
                                       ('train_frac', .5), ('sample', False)])
def test_smoke_rejects_incompatible_plan(saved_plan, field, value):
    saved_plan['identity'][field] = value
    with pytest.raises(ValueError, match='row plan differs'):
        validate(saved_plan)


def test_collapse_regulation_accepts_a_complete_config():
    config = _valid_collapse_config()
    assert run_plan.validate_collapse_regulation(config) is config


def test_collapse_regulation_ignores_unrelated_config_changes():
    config = _valid_collapse_config()
    config['paths'] = {'laya': '/somewhere/else'}
    config['kaggle'] = {'username': 'someone', 'source_code_dir': '/other'}
    config['training']['epochs'] = 999
    assert run_plan.validate_collapse_regulation(config) is config


@pytest.mark.parametrize('mutate,knob', [
    (lambda cfg: cfg['training']['uniformity_regularization'].pop('temperature'),
     'training.uniformity_regularization.temperature'),
    (lambda cfg: cfg['training']['uniformity_regularization'].pop('min_batch_size'),
     'training.uniformity_regularization.min_batch_size'),
    (lambda cfg: cfg['collapse_guardrail'].update(profile='unconfigured_profile'),
     'collapse_guardrail.profile'),
    (lambda cfg: cfg['collapse_guardrail'].pop('crossing_rate_ceiling'),
     'collapse_guardrail.crossing_rate_ceiling'),
])
def test_collapse_regulation_fails_loud_naming_the_knob(mutate, knob):
    config = _valid_collapse_config()
    mutate(config)
    with pytest.raises(ValueError) as error:
        run_plan.validate_collapse_regulation(config)
    assert knob in str(error.value)


def test_validate_run_plan_rejects_config_that_cannot_regulate_collapse(
        monkeypatch, saved_plan):
    broken = _valid_collapse_config()
    broken['training']['uniformity_regularization']['temperature'] = None
    monkeypatch.setattr(run_plan, 'load_config', lambda: broken)
    with pytest.raises(ValueError,
                       match='training.uniformity_regularization.temperature'):
        validate(saved_plan)


@pytest.mark.parametrize('device,batch_size', [('cpu', 2), ('cuda', 4)])
def test_device_plans_preserve_complete_coverage(device, batch_size):
    batches = [[0, 1], [2]] if device == 'cpu' else [[0, 1, 2]]
    assert list(FrozenBatchSampler([batches], expected_rows=3,
                                 batch_size=batch_size)) == batches
    for invalid in ([[0, 1], [1, 2]], [[0, 1]], [[0, 1], [3]]):
        with pytest.raises(ValueError):
            FrozenBatchSampler([invalid], expected_rows=3, batch_size=batch_size)
