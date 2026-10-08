import copy

import pytest

from training import run_plan
from training.sampler import FrozenBatchSampler


@pytest.fixture
def saved_plan(monkeypatch):
    identity = dict(loss='mnrl', train_frac=1., sample=True, seed=1729,
                    data_size='same-data')
    monkeypatch.setattr(run_plan, 'plan_identity', lambda *args, **kwargs: dict(identity, **{
        key: kwargs[key] for key in ('loss', 'train_frac', 'sample', 'seed')}))
    # A frozen plan built before the whole-config hash was retired still
    # carries a legacy `config_size`; the run-plan path must ignore it.
    return dict(version=1, identity=dict(identity, config_size='old-config'),
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
    monkeypatch.setattr(run_plan, 'data_size', lambda bundle: 4242)
    identity = run_plan.plan_identity({}, loss='mnrl', train_frac=1., sample=False)
    assert 'config_size' not in identity
    assert identity == {'loss': 'mnrl', 'train_frac': 1.0, 'sample': False,
                        'seed': run_plan.SEED, 'data_size': 4242}


def test_frozen_inputs_materialize_from_the_bundle_members(tmp_path, monkeypatch):
    """The plan binds the bundle-carried ``*_csv`` bytes, not the live checkout.

    The bug this pins: the materialization compared the bare registry keys
    (``labeled_pairs``) against the bundle, so a bundle carrying the real members
    (``labeled_pairs_csv``) was never materialized and the plan read whatever the
    checkout happened to hold. A bundle with none of them is left untouched.
    """
    from core import common
    monkeypatch.setattr(common, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(common, 'F', {'labeled_pairs': tmp_path / 'data/labeled_pairs.csv',
                                      'canonical_records': tmp_path / 'data/canonical_records.csv',
                                      'gate_results': tmp_path / 'data/gate_results.csv'})
    written = run_plan._materialize_frozen_inputs(
        {'labeled_pairs_csv': b'labeled', 'gate_results_csv': b'gates'})
    assert set(written) == {'labeled_pairs', 'gate_results'}
    assert written['labeled_pairs'].read_bytes() == b'labeled'
    assert common.F['labeled_pairs'] == written['labeled_pairs']
    assert written['labeled_pairs'].parent == tmp_path / 'results' / '_prepared_inputs'

    untouched = dict(common.F)
    assert run_plan._materialize_frozen_inputs({'df': None}) == {}
    assert common.F == untouched


def test_one_frozen_input_materializer_serves_plan_and_trainer():
    """Both callers share ONE implementation (no second literal to drift).

    The plan path and the prepared trainer used to spell the ``*_csv`` member
    names and the ``_prepared_inputs`` destination directory independently; the
    trainer now calls ``run_plan._materialize_frozen_inputs``, so only that
    function declares them.
    """
    import inspect

    from training import train_prepared

    assert '_materialize_frozen_inputs' in inspect.getsource(train_prepared)
    assert '_prepared_inputs' not in inspect.getsource(train_prepared)
    assert '_prepared_inputs' in inspect.getsource(run_plan._materialize_frozen_inputs)


def test_smoke_accepts_config_drift_without_mutating_plan(saved_plan):
    before = copy.deepcopy(saved_plan)
    assert validate(saved_plan) is saved_plan
    assert saved_plan == before


def test_full_training_accepts_unrelated_config_drift(saved_plan):
    saved_plan['identity']['sample'] = False
    assert validate(saved_plan, sample=False) is saved_plan


def test_full_training_still_rejects_data_drift(saved_plan):
    saved_plan['identity']['sample'] = False
    saved_plan['identity']['data_size'] = 'other-data'
    with pytest.raises(ValueError, match='row plan differs'):
        validate(saved_plan, sample=False)


@pytest.mark.parametrize('field,value', [('data_size', 'other-data'),
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


def test_collapse_regulation_gate_is_pure_and_cheap(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            'the collapse gate must not derive data or reload the config')
    monkeypatch.setattr(run_plan, 'data_size', _boom)
    monkeypatch.setattr(run_plan, 'load_config', _boom)
    config = _valid_collapse_config()
    assert run_plan.validate_collapse_regulation(config) is config


def test_validate_run_plan_loads_active_config_once(monkeypatch, saved_plan):
    calls = []
    real = run_plan.load_config
    monkeypatch.setattr(run_plan, 'load_config',
                        lambda: (calls.append(1), real())[1])
    assert validate(saved_plan) is saved_plan
    assert len(calls) == 1


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
