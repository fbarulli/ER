"""Numeric veto configuration and invalid confidence must steer gate decisions."""
import pytest

from core.common import training_cfg
from pipeline import pack_gate, three_way_gate


def attrs(**changes):
    value = dict(canonical='acme vanilla water', volume_set={355.}, pack_set={1},
                 flavor_set={'vanilla'}, package_type_set={'bottle'},
                 volume_confidence=.99, pack_confidence=.99,
                 volume_consistency=1., pack_consistency=1.,
                 attribute_consistency_flags=set())
    value.update(changes)
    return value


@pytest.mark.parametrize('dimension,changes', [('volume', {'volume_set':{1000.}}),
                                               ('pack', {'pack_set':{6}})])
@pytest.mark.parametrize('enabled', [False, True])
def test_numeric_veto_obeys_config(monkeypatch, dimension, changes, enabled):
    import pipeline
    cfg = training_cfg().model_copy(deep=True)
    cfg.rand_matching.targeted_veto_gates.veto_dimensions = ['flavor'] + ([dimension] if enabled else [])
    monkeypatch.setattr(pipeline, 'training_cfg', lambda:cfg)
    a, b = attrs(), attrs(**changes)
    assert pack_gate(0., a, b, trust_threshold=.85) is (not enabled)
    assert three_way_gate(a, b)['decision'] == ('hard_no' if enabled else 'fallback')


@pytest.mark.parametrize('field', ['volume_confidence', 'pack_confidence',
                                  'volume_consistency', 'pack_consistency'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -0.1, 1.1, None, 'invalid'])
@pytest.mark.parametrize('side', [0, 1])
def test_invalid_numeric_reliability_is_review(field, value, side):
    pair = [attrs(), attrs()]
    pair[side][field] = value
    assert three_way_gate(*pair)['decision'] == 'fallback'


@pytest.mark.parametrize('field,changes', [('volume_confidence', {'volume_set':{1000.}}),
                                         ('pack_confidence', {'pack_set':{6}})])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 1.1])
def test_invalid_confidence_cannot_support_numeric_veto(field, changes, value):
    a = attrs(**changes, **{field:value})
    b = attrs()
    assert pack_gate(0., a, b, trust_threshold=.85)
    assert three_way_gate(a, b)['decision'] == 'fallback'
    b['flavor_set'] = {'orange'}
    assert three_way_gate(a, b)['decision'] == 'hard_no'
