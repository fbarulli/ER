"""Definite configured conflicts win over uncertainty; missing stays unknown."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.common import training_cfg
from pipeline import pack_gate, three_way_gate


def _attrs(**changes):
    rec = dict(
        canonical='acme vanilla water', volume_set={355.0}, pack_set={1},
        package_type_set={'bottle'}, package_material_set={'plastic'},
        packaging_level_set=set(), flavor_set={'vanilla'},
        carbonation_set=set(), sweetener_set=set(), pulp_set=set(),
        volume_confidence=.99, pack_confidence=.99,
        volume_consistency=1., pack_consistency=1., attribute_consistency_flags=set(),
    )
    rec.update(changes)
    return rec


@pytest.mark.parametrize('field,left,right', [
    ('flavor_set', {'vanilla'}, {'orange'}),
    ('package_material_set', {'plastic'}, {'glass'}),
    ('package_type_set', {'bottle'}, {'can'}),
    ('sweetener_set', {'sugar'}, {'no_sugar'}),
    ('pulp_set', {'no_pulp'}, {'with_pulp'}),
])
def test_definite_conflicts_win_over_missing_pack_and_uncertain_volume(field, left, right):
    a = _attrs(**{field:left}, volume_confidence=.1)
    b = _attrs(**{field:right}, pack_set=set(), pack_confidence=0.)
    assert three_way_gate(a, b)['decision'] == 'hard_no'


@pytest.mark.parametrize('changes', [
    {'pack_set':set(), 'pack_confidence':0.},
    {'pack_set':set(), 'pack_confidence':.99},
    {'pack_set':{6}, 'pack_confidence':.1},
    {'volume_set':set(), 'volume_confidence':.99},
    {'volume_set':{1000.}, 'volume_confidence':.1},
    {'volume_set':{1000.}, 'attribute_consistency_flags':{'ambiguous_volume'}},
])
def test_unknown_numeric_evidence_is_review(changes):
    assert three_way_gate(_attrs(), _attrs(**changes))['decision'] == 'fallback'


def test_disabled_carbonation_is_not_early_veto():
    a, b = _attrs(carbonation_set={'still'}), _attrs(carbonation_set={'carbonated'})
    assert 'carbonation' not in training_cfg().rand_matching.targeted_veto_gates.veto_dimensions
    assert pack_gate(0, a, b)
    assert three_way_gate(a,b)['decision'] == 'proceed'


def test_categorical_config_steers_both_gate_paths(monkeypatch):
    import pipeline
    cfg = training_cfg().model_copy(deep=True)
    cfg.rand_matching.targeted_veto_gates.veto_dimensions.append('carbonation')
    monkeypatch.setattr(pipeline, 'training_cfg', lambda:cfg)
    a, b = _attrs(carbonation_set={'still'}), _attrs(carbonation_set={'carbonated'})
    assert not pack_gate(0,a,b)
    assert three_way_gate(a,b)['decision'] == 'hard_no'


def test_config_disables_material_and_type_veto(monkeypatch):
    import pipeline
    cfg = training_cfg().model_copy(deep=True)
    cfg.rand_matching.targeted_veto_gates.veto_dimensions = ['volume', 'pack', 'flavor']
    monkeypatch.setattr(pipeline, 'training_cfg', lambda:cfg)
    a = _attrs()
    b = _attrs(package_type_set={'can'}, package_material_set={'glass'})
    assert pack_gate(0,a,b)
    assert three_way_gate(a,b)['decision'] == 'proceed'


def test_ambiguous_volume_never_numeric_veto():
    a = _attrs(volume_set={1000.}, attribute_consistency_flags={'ambiguous_volume'})
    b = _attrs()
    assert pack_gate(0,a,b,trust_threshold=.85)
    assert three_way_gate(a,b)['reason'] == training_cfg().gate.reasons.ambiguous_volume


@pytest.mark.parametrize('flag,field,values', [
    ('volume_sources_disagree', 'volume_set', {1000.}),
    ('pack_sources_disagree', 'pack_set', {6}),
    ('sweetener_source_conflict:sucralose', 'sweetener_type_set', {'sucralose'}),
])
def test_source_conflict_reviews_without_trusting_selected_scalar(flag, field, values):
    a = _attrs(**{field:values}, attribute_consistency_flags={flag})
    b = _attrs()
    assert pack_gate(0,a,b,trust_threshold=.85)
    assert three_way_gate(a,b) == {
        'decision':'fallback', 'reason':training_cfg().gate.reasons.source_conflict,
    }
    # An unrelated certain flavor contradiction must still win.
    b['flavor_set'] = {'orange'}
    assert three_way_gate(a,b)['decision'] == 'hard_no'


def test_committed_before_after_evidence_fixtures():
    fixture = Path(__file__).parent / 'fixtures' / 'gate_evidence_before_after.json'
    cases = json.loads(fixture.read_text())
    for case in cases:
        records = []
        for side in ('left', 'right'):
            rec = dict(case[side])
            for key,value in rec.items():
                if key.endswith('_set') or key.endswith('_flags'):
                    rec[key] = set(value)
            records.append(rec)
        assert three_way_gate(*records) == case['after'], case['name']


def test_projected_registry_preserves_consumed_engine_verdicts():
    from core.attribute_conflicts import canonical_attribute_info, CRITICAL_NAME_BY_CENSUS_KEY
    from core.attribute_decision import AttributeDecisionEngine
    from core.attribute_universe import attribute_registry
    cfg = training_cfg()
    dimensions = set(cfg.rand_matching.targeted_veto_gates.veto_dimensions) - {'volume', 'pack', 'package_type', 'pack_material'}
    projected = {key:spec for key,spec in attribute_registry().items() if CRITICAL_NAME_BY_CENSUS_KEY.get(key) in dimensions}
    fixture = Path(__file__).parent / 'fixtures' / 'gate_evidence_before_after.json'
    engine = AttributeDecisionEngine(
        volume_relative_tolerance=cfg.gate.vol_tolerance,
        volume_absolute_tolerance_ml=cfg.gate.vol_abs_tolerance,
    )
    for case in json.loads(fixture.read_text()):
        a,b = case['left'],case['right']
        left,right = canonical_attribute_info(a),canonical_attribute_info(b)
        full = engine.evaluate(left,right,left_raw=a,right_raw=b)
        reduced = engine.evaluate(left,right,left_raw=a,right_raw=b,registry=projected)
        for key in projected:
            assert full.dimensions[key] == reduced.dimensions[key], (case['name'],key)


def test_source_numeric_disagreement_does_not_hide_other_numeric_conflict():
    a = _attrs(volume_set={1000.}, attribute_consistency_flags={'volume_sources_disagree'})
    b = _attrs(pack_set={6})
    assert three_way_gate(a,b)['decision'] == 'hard_no'
    a = _attrs(pack_set={6}, attribute_consistency_flags={'pack_sources_disagree'})
    b = _attrs(volume_set={1000.})
    assert three_way_gate(a,b)['decision'] == 'hard_no'


def test_numeric_only_config_skips_empty_categorical_engine_registry(monkeypatch):
    import pipeline
    cfg = training_cfg().model_copy(deep=True)
    cfg.rand_matching.targeted_veto_gates.veto_dimensions = ['volume', 'pack']
    monkeypatch.setattr(pipeline, 'training_cfg', lambda:cfg)
    assert three_way_gate(_attrs(),_attrs())['decision'] == 'proceed'
