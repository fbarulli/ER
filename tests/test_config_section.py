from core import common


def test_section_copy_is_isolated_and_observes_source_changes(monkeypatch):
    source = {'training':{'model_input':{'profile':'cleaned'}}}
    monkeypatch.setattr(common,'_load_config_cached',lambda:source)
    section = common.config_section('training','model_input')
    section['profile'] = 'changed by caller'
    assert source['training']['model_input']['profile'] == 'cleaned'
    source['training']['model_input']['profile'] = 'new config'
    assert common.config_section('training','model_input')['profile'] == 'new config'


def test_section_accessor_respects_injected_config_loader():
    override = lambda:{'training':{'model_input':{'profile':'override'}}}
    assert common.config_section('training','model_input',loader=override) == {'profile':'override'}
