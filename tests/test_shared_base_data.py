"""Shared preparation builds once, rejects stale inputs, and isolates mutations."""
import pandas as pd
import pytest

from training.base_data import load_base_data


def test_three_consumers_build_once_and_get_independent_data(tmp_path,monkeypatch):
    import pipeline
    import training.base_data as module
    calls=[]
    monkeypatch.setattr(module,'fingerprint',lambda df,variant:{'rows':df.fillna('').to_json(),'variant':variant})
    def build(df,*,payload_variant):
        calls.append(payload_variant)
        return {'payload':['original'],'pos':[(0,1)]}
    monkeypatch.setattr(pipeline,'build_training_data',build)
    path=tmp_path/'shared.pkl'
    frame=pd.DataFrame({'sku_name_eng':['water'],'description_short_eng':[None]})
    first=load_base_data(frame,cache_path=path)
    first['payload'].append('augmentation')
    second=load_base_data(frame.fillna(''),cache_path=path)
    second['payload'][0]='changed'
    third=load_base_data(frame,cache_path=path)
    assert calls==['full']
    assert third=={'payload':['original'],'pos':[(0,1)]}


def test_changed_fingerprint_rejects_reuse(tmp_path,monkeypatch):
    import pipeline
    import training.base_data as module
    generation=[1]
    monkeypatch.setattr(module,'fingerprint',lambda df,variant:{'generation':generation[0]})
    monkeypatch.setattr(pipeline,'build_training_data',lambda *a,**kw:{'payload':['water']})
    path=tmp_path/'shared.pkl';frame=pd.DataFrame({'sku_name_eng':['water']})
    load_base_data(frame,cache_path=path)
    generation[0]=2
    with pytest.raises(ValueError,match='Stale shared base payload'):
        load_base_data(frame,cache_path=path)


def test_corrupted_payload_is_rejected_before_loading(tmp_path,monkeypatch):
    import pipeline
    import training.base_data as module
    monkeypatch.setattr(module,'fingerprint',lambda *args:{'input':'same'})
    monkeypatch.setattr(pipeline,'build_training_data',lambda *a,**kw:{'payload':['water']})
    path=tmp_path/'shared.pkl';frame=pd.DataFrame({'sku_name_eng':['water']})
    load_base_data(frame,cache_path=path)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum mismatch'):
        load_base_data(frame,cache_path=path)

@pytest.mark.parametrize('changed',['csv','config','parser','frame'])
def test_real_fingerprint_catches_each_kind_of_change(tmp_path,monkeypatch,changed):
    import core.common as common
    import pipeline
    files={}
    for key in ['dataset_deduped','canonical_records','gate_results','labeled_pairs','number_reference']:
        path=tmp_path/(key+'.csv');path.write_text('original');files[key]=path
    config=tmp_path/'config/training.yaml';config.parent.mkdir();config.write_text('setting: original\n')
    parser=tmp_path/'src/parser.py';parser.parent.mkdir();parser.write_text('version = 1\n')
    monkeypatch.setattr(common,'F',files)
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'build_training_data',lambda *a,**kw:{'payload':['water']})
    frame=pd.DataFrame({'sku_name_eng':['water'],'description_short_eng':[None]})
    path=tmp_path/'shared.pkl'
    load_base_data(frame,cache_path=path)
    load_base_data(frame.fillna(''),cache_path=path)
    if changed=='csv':files['canonical_records'].write_text('changed')
    elif changed=='config':config.write_text('setting: changed\n')
    elif changed=='parser':parser.write_text('version = 2\n')
    else:frame.loc[0,'sku_name_eng']='juice'
    with pytest.raises(ValueError,match='Stale shared base payload'):
        load_base_data(frame,cache_path=path)
