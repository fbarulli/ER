"""Shared preparation builds once, rebuilds an incompatible cache, and isolates mutations."""
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


def test_changed_fingerprint_rebuilds_the_cache(tmp_path,monkeypatch):
    """An incompatible cache is REBUILT, never a freshness failure (2026-10-08).

    The fingerprint is the reuse key: when it no longer matches, the payload is
    rebuilt in place. Data is never checked (2026-10-09): the recorded size is a
    report value, never compared.
    """
    import pipeline
    import training.base_data as module
    generation=[1]
    monkeypatch.setattr(module,'fingerprint',lambda df,variant:{'generation':generation[0]})
    builds=[]
    monkeypatch.setattr(pipeline,'build_training_data',
                        lambda *a,**kw:builds.append(1) or {'payload':['water']})
    path=tmp_path/'shared.pkl';frame=pd.DataFrame({'sku_name_eng':['water']})
    load_base_data(frame,cache_path=path)
    generation[0]=2
    load_base_data(frame,cache_path=path)
    assert builds==[1,1]


@pytest.mark.parametrize('changed',['csv','config','parser','frame'])
def test_real_fingerprint_rebuilds_on_each_kind_of_change(tmp_path,monkeypatch,changed):
    """Every kind of changed input rebuilds the payload instead of failing.

    The reuse key is STRUCTURAL (file bytes, frame shape), which is why a
    same-size rewrite is the accepted blind spot: each case below changes the
    size the key reads, so the rebuild is proven without content hashing.
    """
    import core.common as common
    import pipeline
    files={}
    for key in ['dataset_deduped','canonical_records','gate_results','labeled_pairs','number_reference']:
        path=tmp_path/(key+'.csv');path.write_text('original');files[key]=path
    config=tmp_path/'config/training.yaml';config.parent.mkdir();config.write_text('setting: original\n')
    parser=tmp_path/'src/parser.py';parser.parent.mkdir();parser.write_text('version = 1\n')
    monkeypatch.setattr(common,'F',files)
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    builds=[]
    monkeypatch.setattr(pipeline,'build_training_data',
                        lambda *a,**kw:builds.append(1) or {'payload':['water']})
    frame=pd.DataFrame({'sku_name_eng':['water'],'description_short_eng':[None]})
    path=tmp_path/'shared.pkl'
    load_base_data(frame,cache_path=path)
    load_base_data(frame.fillna(''),cache_path=path)
    assert len(builds)==1, 'an unchanged input tree reuses the cached payload'
    if changed=='csv':files['canonical_records'].write_text('changed')
    elif changed=='config':config.write_text('setting: changed\n')
    # A same-size rewrite is invisible to a structural key; these two change the
    # bytes the key reads (the parser file's length, the frame's row count).
    elif changed=='parser':parser.write_text('version = 2\n# newer parser\n')
    else:frame.loc[len(frame)]=['mineral-water',None]
    load_base_data(frame,cache_path=path)
    assert len(builds)==2, f'a changed {changed} input must rebuild the payload'
