"""Preparation must reject stale CSVs, pin measured counts, and stop on failure."""
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from training.prepare_all import refresh_gate_census, verify_stage_manifest, prepare_all


def test_resume_rejects_changed_prerequisite_bytes(tmp_path):
    path=tmp_path/'pairs.csv';path.write_text('gtin1,gtin2\n1,2\n')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':'1','stage':'pairs','started':'2026-10-04',
                                   'status':'complete','inputs':[], 'row_accounting':{},
                                   'environment':{}, 'expected_outputs':[],
                                   'outputs':[{'path':str(path),'sha256':digest}]}))
    verify_stage_manifest(manifest)
    path.write_text('gtin1,gtin2\n1,3\n')
    with pytest.raises(ValueError,match='Stale prerequisite'):
        verify_stage_manifest(manifest)


def test_census_update_is_measured_and_preserves_other_configuration(tmp_path):
    gates=tmp_path/'gates.csv'
    pd.DataFrame({'gtin1':['1','1','2'],'gtin2':['2','3','3'],
                  'gate_decision':['proceed','fallback','hard_no']}).to_csv(gates,index=False)
    config=tmp_path/'training.yaml'
    prefix='rand_matching:\n  # keep this comment\n'
    suffix='  target_recall: 0.9\n'
    config.write_text(prefix+'  gate_census_pin:\n    total_pairs: 9\n    hard_no: 4\n    proceed: 3\n    fallback: 2\n'+suffix)
    report=tmp_path/'census.json'
    counts=refresh_gate_census(gates,config,report)
    assert counts=={'total_pairs':3,'hard_no':1,'proceed':1,'fallback':1}
    assert json.loads(report.read_text())==counts
    assert config.read_text().startswith(prefix) and config.read_text().endswith(suffix)
    frame=pd.read_csv(gates);pd.concat([frame,frame.iloc[:1]]).to_csv(gates,index=False)
    with pytest.raises(ValueError,match='Duplicate candidate'):
        refresh_gate_census(gates,config,report)


def test_failed_stage_stops_preparation_and_retains_smoke(tmp_path,monkeypatch):
    import core.common as common
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    monkeypatch.setattr(common,'RESULTS',tmp_path/'results')
    monkeypatch.setattr(common,'resolve_model',lambda key:tmp_path/'model')
    import training.prepare_all as preparation
    from model_tracks.config import SuiteConfig
    monkeypatch.setattr('model_tracks.config.load_config', lambda path: SuiteConfig(
        setup_dir='data/track_setup', text_bundle='data/track_setup/text_prepared.pkl.gz'))
    monkeypatch.setattr(preparation, 'preparation_provenance', lambda *args: {'source':'test'})
    smoke=tmp_path/'data/prepared/smoke_200/pairs.csv'
    smoke.parent.mkdir(parents=True);smoke.write_text('existing smoke bytes')
    calls=[]
    def failed(command,**kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2,command)
    monkeypatch.setattr(subprocess,'run',failed)
    run_dir=tmp_path/'prep_run'
    with pytest.raises(subprocess.CalledProcessError):
        prepare_all(run_dir=run_dir)
    assert len(calls)==1 and calls[0][-1]=='training.dedupe'
    manifest=json.loads((run_dir/'manifest.json').read_text())
    assert manifest['status']=='failed' and manifest['failed_stage']=='dedupe'
    assert manifest['stages']==[] and not manifest['training_started']
    assert smoke.read_text()=='existing smoke bytes'


def test_resume_rejects_corrupted_prepared_output(tmp_path):
    from training.prepare_all import PreparedFile, verify_reusable_outputs
    artifact = tmp_path / 'graph.npz'
    artifact.write_bytes(b'original')
    inventory = {str(artifact): PreparedFile(
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), bytes=artifact.stat().st_size)}
    verify_reusable_outputs(inventory)
    artifact.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='Stale prepared resume input'):
        verify_reusable_outputs(inventory)


def test_census_rejects_reversed_duplicate(tmp_path):
    gates = tmp_path / 'gates.csv'
    pd.DataFrame({'gtin1':['1','2'], 'gtin2':['2','1'],
                  'gate_decision':['proceed','proceed']}).to_csv(gates, index=False)
    with pytest.raises(ValueError, match='Duplicate candidate'):
        refresh_gate_census(gates, tmp_path/'unused.yaml', tmp_path/'unused.json')
