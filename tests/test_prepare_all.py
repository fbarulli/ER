"""Preparation must reject stale prerequisites, record measured counts, and stop on failure."""
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


def test_census_is_measured_and_recorded_without_config_rewrite(tmp_path):
    # Pins removed 2026-10-06 (owner ruling): the census stage MEASURES and
    # records; it must not touch the config.
    gates=tmp_path/'gates.csv'
    pd.DataFrame({'gtin1':['1','1','2'],'gtin2':['2','3','3'],
                  'gate_decision':['proceed','fallback','hard_no']}).to_csv(gates,index=False)
    config=tmp_path/'training.yaml'
    config.write_text('rand_matching:\n  # keep this comment\n  target_recall: 0.9\n')
    report=tmp_path/'census.json'
    counts=refresh_gate_census(gates,report)
    assert counts=={'total_pairs':3,'hard_no':1,'proceed':1,'fallback':1}
    assert json.loads(report.read_text())==counts
    assert config.read_text()=='rand_matching:\n  # keep this comment\n  target_recall: 0.9\n'
    frame=pd.read_csv(gates);pd.concat([frame,frame.iloc[:1]]).to_csv(gates,index=False)
    with pytest.raises(ValueError,match='Duplicate candidate'):
        refresh_gate_census(gates,report)


def test_failed_stage_stops_preparation_and_retains_smoke(tmp_path,monkeypatch):
    import core.common as common
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    monkeypatch.setattr(common,'RESULTS',tmp_path/'results')
    monkeypatch.setattr(common,'resolve_model',lambda key:tmp_path/'model')
    import training.prepare_all as preparation
    from model_tracks.config import SuiteConfig
    monkeypatch.setattr('model_tracks.config.load_config', lambda path: SuiteConfig(
        setup_dir='data/track_setup', text_bundle='data/track_setup/text_prepared.pkl.gz'))
    monkeypatch.setattr(preparation, 'preparation_provenance', lambda *args: {'source':'0' * 64})
    smoke=tmp_path/'data/prepared/smoke_200/pairs.csv'
    smoke.parent.mkdir(parents=True);smoke.write_text('existing smoke bytes')
    calls=[]
    def failed(command,**kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2,command)
    from training.preparation_run import TrainingPreparation
    monkeypatch.setattr(TrainingPreparation, 'run_stage',
        lambda self, arguments, **kwargs: failed(['python', *arguments], **kwargs))
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
        refresh_gate_census(gates, tmp_path/'unused.json')


def test_inventory_deduplicates_paths_but_never_caches_content(tmp_path, monkeypatch):
    import os
    import training.prepare_all as preparation
    artifact = tmp_path / 'payload'
    artifact.write_bytes(b'old')
    alias = tmp_path / 'alias'
    alias.symlink_to(artifact)
    calls = []
    original = preparation.sha256
    monkeypatch.setattr(preparation, 'sha256', lambda path: (calls.append(path), original(path))[1])
    first = preparation.file_inventory([artifact, artifact, alias])
    assert len(calls) == len(first) == 1
    stat = artifact.stat()
    artifact.write_bytes(b'new')
    os.utime(artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = preparation.file_inventory([artifact])
    assert first != second  # Same length and mtime cannot hide changed content.
    assert len(calls) == 2


def test_provenance_includes_nested_json(tmp_path, monkeypatch):
    import core.common as common
    import training.prepare_all as preparation
    config = tmp_path / 'config'
    config.mkdir()
    training = config / 'training.yaml'
    training.write_text('rand_matching:\n  setting: 1\n')
    policy = config / 'nested' / 'policy.json'
    policy.parent.mkdir()
    policy.write_text('{"version": 1}')
    raw = tmp_path / 'raw.csv'
    raw.write_text('raw')
    for name in ('CONFIG_PATH', 'TRAINING_CONFIG_PATH', 'VOCABULARY_CONFIG_PATH'):
        monkeypatch.setattr(common, name, training)
    monkeypatch.setattr(common, 'DATA_PATH', raw)
    monkeypatch.setattr('graph_tracks.text_cache.checkpoint_hash', lambda path, **kwargs: '0'*64)
    first = preparation.preparation_provenance(tmp_path, training, 'model')
    policy.write_text('{"version": 2}')
    assert preparation.preparation_provenance(tmp_path, training, 'model') != first


def test_full_run_and_suite_resume_force_fresh_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import core.common as common
    import training.prepare_all as preparation
    from model_tracks.config import SuiteConfig
    root = tmp_path
    results = root / 'results'
    run_dir = root / 'run'
    setup = root / 'setup'
    suite = SuiteConfig(setup_dir='setup', text_bundle='setup/text.pkl.gz')
    files = {key: root / f'{key}.csv' for key in preparation.REUSABLE_KEYS}
    for path in files.values():
        path.write_text('original')
    monkeypatch.setattr(common, 'TRAIN_ROOT', root)
    monkeypatch.setattr(common, 'RESULTS', results)
    monkeypatch.setattr(common, 'F', files)
    monkeypatch.setattr(common, 'resolve_model', lambda key: root/'checkpoint')
    monkeypatch.setattr('model_tracks.config.load_config', lambda path: suite)
    monkeypatch.setattr(preparation, 'preparation_provenance', lambda *args: {'text_checkpoint': '0'*64})
    monkeypatch.setattr(preparation, 'refresh_gate_census', lambda *args: {})
    monkeypatch.setenv('ER_DATA_GATE', 'stale inherited trust')
    calls = []

    def run(command, **kwargs):
        assert kwargs['env']['ER_DATA_GATE_ENFORCE'] == '1'
        calls.append(command)
        if 'training.negative_supply' in command:
            directory = results/'negative_supply'/'prep_run'
            directory.mkdir(parents=True)
            for filename in ('pairs.csv', 'manifest.json'):
                (directory/filename).write_text('lane')
        elif any('negative_supply_discriminator.py' in part for part in command):
            (run_dir/'discriminator.json').write_text('{"verdict": "PASS"}')
        elif 'graph_tracks.setup' in command:
            setup.mkdir()
            (setup/'setup_manifest.json').write_text(json.dumps({
                'source_catalog_sha256': preparation.sha256(files['dataset_deduped']),
                'labeled_pairs_sha256': preparation.sha256(files['labeled_pairs']),
                'text_checkpoint_sha256': '0'*64,
            }))
        elif 'training.train' in command:
            bundle = Path(command[command.index('--prepare-bundle') + 1])
            bundle.parent.mkdir(parents=True)
            bundle.write_bytes(b'bundle')
            Path(str(bundle)+'.json').write_text('{}')
        elif 'model_tracks.package' in command:
            (run_dir/'all_tracks_inputs.tar.zst').write_bytes(b'archive')
            # Packaging legitimately updates graph inputs and adds projections.
            (setup/'projection.json').write_text('{"projection": 1}')
        return subprocess.CompletedProcess(command, 0)

    from training.preparation_run import TrainingPreparation
    monkeypatch.setattr(TrainingPreparation, 'run_stage',
        lambda self, arguments, **kwargs: run(['python', *arguments], **kwargs))
    verified = []
    def load(path, *, verify_inputs):
        verified.append(verify_inputs)
        return SimpleNamespace(model_dump=lambda **kwargs: {}), {
            key+'_csv': files[key].read_bytes()
            for key in ('canonical_records', 'gate_results', 'labeled_pairs')}
    monkeypatch.setattr('training.prepared_bundle.load_prepared_bundle', load)
    monkeypatch.setattr('model_tracks.package.verify', lambda path: {'preflight': {}})
    manifest_path = prepare_all(run_dir=run_dir)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'complete'
    assert manifest['stages'][-4:] == ['graph_inputs', 'full_bundle', 'suite_inputs', 'verify_handoff']
    assert verified == [True]
    calls.clear()
    prepare_all(run_dir=run_dir, resume_from='suite_inputs')
    assert len(calls) == 1 and 'model_tracks.package' in calls[0]
    assert verified == [True, True]
    # A reference edit must invalidate resume, even with unchanged size.
    files['number_reference'].write_text('modified')
    with pytest.raises(ValueError, match='Stale prepared resume input'):
        prepare_all(run_dir=run_dir, resume_from='suite_inputs')


def test_bundle_copy_remains_independent(tmp_path):
    from training.prepare_all import copy_bundle
    source, target = tmp_path/'source', tmp_path/'target'
    source.write_bytes(b'original')
    copy_bundle(source, target)
    assert target.read_bytes() == b'original'
    assert source.stat().st_ino != target.stat().st_ino
    source.write_bytes(b'modified')
    assert target.read_bytes() == b'original'
    target.write_bytes(b'separate')
    assert source.read_bytes() == b'modified'


def test_bundle_copy_falls_back_only_when_clone_is_unsupported(tmp_path, monkeypatch):
    import errno
    import training.prepare_all as preparation
    source, target = tmp_path/'source', tmp_path/'target'
    source.write_bytes(b'original')
    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, 'unsupported')
    monkeypatch.setattr(preparation.fcntl, 'ioctl', unsupported)
    preparation.copy_bundle(source, target)
    assert target.read_bytes() == source.read_bytes()
    def full(*args):
        raise OSError(errno.ENOSPC, 'full')
    monkeypatch.setattr(preparation.fcntl, 'ioctl', full)
    with pytest.raises(OSError) as exc:
        preparation.copy_bundle(source, target)
    assert exc.value.errno == errno.ENOSPC
