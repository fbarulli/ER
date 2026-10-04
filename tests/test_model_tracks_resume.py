import json
from types import SimpleNamespace

import pytest

from model_tracks.resume import completed_track, record_completion, validate_suite, graph_checkpoint, suite_identity


def test_completed_tracks_require_intact_portable_artifacts(tmp_path):
    source = tmp_path / 'text'
    source.mkdir()
    (source / 'metrics.csv').write_text('status\nok\n')
    record_completion(source, 'text')
    assert completed_track(source, 'text')
    import shutil
    restored = tmp_path / 'restored'
    shutil.copytree(source, restored)
    assert completed_track(restored, 'text')
    (restored / 'metrics.csv').write_text('modified\n')
    with pytest.raises(ValueError, match='completed artifact changed'):
        completed_track(restored, 'text')


def test_resume_rejects_changed_provenance_and_unverified_marker(tmp_path):
    identity = {'run_tag': 'run', 'inputs': {'sha': 'original'}}
    (tmp_path / 'suite_manifest.json').write_text(json.dumps({'resume_identity': identity}))
    validate_suite(tmp_path, identity)
    with pytest.raises(ValueError, match='provenance mismatch'):
        validate_suite(tmp_path, {'run_tag': 'run', 'inputs': {'sha': 'changed'}})
    (tmp_path / 'track_complete.json').write_text(json.dumps(
        {'track': 'text', 'status': 'ok', 'postprocess_complete': True}))
    with pytest.raises(ValueError, match='inventory missing'):
        completed_track(tmp_path, 'text')


def test_graph_resume_selects_latest_epoch_in_its_own_run(tmp_path):
    for run, epoch in [('run-hybrid', 2), ('run-hybrid', 10), ('other-hybrid', 99)]:
        folder = tmp_path / '_checkpoints/hybrid' / f'{run}_f0' / f'checkpoint-{epoch}'
        folder.mkdir(parents=True)
        (folder / 'hybrid__graph_model.pt').write_bytes(b'checkpoint')
    assert graph_checkpoint(tmp_path, 'hybrid', 'run-hybrid').parent.name == 'checkpoint-10'
    assert graph_checkpoint(tmp_path, 'gnn_only', 'run-gnn_only') is None


def test_graph_resume_finds_trainer_nested_worker_output(tmp_path):
    root = tmp_path / 'hybrid__run-hybrid/_checkpoints/hybrid/run-hybrid_f0/checkpoint-3'
    root.mkdir(parents=True)
    checkpoint = root / 'hybrid__graph_model.pt'
    checkpoint.write_bytes(b'checkpoint')
    assert graph_checkpoint(tmp_path, 'hybrid', 'run-hybrid') == checkpoint


def test_identity_ignores_generated_reports_and_portable_suite_paths(tmp_path, monkeypatch):
    import core.common
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    setup = tmp_path / 'setup'
    setup.mkdir()
    (setup / 'eligible_catalog.csv').write_text('frozen')
    import yaml
    for track in ('gnn_only', 'hybrid'):
        (setup / (track + '.yaml')).write_text(yaml.safe_dump({
            'track': track, 'listings': 'data/listings.json', 'pairs': 'data/pairs.csv',
            'output_dir': 'results/graph_tracks',
            **({'text_cache': 'data/text.npz'} if track == 'hybrid' else {})}))
    (setup / 'text.yaml').write_text(yaml.safe_dump({'track': 'text', 'output_dir': 'results/model_tracks'}))

    (tmp_path / 'config').mkdir()
    for name in ('training.yaml', 'identity_dimensions.yaml', 'identity_reviews.json', 'vocabulary.json'):
        (tmp_path / 'config' / name).write_text('same')
    settings = dict(setup_dir='setup', text_bundle='old/bundle', epochs=1,
                    ablation_config='config/attribute_ablation.yaml')
    monkeypatch.setattr('model_tracks.package.runtime_snapshot_files',
        lambda **kwargs: {path.relative_to(tmp_path).as_posix():path
                          for path in (tmp_path/'config').glob('*')})
    cfg = SimpleNamespace(**settings, model_dump=lambda: settings.copy())
    inputs = {'text': dict(bundle_sha256='sha', payload='full', masking_profile='baseline', rows=1)}
    before = suite_identity(cfg, inputs, 'run')
    (setup / 'preflight.json').write_text('new report')
    (setup / 'suite.yaml').write_text('portable path')
    settings['text_bundle'] = 'new/bundle'
    assert suite_identity(cfg, inputs, 'run') == before
    lane = yaml.safe_load((setup / 'text.yaml').read_text())
    lane['hnsw_m'] = 32
    (setup / 'text.yaml').write_text(yaml.safe_dump(lane))
    assert suite_identity(cfg, inputs, 'run') != before
    lane['hnsw_m'] = 16
    (setup / 'text.yaml').write_text(yaml.safe_dump(lane))
    assert suite_identity(cfg, inputs, 'run') == before
    (setup / 'eligible_catalog.csv').write_text('changed')
    assert suite_identity(cfg, inputs, 'run') != before


def test_resume_runs_only_unfinished_worker_with_fresh_barrier(tmp_path):
    import os
    import sys
    from pathlib import Path
    from model_tracks.parallel import run_parallel
    old = tmp_path / 'barrier'
    old.mkdir()
    (old / 'start').write_text('stale attempt')
    (tmp_path / 'text__worker.log').write_text('previous attempt\n')
    code = '''import os,pathlib
from model_tracks.parallel import wait_for_start
root=pathlib.Path(os.environ['ER_TRACK_BARRIER'])
assert root.name != 'barrier'
assert not (root/'start').exists()
wait_for_start(root,'text',timeout=10)
print('resumed text')
'''
    result = run_parallel({'text': [sys.executable, '-c', code]}, tmp_path,
                          {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')},
                          resume=True, barrier_timeout=10)
    assert result['workers'] == ['text']
    log = (tmp_path / 'text__worker.log').read_text()
    assert 'previous attempt' in log and 'resumed text' in log


def test_suite_archive_reuse_requires_current_worker_generation(tmp_path):
    from core.portable_archive import write_archive
    from model_tracks.resume import TRACKS, verify_suite_archive
    output = tmp_path / 'run'
    output.mkdir()
    identity = {'run_tag': 'run', 'inputs': {'sha': 'original'}}
    (output / 'suite_manifest.json').write_text(json.dumps({'resume_identity': identity}))
    for track in TRACKS:
        folder = output / track
        folder.mkdir()
        (folder / 'checkpoint.bin').write_bytes(b'original')
        record_completion(folder, track, postprocess_complete=False)
    archive = tmp_path / 'run.zip'
    write_archive(archive, {path.relative_to(output).as_posix(): path
                           for path in output.rglob('*') if path.is_file()},
                  manifest_name='suite_bundle_manifest.json', metadata={'run_tag': 'run'})
    verify_suite_archive(archive, output, 'run', identity, postprocess_complete=False)
    (output / 'text/checkpoint.bin').write_bytes(b'retrained')
    record_completion(output / 'text', 'text', postprocess_complete=False)
    with pytest.raises(ValueError, match='stale worker artifacts'):
        verify_suite_archive(archive, output, 'run', identity, postprocess_complete=False)


def test_archive_is_not_published_when_source_changes_during_write(tmp_path, monkeypatch):
    import zipfile
    from core.portable_archive import write_archive
    source = tmp_path / 'checkpoint.bin'
    source.write_bytes(b'original')
    output = tmp_path / 'run.zip'
    original_write = zipfile.ZipFile.write
    def mutate(self, filename, *args, **kwargs):
        source.write_bytes(b'changed')
        return original_write(self, filename, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'write', mutate)
    with pytest.raises(ValueError, match='archive integrity mismatch'):
        write_archive(output, {'checkpoint.bin':source}, manifest_name='manifest.json', metadata={})
    assert not output.exists()
    assert not list(tmp_path.glob('*.partial-*'))
