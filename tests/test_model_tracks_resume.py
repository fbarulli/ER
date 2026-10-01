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


def test_identity_ignores_generated_reports_and_portable_suite_paths(tmp_path, monkeypatch):
    import core.common
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    setup = tmp_path / 'setup'
    setup.mkdir()
    (setup / 'eligible_catalog.csv').write_text('frozen')
    (tmp_path / 'config').mkdir()
    for name in ('training.yaml', 'identity_dimensions.yaml', 'identity_reviews.json', 'vocabulary.json'):
        (tmp_path / 'config' / name).write_text('same')
    settings = dict(setup_dir='setup', text_bundle='old/bundle', epochs=1)
    cfg = SimpleNamespace(**settings, model_dump=lambda: settings.copy())
    inputs = {'text': dict(bundle_sha256='sha', payload='full', masking_profile='baseline', rows=1)}
    before = suite_identity(cfg, inputs, 'run')
    (setup / 'preflight.json').write_text('new report')
    (setup / 'suite.yaml').write_text('portable path')
    settings['text_bundle'] = 'new/bundle'
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
