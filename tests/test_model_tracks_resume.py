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
    # Owner order 2026-10-07: the implementation inventory stays a recorded
    # field but is never compared — a code-only difference cannot block resume.
    recorded = dict(identity, implementation={'src/model_tracks/run.py': 'a' * 64})
    fresh = dict(identity, implementation={'src/model_tracks/run.py': 'b' * 64})
    (tmp_path / 'suite_manifest.json').write_text(json.dumps({'resume_identity': recorded}))
    validate_suite(tmp_path, fresh)
    (tmp_path / 'track_complete.json').write_text(json.dumps(
        {'track': 'text', 'status': 'ok', 'postprocess_complete': True}))
    with pytest.raises(ValueError, match='inventory missing'):
        completed_track(tmp_path, 'text')


def test_graph_resume_selects_latest_epoch_in_its_own_run(tmp_path):
    for run, epoch in [('run-gnn_only', 2), ('run-gnn_only', 10), ('other-gnn_only', 99)]:
        folder = tmp_path / '_checkpoints/gnn_only' / f'{run}_f0' / f'checkpoint-{epoch}'
        folder.mkdir(parents=True)
        (folder / 'gnn_only__graph_model.pt').write_bytes(b'checkpoint')
    assert graph_checkpoint(tmp_path, 'gnn_only', 'run-gnn_only').parent.name == 'checkpoint-10'
    assert graph_checkpoint(tmp_path, 'cascade', 'run-cascade') is None


def test_graph_resume_finds_trainer_nested_worker_output(tmp_path):
    root = tmp_path / 'gnn_only__run-gnn_only/_checkpoints/gnn_only/run-gnn_only_f0/checkpoint-3'
    root.mkdir(parents=True)
    checkpoint = root / 'gnn_only__graph_model.pt'
    checkpoint.write_bytes(b'checkpoint')
    assert graph_checkpoint(tmp_path, 'gnn_only', 'run-gnn_only') == checkpoint


def test_frozen_setup_filenames_come_from_the_declared_layout():
    """``suite_identity`` freezes the DECLARED prepared-setup filenames.

    The five top-level frozen names used to be spelled inline; they now come
    from ``training.preparation.graph_setup`` — the same declaration the
    producers (``graph_tracks.setup`` / ``model_tracks.package``) write. A
    drifted literal would freeze a file nobody writes and miss the real one, so
    the declaration must still name exactly those files.
    """
    from core.common import training_cfg
    from model_tracks import resume

    layout = resume._setup_layout()
    assert layout == training_cfg().preparation.graph_setup
    assert {layout.catalog, layout.splits, layout.pairs,
            layout.shared_embeddings, layout.manifest} == {
        'eligible_catalog.csv', 'listing_splits.csv', 'listing_pairs.csv',
        'shared_minilm__embeddings.npz', 'setup_manifest.json'}


def test_identity_ignores_generated_reports_and_portable_suite_paths(tmp_path, monkeypatch):
    import core.common
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    setup = tmp_path / 'setup'
    setup.mkdir()
    (setup / 'eligible_catalog.csv').write_text('frozen')
    import yaml
    for track in ('gnn_only', 'cascade'):
        lane = {'track': track, 'listings': 'data/listings.json', 'pairs': 'data/pairs.csv',
                'output_dir': 'results/graph_tracks'}
        if track == 'cascade':
            lane['text_index'] = 'results/graph_tracks/text__index'
            lane['gnn_checkpoint'] = 'results/graph_tracks/gnn_only__best_checkpoint.json'
        (setup / (track + '.yaml')).write_text(yaml.safe_dump(lane))
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
    inputs = {'text': dict(bundle_size='sha', payload='full', masking_profile='baseline', rows=1)}
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


def test_suite_archive_reuse_keys_on_identity_not_freshness(tmp_path):
    from core.portable_archive import write_archive
    from model_tracks.resume import TRACKS, expected_postprocess, verify_suite_archive
    output = tmp_path / 'run'
    output.mkdir()
    identity = {'run_tag': 'run', 'inputs': {'sha': 'original'}}
    (output / 'suite_manifest.json').write_text(json.dumps({'resume_identity': identity}))
    for track in TRACKS:
        folder = output / track
        folder.mkdir()
        (folder / 'checkpoint.bin').write_bytes(b'original')
        record_completion(folder, track,
                          postprocess_complete=expected_postprocess(track, gpu_only=True))
    archive = tmp_path / 'run.zip'
    write_archive(archive, {path.relative_to(output).as_posix(): path
                           for path in output.rglob('*') if path.is_file()},
                  manifest_name='suite_bundle_manifest.json', metadata={'run_tag': 'run'})
    assert verify_suite_archive(archive, output, 'run', identity, gpu_only=True) is not None
    # A retrained track is NOT a freshness verdict: reuse keys on the run tag,
    # the suite provenance and the archive's own bundle digest (owner directive
    # 2026-10-08; the output tree is not compared against the archive).
    (output / 'text/checkpoint.bin').write_bytes(b'retrained')
    record_completion(output / 'text', 'text',
                      postprocess_complete=expected_postprocess('text', gpu_only=True))
    assert verify_suite_archive(archive, output, 'run', identity, gpu_only=True) is not None
