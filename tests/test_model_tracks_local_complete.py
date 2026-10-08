import json
from pathlib import Path

import pytest
import yaml

from core.portable_archive import write_archive, verify_archive
from model_tracks.local_complete import complete
from model_tracks.resume import record_completion


def _manifest(path, track, report_test):
    """One calibrated track report manifest per the shared honesty contract."""
    from graph_tracks.report_manifest import build as build_manifest, write as write_manifest
    write_manifest(path, build_manifest(
        track=track, checkpoint='checkpoint-1/model.pt',
        checkpoint_sha256='0' * 64, listings_sha256='1' * 64, pairs_sha256='2' * 64,
        threshold=0.5, threshold_source='dev_youden', test_reported=bool(report_test),
        model_selection='dev_pr_auc', retrieval_ks=[10]))


def suite(tmp_path, monkeypatch, post_training_ablation=False, archive_format='zip'):
    from core import common
    from graph_tracks import preflight, report
    from model_tracks import text_report
    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    cfg = {'setup_dir': 'data/model_tracks/shared',
           'text_bundle': 'data/model_tracks/shared/text.pkl',
           'publish_git': False, 'publish_dvc': False, 'result_archive_format': archive_format,
           'post_training_ablation': post_training_ablation}
    inline = {'data/model_tracks/suite.yaml': yaml.safe_dump(cfg)}
    for track in ('gnn_only', 'cascade'):
        settings = {'track': track, 'listings': 'data/model_tracks/shared/prepared/listings.json',
                    'pairs': 'data/model_tracks/shared/prepared/pairs.csv',
                    'output_dir': 'results/graph_tracks'}
        if track == 'cascade':
            settings['text_index'] = 'results/graph_tracks/text__index'
            settings['gnn_checkpoint'] = 'results/graph_tracks/gnn_only__best_checkpoint.json'
        inline[f'data/model_tracks/shared/{track}.yaml'] = yaml.safe_dump(settings)
    inputs = {'text': {'bundle_sha256': 'bundle'}}
    input_zip = write_archive(tmp_path / 'input.zip', {}, inline=inline,
                              manifest_name='model_tracks_package.json', metadata={'preflight': inputs})
    root = tmp_path / 'remote'
    root.mkdir()
    (root / 'suite_manifest.json').write_text(json.dumps({'run_tag': 'run', 'inputs': inputs, 'config': cfg,
        'resume_identity': {'implementation': {}}}))
    for track in ('text', 'gnn_only', 'cascade'):
        output = root / track
        output.mkdir()
        checkpoint = output / 'checkpoint-1'
        checkpoint.mkdir()
        (checkpoint / 'model.pt').write_bytes(b'trained')
        if track != 'text':
            (output / f'{track}__best_checkpoint.json').write_text(json.dumps({'path': '/remote/checkpoint-1/model.pt'}))
        record_completion(output, track, postprocess_complete=False)
    training_zip = write_archive(tmp_path / f'run.training.{archive_format}',
        {p.relative_to(root).as_posix(): p for p in root.rglob('*') if p.is_file()},
        manifest_name='suite_bundle_manifest.json', metadata={'run_tag': 'run'})
    calls = []
    def text(output, setup, *, device, report_test):
        assert device == 'cpu'
        calls.append('text')
        (output / 'text__training_report.md').write_text('local text report')
        _manifest(output / 'text__completion_manifest.json', 'text', report_test)
    def graph(checkpoint, listings, pairs, output, cfg, **kwargs):
        assert checkpoint.read_bytes() == b'trained'
        assert cfg.device == 'cpu'
        assert listings.is_relative_to(tmp_path / 'run/local_inputs')
        calls.append(cfg.track)
        (output / 'report.md').write_text('local graph report')
        _manifest(output / f'{cfg.track}__report_manifest.json', cfg.track, cfg.report_test)
    monkeypatch.setattr(text_report, 'complete', text)
    monkeypatch.setattr(report, 'complete', graph)
    monkeypatch.setattr(preflight, 'preflight', lambda *_args, **_kwargs: {})
    return training_zip, input_zip, calls, graph


def test_downloaded_checkpoints_complete_locally_without_retraining(tmp_path, monkeypatch):
    training_zip, input_zip, calls, _ = suite(tmp_path, monkeypatch)
    final = complete(training_zip, input_zip, 'run')
    assert calls == ['text', 'gnn_only', 'cascade']
    metadata = verify_archive(final, 'suite_bundle_manifest.json')
    assert metadata['postprocess_location'] == 'local CPU'
    assert all(json.loads((tmp_path / 'run' / track / 'track_complete.json').read_text())['postprocess_complete']
               for track in ('text', 'gnn_only', 'cascade'))
    assert complete(training_zip, input_zip, 'run') == final
    assert len(calls) == 3


def test_local_report_failure_is_retryable_without_retraining(tmp_path, monkeypatch):
    from graph_tracks import report
    training_zip, input_zip, calls, graph = suite(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError('report interrupted')
    monkeypatch.setattr(report, 'complete', fail)
    with pytest.raises(RuntimeError, match='report interrupted'):
        complete(training_zip, input_zip, 'run')
    assert not (tmp_path / 'run.zip').exists()
    monkeypatch.setattr(report, 'complete', graph)
    complete(training_zip, input_zip, 'run')
    assert calls == ['text', 'gnn_only', 'cascade']


def test_interrupted_text_report_preserves_future_artifacts(tmp_path, monkeypatch):
    from model_tracks import text_report
    training_zip, input_zip, calls, _ = suite(tmp_path, monkeypatch)
    attempts = []
    def flaky(output, setup, *, device, report_test):
        attempts.append(len(attempts) + 1)
        # an artifact outside the old fixed allowlist
        (output / 'text__future_artifact.json').write_text(f'attempt {len(attempts)}')
        (output / 'text__training_report.md').write_text(f'attempt {len(attempts)}')
        _manifest(output / 'text__completion_manifest.json', 'text', report_test)
        if len(attempts) == 1:
            raise RuntimeError('text report interrupted')
    monkeypatch.setattr(text_report, 'complete', flaky)
    with pytest.raises(RuntimeError, match='text report interrupted'):
        complete(training_zip, input_zip, 'run')
    assert calls == []  # the text lane failed before the graph lanes ran
    complete(training_zip, input_zip, 'run')
    text_dir = tmp_path / 'run' / 'text'
    preserved = [p.name for p in text_dir.glob('interrupted-*')]
    # every prior-attempt text__* artifact is preserved, not mixed
    assert any(name.endswith('text__future_artifact.json') for name in preserved)
    assert any(name.endswith('text__training_report.md') for name in preserved)
    assert (text_dir / 'text__future_artifact.json').read_text() == 'attempt 2'
    assert (text_dir / 'text__training_report.md').read_text() == 'attempt 2'


def test_completion_runs_configured_post_training_ablation(tmp_path,monkeypatch):
    from model_tracks import local_complete, post_training_ablation
    calls = []
    def complete_saved(destination, settings, *, publisher=None):
        calls.append(('complete', settings))
        for track in ('text', 'gnn_only', 'cascade'):
            folder = destination / track / 'ablation'; folder.mkdir(parents=True)
            (folder / 'report.json').write_text(json.dumps({
                'track': track, 'request_sha256': '0' * 64, 'result_sha256': '0' * 64,
                'threshold': 0.5, 'threshold_provenance': {'dev': True},
                'threshold_binding': {'track': track, 'checkpoint_sha256': '0' * 64, 'verified': True},
                'rows': []}))
        return destination
    monkeypatch.setattr(post_training_ablation, 'complete_saved', complete_saved)
    monkeypatch.setattr(post_training_ablation, 'publish_saved',
                        lambda destination, suite, *, archive: calls.append(('publish', archive)))
    training_zip, input_zip, _, _ = suite(tmp_path, monkeypatch, post_training_ablation=True)
    final = local_complete.complete(training_zip, input_zip, 'run')
    # the ablation runs once, before the publication archive is sealed
    assert [c[0] for c in calls] == ['complete', 'publish']
    assert calls[0][1].post_training_ablation is True
    assert calls[1][1] == final


def test_completed_archive_restores_inputs_and_reports_before_publish(tmp_path,monkeypatch):
    import shutil
    from model_tracks import local_complete
    training_zip,input_zip,calls,_ = suite(tmp_path,monkeypatch)
    final = complete(training_zip,input_zip,'run')
    shutil.rmtree(tmp_path/'run')
    def publish(archive,settings,run_tag,*,ablation_done=False):
        assert (tmp_path/'run/text/text__training_report.md').is_file()
        assert (tmp_path/'run/local_inputs/data/model_tracks/suite.yaml').is_file()
        return archive
    monkeypatch.setattr(local_complete,'_publish',publish)
    assert complete(training_zip,input_zip,'run') == final
    assert calls == ['text','gnn_only','cascade']


def test_zstandard_checkpoints_complete_locally_and_remain_retryable(tmp_path, monkeypatch):
    training, inputs, calls, _ = suite(tmp_path, monkeypatch, archive_format='tar.zst')
    final = complete(training, inputs, 'run')
    assert final.name == 'run.tar.zst'
    assert verify_archive(final, 'suite_bundle_manifest.json')['postprocess_location'] == 'local CPU'
    assert calls == ['text', 'gnn_only', 'cascade']
    assert complete(training, inputs, 'run') == final
    assert len(calls) == 3
