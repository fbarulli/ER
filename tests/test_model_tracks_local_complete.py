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


def suite(tmp_path, monkeypatch, post_training_ablation=False, archive_format='zip',
          cascade_complete=False, ablation_skip_event=False, with_runtime_source=False):
    from core import common
    from graph_tracks import preflight, report
    # Bind the graph modules that import ``load_records``/``load_pairs`` BY VALUE
    # (module-level ``from graph_tracks.data import load_records``) before the
    # fixture patches the shared module attributes, so the stub never leaks into
    # a later test's already-imported reference (test isolation).
    from graph_tracks import prepare as _prepare, prepared_inputs as _prepared_inputs  # noqa: F401
    from model_tracks import text_report, worker
    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    cfg = {'setup_dir': 'data/model_tracks/shared',
           'text_bundle': 'data/model_tracks/shared/text.pkl',
           'publish_git': False, 'publish_dvc': False, 'result_archive_format': archive_format,
           'post_training_ablation': post_training_ablation}
    inline = {'data/model_tracks/suite.yaml': yaml.safe_dump(cfg),
              # The prepared inputs the lanes and the cascade manifest read.
              'data/model_tracks/shared/prepared/listings.json': '{"listings": []}',
              'data/model_tracks/shared/prepared/pairs.csv': 'sku_id1,sku_id2,label,split\n'}
    if with_runtime_source:
        # The frozen completion runtime the snapshot wrapper materializes.
        inline['src/model_tracks/local_complete.py'] = '# frozen runtime\n'
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
    if ablation_skip_event:
        # A GPU session that shipped no ablation templates records the skip.
        (root / 'suite_events.jsonl').write_text(json.dumps({
            'phase': 'attribute_ablation_export', 'status': 'skipped',
            'reason': 'bundle shipped no ablation templates'}) + '\n')
    for track in ('text', 'gnn_only', 'cascade'):
        output = root / track
        output.mkdir()
        checkpoint = output / 'checkpoint-1'
        checkpoint.mkdir()
        (checkpoint / 'model.pt').write_bytes(b'trained')
        if track != 'text':
            (output / f'{track}__best_checkpoint.json').write_text(json.dumps({'path': '/remote/checkpoint-1/model.pt'}))
        if track == 'cascade' and cascade_complete:
            # A finished cascade already ships the composed report and its
            # calibrated manifest; local completion leaves both alone.
            (output / 'cascade__cascade_report.json').write_text('{}')
            _manifest(output / 'cascade__report_manifest.json', 'cascade', False)
        # The trained lanes deferred their CPU report (the GPU-only contract);
        # the cascade did or did not finish composing.
        record_completion(output, track,
                          postprocess_complete=(track == 'cascade') and cascade_complete)
    training_zip = write_archive(tmp_path / f'run.training.{archive_format}',
        {p.relative_to(root).as_posix(): p for p in root.rglob('*') if p.is_file()},
        manifest_name='suite_bundle_manifest.json', metadata={'run_tag': 'run'})
    calls = []
    def text(output, setup, *, device, report_test):
        assert device == 'cpu'
        calls.append('text')
        (output / 'text__training_report.md').write_text('local text report')
        # The text lane's saved catalog export: the cascade's ranker input.
        (output / 'text__index').mkdir()
        (output / 'text__vectors.npz').write_bytes(b'text-vectors')
        _manifest(output / 'text__completion_manifest.json', 'text', report_test)
    def graph(checkpoint, listings, pairs, output, cfg, **kwargs):
        assert checkpoint.read_bytes() == b'trained'
        assert cfg.device == 'cpu'
        assert listings.is_relative_to(tmp_path / 'run/local_inputs')
        calls.append(cfg.track)
        (output / 'report.md').write_text('local graph report')
        # The gnn lane's saved forward export: the cascade's decider input.
        inference = output / f'{cfg.track}__inference'
        inference.mkdir()
        (inference / f'{cfg.track}__vectors.npz').write_bytes(b'gnn-vectors')
        _manifest(output / f'{cfg.track}__report_manifest.json', cfg.track, cfg.report_test)

    def cascade_roles(records, pairs, artifacts):
        return 'RANKED', ['relevant'], 'DECISIONS'

    def cascade_report(ranked, relevant, decisions, output, *, track, ks):
        assert (ranked, relevant, decisions) == ('RANKED', ['relevant'], 'DECISIONS')
        assert track == 'cascade'
        calls.append('cascade')
        (output / 'cascade__cascade_report.json').write_text('{}')

    monkeypatch.setattr(worker, '_cascade_roles', cascade_roles)
    monkeypatch.setattr('graph_tracks.report.report_cascade', cascade_report)
    # The cascade branch loads the suite's frozen pair/listings inputs for real;
    # this fixture ships them empty, so the loaders are stubbed to their shape.
    monkeypatch.setattr('graph_tracks.data.load_records', lambda path: [])
    monkeypatch.setattr('graph_tracks.train.load_pairs', lambda path, records: {})
    monkeypatch.setattr(text_report, 'complete', text)
    monkeypatch.setattr(report, 'complete', graph)
    monkeypatch.setattr(preflight, 'preflight', lambda *_args, **_kwargs: {})
    return training_zip, input_zip, calls, graph

def test_cascade_completes_locally_by_composing_the_trained_lanes(tmp_path, monkeypatch):
    """An interrupted cascade is re-composed, never re-trained.

    The cascade has no checkpoint, so the trained-lane branch cannot apply: its
    completion composes the text ranker export and the gnn_only decider export
    that the trained lanes already wrote, and identifies itself by that scorer.
    """
    from graph_tracks.data import file_hash
    training_zip, input_zip, calls, _ = suite(tmp_path, monkeypatch, cascade_complete=False)
    final = complete(training_zip, input_zip, 'run')
    assert calls == ['text', 'gnn_only', 'cascade']
    cascade = tmp_path / 'run' / 'cascade'
    assert json.loads((cascade / 'track_complete.json').read_text())['postprocess_complete'] is True
    report = json.loads((cascade / 'cascade__report_manifest.json').read_text())
    assert report['track'] == 'cascade'
    assert report['checkpoint_sha256'] == file_hash(cascade / 'checkpoint-1/model.pt')


def test_a_completed_cascade_is_never_recomposed(tmp_path, monkeypatch):
    training_zip, input_zip, calls, _ = suite(tmp_path, monkeypatch, cascade_complete=True)
    complete(training_zip, input_zip, 'run')
    assert calls == ['text', 'gnn_only']


def test_verified_cascade_archive_needs_no_ablation(tmp_path, monkeypatch):
    """``post_training_ablation=True`` covers the trained lanes, not the cascade."""
    from model_tracks import archive_verification
    from model_tracks.config import SuiteConfig
    training_zip, input_zip, _, _ = suite(tmp_path, monkeypatch, cascade_complete=True,
                                         post_training_ablation=True,
                                         ablation_skip_event=True)
    final = complete(training_zip, input_zip, 'run')
    settings = SuiteConfig.model_validate({'setup_dir': 'data/model_tracks/shared',
                                           'text_bundle': 'data/model_tracks/shared/text.pkl',
                                           'publish_git': False, 'publish_dvc': False,
                                           'result_archive_format': 'zip',
                                           'post_training_ablation': True})
    result = archive_verification.verification_result(final, 'run', settings=settings)
    assert result['status'] == 'verified', result.get('error')
    assert set(result['tracks']) == {'text', 'gnn_only', 'cascade'}
    assert 'ablation' not in result['tracks']['cascade']


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
        # A real retry rebuilds the ranker export its consumers read.
        (output / 'text__index').mkdir(exist_ok=True)
        (output / 'text__vectors.npz').write_bytes(b'text-vectors')
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
                        lambda destination, suite, *, archive, bundle=None: calls.append(('publish', archive)))
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
    def publish(archive,settings,run_tag,*,ablation_done=False,bundle=None):
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


def test_legacy_source_pin_uses_the_runtime_inventory_ssot(tmp_path, monkeypatch):
    """Legacy mode pins code/config/scripts, never the gitignored registry.

    The pinned surface is ``resume.runtime_source_inventory`` minus the registry:
    a changed runtime member fails loud, a changed registry is ignored, and the
    whole check is a no-op outside legacy mode.
    """
    from types import SimpleNamespace

    from core.bundle import _bundle_spec
    from graph_tracks.data import file_hash
    from model_tracks import local_complete
    from model_tracks.config import SuiteConfig

    monkeypatch.setattr('core.common.TRAIN_ROOT', tmp_path)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'a.py').write_text('code')
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts' / 'registry.json').write_text('registry')
    settings = SuiteConfig.model_validate({
        'setup_dir': 's', 'text_bundle': 'b',
        'publish_git': False, 'publish_dvc': False})
    files = {
        'src/a.py': file_hash(tmp_path / 'src' / 'a.py'),
        # Present but deliberately wrong: the registry is not on the pin surface.
        'artifacts/registry.json': '0' * 64,
    }
    inputs = SimpleNamespace(manifest={_bundle_spec().files_key: files})

    monkeypatch.setattr('core.perf_switches.legacy_mode', lambda: False)
    local_complete._require_legacy_source_pin(inputs, settings)  # no-op outside legacy

    monkeypatch.setattr('core.perf_switches.legacy_mode', lambda: True)
    local_complete._require_legacy_source_pin(inputs, settings)  # registry drift ignored

    files['src/a.py'] = '1' * 64
    with pytest.raises(ValueError, match='differs from training'):
        local_complete._require_legacy_source_pin(inputs, settings)


def test_snapshot_completion_checks_the_final_archive_exactly_once(tmp_path, monkeypatch):
    """The frozen-runtime wrapper reuses the boundary handle it already holds.

    ``snapshot_completion.complete`` verifies the final archive once, then hands
    that same handle to publication. Without the handle, publication re-loads
    (and re-hashes) the archive again in the same process, violating the
    one-integrity-check-per-VM-crossing contract.
    """
    from types import SimpleNamespace

    from core.bundle import Bundle
    from model_tracks import snapshot_completion

    training_zip, input_zip, _, _ = suite(tmp_path, monkeypatch, with_runtime_source=True)
    final = complete(training_zip, input_zip, 'run')

    loads = []
    real_load = Bundle.load.__func__

    def counting(cls, path, role, **kwargs):
        loads.append(str(path))
        return real_load(cls, path, role, **kwargs)

    monkeypatch.setattr(Bundle, 'load', classmethod(counting))

    def fake_frozen_run(command, cwd=None, env=None, check=False):
        Path(command[6]).write_text(json.dumps({'final': str(final)}))
        assert Path(cwd).is_dir()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(snapshot_completion.subprocess, 'run', fake_frozen_run)

    assert snapshot_completion.complete(training_zip, input_zip, 'run') == final
    # Two boundary loads for the two transported archives, exactly one for the
    # freshly sealed final archive (the load that produces the receipt).
    assert loads.count(str(final)) == 1, loads
