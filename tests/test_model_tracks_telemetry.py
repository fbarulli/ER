import json

import pytest

from model_tracks.telemetry import WorkerEvents


def test_worker_events_are_durable_per_attempt_and_redact_secrets(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('DVC_API_KEY', 'secret-value-123')
    first = WorkerEvents(tmp_path, 'text', 'run-text')
    first.emit('command', 'prepared', detail='secret-value-123 Bearer abcdef123 api_key=foo')
    second = WorkerEvents(tmp_path, 'text', 'run-text')
    second.emit('training', 'started')
    rows = [json.loads(line) for line in first.path.read_text().splitlines()]
    assert rows[0]['attempt'] != rows[1]['attempt']
    assert rows[0]['track'] == 'text' and rows[0]['run_tag'] == 'run-text'
    assert rows[0]['timestamp'].endswith('+00:00')
    logs = first.path.read_text() + capsys.readouterr().out
    assert 'secret-value-123' not in logs and 'abcdef123' not in logs
    assert 'api_key=foo' not in logs


@pytest.mark.parametrize('track', ['text', 'gnn_only', 'cascade'])
def test_worker_failure_records_traceback_then_reraises(tmp_path, monkeypatch, track):
    from model_tracks import worker
    monkeypatch.setenv('EUROMONITOR_RESULTS_DIR', str(tmp_path))
    def fail(*_, **__):
        raise RuntimeError('failed input validation')
    monkeypatch.setattr(worker, '_run', fail)
    with pytest.raises(RuntimeError, match='failed input validation'):
        worker.run(tmp_path / 'suite.yaml', track, f'run-{track}')
    rows = [json.loads(line) for line in (tmp_path / 'worker_events.jsonl').read_text().splitlines()]
    assert [row['phase'] for row in rows] == ['input_validation', 'failure']
    assert rows[-1]['error_type'] == 'RuntimeError'
    assert 'RuntimeError: failed input validation' in rows[-1]['traceback']


def test_supervisor_preflight_failure_retains_final_event_log(tmp_path, monkeypatch):
    from model_tracks import run
    from types import SimpleNamespace
    monkeypatch.setattr(run, 'load_config', lambda _: SimpleNamespace(dvc_enabled=False, profiling=False, result_archive_format='zip', post_training_ablation=False))
    # Accepts the call's keyword arguments: run.py passes
    # `allow_gpu_pending=cfg.post_training_ablation` so the CPU-only
    # preparation preflight may tolerate the declared GPU-pending hybrid cache.
    # The stub only needs to fail.
    def fail(*_args, **_kwargs):
        raise ValueError('stale frozen input')
    monkeypatch.setattr(run, 'preflight', fail)
    output = tmp_path / 'suite'
    with pytest.raises(ValueError, match='stale frozen input'):
        run.run(tmp_path / 'suite.yaml', output, 'suite')
    saved = output.with_suffix('.events.jsonl')
    assert saved.read_bytes() == (output / 'suite_events.jsonl').read_bytes()
    rows = [json.loads(line) for line in saved.read_text().splitlines()]
    assert [(row['phase'], row['status']) for row in rows] == [
        ('suite', 'starting'), ('preflight', 'starting'), ('suite', 'failed')]
    assert 'ValueError: stale frozen input' in rows[-1]['traceback']


def test_failure_logs_collect_without_suite_manifest(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import core.common
    from model_tracks.colab import TracksLane
    monkeypatch.setattr(core.common, 'RESULTS', tmp_path)
    contents = b'{"phase":"preflight","status":"failed"}\n'
    backend = SimpleNamespace(SESSION='session',
        run_colab_exec_capture=lambda *a, **k: json.dumps(['suite_events.jsonl']),
        _download_one_remote_file=lambda remote, target: target.write_bytes(contents))
    TracksLane(archive=tmp_path / 'inputs.tar.zst', run_tag='run').collect_failure_logs(
        backend, '/remote/suite')
    assert (tmp_path / 'model_tracks/run__logs/suite_events.jsonl').read_bytes() == contents
