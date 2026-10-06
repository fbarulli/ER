import threading
from core.archive_reader import open_archive, archive_sidecar

import pytest

from model_tracks import incremental


def test_upload_starts_before_close_and_uses_immutable_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv('ER_INCREMENTAL_DVC', '1')
    monkeypatch.setenv('EUROMONITOR_RUN_ID', 'suite-gnn_only')
    started, release = threading.Event(), threading.Event()
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'completed checkpoint')
    def upload(archive, tag):
        started.set()
        assert release.wait(5)
        with open_archive(archive) as bundle:
            assert bundle.read('checkpoint.pt') == b'completed checkpoint'
        assert tag == 'suite-gnn_only-checkpoint-1'
        archive_sidecar(archive, '.publication').mkdir()
    monkeypatch.setattr(incremental, 'persist_results', upload)
    publisher = incremental.ArtifactPublisher(tmp_path)
    try:
        publisher.submit('checkpoint-1', [checkpoint])
        assert started.wait(5)
        checkpoint.write_bytes(b'changed after save')
    finally:
        release.set()
        publisher.close()


def test_upload_failure_propagates_and_disabled_smoke_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setenv('ER_INCREMENTAL_DVC', '0')
    with incremental.ArtifactPublisher(tmp_path) as publisher:
        publisher.submit('checkpoint-1', [])
    assert not (tmp_path/'_artifact_publications').exists()
    monkeypatch.setenv('ER_INCREMENTAL_DVC', '1')
    monkeypatch.setenv('EUROMONITOR_RUN_ID', 'suite-text')
    def fail(*_):
        raise RuntimeError('upload failed')
    monkeypatch.setattr(incremental, 'persist_results', fail)
    publisher = incremental.ArtifactPublisher(tmp_path)
    publisher.submit('checkpoint-1', [])
    with pytest.raises(RuntimeError, match='upload failed'):
        publisher.close()
