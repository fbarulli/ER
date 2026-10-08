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
    def upload(archive, tag, *, bundle=None):
        started.set()
        assert release.wait(5)
        # The sealing writer's verified handle rides the publish call: the
        # publisher reads the transport digest/run tag off it instead of
        # re-verifying bytes this process just hashed while writing.
        assert bundle is not None and bundle.path == archive
        with open_archive(archive) as snapshot:
            assert snapshot.read('checkpoint.pt') == b'completed checkpoint'
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
    def fail(*_args, **_kwargs):
        raise RuntimeError('upload failed')
    monkeypatch.setattr(incremental, 'persist_results', fail)
    # An empty generation is an explicit no-op (an empty bundle is refused by
    # Bundle.seal_archive), so the failure this test propagates must come from a
    # generation that actually has bytes.
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'completed checkpoint')
    publisher = incremental.ArtifactPublisher(tmp_path)
    publisher.submit('checkpoint-1', [])
    assert not (tmp_path/'_artifact_publications').exists(), \
        'an empty generation publishes nothing'
    publisher.submit('checkpoint-1', [checkpoint])
    with pytest.raises(RuntimeError, match='upload failed'):
        publisher.close()
