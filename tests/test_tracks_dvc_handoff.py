import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from graph_tracks.data import file_hash
from model_tracks import dvc_handoff


def receipt():
    return dict(schema='er-training-dvc-handoff-v1', run_tag='run',
                archive_name='run.zip', archive_size=5, archive_sha256='digest',
                remote='https://example.test/data', verified_download=True,
                pointer='outs:\n- path: run.zip\n  md5: abc\n')


@pytest.mark.parametrize('change', [dict(run_tag='other'), dict(verified_download=False),
                                   dict(remote='https://wrong.test'), dict(archive_sha256='bad'),
                                   dict(pointer='outs:\n- path: ../run.zip\n  md5: abc\n')])
def test_handoff_rejects_untrusted_receipt(change):
    data = {**receipt(), **change}
    with pytest.raises(ValueError):
        dvc_handoff.validate(data, 'run', 'digest', 'https://example.test/data')


@pytest.mark.parametrize('publishing,released,bad_receipt,bad_pull', [
    (True, True, False, False), (False, True, False, False),
    (True, False, False, False), (True, True, True, False),
    (True, True, False, True)])
def test_launcher_push_release_pull_then_complete(tmp_path, monkeypatch, publishing, released, bad_receipt, bad_pull):
    from core import common
    from cli import colab as backend
    from model_tracks import colab, resume, snapshot_completion
    import time
    events = []
    settings = dict(setup_dir='data/setup', text_bundle='data/text.pkl', device='cpu',
                    publish_dvc=publishing, publish_git=False)
    inputs = tmp_path / 'inputs.zip'
    with zipfile.ZipFile(inputs, 'w') as archive:
        archive.writestr('data/model_tracks/suite.yaml', json.dumps(settings))
    transport = tmp_path / 'transport.tar.gz'
    transport.write_bytes(b'inputs')
    payload = tmp_path / 'payload.zip'
    with zipfile.ZipFile(payload, 'w') as archive:
        archive.writestr('test', 'result')
    data = receipt()
    data.update(archive_sha256=file_hash(payload), archive_size=payload.stat().st_size)
    monkeypatch.setattr(common, 'RESULTS', tmp_path)
    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(common, 'training_cfg', lambda: SimpleNamespace(colab=SimpleNamespace(dvc_remote_url=data['remote'])))
    monkeypatch.setattr(colab, 'verify', lambda _: dict(revision='revision', files={}))
    monkeypatch.setattr(colab, 'verify_archive', lambda *a: dict(run_tag='run'))
    monkeypatch.setattr(backend, 'GPU', 'CPU')
    monkeypatch.setattr(backend, '_env_value', lambda _: 'token')
    monkeypatch.setattr(backend, '_remote_auth_env_script', lambda **kw: '')
    def stage(name, command, **kw):
        assert 'from model_tracks.dvc_handoff import publish' in command[-1]
        compile(command[-1], '<remote-stage>', 'exec')
        import ast
        tree = ast.parse(command[-1])
        # Publication belongs to its own top-level conditional AFTER the
        # existing-archive retry/new-training branch, so retries still push.
        publication = tree.body[-1]
        assert isinstance(publication, ast.If)
        assert publication.test.value is publishing
        assert isinstance(publication.body[-1], ast.Expr)
        assert publication.body[-1].value.func.id == 'publish'
        events.append('remote_push' if publishing else 'train')
    monkeypatch.setattr(backend, 'run_detached_stage', stage)
    if bad_receipt:
        data['run_tag'] = 'wrong'
    def remote_text(path):
        if path.endswith('.publication.json'):
            events.append('receipt')
            return json.dumps(data)
        return data['archive_sha256']
    monkeypatch.setattr(backend, '_read_remote_text', remote_text)
    monkeypatch.setattr(backend, 'run_colab_exec_capture', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    def download(remote, target):
        assert not publishing, 'publishing suites must not directly download the archive'
        events.append('download')
        target.write_bytes(payload.read_bytes())
    monkeypatch.setattr(backend, '_download_one_remote_file', download)
    monkeypatch.setattr(backend, 'stop', lambda: events.append('stop') or released)
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    def pull(data, destination):
        assert events[-1] == 'stop'
        events.append('pull')
        destination.write_bytes(b'corrupt' if bad_pull else payload.read_bytes())
    monkeypatch.setattr(dvc_handoff, 'pull', pull)
    monkeypatch.setattr(resume, 'validate_archived_track', lambda *a, **kw: events.append('validate'))
    monkeypatch.setattr(snapshot_completion, 'complete', lambda *a: events.append('complete') or 'final.zip')
    if bad_receipt or bad_pull:
        with pytest.raises(ValueError, match='handoff identity|collection mismatch'):
            colab.run(inputs, 'run', git_inputs=transport)
        assert 'complete' not in events
        if bad_receipt:
            assert 'pull' not in events
        return
    if not released:
        with pytest.raises(RuntimeError, match='termination could not be verified'):
            colab.run(inputs, 'run', git_inputs=transport)
        assert 'pull' not in events and 'complete' not in events
        return
    assert colab.run(inputs, 'run', git_inputs=transport) == 'final.zip'
    assert events == (['remote_push', 'receipt', 'stop', 'pull'] if publishing
                      else ['train', 'download', 'stop']) + ['validate'] * 3 + ['complete']


@pytest.mark.parametrize('corrupt', [False, True])
def test_pull_uses_clean_workspace_and_checks_download(tmp_path, monkeypatch, corrupt):
    from core.portable_archive import write_archive
    from training import dvc_store
    source = tmp_path / 'source.txt'
    source.write_text('trained checkpoint')
    archive = tmp_path / 'run.zip'
    write_archive(archive, {'source.txt': source}, manifest_name='suite_bundle_manifest.json',
                  metadata={'run_tag': 'run'})
    data = receipt()
    data.update(archive_sha256=file_hash(archive), archive_size=archive.stat().st_size)
    monkeypatch.setenv('DVC_API_KEY', 'test-token')
    events = []
    def configure(root, **kwargs):
        assert kwargs == dict(token='test-token', remote=data['remote'])
        assert not (root / 'run.zip').exists()
        events.append('configure')
    def run(command, root):
        assert command == ['dvc', 'pull', '--force', 'run.zip.dvc']
        assert (root / 'run.zip.dvc').read_text() == data['pointer']
        (root / 'run.zip').write_bytes(b'bad' if corrupt else archive.read_bytes())
        events.append('pull')
    monkeypatch.setattr(dvc_store, '_configure_workspace', configure)
    monkeypatch.setattr(dvc_store, '_run', run)
    destination = tmp_path / 'results/run.training.zip'
    if corrupt:
        with pytest.raises(ValueError, match='checksum mismatch'):
            dvc_handoff.pull(data, destination)
        assert not destination.exists()
    else:
        assert dvc_handoff.pull(data, destination) == destination
        assert destination.read_bytes() == archive.read_bytes()
    assert events == ['configure', 'pull']
