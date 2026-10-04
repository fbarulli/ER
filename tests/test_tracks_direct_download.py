import json
import zipfile
import pytest
from graph_tracks.data import file_hash


@pytest.mark.parametrize('publishing,released,corrupt', [(True, True, False), (False, True, False), (True, False, False), (True, True, True)])
def test_direct_download_verified_before_release_and_completion(tmp_path, monkeypatch, publishing, released, corrupt):
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
    monkeypatch.setattr(common, 'RESULTS', tmp_path)
    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(colab, 'verify', lambda _: dict(revision='revision', files={}))
    monkeypatch.setattr(colab, 'verify_archive', lambda *a: dict(run_tag='run'))
    monkeypatch.setattr(backend, 'GPU', 'CPU')
    monkeypatch.setattr(backend, '_env_value', lambda _: None)
    monkeypatch.setattr(backend, '_wandb_env_script', lambda: '')
    def stage(name, command, **kw):
        assert 'dvc' not in command[-1].lower()
        compile(command[-1], '<remote-stage>', 'exec')
        events.append('train')
    monkeypatch.setattr(backend, 'run_detached_stage', stage)
    monkeypatch.setattr(backend, '_read_remote_text', lambda _: file_hash(payload))
    monkeypatch.setattr(backend, 'run_colab_exec_capture', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    def download(remote, target):
        events.append('download')
        target.write_bytes(b'corrupt' if corrupt else payload.read_bytes())
    monkeypatch.setattr(backend, '_download_one_remote_file', download)
    monkeypatch.setattr(backend, 'stop', lambda: events.append('stop') or released)
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    monkeypatch.setattr(resume, 'validate_archived_track', lambda *a, **kw: events.append('validate'))
    monkeypatch.setattr(snapshot_completion, 'complete', lambda *a, **kw: events.append(('complete', kw)) or 'final.zip')
    if corrupt:
        with pytest.raises(ValueError, match='download mismatch'):
            colab.run(inputs, 'run', git_inputs=transport)
        assert events == ['train', 'download']
        return
    if not released:
        with pytest.raises(RuntimeError, match='termination could not be verified'):
            colab.run(inputs, 'run', git_inputs=transport)
        assert 'pull' not in events and 'complete' not in events
        return
    assert colab.run(inputs, 'run', git_inputs=transport) == 'final.zip'
    assert events == ['train', 'download', 'stop'] + ['validate'] * 3 + [('complete', {'publish': False})]


def test_suite_runtime_does_not_require_dvc_distribution(monkeypatch):
    from types import SimpleNamespace
    from graph_tracks import preflight
    cfg = SimpleNamespace(build_index=False, postprocess=False,
                          wandb=SimpleNamespace(mode='disabled'),
                          dvc=SimpleNamespace(enabled=True))
    def version(package):
        assert package != 'dvc'
        return 'installed'
    monkeypatch.setattr(preflight.importlib.metadata, 'version', version)
    assert 'dvc' not in preflight.runtime_versions(cfg, require_dvc=False)
