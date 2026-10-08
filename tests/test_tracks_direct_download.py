import json
import zipfile
import pytest
from graph_tracks.data import file_size


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
    # A REAL verified package: `colab.run` opens it through the ONE inputs
    # `Bundle.load` boundary (manifest + every member checked once), so a zip
    # holding only the suite config no longer passes. `revision` is the
    # package's own revision, which the checkout script fetches separately.
    from model_tracks.package import package_member as _member
    _members = {_member('suite_package_config'): json.dumps(settings).encode()}
    _inventory = {name: len(blob)
                  for name, blob in _members.items()}
    with zipfile.ZipFile(inputs, 'w') as archive:
        for name, blob in _members.items():
            archive.writestr(name, blob)
        # `revision` is required: colab.run's checkout script fetches and
        # detach-checks-out the PACKAGE's own revision, because the
        # immutable package can predate the commit that published its
        # transport, and a depth-one branch clone need not contain it.
        archive.writestr('model_tracks_package.json',
                         json.dumps({'files': _inventory, 'revision': 'revision'}))
    transport = tmp_path / 'transport.tar.gz'
    transport.write_bytes(b'inputs')
    payload = tmp_path / 'payload.zip'
    with zipfile.ZipFile(payload, 'w') as archive:
        archive.writestr('test', 'result')
    monkeypatch.setattr(common, 'RESULTS', tmp_path)
    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(colab, 'verify', lambda _: dict(revision='revision', files={}))
    # The result download trusts one named single-pass verify seam. Stub it so
    # a tiny fixture archive is never parsed as a real tar.zst; the returned
    # whole-file digest still drives the corruption check.
    monkeypatch.setattr(colab, 'verify_result_archive',
                        lambda path: (dict(run_tag='run', files={}), file_size(path)))
    monkeypatch.setattr(backend, 'GPU', 'CPU')
    monkeypatch.setattr(backend, '_env_value', lambda _: None)
    monkeypatch.setattr(backend, '_wandb_env_script', lambda: '')
    def stage(name, command, **kw):
        assert 'dvc' not in command[-1].lower()
        compile(command[-1], '<remote-stage>', 'exec')
        events.append('train')
    monkeypatch.setattr(backend, 'run_detached_stage', stage)
    monkeypatch.setattr(backend, '_read_remote_text', lambda _: file_size(payload))
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


def test_remote_auth_never_requests_dvc_credentials(monkeypatch):
    from cli import colab
    monkeypatch.setattr(colab, '_wandb_env_script', lambda: '')
    monkeypatch.setattr(colab, '_optuna_env_script', lambda: '')
    monkeypatch.setattr(colab, '_env_value', lambda key: (_ for _ in ()).throw(AssertionError(key)))
    source = colab._remote_auth_env_script(include_optuna=True)
    scope = {'os': type('FakeOS', (), {'environ': {}})()}
    exec(source, scope)
    assert scope['os'].environ == {'EUROMONITOR_DISABLE_DVC_CHECKPOINTS': '1', 'ER_INCREMENTAL_DVC': '0'}

def test_colab_hpo_rejects_dvc_before_any_launch():
    from cli import colab
    with pytest.raises(ValueError, match='local/none persistence'):
        colab.run_hpo(persistence='dvc')
