import json
from pathlib import Path
import zipfile
import sys
from types import SimpleNamespace

import pytest

from core.portable_archive import verify_archive
from model_tracks.package import recovery_package, restore_recovery


def test_recovery_preserves_checkpoint_and_provenance(tmp_path):
    output = tmp_path / 'interrupted'
    (output / 'text' / '_checkpoints').mkdir(parents=True)
    (output / 'suite_manifest.json').write_text(json.dumps({'run_tag': 'cpu-smoke'}))
    checkpoint = output / 'text' / '_checkpoints' / 'last.pt'
    checkpoint.write_bytes(b'checkpoint')
    (output / '.env').write_text('private')
    (output / 'wandb').mkdir()
    (output / 'wandb' / 'secret').write_text('private')
    (output / 'escape').symlink_to(tmp_path / 'outside')
    original = {'revision': 'abc', 'files': {'src/model_tracks/run.py': 'digest'}}
    archive = recovery_package(output, tmp_path / 'recovery.zip', 'cpu-smoke', input_package=original)
    metadata = verify_archive(archive, 'suite_recovery_manifest.json')
    assert metadata['input_package'] == original
    assert '.env' not in metadata['files']
    assert not any(member.startswith('wandb/') for member in metadata['files'])
    restored = restore_recovery(archive, tmp_path / 'restored', 'cpu-smoke')
    assert (restored / 'text/_checkpoints/last.pt').read_bytes() == b'checkpoint'
    with pytest.raises(FileExistsError):
        restore_recovery(archive, restored, 'cpu-smoke')
    with pytest.raises(ValueError, match='run mismatch'):
        restore_recovery(archive, tmp_path / 'wrong', 'another-run')


def test_recovery_rejects_unlisted_members_before_writing(tmp_path):
    output = tmp_path / 'interrupted'
    output.mkdir()
    (output / 'suite_manifest.json').write_text(json.dumps({'run_tag': 'smoke'}))
    archive = recovery_package(output, tmp_path / 'recovery.zip', 'smoke')
    with zipfile.ZipFile(archive, 'a') as source:
        source.writestr('unlisted.txt', 'unexpected')
    restored = tmp_path / 'restored'
    with pytest.raises(ValueError, match='unlisted'):
        restore_recovery(archive, restored, 'smoke')
    assert not restored.exists()


def test_recovery_rejects_traversal(tmp_path):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as source:
        source.writestr('../escaped', 'unexpected')
    with pytest.raises(ValueError, match='unsafe archive path'):
        restore_recovery(archive, tmp_path / 'restored', 'smoke')


def test_colab_failure_collects_verified_recovery_before_reraising(tmp_path, monkeypatch):
    import core.common
    from model_tracks import colab
    from graph_tracks.data import file_hash
    monkeypatch.setattr(core.common, 'RESULTS', tmp_path / 'results')
    inputs = tmp_path / 'inputs.zip'
    with zipfile.ZipFile(inputs, 'w') as archive:
        archive.writestr('data/model_tracks/suite.yaml',
                         'setup_dir: shared\ntext_bundle: shared/text.pkl.gz\ndevice: cpu\n'
                         'publish_git: false\npublish_dvc: false\n')
    metadata = {'revision': 'original', 'files': {'source': 'hash'}}
    monkeypatch.setattr(colab, 'verify', lambda _: metadata)
    output = tmp_path / 'stopped'
    output.mkdir()
    (output / 'suite_manifest.json').write_text('{"run_tag":"smoke"}')
    saved = recovery_package(output, tmp_path / 'saved.zip', 'smoke', input_package=metadata)
    scripts = []

    def exec_remote(session, script, **kwargs):
        compile(script, '<remote>', 'exec')
        scripts.append(script)

    def detached(stage, command, **kwargs):
        compile(command[-1], '<remote-stage>', 'exec')
        raise RuntimeError('worker failure')

    def download(remote, local):
        local.write_bytes(saved.read_bytes())

    backend = SimpleNamespace(REMOTE_ROOT='/remote/root', SESSION='session', GPU='CPU',
        GIT_REMOTE_NAME='origin', BRANCH='training', _BOOTSTRAP='', _RESULT_DOWNLOAD_TIMEOUT_SECONDS=30,
        _WORKER_TIMEOUT_SECONDS=30, run_colab_exec_stream=exec_remote,
        _upload_with_retries=lambda *a, **k: None, _remote_auth_env_script=lambda **k: '',
        run_detached_stage=detached, _read_remote_text=lambda _: file_hash(saved),
        _download_one_remote_file=download)
    # `model_tracks.colab.run` does `from cli import colab as backend`, which
    # resolves the attribute on the ALREADY-IMPORTED cli package before
    # consulting sys.modules — so with cli.colab imported by an earlier test
    # (e.g. test_colab_keep_alive_boundary), patching only sys.modules left the
    # real backend running a live session probe ("Session
    # 'my-highram-session' not found") instead of this fake. Patch BOTH the
    # attribute and the sys.modules entry (order-independent).
    import cli
    monkeypatch.setattr(cli, 'colab', backend)
    monkeypatch.setitem(sys.modules, 'cli.colab', backend)
    with pytest.raises(RuntimeError, match='worker failure'):
        colab.run(inputs, 'smoke')
    recovery = tmp_path / 'results/model_tracks/smoke.recovery.zip'
    assert verify_archive(recovery, 'suite_recovery_manifest.json')['input_package'] == metadata
    assert any('recovery_package' in script for script in scripts)
    monkeypatch.setattr(colab, 'verify', lambda _: {**metadata, 'revision': 'changed'})
    with pytest.raises(ValueError, match='differs from interrupted'):
        colab.run(inputs, 'smoke', resume=True)
    backend.GPU = 'T4'
    with pytest.raises(ValueError, match='device differs'):
        colab.run(inputs, 'smoke', resume=True)
