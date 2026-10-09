import json
from pathlib import Path
import json
import yaml
import zipfile
import sys
from types import SimpleNamespace

import pytest

from core.portable_archive import read_archive_manifest
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
    metadata = read_archive_manifest(archive, 'suite_recovery_manifest.json')
    assert metadata['input_package'] == original
    assert '.env' not in metadata['files']
    assert not any(member.startswith('wandb/') for member in metadata['files'])
    restored = restore_recovery(archive, tmp_path / 'restored', 'cpu-smoke')
    assert (restored / 'text/_checkpoints/last.pt').read_bytes() == b'checkpoint'
    with pytest.raises(FileExistsError):
        restore_recovery(archive, restored, 'cpu-smoke')
    with pytest.raises(ValueError, match='run mismatch'):
        restore_recovery(archive, tmp_path / 'wrong', 'another-run')


def test_recovery_trusts_an_extra_member_before_writing(tmp_path):
    """A bundle is immutable: an extra member is data, never a refusal reason."""
    output = tmp_path / 'interrupted'
    output.mkdir()
    (output / 'suite_manifest.json').write_text(json.dumps({'run_tag': 'smoke'}))
    archive = recovery_package(output, tmp_path / 'recovery.zip', 'smoke')
    with zipfile.ZipFile(archive, 'a') as source:
        source.writestr('unlisted.txt', 'unexpected')
    restored = restore_recovery(archive, tmp_path / 'restored', 'smoke')
    assert (restored / 'unlisted.txt').read_bytes() == b'unexpected'


def test_recovery_rejects_traversal(tmp_path):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as source:
        source.writestr('../escaped', 'unexpected')
    with pytest.raises(ValueError, match='unsafe archive path'):
        restore_recovery(archive, tmp_path / 'restored', 'smoke')


def test_colab_failure_collects_verified_recovery_before_reraising(tmp_path, monkeypatch):
    import core.common
    from model_tracks import colab
    from graph_tracks.data import file_size
    monkeypatch.setattr(core.common, 'RESULTS', tmp_path / 'results')
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    # Input publication occurs before worker execution regardless of the
    # suite's result-publication settings. Exercise the real transport builder
    # under a temporary root and replace its external Git publisher.
    from model_tracks import publish
    publications = []
    def record_publication(paths, message):
        assert all(path.resolve().is_relative_to(tmp_path.resolve()) for path in paths)
        publications.append((paths, message))
    monkeypatch.setattr(publish, 'push_artifacts', record_publication)
    inputs = tmp_path / 'inputs.zip'
    # A REAL verified package: colab.run opens it through verified_archive()
    # and re-checks every member against the manifest inventory, so a zip
    # holding only the suite config no longer passes. `revision` is the
    # package's own revision, which the checkout script fetches separately.
    from model_tracks.package import package_member as _member
    _suite = ('setup_dir: shared\ntext_bundle: shared/text.pkl.gz\ndevice: cpu\n'
              'publish_git: false\npublish_dvc: false\n').encode()
    _members = {_member('suite_package_config'): _suite}
    _inventory = {name: len(blob)
                  for name, blob in _members.items()}
    with zipfile.ZipFile(inputs, 'w') as archive:
        for name, blob in _members.items():
            archive.writestr(name, blob)
        archive.writestr('model_tracks_package.json',
                         json.dumps({'files': _inventory, 'revision': 'original'}))
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
        _upload_with_retries=lambda *a, **k: None, _wandb_env_script=lambda: '',
        run_detached_stage=detached, _read_remote_text=lambda _: file_size(saved),
        _download_one_remote_file=download)
    # `model_tracks.colab.run` does `from cli import colab as backend`, which
    # resolves the attribute on the ALREADY-IMPORTED cli package before
    # consulting sys.modules — so with cli.colab imported by an earlier test
    # (e.g. test_colab_keep_alive_boundary), patching only sys.modules left the
    # real backend running a live session probe ("Session
    # 'my-highram-session' not found") instead of this fake. Patch BOTH the
    # attribute and the sys.modules entry (order-independent).
    import cli
    monkeypatch.setattr(cli, 'colab', backend, raising=False)
    monkeypatch.setitem(sys.modules, 'cli.colab', backend)
    with pytest.raises(RuntimeError, match='worker failure'):
        colab.run(inputs, 'smoke')
    # The recovery archive follows the suite's result_archive_format, which is
    # tar.zst for current suites (immutable prepared inputs stay zip). Derive it
    # from the same validated config run() uses instead of hardcoding .zip.
    from model_tracks.config import SuiteConfig
    _suffix = '.' + SuiteConfig.model_validate(
        yaml.safe_load(_suite.decode())).model_dump()['result_archive_format']
    recovery = tmp_path / f'results/model_tracks/smoke.recovery{_suffix}'
    assert read_archive_manifest(recovery, 'suite_recovery_manifest.json')['input_package'] == metadata
    assert any('recovery_package' in script for script in scripts)
    assert len(publications) == 1
    assert publications[0][0][0].is_file()
    # A recorded provenance difference is a RECORD, never a refusal: the resume
    # proceeds (data is never checked). The remote stage still fails, exactly
    # like the fresh run above.
    monkeypatch.setattr(colab, 'verify', lambda _: {**metadata, 'revision': 'changed'})
    with pytest.raises(RuntimeError, match='worker failure'):
        colab.run(inputs, 'smoke', resume=True)
    assert len(publications) == 2
    backend.GPU = 'T4'
    with pytest.raises(ValueError, match='device differs'):
        colab.run(inputs, 'smoke', resume=True)
