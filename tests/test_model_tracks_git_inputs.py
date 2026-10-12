from core.portable_archive import ByteCount
import subprocess
import types
from core.archive_reader import tar_archive

import pytest

from core.portable_archive import write_archive
from model_tracks.colab import prepare_git_inputs


def test_suite_inputs_use_verified_git_tar_and_reuse_it(tmp_path,monkeypatch):
    from core import common
    from core.common import training_cfg
    from model_tracks import colab
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    source = tmp_path/'data.txt'
    source.write_text('fixed local input')
    archive = write_archive(tmp_path/'input.zip',{'data.txt':source},
        manifest_name='model_tracks_package.json',metadata={'revision':'frozen'})
    steps = []
    monkeypatch.setattr(colab,'subprocess',types.SimpleNamespace(
        run=lambda command,**kwargs:(steps.append(list(command)),
                                     subprocess.CompletedProcess(command,0))[1]))
    transport = prepare_git_inputs(archive,'run')
    with tar_archive(transport) as package:
        assert package.getnames() == ['inputs.tar.zst']
        assert package.extractfile('inputs.tar.zst').read() == archive.read_bytes()
    digest = ByteCount(transport.read_bytes()).total
    assert prepare_git_inputs(archive,'run') == transport
    assert ByteCount(transport.read_bytes()).total == digest
    assert steps == [['git','push','origin',
                      f'HEAD:{training_cfg().colab.branch}']]*2


def test_suite_git_inputs_reuse_the_supplied_recovery_manifest(tmp_path,monkeypatch):
    """A resume run opens the recovery Bundle ONCE, not once per entry point.

    `colab.run` loads the recovery handle at its own boundary and hands the
    manifest to the transport builder; the builder must then never re-verify
    the archive (a second load is a second multi-hundred-MB integrity pass).
    """
    from core import common
    from core.bundle import Bundle, BundleRole
    from model_tracks import colab
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    source = tmp_path/'data.txt'
    source.write_text('fixed local input')
    archive = write_archive(tmp_path/'input.zip',{'data.txt':source},
        manifest_name='model_tracks_package.json',metadata={'revision':'frozen'})
    metadata = Bundle.load(archive,BundleRole.inputs).manifest
    recovery = tmp_path/'recovery.zip'
    recovery.write_bytes(b'recovery bytes')

    def forbidden(*args,**kwargs):
        raise AssertionError('the recovery archive was verified a second time')

    monkeypatch.setattr(colab,'Bundle',types.SimpleNamespace(load=forbidden))
    monkeypatch.setattr(colab,'subprocess',types.SimpleNamespace(
        run=lambda command,**kwargs: subprocess.CompletedProcess(command,0)))
    transport = prepare_git_inputs(
        archive,'run',resume_archive=recovery,
        recovery={'run_tag':'run','input_package':{'revision':metadata['revision'],
                                                   'files':metadata['files']}})
    with tar_archive(transport) as package:
        assert set(package.getnames()) == {'inputs.tar.zst','recovery.tar.zst'}


def test_tracks_launch_saves_inputs_without_any_dvc_push(tmp_path,monkeypatch):
    """GPU tracks launch: transport bytes + branch sync, never a DVC publish.

    The owner ruling 2026-10-12 removes DVC from the training provisioning
    path: `prepare_git_inputs` (the ONE blocking step before ensure_session in
    `cli.colab --what tracks`) must complete with every DVC entry point
    tripwired, leaving only the fast-forward branch push that makes the
    packaged revision visible to the VM's clone.
    """
    from core import common
    from core.common import training_cfg
    from model_tracks import colab, run_retention
    from training import dvc_store
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    source = tmp_path/'data.txt'
    source.write_text('fixed local input')
    archive = write_archive(tmp_path/'input.zip',{'data.txt':source},
        manifest_name='model_tracks_package.json',metadata={'revision':'frozen'})
    def tripwire(*args,**kwargs):
        raise AssertionError(f'DVC publication ran on the GPU launch path: {args}')
    monkeypatch.setattr(run_retention,'publish_result_paths',tripwire)
    monkeypatch.setattr(dvc_store,'_run',tripwire)
    monkeypatch.setattr(dvc_store,'push_targets',tripwire)
    steps = []
    def fake_run(command,**kwargs):
        steps.append(list(command))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(colab,'subprocess',types.SimpleNamespace(run=fake_run))
    transport = prepare_git_inputs(archive,'run')
    assert transport.is_file()
    branch = training_cfg().colab.branch
    assert steps == [['git','push','origin',f'HEAD:{branch}']]
