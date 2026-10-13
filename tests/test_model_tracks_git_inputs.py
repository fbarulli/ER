from core.portable_archive import ByteCount
import types
from core.archive_reader import tar_archive

import pytest

from core.portable_archive import write_archive
from model_tracks.colab import prepare_git_inputs


def test_suite_inputs_use_verified_git_tar_and_reuse_it(tmp_path,monkeypatch):
    from core import common
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    source = tmp_path/'data.txt'
    source.write_text('fixed local input')
    archive = write_archive(tmp_path/'input.zip',{'data.txt':source},
        manifest_name='model_tracks_package.json',metadata={'revision':'frozen'})
    published = []
    def publish(paths,message):
        published.append(paths[0])
    transport = prepare_git_inputs(archive,'run',publisher=publish)
    with tar_archive(transport) as package:
        assert package.getnames() == ['inputs.tar.zst']
        assert package.extractfile('inputs.tar.zst').read() == archive.read_bytes()
    digest = ByteCount(transport.read_bytes()).total
    assert prepare_git_inputs(archive,'run',publisher=publish) == transport
    assert ByteCount(transport.read_bytes()).total == digest
    assert published == [transport,transport]


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
    transport = prepare_git_inputs(
        archive,'run',resume_archive=recovery,
        recovery={'run_tag':'run','input_package':{'revision':metadata['revision'],
                                                   'files':metadata['files']}},
        publisher=lambda paths,message: None)
    with tar_archive(transport) as package:
        assert set(package.getnames()) == {'inputs.tar.zst','recovery.tar.zst'}
