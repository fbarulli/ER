import hashlib
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
    digest = hashlib.sha256(transport.read_bytes()).hexdigest()
    assert prepare_git_inputs(archive,'run',publisher=publish) == transport
    assert hashlib.sha256(transport.read_bytes()).hexdigest() == digest
    assert published == [transport,transport]


def test_suite_git_inputs_refuse_unbound_recovery_before_publication(tmp_path,monkeypatch):
    from core import common
    monkeypatch.setattr(common,'TRAIN_ROOT',tmp_path)
    source = tmp_path/'data.txt'
    source.write_text('fixed local input')
    archive = write_archive(tmp_path/'input.zip',{'data.txt':source},
        manifest_name='model_tracks_package.json',metadata={'revision':'frozen'})
    recovery = write_archive(tmp_path/'recovery.zip',{'state':source},
        manifest_name='suite_recovery_manifest.json',
        metadata={'run_tag':'run','input_package':{'revision':'other','files':{}}})
    def forbidden(*args):
        raise AssertionError('unbound inputs must never be published')
    with pytest.raises(ValueError,match='differs'):
        prepare_git_inputs(archive,'run',resume_archive=recovery,publisher=forbidden)
