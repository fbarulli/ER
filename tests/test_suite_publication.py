import json
import zipfile

import pytest

from core.portable_archive import write_archive
from model_tracks.publish import persist_results


def test_suite_persists_binary_checkpoints_and_collects_restore_references(tmp_path, monkeypatch):
    from core import common
    from training import dvc_store
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'checkpoint binary')
    archive = write_archive(tmp_path/'suite.zip', {'hybrid/model.pt': checkpoint},
                            manifest_name='suite_bundle_manifest.json', metadata={'run_tag':'suite'})
    monkeypatch.setenv('DVC_API_KEY', 'test-placeholder')
    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    def artifact(kind, values):
        return tmp_path/'dvc_refs'/values['run_id']/f"worker_{values['worker']}"/'publication_manifest.json'
    monkeypatch.setattr(common, 'artifact', artifact)
    monkeypatch.setattr(dvc_store, '_configure', lambda *_: None)
    calls = []
    monkeypatch.setattr(dvc_store, '_run', lambda command, cwd: calls.append((command,cwd)))
    def publish(source, run_id, worker):
        from core.schemas import DvcPublicationManifest
        DvcPublicationManifest(schema_version='1', run_id=run_id, worker=worker,
            remote='https://example.test/dvc', verified_download=True,
            pointers=[{'pointer':'suite.zip.dvc','outputs':['suite.zip']}])
        assert calls == [(['dvc','add','suite.zip'], source)]
        with zipfile.ZipFile(source/'suite.zip') as bundle:
            assert bundle.read('hybrid/model.pt') == checkpoint.read_bytes()
        index = artifact('', {'run_id':run_id,'worker':worker})
        index.parent.mkdir(parents=True)
        pointer = index.parent/'suite.zip.dvc'
        pointer.write_text('outs: []\n')
        index.write_text(json.dumps({'pointers':[{'pointer':pointer.relative_to(tmp_path).as_posix()}]}))
    monkeypatch.setattr(dvc_store, 'publish', publish)
    receipt = json.loads(persist_results(archive,'suite').read_text())
    assert receipt['verified_download'] is True
    assert set(receipt['references']) == {'dvc_refs/suite/worker_1/publication_manifest.json',
                                        'dvc_refs/suite/worker_1/suite.zip.dvc'}


def test_suite_publication_requires_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv('DVC_API_KEY', raising=False)
    archive = write_archive(tmp_path/'suite.zip', {}, manifest_name='suite_bundle_manifest.json',
                            metadata={'run_tag':'suite'})
    with pytest.raises(RuntimeError, match='DVC_API_KEY'):
        persist_results(archive,'suite')
    assert (tmp_path/'suite.publication'/'suite.zip').is_file()
