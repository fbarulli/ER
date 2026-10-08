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


RUN = 'boundaryrun01'


def _completion_manifest(track):
    from graph_tracks.report_manifest import build as build_manifest
    return json.dumps(build_manifest(
        track=track, checkpoint='checkpoint-1/model.pt',
        checkpoint_sha256='0' * 64, listings_sha256='1' * 64, pairs_sha256='2' * 64,
        threshold=0.5, threshold_source='dev_youden', test_reported=False,
        model_selection='dev_pr_auc', retrieval_ks=[10]))


def _sealed_run(tmp_path, *, markers, marker_paths=None, extra_members=None,
                include_checkpoint=True):
    """A sealed result archive satisfying the completion contract, as ``markers`` says.

    ``markers`` names the gnn_only selected-checkpoint markers the tree records;
    ``marker_paths`` overrides the path each marker records (the default is the
    remote absolute path shape a real worker writes); ``extra_members`` adds
    further members and ``include_checkpoint`` drops the selected checkpoint
    itself. Publication must resolve exactly one selected member under the
    Bundle's documented rule, whatever the run left behind.
    """
    import hashlib, io
    import torch
    from model_tracks.config import SuiteConfig
    files = {
        'suite_manifest.json': json.dumps({
            'run_tag': RUN, 'inputs': {}, 'resume_identity': {'implementation': {}},
            'config': SuiteConfig(setup_dir='setup', text_bundle='bundle',
                                  publish_git=False, publish_dvc=False).model_dump()}),
        'text/track_complete.json': json.dumps(
            {'track': 'text', 'status': 'ok', 'postprocess_complete': True}),
        'text/text__completion_manifest.json': _completion_manifest('text'),
        'text/checkpoint-1/trainer_state.json': json.dumps(
            {'best_model_checkpoint': 'checkpoint-1', 'best_metric': 0.5, 'global_step': 10}),
        'text/checkpoint-1/model.pt': b'text-weights',
        'gnn_only/track_complete.json': json.dumps(
            {'track': 'gnn_only', 'status': 'ok', 'postprocess_complete': True}),
        'gnn_only/gnn_only__report_manifest.json': _completion_manifest('gnn_only'),
        'cascade/track_complete.json': json.dumps(
            {'track': 'cascade', 'status': 'ok', 'postprocess_complete': True}),
        'cascade/cascade__report_manifest.json': _completion_manifest('cascade'),
    }
    buffer = io.BytesIO()
    torch.save({key: 0 for key in ('schema', 'manifest', 'vocabulary', 'support_records',
                                   'support_text', 'text_dim', 'model', 'scorer')}, buffer)
    payload = buffer.getvalue()
    checkpoint = 'gnn_only/_checkpoints/gnn_only/run_f0/checkpoint-1/gnn_only__graph_model.pt'
    if include_checkpoint:
        files[checkpoint] = payload
        files[checkpoint.rsplit('/', 1)[0] + '/gnn_only__checkpoint_manifest.json'] = json.dumps({
            'schema': 'er-graph-checkpoint-v1', 'track': 'gnn_only',
            'files': {checkpoint.rsplit('/', 1)[1]: hashlib.sha256(payload).hexdigest()},
            'inference_only': True})
    files.update(extra_members or {})
    for index, marker in enumerate(markers, start=1):
        recorded = (marker_paths or {}).get(
            marker, f'/remote/checkpoint-{index}/gnn_only__graph_model.pt')
        files[marker] = json.dumps({'path': recorded}).encode()
    for track in ('text', 'gnn_only', 'cascade'):
        files[f'{track}/track_inventory.json'] = json.dumps({
            'track': track, 'files': {
                name.split('/', 1)[1]: hashlib.sha256(
                    value if isinstance(value, bytes) else value.encode()).hexdigest()
                for name, value in files.items() if name.startswith(track + '/')}}).encode()
    root = tmp_path / 'tree'
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else value.encode())
    return write_archive(tmp_path / f'{RUN}.zip',
                         {path.relative_to(root).as_posix(): path
                          for path in root.rglob('*') if path.is_file()},
                         manifest_name='suite_bundle_manifest.json',
                         metadata={'run_tag': RUN})


def test_materialize_resolves_exactly_one_recorded_graph_checkpoint(tmp_path, monkeypatch):
    """Publication publishes the one recorded selection, or fails loud."""
    from core.bundle import Bundle, BundleRole
    from model_tracks import publish
    monkeypatch.setattr('core.common.TRAIN_ROOT', tmp_path)
    ambiguous = _sealed_run(tmp_path / 'ambiguous', markers=[
        'gnn_only/gnn_only__best_checkpoint.json',
        'gnn_only/gnn_only__old__best_checkpoint.json'])
    with pytest.raises(ValueError, match='ambiguous selected checkpoint: gnn_only'):
        publish.materialize(ambiguous, RUN,
                            bundle=Bundle.load(ambiguous, BundleRole.result))
    assert not (tmp_path / 'artifacts/models/tracks' / RUN).exists()

    selected = _sealed_run(tmp_path / 'selected',
                           markers=['gnn_only/gnn_only__best_checkpoint.json'])
    destination = publish.materialize(selected, RUN,
                                      bundle=Bundle.load(selected, BundleRole.result))
    manifest = json.loads((destination / 'models_manifest.json').read_text())
    assert manifest['run_tag'] == RUN
    assert manifest['tracks'] == ['text', 'gnn_only']
    assert set(manifest['files']) >= {'text/model.pt', 'gnn_only/gnn_only__graph_model.pt'}


def test_materialize_refuses_a_marker_that_names_two_members(tmp_path, monkeypatch):
    """One recorded selection matching two members is ambiguity, not first-match.

    ``Bundle.checkpoint`` resolves such a state to the first match of its walk;
    publication must fail loud instead of shipping whichever copy sorted first.
    """
    from core.bundle import Bundle, BundleRole
    from model_tracks import publish
    monkeypatch.setattr('core.common.TRAIN_ROOT', tmp_path)
    archive = _sealed_run(
        tmp_path / 'duplicate', markers=['gnn_only/gnn_only__best_checkpoint.json'],
        extra_members={'gnn_only/staging/checkpoint-1/gnn_only__graph_model.pt': b'stale-copy'})
    with pytest.raises(ValueError, match='ambiguous selected checkpoint: gnn_only'):
        publish.materialize(archive, RUN, bundle=Bundle.load(archive, BundleRole.result))
    assert not (tmp_path / 'artifacts/models/tracks' / RUN).exists()


def test_materialize_keeps_a_marker_without_a_member_unavailable(tmp_path, monkeypatch):
    """A recorded selection whose member is absent stays 'unavailable'."""
    from core.bundle import Bundle, BundleRole
    from model_tracks import publish
    monkeypatch.setattr('core.common.TRAIN_ROOT', tmp_path)
    archive = _sealed_run(tmp_path / 'missing',
                          markers=['gnn_only/gnn_only__best_checkpoint.json'],
                          include_checkpoint=False)
    with pytest.raises(ValueError, match='selected checkpoint unavailable: gnn_only'):
        publish.materialize(archive, RUN, bundle=Bundle.load(archive, BundleRole.result))
