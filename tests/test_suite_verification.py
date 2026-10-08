"""Post-download verification of sealed suite archives, plus its dashboard surface."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from core.portable_archive import write_archive
from graph_tracks.data import file_hash
from graph_tracks.report_manifest import build as build_manifest
from model_tracks import archive_verification
from model_tracks.config import SuiteConfig
from model_tracks.resume import TRACKS

RUN = 'verifyrun01'


def _report(track, threshold=0.5):
    return build_manifest(
        track=track, checkpoint='checkpoint-1/model.pt',
        checkpoint_sha256='0' * 64, listings_sha256='1' * 64, pairs_sha256='2' * 64,
        threshold=threshold, threshold_source='dev_youden', test_reported=False,
        model_selection='dev_pr_auc', retrieval_ks=[10])


def _ablation(track):
    return {'track': track, 'request_sha256': '3' * 64, 'result_sha256': '4' * 64,
            'threshold': 0.5, 'threshold_provenance': {'sha256': '5' * 64},
            'threshold_binding': {'track': track, 'checkpoint_sha256': '0' * 64, 'verified': True},
            'rows': []}


def _binding(post_training_ablation):
    settings = SuiteConfig(setup_dir='setup', text_bundle='bundle',
                           publish_git=False, publish_dvc=False,
                           post_training_ablation=post_training_ablation)
    return {'run_tag': RUN, 'inputs': {}, 'resume_identity': {'implementation': {}},
            'config': settings.model_dump()}


def _build_sealed(tmp_path, ablation=False, name=RUN):
    """A minimal archive that satisfies the sealing-time contract end to end."""
    inline = {'suite_manifest.json': json.dumps(_binding(ablation))}
    for track in TRACKS:
        suffix = ('text__completion_manifest.json' if track == 'text'
                  else f'{track}__report_manifest.json')
        rel = suffix if track == 'text' else f'{track}__local_completion/{suffix}'
        inline[f'{track}/{rel}'] = json.dumps(_report(track))
        inline[f'{track}/track_complete.json'] = json.dumps(
            {'track': track, 'status': 'ok', 'postprocess_complete': True})
        listed = [rel, 'track_complete.json']
        if ablation and track in ('text', 'gnn_only'):
            # The cascade trains nothing and ships no ablation templates, so a
            # completed suite with post_training_ablation=True legitimately has
            # no cascade ablation report.
            inline[f'{track}/ablation/report.json'] = json.dumps(_ablation(track))
            listed.append('ablation/report.json')
        inventory = {'track': track,
                     'files': {member: hashlib.sha256(inline[f'{track}/{member}'].encode()).hexdigest()
                               for member in listed}}
        inline[f'{track}/track_inventory.json'] = json.dumps(inventory)
    out = tmp_path / f'{name}.zip'
    write_archive(out, {}, inline=inline, manifest_name='suite_bundle_manifest.json',
                  metadata={'run_tag': RUN})
    (out.with_suffix('.sha256')).write_text(file_hash(out) + '\n')
    return out


def test_verified_archive_reports_sha_and_tracks(tmp_path):
    archive = _build_sealed(tmp_path, ablation=False)
    result = archive_verification.verification_result(archive)
    assert result['status'] == 'verified'
    assert result['run_tag'] == RUN
    assert result['zip_sha256']['match'] is True
    assert result['tracks']['text']['threshold'] == 0.5
    assert result['tracks']['text']['test_reported'] is False
    assert result['tracks']['cascade']['threshold'] == 0.5
    assert result['tracks']['cascade']['test_reported'] is False


def test_settings_pin_rejects_a_different_suite(tmp_path):
    archive = _build_sealed(tmp_path)
    other = SuiteConfig(setup_dir='setup', text_bundle='BUNDLE-CHANGED',
                        publish_git=False, publish_dvc=False)
    result = archive_verification.verification_result(archive, settings=other)
    assert result['status'] == 'failed'
    assert 'configuration differs' in result['error']


def test_tampered_member_fails_with_integrity_message(tmp_path):
    import shutil
    import zipfile
    archive = _build_sealed(tmp_path)
    tampered = tmp_path / 'tampered.zip'
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(tampered, 'w') as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == 'text/text__completion_manifest.json':
                data = data.replace(b'"threshold": 0.5', b'"threshold": 0.51')
            dst.writestr(info, data)
    result = archive_verification.verification_result(tampered)
    assert result['status'] == 'failed'
    assert 'integrity' in result['error']


def test_missing_archive_is_unreadable(tmp_path):
    result = archive_verification.verification_result(tmp_path / 'absent.zip')
    assert result['status'] == 'unreadable'
    assert 'absent.zip' in result['error']


def test_verified_archive_with_ablation_skips_cascade(tmp_path):
    """A completed suite with ablation enabled needs no cascade ablation.

    Only the trained lanes export ablation templates; the cascade is a
    combinator with no checkpoint to ablate, so verification must accept its
    absence while still requiring the trained lanes' saved ablations.
    """
    archive = _build_sealed(tmp_path, ablation=True)
    result = archive_verification.verification_result(archive)
    assert result['status'] == 'verified'
    assert 'ablation' in result['tracks']['text']
    assert 'ablation' in result['tracks']['gnn_only']
    assert 'ablation' not in result['tracks']['cascade']


def test_cascade_manifest_folds_both_report_roles(tmp_path):
    """The cascade's per-track manifest carries report_cascade's two role metrics.

    ``report_cascade`` writes ``cascade__cascade_report.json`` with ranker and
    decider sections; the suite manifest must expose them too, so a cascade
    archive has the same traceability shape as the trained lanes.
    """
    import numpy as np
    from graph_tracks import report as graph_report
    from graph_tracks.report_manifest import build as build_manifest, write as write_manifest
    from model_tracks.cascade import Decisions, Ranked

    ranked = Ranked(query_ids=('a',),
                    candidate_ids=np.array([['b', 'c']], dtype=object),
                    similarities=np.array([[1.0, 0.9]], dtype=np.float32))
    decisions = Decisions(query_ids=('a',),
                          candidate_ids=np.array([['b', 'c']], dtype=object),
                          scores=np.array([[0.9, 0.1]], dtype=np.float32),
                          order=np.array([[0, 1]], dtype=np.int64),
                          similarities=None)
    report_cascade = graph_report.report_cascade(
        ranked, [{'b'}], decisions, tmp_path, ks=(1, 2))
    assert set(report_cascade['roles']) == {'ranker', 'decider'}
    # The cascade carries the same traceability keys as the trained lanes; it
    # has no population to slice, so they are empty with an explicit reason.
    assert report_cascade['slices'] == [] and report_cascade['attributes'] == []
    assert report_cascade['traceability']['slices'].startswith('not applicable')
    assert report_cascade['traceability']['attributes'].startswith('not applicable')

    manifest = build_manifest(
        track='cascade', checkpoint='gnn.json', checkpoint_sha256='0' * 64,
        listings_sha256='1' * 64, pairs_sha256='2' * 64, threshold=0.5,
        threshold_source='dev_youden', test_reported=False,
        model_selection='dev_pr_auc', retrieval_ks=[1, 2])
    written = write_manifest(tmp_path / 'cascade__report_manifest.json', manifest)
    folded = json.loads(written.read_text())
    assert folded['roles'] == report_cascade['roles']
    assert folded['slices'] == report_cascade['slices']
    assert folded['attributes'] == report_cascade['attributes']
    assert folded['traceability'] == report_cascade['traceability']


def test_traceability_keys_are_identical_across_tracks(tmp_path):
    """text, gnn_only and cascade all expose slices/attributes + a reason.

    The cascade computes no table, but a missing key would let a reader
    confuse "the cascade did not measure slices" with "slices vanished".
    """
    from graph_tracks.report_manifest import TRACEABILITY_KEYS

    manifests = {track: _report(track) for track in ('text', 'gnn_only', 'cascade')}
    for track, manifest in manifests.items():
        for key in TRACEABILITY_KEYS:
            assert key in manifest, f'{track} manifest is missing {key}'
        assert set(manifest['traceability']) == set(TRACEABILITY_KEYS)
    # A trained lane's attribute table lives next to the report; the cascade
    # records an explicit not-applicable rather than pointing at a file it
    # never writes.
    assert 'attribute_separation_summary' in manifests['gnn_only']['traceability']['attributes']
    assert manifests['cascade']['traceability']['attributes'].startswith('not applicable')
    assert manifests['cascade']['slices'] == []


def test_verification_sidecar_roundtrip(tmp_path):
    archive = _build_sealed(tmp_path)
    result = archive_verification.verification_result(archive)
    path = archive_verification.write_verification(archive, result)
    assert path.name == f'{RUN}.verification.json'
    assert archive_verification.load_verification(archive) == result
    assert archive_verification.load_verification(tmp_path / 'absent.zip') is None


class RouteClient:
    """Call the panel routes directly; the sandbox blocks TestClient's thread portal."""

    def __init__(self, module):
        self.module = module

    def get(self, route, params=None):
        handlers = {'/training': self.module.training, '/training/verify': self.module.verify_run}
        from fastapi import HTTPException
        try:
            response = handlers[route](**(params or {}))
        except HTTPException as error:
            return SimpleNamespace(status_code=error.status_code, text=error.detail, headers={})
        if isinstance(response, str):
            return SimpleNamespace(status_code=200, text=response, headers={})
        return SimpleNamespace(status_code=response.status_code,
                               text=getattr(response, 'body', b'').decode(errors='replace'),
                               headers=response.headers)


from types import SimpleNamespace  # noqa: E402


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'er_training_reports_verification',
        Path(__file__).parents[1] / 'dashboard/training_reports.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'PROJECT', tmp_path)
    root = tmp_path / 'results' / 'model_tracks'
    root.mkdir(parents=True)
    (root / f'{RUN}.zip').write_bytes(b'zip-bytes')
    return RouteClient(module), root, f'results/model_tracks/{RUN}.zip'


def test_verification_panel_renders_from_sidecar(dashboard):
    client, root, key = dashboard
    result = {'schema': archive_verification.VERIFICATION_SCHEMA, 'status': 'verified',
              'verified_at': '2026-10-04T16:00:00+00:00', 'run_tag': RUN,
              'archive': f'{RUN}.zip', 'zip_sha256': {'match': True},
              'tracks': {'text': {'report': 'text/text__completion_manifest.json',
                                  'threshold': 0.5, 'checkpoint_sha256': '0' * 64,
                                  'test_reported': False}}}
    (root / f'{RUN}.verification.json').write_text(json.dumps(result))
    html = client.get('/training', {'run': key}).text
    assert 'Archive verification: verified' in html
    assert 'sha256 sidecar: match' in html
    assert 'text__completion_manifest.json' in html


def test_failed_verification_shows_the_error_inline(dashboard):
    client, root, key = dashboard
    result = {'status': 'failed', 'verified_at': 'x', 'run_tag': RUN, 'archive': f'{RUN}.zip',
              'zip_sha256': {'match': False, 'expected': 'a' * 64, 'actual': 'b' * 64},
              'error': 'ValueError: archive integrity mismatch: text/x.json', 'tracks': {}}
    (root / f'{RUN}.verification.json').write_text(json.dumps(result))
    html = client.get('/training', {'run': key}).text
    assert 'Archive verification: failed' in html
    assert 'MISMATCH' in html
    assert 'archive integrity mismatch' in html


def test_missing_sidecar_offers_the_verify_link(dashboard):
    client, _, key = dashboard
    html = client.get('/training', {'run': key}).text
    assert 'Verify archive' in html
    assert '/training/verify?' in html
    assert f'{RUN}.verification.json' in html


def test_verify_route_records_outcome_and_redirects(dashboard, monkeypatch):
    client, root, key = dashboard
    seen = []
    monkeypatch.setattr(archive_verification, 'verification_result',
                        lambda path, **kwargs: seen.append(path)
                        or {'schema': archive_verification.VERIFICATION_SCHEMA, 'status': 'verified'})
    monkeypatch.setattr(archive_verification, 'write_verification',
                        lambda path, result: (root / f'{RUN}.verification.json')
                        .write_text(json.dumps(result)))
    response = client.get('/training/verify', {'run': key})
    assert response.status_code == 302
    assert response.headers['location'] == f'/training?run={key.replace("/", "%2F")}'
    assert (root / f'{RUN}.verification.json').is_file()
    assert seen and seen[0].name == f'{RUN}.zip'


def test_verify_route_rejects_local_run_folders(dashboard):
    client, root, _ = dashboard
    folder = root / 'localrun'
    folder.mkdir()
    (folder / 'model_evaluation_summary.csv').write_text('split,pr_auc\ndev,0.7\n')
    response = client.get('/training/verify', {'run': 'results/model_tracks/localrun'})
    assert response.status_code == 400
