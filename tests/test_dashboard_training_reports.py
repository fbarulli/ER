import importlib.util
import json
import stat
import asyncio
from types import SimpleNamespace
from pathlib import Path
import zipfile

import pytest
from fastapi import HTTPException


class RouteClient:
    """Call route handlers directly; sandbox blocks TestClient's thread portal."""
    def __init__(self, module):
        self.module = module

    def get(self, route, params=None):
        handlers = {'/training': self.module.training, '/training/plot': self.module.plot,
                    '/training/log': self.module.log}
        try:
            response = handlers[route](**(params or {}))
        except HTTPException as error:
            return SimpleNamespace(status_code=error.status_code, text=error.detail)
        if isinstance(response, str):
            return SimpleNamespace(status_code=200, text=response)
        return SimpleNamespace(status_code=response.status_code, content=response.body,
                               text=response.body.decode(errors='replace'), headers=response.headers)


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('er_training_reports', Path(__file__).parents[1] / 'dashboard/training_reports.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'PROJECT', tmp_path)
    root = tmp_path / 'results/model_tracks'
    root.mkdir(parents=True)
    with zipfile.ZipFile(root / 'smoke.zip', 'w') as archive:
        archive.writestr('suite_manifest.json', json.dumps({'config': {'device': 'cpu', 'epochs': 1, 'report_test': False}}))
        archive.writestr('suite_result.json', '{"status":"ok"}')
        archive.writestr('text/text__reports/text__score_distribution_and_pr.png', b'png-test')
        archive.writestr('text/text__reports/text__model_evaluation_summary.csv', 'split,pr_auc\ndev,0.7\n')
        archive.writestr('text/checkpoint/model.safetensors', b'private-checkpoint')
    return RouteClient(module), root


def test_downloaded_suite_plots_and_metrics_render(dashboard):
    client, _ = dashboard
    response = client.get('/training')
    assert response.status_code == 200
    assert 'dev' in response.text and '0.7' in response.text
    assert 'cpu' in response.text and 'Complete suite' in response.text
    # The live track set is text / gnn_only / cascade; the retired hybrid name
    # must not reappear on the public page.
    assert 'cascade' in response.text and 'hybrid' not in response.text
    plot = client.get('/training/plot', params={'run': 'results/model_tracks/smoke.zip',
        'artifact': 'text/text__reports/text__score_distribution_and_pr.png'})
    assert plot.content == b'png-test' and plot.headers['content-type'] == 'image/png'


def test_plot_route_rejects_other_files_and_traversal(dashboard):
    client, root = dashboard
    for artifact in ('text/checkpoint/model.safetensors', '../secret.png'):
        assert client.get('/training/plot', params={'run': 'results/model_tracks/smoke.zip', 'artifact': artifact}).status_code == 404
    assert client.get('/training', params={'run': '../../secret.zip'}).status_code == 404
    folder = root / 'local'
    folder.mkdir()
    outside = root.parent / 'outside.png'
    outside.write_bytes(b'secret')
    (folder / 'escape.png').symlink_to(outside)
    assert client.get('/training/plot', params={'run': 'results/model_tracks/local', 'artifact': 'escape.png'}).status_code == 404


def test_local_run_with_metrics_and_plots(dashboard):
    client, root = dashboard
    folder = root / 'local' / 'hybrid' / 'reports'
    folder.mkdir(parents=True)
    (folder / 'hybrid__fold_metrics.csv').write_text('split,loss\nvalidation,0.2\n')
    (folder / 'hybrid__learning_curve.png').write_bytes(b'local-plot')
    response = client.get('/training', params={'run': 'results/model_tracks/local'})
    assert response.status_code == 200
    assert 'validation' in response.text and '0.2' in response.text
    assert client.get('/training/plot', params={'run': 'results/model_tracks/local',
        'artifact': 'hybrid/reports/hybrid__learning_curve.png'}).content == b'local-plot'


def test_malformed_archive_and_metadata_are_readable_errors(dashboard):
    client, root = dashboard
    (root / 'broken.zip').write_bytes(b'not a zip')
    response = client.get('/training', params={'run': 'results/model_tracks/broken.zip'})
    assert response.status_code == 200 and 'could not be read' in response.text
    assert client.get('/training/plot', params={'run': 'results/model_tracks/broken.zip',
        'artifact': 'plot.png'}).status_code == 404
    with zipfile.ZipFile(root / 'bad-metadata.zip', 'w') as archive:
        archive.writestr('suite_manifest.json', '{bad json')
        archive.writestr('model_evaluation_summary.csv', b'\xff')
        archive.writestr('plot.png', b'valid-plot')
    response = client.get('/training', params={'run': 'results/model_tracks/bad-metadata.zip'})
    assert response.status_code == 200
    assert 'Suite metadata could not be read' in response.text
    assert 'Metrics could not be read' in response.text
    assert 'plot.png' in response.text


def test_archive_unsafe_members_and_run_symlinks_are_hidden(dashboard):
    client, root = dashboard
    with zipfile.ZipFile(root / 'unsafe.zip', 'w') as archive:
        archive.writestr('../escaped.png', b'secret')
        archive.writestr('/absolute.png', b'secret')
        link = zipfile.ZipInfo('linked.png')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, '/outside.png')
    response = client.get('/training', params={'run': 'results/model_tracks/unsafe.zip'})
    assert 'escaped.png' not in response.text and 'linked.png' not in response.text
    for artifact in ('../escaped.png', '/absolute.png', 'linked.png'):
        assert client.get('/training/plot', params={'run': 'results/model_tracks/unsafe.zip',
            'artifact': artifact}).status_code == 404
    outside = root.parent / 'external'
    outside.mkdir()
    (outside / 'plot.png').write_bytes(b'secret')
    (root / 'linked-run').symlink_to(outside, target_is_directory=True)
    assert client.get('/training', params={'run': 'results/model_tracks/linked-run'}).status_code == 404


def test_empty_and_metrics_only_runs(dashboard):
    client, root = dashboard
    (root / 'input-only').mkdir()
    (root / 'input-only' / 'listing_pairs.csv').write_text('left,right\na,b\n')
    assert 'input-only' not in client.get('/training').text
    (root / 'metrics-only').mkdir()
    (root / 'metrics-only' / 'retrieval_summary.csv').write_text('k,recall\n10,0.8\n')
    response = client.get('/training', params={'run': 'results/model_tracks/metrics-only'})
    assert response.status_code == 200 and '0.8' in response.text and 'no saved plots' in response.text
    for path in root.iterdir():
        if path.is_file():
            path.unlink()
    (root / 'metrics-only' / 'retrieval_summary.csv').unlink()
    (root / 'metrics-only').rmdir()
    assert 'No downloaded training reports yet' in client.get('/training').text


def download_bytes(response):
    # Consume the original sync iterator directly instead of using ASGI's
    # thread pool; downloads remain byte-for-byte and unbounded in production.
    async def collect():
        return b''.join([chunk async for chunk in response.body_iterator])
    return asyncio.run(collect())


def test_structured_events_and_raw_logs_preview_from_zip(dashboard):
    client, root = dashboard
    with zipfile.ZipFile(root / 'logged.zip', 'w') as archive:
        archive.writestr('suite_events.jsonl', '{"event":"suite_started","tracks":3}\n')
        archive.writestr('text/worker_events.jsonl', '{"event":"epoch_completed","epoch":1}\n')
        archive.writestr('hybrid__worker.log', '<script>unsafe HTML</script>\nworker finished\n')
    key = 'results/model_tracks/logged.zip'
    page = client.get('/training', {'run': key})
    assert page.status_code == 200
    assert 'suite_started' in page.text and 'epoch_completed' in page.text
    assert '&lt;script&gt;unsafe HTML&lt;/script&gt;' in page.text
    assert 'Download full log' in page.text
    response = client.get('/training/log', {'run': key, 'artifact': 'text/worker_events.jsonl'})
    assert response.content == b'{"event":"epoch_completed","epoch":1}\n'


def test_local_logs_only_run_and_bounded_preview(dashboard):
    client, root = dashboard
    module = client.module
    folder = root / 'logged-local'
    folder.mkdir()
    payload = b'initial event\n' + b'x' * (module.LOG_PREVIEW_BYTES + 100)
    (folder / 'gnn_only__worker.log').write_bytes(payload)
    key = 'results/model_tracks/logged-local'
    page = client.get('/training', {'run': key})
    assert page.status_code == 200 and 'Preview truncated' in page.text
    preview = client.get('/training/log', {'run': key, 'artifact': 'gnn_only__worker.log'})
    assert '65,536 bytes' in preview.text and 'Download the full log' in preview.text
    assert len(preview.content) < len(payload)
    response = module.log(key, 'gnn_only__worker.log', download=True)
    assert "filename*=UTF-8''gnn_only__worker.log" in response.headers['content-disposition']
    # Read through the route's safe bounded helper; streaming iterator is
    # verified separately without any thread-portal scheduling.
    assert module.read(folder, 'gnn_only__worker.log') == payload


def test_log_route_rejects_traversal_symlinks_and_unrelated_artifacts(dashboard):
    client, root = dashboard
    key = 'results/model_tracks/smoke.zip'
    for artifact in ('../secret.log', '/secret.log', 'text/checkpoint/model.safetensors'):
        assert client.get('/training/log', {'run': key, 'artifact': artifact}).status_code == 404
    folder = root / 'logs'
    folder.mkdir()
    (folder / 'suite_events.jsonl').write_text('{}\n')
    (folder / 'real.log').write_text('inside')
    (folder / 'linked.log').symlink_to(folder / 'real.log')
    outside = root.parent / 'outside'
    outside.mkdir()
    (outside / 'secret.log').write_text('private')
    (folder / 'linked-folder').symlink_to(outside, target_is_directory=True)
    key = 'results/model_tracks/logs'
    for artifact in ('linked.log', 'linked-folder/secret.log'):
        assert client.get('/training/log', {'run': key, 'artifact': artifact}).status_code == 404


def test_zip_logs_truncated_and_download_stream_is_complete(dashboard, monkeypatch):
    client, root = dashboard
    module = client.module
    payload = b'event\n' * 14000
    with zipfile.ZipFile(root / 'large.zip', 'w') as archive:
        archive.writestr('text__worker.log', payload)
    key = 'results/model_tracks/large.zip'
    assert 'Preview truncated' in client.get('/training/log', {'run': key, 'artifact': 'text__worker.log'}).text
    # Replace Starlette's threadpool bridge only for consumption in this
    # sandbox test. The download generator itself is production code.
    async def iterate(iterator):
        for item in iterator:
            yield item
    monkeypatch.setattr('starlette.responses.iterate_in_threadpool', iterate)
    response = module.log(key, 'text__worker.log', download=True)
    assert download_bytes(response) == payload


def test_collected_sibling_events_visible_without_modifying_archive(dashboard):
    client, root = dashboard
    sidecar = root / 'smoke.events.jsonl'
    sidecar.write_text('{"phase":"publication","status":"complete"}\n')
    key = 'results/model_tracks/smoke.zip'
    page = client.get('/training', {'run': key})
    assert '__collected__/suite_events.jsonl' in page.text
    assert 'publication' in page.text
    log = client.get('/training/log', {'run': key, 'artifact': '__collected__/suite_events.jsonl'})
    assert log.content == sidecar.read_bytes()
    sidecar.unlink()
    sidecar.symlink_to(root / 'external-events.jsonl')
    (root / 'external-events.jsonl').write_text('secret')
    assert client.get('/training/log', {'run': key, 'artifact': '__collected__/suite_events.jsonl'}).status_code == 404
