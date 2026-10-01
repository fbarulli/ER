import importlib.util
import json
import stat
from pathlib import Path
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


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
    app = FastAPI()
    app.include_router(module.router)
    return TestClient(app), root


def test_downloaded_suite_plots_and_metrics_render(dashboard):
    client, _ = dashboard
    response = client.get('/training')
    assert response.status_code == 200
    assert 'dev' in response.text and '0.7' in response.text
    assert 'cpu' in response.text and 'Complete suite' in response.text
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
