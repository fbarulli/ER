import json
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
import yaml

from test_graph_tracks import inputs, disable_tracking, population
from graph_tracks.artifacts import name


@pytest.mark.parametrize('hybrid', [False, True])
def test_graph_training_reports_decisions_phases_and_test_skip(tmp_path, monkeypatch, hybrid):
    from graph_tracks.train import train
    disable_tracking(monkeypatch)
    _, _, _, config = inputs(tmp_path, hybrid)
    settings = yaml.safe_load(config.read_text())
    settings.update(epochs=1, report_test=False)
    config.write_text(yaml.safe_dump(settings))
    checkpoint = train(config, run_tag='visible')
    track = 'hybrid' if hybrid else 'gnn_only'
    log = (tmp_path / 'run' / name(track, 'visible') / name(track, 'training.log')).read_text()
    for evidence in ('input_validation complete', 'features complete', 'epoch=1/1',
                     'classification_loss=', 'dev_pr_auc=', 'reason=strictly higher dev_pr_auc',
                     'criterion=max_dev_pr_auc', 'test skipped reason=report_test_false',
                     'postprocess complete', str(checkpoint)):
        assert evidence in log


def test_text_postprocess_reports_artifact_paths_and_calibration(tmp_path, monkeypatch, capsys):
    import pandas as pd
    import graph_tracks.text_cache
    import graph_tracks.data
    import graph_tracks.report
    import graph_tracks.report_attributes
    import training.validation_inference
    import training.hnsw_index
    from model_tracks.text_report import complete
    setup = tmp_path / 'setup'
    setup.mkdir()
    listings, pairs, _, config = inputs(setup)
    (setup / 'prepared').mkdir()
    listings.rename(setup / 'prepared/listings.json')
    pairs.rename(setup / 'prepared/pairs.csv')
    (setup / 'gnn_only.yaml').write_text(config.read_text())
    settings = yaml.safe_load((setup / 'gnn_only.yaml').read_text())
    settings.update(hnsw_ef_construction=20, hnsw_m=4, hnsw_ef_search=10, retrieval_ks=[1])
    (setup / 'gnn_only.yaml').write_text(yaml.safe_dump(settings))
    output = tmp_path / 'out'
    output.mkdir()
    checkpoint = tmp_path / 'checkpoint-1'
    vectors = np.random.default_rng(0).normal(size=(9, 6)).astype('float32')
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    monkeypatch.setattr(training.validation_inference, 'resolve_best_checkpoint',
                        lambda _: (checkpoint, {'best_metric': .8, 'global_step': 1}))
    monkeypatch.setattr(graph_tracks.text_cache, 'create_cache', lambda *a, **k: output / 'text__vectors.npz')
    monkeypatch.setattr(graph_tracks.data, 'load_text_cache', lambda *a, **k: (vectors, {}))
    monkeypatch.setattr(training.hnsw_index, 'PersistentHnswIndex', lambda *a, **k: Mock())
    monkeypatch.setattr(graph_tracks.report, '_plots', lambda *a: None)
    monkeypatch.setattr(graph_tracks.report, 'retrieval_report', lambda *a: {})
    monkeypatch.setattr(graph_tracks.report_attributes, 'write_reports', lambda *a: None)
    complete(output, setup, device='cpu', report_test=False)
    log = capsys.readouterr().out
    for evidence in ('reason=trainer_recorded_best', 'best_metric=0.8', 'vector_export complete',
                     'index_build complete', 'threshold=', 'source=dev_youden',
                     'test skipped reason=report_test_false', 'reports complete',
                     'text__completion_manifest.json'):
        assert evidence in log
    assert set(pd.read_csv(output / 'text__reports/text__scored_pairs.csv').split) == {'dev'}
