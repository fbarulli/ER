"""Leakage/alignment guards and local worker lifecycle; no external telemetry."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from graph_tracks.artifacts import name
from graph_tracks.data import NUMERIC, RELATIONS, file_hash, fit_vocabulary, load_records, load_text_cache, tensorize
from graph_tracks.model import AttributeGNN
from graph_tracks.train import load_pairs, train
from graph_tracks.infer import GraphEncoder, export


def population():
    return [{'sku_id': f'{split}-{i}', 'split': split,
             'attribute': {'brand': ['alpha' if i < 2 else 'beta'],
                            'flavor': ['lemon' if i < 2 else 'orange']},
             'numeric': {'volume_ml': [330], 'pack': [1]}}
            for split in ('train', 'dev', 'test') for i in range(3)]


def inputs(tmp_path, hybrid=False):
    records = population()
    listings = tmp_path / 'listings.json'
    listings.write_text(json.dumps({'schema': 'er-graph-listings-v1', 'listings': records}))
    pairs = tmp_path / 'pairs.csv'
    pd.DataFrame([{'sku_id1': f'{s}-0', 'sku_id2': f'{s}-{j}',
                   'label': int(j == 1), 'split': s}
                  for s in ('train', 'dev', 'test') for j in (1, 2)]).to_csv(pairs, index=False)
    cfg = {'track': 'hybrid' if hybrid else 'gnn_only', 'listings': str(listings),
           'pairs': str(pairs), 'output_dir': str(tmp_path / 'run'),
           'hidden_dim': 8, 'output_dim': 8, 'epochs': 2, 'device': 'cpu',
           'allow_unmanifested_inputs': True, 'wandb': {'mode': 'disabled'}, 'dvc': {'enabled': False}}
    cache = None
    if hybrid:
        cache = tmp_path / 'text.npz'
        np.savez(cache, ids=np.asarray([r['sku_id'] for r in records]),
                 embeddings=np.random.default_rng(42).normal(size=(len(records), 6)).astype('float32'),
                 metadata=json.dumps({'checkpoint_sha256': 'test-checkpoint', 'composition': {'profile': 'test'}}))
        cfg['text_cache'] = str(cache)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    return listings, pairs, cache, config


def disable_tracking(monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'off')
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.delenv('EUROMONITOR_RESULTS_DIR', raising=False)


def test_split_and_label_guards(tmp_path):
    listings, pairs, _, _ = inputs(tmp_path)
    records = load_records(listings)
    frame = pd.read_csv(pairs)
    frame.loc[0, 'sku_id2'] = 'dev-1'
    frame.to_csv(pairs, index=False)
    with pytest.raises(ValueError, match='split boundary'):
        load_pairs(pairs, records)
    records[0]['attribute']['gtin'] = ['123']
    listings.write_text(json.dumps({'schema': 'er-graph-listings-v1', 'listings': records}))
    with pytest.raises(ValueError, match='unsupported attribute'):
        load_records(listings)


def test_vocabulary_and_inductive_batch_invariance():
    records = population()
    records[-1]['attribute']['brand'] = ['held-out-only']
    vocab = fit_vocabulary(records)
    assert 'held-out-only' not in vocab['brand']
    model = AttributeGNN(vocab, hidden=8, output=8).eval()
    support = tensorize([r for r in records if r['split'] == 'train'], vocab, 'cpu')
    with torch.no_grad():
        states = model.context(support)
        assert torch.count_nonzero(states['brand'][0]) == 0
        alone = model.encode(tensorize([records[-1]], vocab, 'cpu'), states)
        together = model.encode(tensorize(records[-3:], vocab, 'cpu'), states)[-1:]
    torch.testing.assert_close(alone, together)


def test_cache_alignment_and_zero_guard(tmp_path):
    path = tmp_path / 'cache.npz'
    metadata = json.dumps({'checkpoint_sha256': 'x', 'composition': 'test'})
    np.savez(path, ids=['b', 'a'], embeddings=np.eye(2, dtype='float32'), metadata=metadata)
    vectors, _ = load_text_cache(path, ['a', 'b'])
    np.testing.assert_array_equal(vectors, np.eye(2)[::-1])
    with pytest.raises(ValueError, match='missing'):
        load_text_cache(path, ['unknown'])
    np.savez(path, ids=['a'], embeddings=np.zeros((1, 2), dtype='float32'), metadata=metadata)
    with pytest.raises(ValueError, match='zero'):
        load_text_cache(path, ['a'])
    np.savez(path, ids=['a'], embeddings=np.ones((1, 2), dtype='float64'), metadata=metadata)
    with pytest.raises(ValueError, match='float32'):
        load_text_cache(path, ['a'])


def test_worker_export_and_resume(tmp_path, monkeypatch, hybrid=False):
    disable_tracking(monkeypatch)
    listings, pairs, cache, config = inputs(tmp_path, hybrid)
    checkpoint = train(config, run_tag='smoke')
    track = 'hybrid' if hybrid else 'gnn_only'
    run = tmp_path / 'run' / name(track, 'smoke')
    assert json.loads((run / name(track, 'graph_worker_result.json')).read_text())['status'] == 'ok'
    assert (run / name(track, 'gradient_metrics.jsonl')).is_file()
    scoring = tmp_path / 'query_pairs.csv'
    pd.read_csv(pairs)[['sku_id1', 'sku_id2']].to_csv(scoring, index=False)
    target = export(checkpoint, listings, tmp_path / 'export', text_cache=cache,
                    pairs=scoring, build_index=True, batch_size=2)
    vectors = np.load(target / name(track, 'vectors.npz'), allow_pickle=False)['embeddings']
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    assert pd.read_csv(target / name(track, 'pair_scores.csv')).score.between(0, 1).all()
    from training.hnsw_index import PersistentHnswIndex
    index = PersistentHnswIndex(target / name(track, 'index'), ef_construction=200, M=16, ef_search=100)
    index.load(ids=[r['sku_id'] for r in population()], dim=8, checkpoint=checkpoint,
               model_name='hybrid' if hybrid else 'gnn_only', preprocessing_fingerprint=file_hash(listings))
    labels, distances = index.query(vectors[:1], top_k=3)
    assert labels.shape == (1, 3)
    assert np.isfinite(distances).all()
    encoder = GraphEncoder(checkpoint)
    text = None if cache is None else load_text_cache(cache, [r['sku_id'] for r in population()])[0]
    np.testing.assert_allclose(vectors, encoder.encode(population(), text, 1), atol=1e-6)
    # Move the whole checkpoint tree, removing original absolute best paths.
    relocated = tmp_path / 'relocated'
    shutil.copytree(run / '_checkpoints', relocated)
    resume = relocated / 'gnn_only' if not hybrid else relocated / 'hybrid'
    resume = resume / 'smoke_f0' / 'checkpoint-2' / name(track, 'graph_model.pt')
    shutil.rmtree(run)
    cfg = yaml.safe_load(config.read_text())
    cfg.update(epochs=3, output_dir=str(tmp_path / 'resumed'))
    config.write_text(yaml.safe_dump(cfg))
    resumed_best = train(config, run_tag='continued', resume=resume)
    assert resumed_best.is_relative_to(tmp_path / 'resumed')
    GraphEncoder(resumed_best)
    cfg['seed'] = 99
    cfg['output_dir'] = str(tmp_path / 'bad-resume')
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='resume config mismatch'):
        train(config, run_tag='bad', resume=resume)


def test_prepared_export_shared_identity(tmp_path):
    from graph_tracks.prepare import prepare
    listings, pairs, _, _ = inputs(tmp_path)
    records = load_records(listings)
    catalog = tmp_path / 'catalog.csv'
    splits = tmp_path / 'splits.csv'
    pd.DataFrame([{'sku_id': r['sku_id'], 'sku_name_eng': 'Lemon drink 330 ml',
                   'brand': 'Example', 'attribute': 'Flavor: Lemon; Pack Size: 1',
                   'gtin': '4006381333931'} for r in records]).to_csv(catalog, index=False)
    pd.DataFrame([{'sku_id': r['sku_id'], 'split': r['split']} for r in records]).to_csv(splits, index=False)
    prepared = prepare(catalog, splits, pairs, tmp_path / 'prepared')
    out = load_records(prepared)
    assert out[0]['attribute']['brand']
    assert 'gtin' not in prepared.read_text()
    manifest = json.loads((prepared.parent / 'input_manifest.json').read_text())
    assert manifest['identity_extractor'] == 'core.sku_identity.row_identity'
    assert manifest['feature_scope'].startswith('derived from core.sku_identity.graph_schema')
    assert manifest['relations'] == list(RELATIONS)
    assert manifest['numeric'] == list(NUMERIC)
    # Inference-only batches do not need a dummy training node.
    listings.write_text(json.dumps({'schema': 'er-graph-listings-v1', 'listings': out[-3:]}))
    assert len(load_records(listings, require_training=False)) == 3
    with pytest.raises(ValueError, match='training listing'):
        load_records(listings)


def test_complete_offline_wandb_dvc_lifecycle(tmp_path, monkeypatch, hybrid=False):
    """Exercise real W&B SDK and DVC add/push/clean-pull without external services."""
    disable_tracking(monkeypatch)
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_SILENT', 'true')
    monkeypatch.setenv('WANDB_CONSOLE', 'off')
    listings, pairs, cache, config = inputs(tmp_path, hybrid)
    cfg = yaml.safe_load(config.read_text())
    cfg['wandb'] = {'project': 'e-r-graph-smoke', 'mode': 'offline'}
    cfg['dvc'] = {'enabled': True, 'remote': str(tmp_path / 'local-remote'), 'push': True}
    cfg['epochs'] = 1
    config.write_text(yaml.safe_dump(cfg))
    best = train(config, run_tag='offline-complete')
    track = cfg['track']
    run = tmp_path / 'run' / name(track, 'offline-complete')
    result = json.loads((run / name(track, 'graph_worker_result.json')).read_text())
    assert result['postprocess_complete'] is True
    assert result['wandb_run_id']
    assert list((run / 'wandb').glob('offline-run-*/run-*.wandb'))
    completion = run / name(track, 'completion-epoch-1')
    reports = completion / name(track, 'reports')
    summary = pd.read_csv(reports / name(track, 'model_evaluation_summary.csv'))
    assert set(summary.split) == {'dev', 'test'}
    assert summary.threshold.nunique() == 1
    assert summary.threshold_source.eq('dev_youden').all()
    assert (reports / name(track, 'score_distribution_and_pr.png')).is_file()
    manifest = json.loads((reports / name(track, 'report_manifest.json')).read_text())
    assert manifest['test_used_for_selection'] is False
    assert manifest['trained_endpoints_scored'] is False
    from graph_tracks.dvc import restore
    project = run / name(track, 'dvc-epoch-1')
    restored = restore(project, tmp_path / 'restored')
    restored_best = restored / best.relative_to(run)
    GraphEncoder(restored_best)
    assert restored_best.name.startswith(track + '__')
    assert list(restored.rglob(name(track, 'model_evaluation_summary.csv')))


def test_dev_threshold_handles_ties_and_test_does_not_fit():
    from graph_tracks.report import dev_threshold, pair_metrics
    labels = np.asarray([0, 1, 0, 1], dtype=float)
    threshold = dev_threshold(labels, np.full(4, .5))
    assert threshold == .5
    metrics = pair_metrics(labels, np.asarray([.9, .1, .8, .2]), threshold, [1])
    assert metrics['threshold'] == .5
    assert metrics['f1'] == 0


def test_track_checkpoint_name_and_integrity(tmp_path, monkeypatch):
    disable_tracking(monkeypatch)
    _, _, _, config = inputs(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg.update(epochs=1, postprocess=False)
    config.write_text(yaml.safe_dump(cfg))
    checkpoint = train(config, run_tag='integrity')
    wrong = checkpoint.with_name(name('cascade', 'graph_model.pt'))
    shutil.copy2(checkpoint, wrong)
    with pytest.raises(ValueError, match='checkpoint filename'):
        GraphEncoder(wrong)
    with checkpoint.open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        GraphEncoder(checkpoint)


def test_dvc_no_remote_restore_and_credential_url_guard(tmp_path):
    from graph_tracks.dvc import snapshot, restore
    from graph_tracks.config import DvcSpec
    from pydantic import ValidationError
    source = tmp_path / 'run'
    source.mkdir()
    (source / name('gnn_only', 'example.json')).write_text('{"test": true}')
    project = snapshot(source, 'gnn_only')
    restored = restore(project, tmp_path / 'restored')
    assert (restored / name('gnn_only', 'example.json')).read_text() == '{"test": true}'
    with pytest.raises(ValidationError, match='must not contain credentials'):
        DvcSpec(remote='https://username:password@example.com/store', push=True)


def test_prepared_manifest_hash_drift_fails(tmp_path, monkeypatch):
    disable_tracking(monkeypatch)
    _, _, _, config = inputs(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    manifest = tmp_path / 'input_manifest.json'
    manifest.write_text(json.dumps({'listings_sha256': 'wrong'}))
    cfg['input_manifest'] = str(manifest)
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='prepared input mismatch'):
        train(config, run_tag='stale-input')


def test_same_run_resume_preserves_history_and_new_completion(tmp_path, monkeypatch):
    disable_tracking(monkeypatch)
    _, _, _, config = inputs(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg.update(epochs=1)
    config.write_text(yaml.safe_dump(cfg))
    checkpoint = train(config, run_tag='same-run')
    cfg['epochs'] = 2
    config.write_text(yaml.safe_dump(cfg))
    train(config, run_tag='same-run', resume=checkpoint)
    run = tmp_path / 'run' / name('gnn_only', 'same-run')
    history = (run / name('gnn_only', 'epoch_metrics.jsonl')).read_text().splitlines()
    assert [json.loads(line)['epoch'] for line in history] == [1, 2]
    assert (run / name('gnn_only', 'completion-epoch-1')).is_dir()
    assert (run / name('gnn_only', 'completion-epoch-2')).is_dir()


def test_inference_only_split_does_not_enter_training(tmp_path):
    listings, _, _, _ = inputs(tmp_path)
    records = population()[-3:]
    for record in records:
        record['split'] = 'inference'
    listings.write_text(json.dumps({'schema': 'er-graph-listings-v1', 'listings': records}))
    assert len(load_records(listings, require_training=False)) == 3
    with pytest.raises(ValueError, match='encoding only'):
        load_records(listings)


def test_portable_bundle_inputs_resume_and_secret_exclusion(tmp_path, monkeypatch):
    import zipfile
    from graph_tracks.bundle import bundle
    disable_tracking(monkeypatch)
    listings, pairs, _, config = inputs(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg.update(epochs=1, postprocess=False)
    config.write_text(yaml.safe_dump(cfg))
    train(config, run_tag='bundle')
    run = tmp_path / 'run' / name('gnn_only', 'bundle')
    secret_dir = run / name('gnn_only', 'dvc-local') / '.dvc'
    secret_dir.mkdir(parents=True)
    (secret_dir / 'config.local').write_text('password=do-not-export')
    archive = bundle(run, tmp_path / name('gnn_only', 'bundle.zip'))
    restored = tmp_path / 'restored-run'
    with zipfile.ZipFile(archive) as saved:
        assert not any(key.endswith('config.local') for key in saved.namelist())
        assert all(b'do-not-export' not in saved.read(key) for key in saved.namelist())
        saved.extractall(restored)
    manifest = json.loads((restored / name('gnn_only', 'run_manifest.json')).read_text())
    shutil.rmtree(run)
    listings.unlink()
    pairs.unlink()
    cfg.update(epochs=2, output_dir=str(tmp_path / 'resumed-bundle'))
    for key, relative in manifest['input_artifacts'].items():
        cfg[key] = str(restored / relative)
    config.write_text(yaml.safe_dump(cfg))
    checkpoint = restored / '_checkpoints' / 'gnn_only' / 'bundle_f0' / 'checkpoint-1' / name('gnn_only', 'graph_model.pt')
    selected = train(config, run_tag='resumed-bundle', resume=checkpoint)
    assert selected.is_file()


def test_prepare_rejects_scoped_hold_without_blocking_gtin_peers(tmp_path):
    from graph_tracks.prepare import prepare
    from core.identity_policy import review_policy
    holds = review_policy().quarantined_listings
    assert holds
    sku_id, hold = next(iter(holds.items()))
    listings, pairs, _, _ = inputs(tmp_path)
    records = load_records(listings)
    catalog = tmp_path / 'catalog.csv'
    splits = tmp_path / 'splits.csv'
    rows = [{'sku_id': r['sku_id'], 'sku_name_eng': 'Example drink', 'brand': 'Example',
             'gtin': hold.expected_gtin} for r in records]
    assignments = [{'sku_id': r['sku_id'], 'split': r['split']} for r in records]
    pd.DataFrame(rows).to_csv(catalog, index=False)
    pd.DataFrame(assignments).to_csv(splits, index=False)
    assert prepare(catalog, splits, pairs, tmp_path / 'good-peers').is_file()
    rows[0]['sku_id'] = sku_id
    pd.DataFrame(rows).to_csv(catalog, index=False)
    with pytest.raises(ValueError, match='quarantined identity groups/listings'):
        prepare(catalog, splits, pairs, tmp_path / 'blocked')


def test_plateau_stops_and_resume_retains_control_state(tmp_path, monkeypatch, hybrid=False):
    import importlib
    worker = importlib.import_module('graph_tracks.train')
    disable_tracking(monkeypatch)
    monkeypatch.setattr(worker, 'quality', lambda *_: {'dev_pr_auc': 0.5,
                                                       'dev_precision_at_recall': 0.4,
                                                       'dev_p_at_r95': 0.4,
                                                       'agreed_recall': 0.95})
    _, _, _, config = inputs(tmp_path, hybrid=hybrid)
    cfg = yaml.safe_load(config.read_text())
    cfg.update(epochs=10, postprocess=False)
    config.write_text(yaml.safe_dump(cfg))
    selected = train(config, run_tag='plateau')
    assert selected.parent.name == 'checkpoint-1'
    output = Path(cfg['output_dir']) / f"{cfg['track']}__plateau"
    rows = [json.loads(line) for line in (output / name(cfg['track'], 'epoch_metrics.jsonl')).read_text().splitlines()]
    assert len(rows) == 4
    assert rows[2]['next_learning_rate'] == pytest.approx(cfg.get('learning_rate', 0.001) * 0.5)
    last = selected.parent.parent / 'checkpoint-4' / selected.name
    saved = torch.load(last, weights_only=False)
    assert saved['bad_epochs'] == 3
    assert saved['scheduler']['last_epoch'] == 4
    train(config, run_tag='plateau', resume=last)
    assert len((output / name(cfg['track'], 'epoch_metrics.jsonl')).read_text().splitlines()) == 4
    result = json.loads((output / name(cfg['track'], 'graph_worker_result.json')).read_text())
    assert result['completed_epochs'] == 4
    assert result['early_stopped'] is True
