"""Leakage/alignment guards and local worker lifecycle; no external telemetry."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from graph_tracks.data import file_hash, fit_vocabulary, load_records, load_text_cache, tensorize
from graph_tracks.model import AttributeGNN
from graph_tracks.train import load_pairs, train
from graph_tracks.infer import GraphEncoder, export


def population():
    return [{'product_id': f'{split}-{i}', 'split': split,
             'attributes': {'brand': ['alpha' if i < 2 else 'beta'],
                            'flavor': ['lemon' if i < 2 else 'orange']},
             'numeric': {'volume_ml': [330], 'pack': [1]}}
            for split in ('train', 'dev', 'test') for i in range(3)]


def inputs(tmp_path, hybrid=False):
    records = population()
    listings = tmp_path / 'listings.json'
    listings.write_text(json.dumps({'schema': 'er-graph-listings-v1', 'listings': records}))
    pairs = tmp_path / 'pairs.csv'
    pd.DataFrame([{'product_id1': f'{s}-0', 'product_id2': f'{s}-{j}',
                   'label': int(j == 1), 'split': s}
                  for s in ('train', 'dev', 'test') for j in (1, 2)]).to_csv(pairs, index=False)
    cfg = {'track': 'hybrid' if hybrid else 'gnn_only', 'listings': str(listings),
           'pairs': str(pairs), 'output_dir': str(tmp_path / 'run'),
           'hidden_dim': 8, 'output_dim': 8, 'epochs': 2, 'device': 'cpu'}
    cache = None
    if hybrid:
        cache = tmp_path / 'text.npz'
        np.savez(cache, ids=np.asarray([r['product_id'] for r in records]),
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
    frame.loc[0, 'product_id2'] = 'dev-1'
    frame.to_csv(pairs, index=False)
    with pytest.raises(ValueError, match='split boundary'):
        load_pairs(pairs, records)
    records[0]['attributes']['barcode'] = ['123']
    listings.write_text(json.dumps({'schema': 'er-graph-listings-v1', 'listings': records}))
    with pytest.raises(ValueError, match='unsupported attribute'):
        load_records(listings)


def test_vocabulary_and_inductive_batch_invariance():
    records = population()
    records[-1]['attributes']['brand'] = ['held-out-only']
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
    np.savez(path, ids=['a'], embeddings=np.zeros((1, 2)), metadata=metadata)
    with pytest.raises(ValueError, match='zero'):
        load_text_cache(path, ['a'])


@pytest.mark.parametrize('hybrid', [False, True])
def test_worker_export_and_resume(tmp_path, monkeypatch, hybrid):
    disable_tracking(monkeypatch)
    listings, pairs, cache, config = inputs(tmp_path, hybrid)
    checkpoint = train(config, run_tag='smoke')
    run = tmp_path / 'run'
    assert json.loads((run / 'graph_worker_result.json').read_text())['status'] == 'ok'
    assert (run / 'gradient_metrics.jsonl').is_file()
    scoring = tmp_path / 'query_pairs.csv'
    pd.read_csv(pairs)[['product_id1', 'product_id2']].to_csv(scoring, index=False)
    target = export(checkpoint, listings, tmp_path / 'export', text_cache=cache,
                    pairs=scoring, build_index=True, batch_size=2)
    vectors = np.load(target / 'vectors.npz', allow_pickle=False)['embeddings']
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    assert pd.read_csv(target / 'pair_scores.csv').score.between(0, 1).all()
    from training.hnsw_index import PersistentHnswIndex
    index = PersistentHnswIndex(target / 'index', ef_construction=200, M=16, ef_search=100)
    index.load(ids=[r['product_id'] for r in population()], dim=8, checkpoint=checkpoint,
               model_name='hybrid' if hybrid else 'gnn_only', preprocessing_fingerprint=file_hash(listings))
    labels, distances = index.query(vectors[:1], top_k=3)
    assert labels.shape == (1, 3)
    assert np.isfinite(distances).all()
    encoder = GraphEncoder(checkpoint)
    text = None if cache is None else load_text_cache(cache, [r['product_id'] for r in population()])[0]
    np.testing.assert_allclose(vectors, encoder.encode(population(), text, 1), atol=1e-6)
    # Move the whole checkpoint tree, removing original absolute best paths.
    relocated = tmp_path / 'relocated'
    shutil.copytree(run / '_checkpoints', relocated)
    resume = relocated / 'gnn_only' if not hybrid else relocated / 'hybrid'
    resume = resume / 'smoke_f0' / 'checkpoint-2' / 'graph_model.pt'
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
    pd.DataFrame([{'product_id': r['product_id'], 'title': 'Lemon drink 330 ml',
                   'brand': 'Example', 'attributes': 'Flavor: Lemon; Pack Size: 1',
                   'barcode': '4006381333931'} for r in records]).to_csv(catalog, index=False)
    pd.DataFrame([{'product_id': r['product_id'], 'split': r['split']} for r in records]).to_csv(splits, index=False)
    prepared = prepare(catalog, splits, pairs, tmp_path / 'prepared')
    out = load_records(prepared)
    assert out[0]['attributes']['brand']
    assert 'barcode' not in prepared.read_text()
    manifest = json.loads((prepared.parent / 'input_manifest.json').read_text())
    assert manifest['identity_extractor'] == 'core.product_identity.row_identity'
    assert 'not all' in manifest['feature_scope']
    # Inference-only batches do not need a dummy training node.
    listings.write_text(json.dumps({'schema': 'er-graph-listings-v1', 'listings': out[-3:]}))
    assert len(load_records(listings, require_training=False)) == 3
    with pytest.raises(ValueError, match='training listing'):
        load_records(listings)
