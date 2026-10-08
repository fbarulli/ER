import json
import numpy as np
import pytest
import torch
from graph_tracks.data import fit_vocabulary, tensorize
from graph_tracks.model import AttributeGNN
from graph_tracks.prepared_inputs import prepare_training, load_plan, load_batch, save_batch
from test_graph_tracks import inputs, population, disable_tracking


@pytest.mark.parametrize('mutation', ['missing', 'reordered', 'repeated', 'wrong_pairs', 'wrong_labels'])
def test_prepared_catalog_rejects_lost_or_misaligned_rows(tmp_path, mutation):
    from graph_tracks.prepared_inputs import PreparedGraphInputs
    from graph_tracks.train import load_pairs
    listings, pairs, _, _ = inputs(tmp_path)
    prepare_training(listings, pairs, batch_size=2)
    plan, archive = load_plan(listings, pairs)
    arrays = {key: archive[key].copy() for key in archive.files}
    archive.close()
    if mutation == 'missing':
        plan['query_batches'].pop()
    elif mutation == 'reordered':
        plan['query_batches'].reverse()
    elif mutation == 'repeated':
        plan['query_batches'].append(plan['query_batches'][-1])
    elif mutation == 'wrong_pairs':
        arrays['train/catalog_pairs'][0] = arrays['train/catalog_pairs'][0][::-1]
    else:
        arrays['train/labels'][0] = 1 - arrays['train/labels'][0]
    with pytest.raises(ValueError, match='prepared graph'):
        PreparedGraphInputs(plan=plan, arrays=arrays).validate_catalog(
            population(), load_pairs(pairs, population()))


def test_graph_only_export_rejects_partial_embeddings(tmp_path, monkeypatch):
    from graph_tracks.train import train
    from graph_tracks.infer import export, GraphEncoder
    import yaml
    disable_tracking(monkeypatch)
    listings, _, _, config = inputs(tmp_path)
    settings = yaml.safe_load(config.read_text())
    settings.update(postprocess=False, epochs=1)
    config.write_text(yaml.safe_dump(settings))
    checkpoint = train(config, run_tag='partial')
    encoder = GraphEncoder(checkpoint)
    monkeypatch.setattr(GraphEncoder, 'encode', lambda *args, **kwargs: np.ones((1, 8), dtype=np.float32))
    destination = tmp_path/'export'
    with pytest.raises(ValueError, match='embedding/ID population'):
        export(checkpoint, listings, destination, encoder=encoder)
    assert not destination.exists()


def test_prepared_graph_matches_values_and_gradients(tmp_path):
    listings, pairs, _, _ = inputs(tmp_path)
    prepare_training(listings, pairs, batch_size=2)
    plan, arrays = load_plan(listings, pairs)
    records = population()
    vocab = fit_vocabulary(records)
    raw = tensorize(records[:3], vocab, 'cpu')
    prepared = load_batch(arrays, 'train', 'cpu', vocab)
    model = AttributeGNN(vocab, 8, 8)
    expected = model.encode(raw, model.context(raw))
    expected.sum().backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    initial = model.initial(prepared)
    actual = model.encode(prepared, model.context(prepared, initial=initial), initial=initial)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    for n, p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(p.grad, gradients[n])
    assert plan['ids'] == [r['sku_id'] for r in records]
    assert len(plan['query_batches']) == 5
    arrays.close()
    pairs.write_text(pairs.read_text() + '\n')
    with pytest.raises(ValueError, match='pair mismatch'):
        load_plan(listings, pairs)


def test_prepared_graph_rejects_corrupt_topology():
    vocab = fit_vocabulary(population())
    arrays = {}
    save_batch(arrays, 'x', tensorize(population()[:3], vocab, 'cpu'), vocab)
    arrays['x/brand/value'][0] = len(vocab['brand']) + 1
    with pytest.raises(ValueError, match='bounds'):
        load_batch(arrays, 'x', 'cpu', vocab)


def test_saved_graph_report_performs_no_forward(tmp_path, monkeypatch):
    from graph_tracks.train import train
    from graph_tracks.config import load_config
    from graph_tracks.infer import forward_outputs, GraphEncoder
    from graph_tracks.report import complete
    disable_tracking(monkeypatch)
    listings, pairs, _, config = inputs(tmp_path)
    settings = load_config(config)
    settings.postprocess = False
    settings.build_index = False
    import yaml
    config.write_text(yaml.safe_dump(settings.model_dump()))
    prepare_training(listings, pairs, batch_size=2)
    checkpoint = train(config, run_tag='prepared')
    saved = forward_outputs(checkpoint, listings, pairs, tmp_path/'forward', settings)
    def fail(*args, **kwargs):
        raise AssertionError('report must not construct model')
    monkeypatch.setattr(GraphEncoder, '__init__', fail)
    output = tmp_path/'report'
    output.mkdir()
    result = complete(checkpoint, listings, pairs, output, settings, saved_inference=saved)
    assert {r['split'] for r in result['summary']} == {'dev', 'test'}
    manifest = json.loads((saved/'gnn_only__export_manifest.json').read_text())
    assert manifest['forward_only'] and manifest['embedding_dtype'] == 'float32'


def test_new_query_preparation_preserves_inductive_contract(tmp_path, monkeypatch):
    from graph_tracks.train import train
    from graph_tracks.infer import export, GraphEncoder
    from graph_tracks.prepared_inputs import prepare_inference
    import yaml
    disable_tracking(monkeypatch)
    _, _, _, config = inputs(tmp_path)
    settings = yaml.safe_load(config.read_text())
    settings.update(postprocess=False, epochs=1)
    config.write_text(yaml.safe_dump(settings))
    checkpoint = train(config, run_tag='query')
    records = [dict(population()[-1], sku_id='new-query', split='inference')]
    query = tmp_path/'new'/ 'listings.json'
    query.parent.mkdir()
    query.write_text(json.dumps({'schema':'er-graph-listings-v1','listings':records}))
    expected = GraphEncoder(checkpoint).encode(records)
    prepare_inference(query, checkpoint, batch_size=1)
    result = export(checkpoint, query, tmp_path/'new-output')
    with np.load(result/'gnn_only__vectors.npz') as cache:
        np.testing.assert_allclose(cache['embeddings'], expected, atol=1e-6)
        assert cache['ids'].tolist() == ['new-query']

