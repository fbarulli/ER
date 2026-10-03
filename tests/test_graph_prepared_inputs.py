import json
import numpy as np
import pytest
import torch
from graph_tracks.data import fit_vocabulary, tensorize
from graph_tracks.model import AttributeGNN
from graph_tracks.prepared_inputs import prepare_training, load_plan, load_batch, save_batch
from test_graph_tracks import inputs, population, disable_tracking


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


def test_manifested_hybrid_copied_inputs_relocate(tmp_path, monkeypatch):
    import pandas as pd
    import yaml
    import shutil
    from graph_tracks.prepare import prepare
    from graph_tracks.train import train
    from graph_tracks.preflight import preflight
    from graph_tracks.text_cache import checkpoint_hash
    from training.prepare_embeddings import prepare_request
    disable_tracking(monkeypatch)
    setup = tmp_path/'setup'
    setup.mkdir()
    catalog, splits, pairs = [setup/f'{s}.csv' for s in ('eligible_catalog','splits','pairs')]
    records = population()
    pd.DataFrame([{'sku_id':r['sku_id'],'sku_name_eng':'Lemon drink 330 ml','gtin':''} for r in records]).to_csv(catalog,index=False)
    pd.DataFrame([{'sku_id':r['sku_id'],'split':r['split']} for r in records]).to_csv(splits,index=False)
    pd.DataFrame([{'sku_id1':f'{s}-0','sku_id2':f'{s}-{i}','label':int(i==1),'split':s}
                  for s in ('train','dev','test') for i in (1,2)]).to_csv(pairs,index=False)
    listings = prepare(catalog,splits,pairs,setup/'prepared')
    checkpoint = tmp_path/'baseline'
    checkpoint.mkdir()
    (checkpoint/'weights').write_text('frozen-baseline')
    digest = checkpoint_hash(checkpoint)
    (setup/'setup_manifest.json').write_text(json.dumps({'text_checkpoint_sha256':digest}))
    request = prepare_request(setup,checkpoint)
    (setup/'embedding_inputs.json').write_text(json.dumps(request))
    text = setup/'shared_minilm__embeddings.npz'
    np.savez(text,ids=request['ids'],embeddings=np.ones((len(records),4),dtype=np.float32)/2,
             metadata=json.dumps(request['metadata']))
    config = tmp_path/'hybrid.yaml'
    settings = {'track':'hybrid','listings':str(listings),'pairs':str(listings.parent/'pairs.csv'),
                'input_manifest':str(listings.parent/'input_manifest.json'),'text_cache':str(text),
                'text_checkpoint_sha256':digest,'output_dir':str(tmp_path/'runs'), 'device':'cpu',
                'hidden_dim':8,'output_dim':8,'epochs':1,'postprocess':False,
                'wandb':{'mode':'disabled'},'dvc':{'enabled':False}}
    config.write_text(yaml.safe_dump(settings))
    train(config,run_tag='portable')
    run = tmp_path/'runs/hybrid__portable'
    copied = tmp_path/'relocated'
    shutil.copytree(run,copied)
    manifest = json.loads((copied/'hybrid__run_manifest.json').read_text())
    for key,relative in manifest['input_artifacts'].items():
        if key != 'report_attributes':
            settings[key] = str(copied/relative)
    shutil.rmtree(setup)
    config.write_text(yaml.safe_dump(settings))
    assert preflight(config)['text_dimension'] == 4
