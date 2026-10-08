import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest


def exporter():
    path = Path(__file__).resolve().parents[1]/'scripts/export_atlas_embeddings.py'
    spec = importlib.util.spec_from_file_location('atlas_export',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_export_consumes_saved_vectors_in_listing_order_without_model(tmp_path,monkeypatch):
    module = exporter()
    source = tmp_path/'catalog.csv';source.write_text('sku_id,name\nb,B\na,A\n')
    cache = tmp_path/'vectors.npz'
    np.savez(cache,ids=['a','b'],embeddings=np.eye(2,dtype=np.float32),metadata=json.dumps({
        'checkpoint_size':'selected','composition':{'profile':'frozen'},'embedding_dtype':'float32'}))
    out = tmp_path/'atlas'
    monkeypatch.setattr(sys,'argv',['export','--input',str(source),'--embeddings',str(cache),'--output-dir',str(out)])
    module.main()
    assert np.array_equal(np.load(out/'embeddings.npy'),np.eye(2,dtype=np.float32)[[1,0]])
    assert pd.read_csv(out/'metadata.csv').atlas_id.tolist() == ['b','a']
    assert json.loads((out/'embedding_provenance.json').read_text())['metadata']['checkpoint_size'] == 'selected'


def test_duplicate_prediction_listing_ids_require_explicit_aggregation(tmp_path,monkeypatch):
    module = exporter()
    source = tmp_path/'catalog.csv';source.write_text('sku_id\na\n')
    predictions = tmp_path/'predictions.csv';predictions.write_text('sku_id,score\na,.1\na,.9\n')
    cache = tmp_path/'vectors.npz'
    np.savez(cache,ids=['a'],embeddings=np.asarray([[1,0]],dtype=np.float32),metadata=json.dumps({
        'checkpoint_size':'selected','composition':{'profile':'frozen'},'embedding_dtype':'float32'}))
    monkeypatch.setattr(sys,'argv',['export','--input',str(source),'--embeddings',str(cache),'--predictions',str(predictions),'--output-dir',str(tmp_path/'atlas')])
    with pytest.raises(ValueError,match='explicit aggregation'):
        module.main()
