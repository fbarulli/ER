import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


def worker():
    path = Path(__file__).resolve().parents[1] / 'scripts/encode_prepared_embeddings.py'
    spec = importlib.util.spec_from_file_location('gpu_embedding_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_worker_encodes_exact_local_texts_without_composition(tmp_path, monkeypatch):
    import torch
    module = worker()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'model').write_bytes(b'frozen-checkpoint')
    request = {'schema': 'er-embedding-request-v2', 'ids': ['a', 'b'],
               'texts': ['locally composed a', 'locally composed b'],
               'metadata': {'checkpoint_sha256': module.checkpoint_hash(checkpoint)}}
    from core import encoding_inputs
    monkeypatch.setattr(encoding_inputs,'tokenization_policy',lambda model:{'truncation':False})
    tokenfile = tmp_path/'prepared_text.npz'
    np.savez(tokenfile,**{'text/0/input_ids':np.arange(len(request['ids'])).reshape(-1,1)})
    request['prepared_text'] = {'sha256':hashlib.sha256(tokenfile.read_bytes()).hexdigest(), 'tokenization':{'truncation':False},
        'token_batches':[{'prefix':'text/0','keys':['input_ids'],'constants':{}}]}
    source = tmp_path / 'request.json'
    source.write_text(json.dumps(request))
    output = tmp_path / 'vectors.npz'
    calls = []
    class Encoder:
        def __init__(self, path, **kwargs):
            assert path == str(checkpoint)
            assert kwargs == {'device': 'cuda', 'local_files_only': True}
        def eval(self):
            pass
        def __call__(self, features):
            calls.append(features['input_ids'].tolist())
            return {'sentence_embedding':torch.eye(2)}
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', types.SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setattr(sys, 'argv', ['encode.py', '--request', str(source),
                                    '--checkpoint', str(checkpoint), '--output', str(output)])
    native_tensor = torch.as_tensor
    monkeypatch.setattr(torch,'as_tensor',lambda data,**kwargs:native_tensor(data,device='cpu'))
    module.main()
    assert calls == [[[0],[1]]]
    with np.load(output, allow_pickle=False) as result:
        assert result['ids'].tolist() == request['ids']
        assert json.loads(result['metadata'].item())['request_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert output.with_suffix('.sha256').read_text() == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match='never reuses'):
        module.main()


def test_explicit_cpu_smoke_encodes_without_cuda(tmp_path, monkeypatch):
    import torch
    module = worker()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'model').write_bytes(b'frozen-checkpoint')
    request = {'schema':'er-embedding-request-v2','ids':['a'], 'texts':['prepared locally'],
               'metadata':{'checkpoint_sha256':module.checkpoint_hash(checkpoint)}}
    from core import encoding_inputs
    monkeypatch.setattr(encoding_inputs,'tokenization_policy',lambda model:{'truncation':False})
    tokenfile = tmp_path/'prepared_text.npz'
    np.savez(tokenfile,**{'text/0/input_ids':np.arange(len(request['ids'])).reshape(-1,1)})
    request['prepared_text'] = {'sha256':hashlib.sha256(tokenfile.read_bytes()).hexdigest(), 'tokenization':{'truncation':False},
        'token_batches':[{'prefix':'text/0','keys':['input_ids'],'constants':{}}]}
    source = tmp_path / 'request.json'
    source.write_text(json.dumps(request))
    output = tmp_path / 'vectors.npz'
    class Encoder:
        def __init__(self,path,**kwargs):
            assert kwargs == {'device':'cpu','local_files_only':True}
        def eval(self):
            pass
        def __call__(self,features):
            return {'sentence_embedding':torch.tensor([[1.,0.]])}
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(SentenceTransformer=Encoder))
    monkeypatch.setattr(sys,'argv',['encode.py','--request',str(source),'--checkpoint',str(checkpoint),
                                  '--output',str(output),'--device','cpu'])
    module.main()
    assert output.is_file()
