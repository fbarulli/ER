import json
from types import SimpleNamespace
import pytest
import torch
from core.encoding_inputs import prepare_text_features, tokenization_policy
from graph_tracks.data import RELATIONS, tensorize
from graph_tracks.pooling import topology
from model_tracks.ablation_inputs import save_batch, load_batch


class NativeEncoder:
    def __init__(self,limit=8):
        self.tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:json.dumps({'vocab':'native'})),special_tokens_map={'pad_token':'[PAD]'})
        self.module = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=limit),preprocess=self.preprocess)
        self.truncate_dim = None
        self.default_prompt_name = 'query'
        self.prompts = {'query':'prefix '}
    def __getitem__(self,n):
        return self.module
    def preprocess(self,texts,prompt=None,processing_kwargs=None):
        assert processing_kwargs['text']['truncation'] is False
        rows = [[101]+list(range(len((prompt+text).split())))+[102] for text in texts]
        width = max(map(len,rows))
        return {'input_ids':torch.tensor([row+[0]*(width-len(row)) for row in rows]),
                'attention_mask':torch.tensor([[1]*len(row)+[0]*(width-len(row)) for row in rows]),
                'prompt_length':2,'modality':'text'}


def test_native_prompt_special_tokens_and_padding_without_truncation():
    model = NativeEncoder()
    features = prepare_text_features(model,['one','one two three'])
    assert features['attention_mask'].sum(-1).tolist() == [4,6]
    assert features['input_ids'][1,5].item() == 102
    assert features['prompt_length'] == 2
    assert tokenization_policy(model)['truncation'] is False


def test_long_complete_input_fails_instead_of_truncating():
    with pytest.raises(ValueError,match='zero truncation required'):
        prepare_text_features(NativeEncoder(limit=4),['one two three four'])


def test_embedding_dimension_truncation_is_rejected():
    model = NativeEncoder(); model.truncate_dim = 2
    with pytest.raises(ValueError,match='embedding truncation'):
        tokenization_policy(model)


def test_prepared_graph_topology_roundtrip_matches_existing():
    vocabulary = {r:['alpha'] for r in RELATIONS}
    records = [{'sku_id':'a','split':'dev','attribute':{'brand':['alpha']},'numeric':{'volume_ml':[500]}},
               {'sku_id':'b','split':'dev','attribute':{},'numeric':{}}]
    baseline = tensorize(records,vocabulary,'cpu')
    arrays = {}; save_batch(arrays,'query',baseline,vocabulary)
    prepared = load_batch(arrays,'query','cpu',vocabulary)
    torch.testing.assert_close(baseline.numeric,prepared.numeric)
    for relation in RELATIONS:
        for attribute in (False,True):
            count = len(vocabulary[relation])+1 if attribute else len(records)
            expected = topology(baseline,relation,attribute=attribute,count=count)
            actual = topology(prepared,relation,attribute=attribute,count=count)
            for left,right in zip(expected,actual):
                torch.testing.assert_close(left,right)


def test_native_training_calls_are_guarded_against_truncation():
    from core.encoding_inputs import enable_zero_truncation
    model = NativeEncoder(limit=4)
    enable_zero_truncation(model)
    features = model.preprocess(['one'],prompt='')
    assert features['attention_mask'].sum().item() == 3
    with pytest.raises(ValueError,match='zero truncation required'):
        model.preprocess(['one two three four'],prompt='')
