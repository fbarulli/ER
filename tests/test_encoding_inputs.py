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


def test_tokenizer_policy_binds_semantics_but_not_transient_batch_settings():
    model = NativeEncoder()
    model.tokenizer.backend_tokenizer.to_str = lambda: json.dumps({
        'vocab':'native', 'padding':{'length':12}, 'truncation':{'max_length':4}})
    baseline = tokenization_policy(model)
    model.tokenizer.backend_tokenizer.to_str = lambda: json.dumps({'vocab':'native'})
    assert tokenization_policy(model) == baseline
    model.tokenizer.padding_side = 'left'
    assert tokenization_policy(model) != baseline
    model.tokenizer.padding_side = 'right'
    model.module.do_lower_case = True
    assert tokenization_policy(model) != baseline


def test_native_preparation_rejects_silent_row_drops():
    model = NativeEncoder()
    original = model.preprocess
    def dropping(texts,prompt=None,processing_kwargs=None):
        return original(texts[:1],prompt=prompt,processing_kwargs=processing_kwargs)
    model.preprocess = dropping
    with pytest.raises(ValueError,match='retain every input row'):
        prepare_text_features(model,['one','two'])


def test_legacy_tokenize_guard_does_not_add_prompt_twice():
    from core.encoding_inputs import enable_zero_truncation
    class Legacy(NativeEncoder):
        def __init__(self):
            super().__init__()
            del self.module.preprocess
            self.module.tokenizer = self.tokenizer
        def tokenize(self,texts):
            return NativeEncoder.preprocess(self,texts,prompt='',
                processing_kwargs={'text':{'truncation':False}})
    model = enable_zero_truncation(Legacy())
    assert model.tokenize(['prefix one'])['attention_mask'].sum().item() == 4


def test_prepared_token_loader_preserves_native_features_and_rejects_dtype_drift():
    from core.encoding_inputs import prepare_token_batches, load_token_features
    arrays = {}
    plan = prepare_token_batches(NativeEncoder(),['one','two three'],arrays,batch_size=2)
    batch = plan['token_batches'][0]
    loaded = load_token_features(arrays,batch,'cpu')
    assert loaded['prompt_length'] == 2
    assert loaded['input_ids'].dtype == torch.long
    arrays[batch['prefix']+'/input_ids'] = arrays[batch['prefix']+'/input_ids'].astype('int32')
    with pytest.raises(ValueError,match='int64'):
        load_token_features(arrays,batch,'cpu')


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


def test_cross_encoder_rejects_complete_overlength_pairs_before_predict():
    from core.encoding_inputs import enable_cross_encoder_zero_truncation
    calls = []
    class Cross:
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=6))
        max_length = 3
        def tokenizer(self,left,right,**kwargs):
            assert kwargs['truncation'] is False
            return {'input_ids':[101]+left.split()+[102]+right.split()+[102]}
        def predict(self,pairs):
            calls.append(pairs)
            return [.7]*len(pairs)
    encoder = enable_cross_encoder_zero_truncation(Cross())
    assert encoder.max_length == 6
    assert encoder.predict([['a','b']]) == [.7]
    with pytest.raises(ValueError,match='zero truncation required'):
        encoder.predict([['one two three','four five']])
    assert len(calls) == 1
