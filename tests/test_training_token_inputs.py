from types import SimpleNamespace
import copy
import json
import numpy as np
import pytest
import torch
from datasets import Dataset
from training.token_inputs import prepare_training_tokens, PreparedTokenLookup, ObjectiveDataCollator, validate_training_tokens
from training.sampler import ControlledBatchSampler, resolve_composition


class Encoder:
    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps({'vocab':'native'})),
                                         special_tokens_map={'pad_token':'[PAD]'}, padding_side='right', pad_token_id=0)
        self.module = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8), preprocess=self.preprocess,
                                      get_config_dict=lambda: {'do_lower_case':False, 'max_seq_length':4})
        self.truncate_dim = None
        self.default_prompt_name = 'query'
        self.prompts = {'query':'prefix '}
    def __getitem__(self, index):
        return self.module
    def preprocess(self, texts, prompt=None, processing_kwargs=None, **kwargs):
        self.calls.append((list(texts), prompt))
        rows = [[101] + [10 + len(word) for word in ((prompt or '') + text).split()] + [102] for text in texts]
        width = max(map(len, rows))
        return {'input_ids':torch.tensor([row + [0]*(width-len(row)) for row in rows]),
                'attention_mask':torch.tensor([[1]*len(row)+[0]*(width-len(row)) for row in rows]),
                'prompt_length':1 if prompt else 0, 'modality':'text'}


def test_fixed_lookup_matches_native_prompts_padding_and_never_tokenizes():
    model = Encoder(); payload = ['one', 'one two', 'one']
    table = prepare_training_tokens(model, payload, batch_size=1)
    assert len(table['texts']) == 2
    expected = model.preprocess(['one two', 'one'], prompt='prefix ')
    lookup = PreparedTokenLookup(model, table, payload)
    model.calls.clear()
    actual = model.preprocess(['one two', 'one'], prompt='prefix ')
    for key in ('input_ids', 'attention_mask'):
        torch.testing.assert_close(actual[key], expected[key])
        assert actual[key].dtype == torch.long
    assert actual['prompt_length'] == expected['prompt_length']
    assert model.calls == []
    with pytest.raises(ValueError, match='missing local tokens'):
        model.preprocess(['unprepared fixed'])
    assert model.calls == []


def test_generated_only_registration_is_consumed_and_overflow_rejected():
    from core.encoding_inputs import enable_zero_truncation
    model = Encoder(); payload = ['one']
    table = prepare_training_tokens(model, payload)
    enable_zero_truncation(model)
    lookup = PreparedTokenLookup(model, table, payload)
    lookup.register_generated('fresh'); lookup.register_generated('fresh two')
    model.calls.clear()
    batch = model.preprocess(['one', 'fresh', 'fresh two'])
    assert batch['input_ids'].shape == (3,4)
    assert model.calls == [(['fresh', 'fresh two'], '')]
    assert not lookup.generated
    with pytest.raises(ValueError, match='missing local tokens'):
        model.preprocess(['fresh'])
    long = 'a b c d e f g'; lookup.register_generated(long)
    with pytest.raises(ValueError, match='zero truncation'):
        model.preprocess([long])


def test_native_collator_excludes_all_metadata_and_preserves_features():
    model = Encoder()
    collator = ObjectiveDataCollator(preprocess_fn=model.preprocess)
    rows = [{'anchor':'one','positive':'two','negative':'three','pair_id':7,
             'population':'twin','sampler_population':'twin','structured_features':[[1.],[2.],[3.]]}]
    batch = collator(rows)
    assert {key.split('_input_ids')[0] for key in batch if key.endswith('_input_ids')} == {'anchor','positive','negative'}
    assert [texts for texts, prompt in model.calls] == [['one'],['two'],['three']]
    assert batch['pair_id'].tolist() == [7]
    assert batch['structured_features'].dtype == torch.float32


def test_token_policy_corruption_overflow_and_dtype_are_rejected():
    model = Encoder(); payload=['one']; table=prepare_training_tokens(model,payload)
    bad=copy.deepcopy(table); bad['texts'].append('one')
    with pytest.raises(ValueError,match='unique'): validate_training_tokens(bad)
    bad=copy.deepcopy(table); bad['variants']['']['rows'][0]['input_ids']=np.array([1.,2.,3.])
    with pytest.raises(ValueError,match='integer'): validate_training_tokens(bad)
    bad=copy.deepcopy(table); bad['policy']['input_token_limit']=2
    with pytest.raises(ValueError,match='policy mismatch'): PreparedTokenLookup(model,bad,payload)
    with pytest.raises(ValueError,match='zero truncation'): prepare_training_tokens(model,['a b c d e f g'])


def test_sampler_covers_imbalanced_populations_defers_shared_text_and_ignores_transform():
    ds = Dataset.from_dict({'anchor':['shared','shared','c','d','e'], 'positive':['a','b','p','q','r'],
                            'negative':['n1','n2','n3','n4','n5'], 'pair_id':list(range(5)),
                            'population':['base','base','base','base','twin']})
    ds.set_transform(lambda batch: (_ for _ in ()).throw(AssertionError('must not materialize dynamic transform')))
    sampler = ControlledBatchSampler(ds, batch_size=3, composition={'base':2,'twin':1},seed=42)
    batches=list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(5))
    assert not any(0 in batch and 1 in batch for batch in batches)
    assert len(sampler)==len(batches)
    assert list(sampler)==batches
    sampler.set_epoch(1)
    assert sorted(index for batch in sampler for index in batch)==list(range(5))
    assert resolve_composition({'base':8,'masked':4,'twin':4}, ['base','twin'],64)=={'base':43,'twin':21}


def test_ann_search_once_train_only_and_existing_does_not_consume_quota(monkeypatch,tmp_path):
    import pandas as pd
    import training.ann_refresh as ann
    calls=[]
    frame=pd.DataFrame({'gtin':['train-a','heldout','train-b','train-c'], 'brand':['a','h','b','c']})
    def mine(df,emb,**kwargs):
        calls.append((df.copy(),emb.copy(),kwargs))
        return np.array([[0,1],[0,2],[1,2]]),np.array([.95,.9,.85])
    monkeypatch.setattr(ann,'mine_hard_negatives',mine)
    monkeypatch.setattr(ann,'pairs_in_set',lambda pairs,row_gtins,allowed:
                        np.array([str(row_gtins[a]) in allowed and str(row_gtins[b]) in allowed for a,b in pairs]))
    pairs,stats=ann.refresh_finetuned_ann(None,frame,['a','held','b','c'],np.array(frame.gtin),
        train_gtins={'train-a','train-b','train-c'},existing=np.array([[0,2]]),step=1,epoch=1,
        output_path=tmp_path/'ann.csv',target=2,configured_band=(.8,1.),band_mode='fixed',k=3,
        candidate_multiplier=4,score_quantiles=(.1,.9),max_per_canonical=3,max_per_brand=3,
        batch_size=2,max_seq_length=512,exclude_conflicting=False,embeddings=np.eye(4,dtype=np.float32))
    assert len(calls)==1
    assert calls[0][0].gtin.tolist()==['train-a','train-b','train-c']
    assert pairs.tolist()==[[0,3],[2,3]]
    assert stats['rejected_existing']==1
    assert stats['shortfall_count']==0
    assert pd.read_csv(tmp_path/'ann.csv').row_a.tolist()==[0,2]


def test_objective_plan_freezes_native_mnrl_rows_and_complete_epoch_order():
    from training.training import _prepare_objective_plan
    payload=['anchor a','positive a','negative a','anchor b','positive b','negative b']
    result=_prepare_objective_plan(loss='mnrl',payload=payload,structured_features=np.zeros((6,2),np.float32),
        train_all=np.array([[0,1],[3,4]]),tr_negs=np.array([[0,2],[3,5]]),tr_neg_sources=np.array(['gate','gate']),
        hp_pairs=np.empty((0,2),int),row_bc=np.array(['a','a','c','b','b','d']),tr_bc={'a','b','c','d'},
        use_hp=False,mask_audit=[],hard_negative_mask_audit=[],hard_train=np.empty((0,2),int),seed=42,fold_i=0)
    plan=result['objective']
    assert plan['triples']==[(0,1,2),(3,4,5)]
    assert plan['dataset']['population']==['base','base']
    for device in ('cpu','cuda'):
        for batches in plan['sampler'][device]['epochs']:
            assert sorted(index for batch in batches for index in batch)==[0,1]


def test_gpu_prepared_path_does_not_rebuild_fixed_inputs(monkeypatch,tmp_path):
    import pandas as pd
    import training.training as trainer
    from training.run_plan import training_config
    empty=np.empty((0,2),int)
    fold={key:empty for key in ('test_pos','train_pos','dev_pos','hard_train','hard_dev','hard_test','tr_negs',
                                'calibration_pos','calibration_neg','train_all','_fold_canonical_rows')}
    fold.update(fold_i=0,test_bc={'test'},tr_bc={'train'},tr_neg_sources=np.array([],object),
                n_train_hard_neg=0,random_easy_unique_candidates=0,n_train_random_easy_neg=0,
                train_neg_source_counts={},n_gate_kept=0,static_masked_pos=0,static_positive_pct=0.,
                dev_pairs=[],dev_neg_pairs=[],dev_structured=[],objective={})
    plan={'inputs':{'shared':{'payload_metadata':[],'gate_lookup':{},'_retrieval_row_component':np.array([]),
                            '_retrieval_true_match_pairs':frozenset(),'_train_neg_source':empty,'country':np.array(['US'])},
                   'folds':[fold],'skipped':[]}}
    monkeypatch.setattr(trainer,'prepare_fixed_training_inputs',lambda *a,**k:pytest.fail('remote preparation called'))
    monkeypatch.setattr(trainer,'artifact',lambda *a,**k:tmp_path/'checkpoint'/'file')
    def load(*a,**k): raise RuntimeError('reached native model load without rebuilding')
    monkeypatch.setattr(trainer,'load_local_sentence_transformer',load)
    data=(pd.DataFrame({'sku_id':['a']}),['text'],np.zeros((1,2),np.float32),np.array(['a']),np.array(['US']),empty,empty,np.empty((0,0),np.float32))
    result=trainer.train_one_config(training_config(),loss='mnrl',model_id='local',use_hp=False,band=(.4,.8),
                                   data=data,seed=42,on_cuda=False,dynamic_mask_lo=.1,dynamic_mask_hi=.2,prepared_plan=plan)
    assert len(result)==1 and result[0]['status']=='failed'
    assert 'reached native model load without rebuilding' in result[0]['traceback']


def test_lookup_preserves_internal_mask_holes_and_token_positions():
    class HoleEncoder(Encoder):
        def preprocess(self,texts,prompt=None,processing_kwargs=None,**kwargs):
            result=super().preprocess(texts,prompt=prompt,processing_kwargs=processing_kwargs,**kwargs)
            result['attention_mask'][:,1]=0
            return result
    model=HoleEncoder();payload=['one two']
    expected=model.preprocess(payload)
    table=prepare_training_tokens(model,payload)
    PreparedTokenLookup(model,table,payload)
    actual=model.preprocess(payload)
    assert actual['input_ids'].shape==expected['input_ids'].shape
    torch.testing.assert_close(actual['input_ids'],expected['input_ids'])
    torch.testing.assert_close(actual['attention_mask'],expected['attention_mask'])


def test_token_preparation_materializes_iterable_before_fingerprinting():
    model = Encoder()
    table = prepare_training_tokens(model, iter(['one', 'two', 'one']))
    PreparedTokenLookup(model, table, ['one', 'two', 'one'])


@pytest.mark.parametrize('metadata,value', [('pair_id', 7), ('structured_features', [[1.], [2.]])])
def test_collator_rejects_metadata_missing_from_first_row(metadata, value):
    model = Encoder()
    collator = ObjectiveDataCollator(preprocess_fn=model.preprocess)
    rows = [{'anchor': 'one', 'positive': 'two'},
            {'anchor': 'three', 'positive': 'four', metadata: value}]
    with pytest.raises(ValueError, match=f'inconsistent {metadata}'):
        collator(rows)


def test_uncontrolled_mnrl_sampler_ignores_shared_population_metadata(monkeypatch):
    import training.training as trainer
    original = trainer.load_config
    def configuration():
        cfg = copy.deepcopy(original())
        cfg['training']['batch_sampler']['enabled'] = False
        cfg['training']['epochs'] = 1
        return cfg
    monkeypatch.setattr(trainer, 'load_config', configuration)
    payload = ['anchor a', 'positive a', 'negative a', 'anchor b', 'positive b', 'negative b']
    result = trainer._prepare_objective_plan(
        loss='mnrl', payload=payload, structured_features=np.zeros((6, 2), np.float32),
        train_all=np.array([[0, 1], [3, 4]]), tr_negs=np.array([[0, 2], [3, 5]]),
        tr_neg_sources=np.array(['gate', 'gate']), hp_pairs=np.empty((0, 2), int),
        row_bc=np.array(['a', 'a', 'c', 'b', 'b', 'd']), tr_bc={'a', 'b', 'c', 'd'},
        use_hp=False, mask_audit=[], hard_negative_mask_audit=[],
        hard_train=np.empty((0, 2), int), seed=42, fold_i=0)
    assert result['objective']['dataset']['population'] == ['base', 'base']
    for plan in result['objective']['sampler'].values():
        assert len(plan['epochs'][0]) == 1
        assert sorted(plan['epochs'][0][0]) == [0, 1]
