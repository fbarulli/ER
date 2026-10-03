"""Checkpoint-native text preparation shared by text and hybrid encoders.

No token truncation or embedding-dimension truncation is permitted. Model
forward passes consume these features directly, without a second tokenizer.
"""
import hashlib
import json
import inspect


def tokenization_policy(model):
    tokenizer = model.tokenizer
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None:
        raise ValueError('zero-truncation preparation requires a serialized fast tokenizer')
    serialized = json.dumps(json.loads(backend.to_str()),sort_keys=True).encode()
    module = model[0]
    config = getattr(module, 'config', None) or module.auto_model.config
    limit = getattr(config, 'max_position_embeddings', None)
    if not isinstance(limit,int) or limit < 1:
        limit = model.max_seq_length
    if getattr(model,'truncate_dim',None) is not None:
        raise ValueError('checkpoint requests embedding truncation; zero truncation required')
    prompt = None
    if getattr(model,'default_prompt_name',None):
        prompt = model.prompts[model.default_prompt_name]
    return {'tokenizer_sha256':hashlib.sha256(serialized).hexdigest(),
            'input_token_limit':int(limit), 'prompt':prompt, 'truncation':False,
            'truncate_dim':None, 'padding':'longest',
            'special_tokens':tokenizer.special_tokens_map}


def prepare_text_features(model, texts, *, policy=None):
    policy = policy or tokenization_policy(model)
    module = model[0]
    # Modern ST preserves its input formatting, normalization and prompt pooling.
    method = getattr(module,'preprocess',None)
    if method is not None and 'processing_kwargs' in inspect.signature(method).parameters:
        features = model.preprocess(texts,prompt=policy['prompt'],
            processing_kwargs={'text':{'truncation':False,'max_length':None,'padding':True}})
    else:
        # Older ST hard-codes truncation=True in tokenize; override only this
        # tokenizer call while keeping the checkpoint's own formatting path.
        original = module.tokenizer
        class FullTokenizer:
            def __getattr__(self,key):
                return getattr(original,key)
            def __call__(self,*args,**kwargs):
                kwargs['truncation'] = False
                kwargs.pop('max_length',None)
                return original(*args,**kwargs)
        module.tokenizer = FullTokenizer()
        try:
            inputs = [policy['prompt']+text for text in texts] if policy['prompt'] else texts
            features = model.tokenize(inputs)
            if policy['prompt']:
                features['prompt_length'] = model.tokenize([policy['prompt']])['input_ids'].shape[-1]-1
        finally:
            module.tokenizer = original
    if 'input_ids' not in features or 'attention_mask' not in features:
        raise ValueError('encoder must expose complete token IDs and attention masks')
    lengths = features['attention_mask'].sum(-1).tolist()
    over = [(n,int(length)) for n,length in enumerate(lengths) if length > policy['input_token_limit']]
    if over:
        raise ValueError(f'zero truncation required: inputs exceed checkpoint token limit {policy["input_token_limit"]}: {over[:10]}')
    return features


def prepare_token_batches(model, texts, arrays, *, batch_size, prefix='text'):
    """One shared policy for ordinary embeddings and text/hybrid ablations."""
    import torch
    policy = tokenization_policy(model)
    batches, lengths = [], []
    for start in range(0,len(texts),batch_size):
        features = prepare_text_features(model,texts[start:start+batch_size],policy=policy)
        stem = f'{prefix}/{start}'
        keys = [key for key,value in features.items() if isinstance(value,torch.Tensor)]
        constants = {key:value for key,value in features.items() if key not in keys}
        for key in keys:
            arrays[stem+'/'+key] = features[key].detach().cpu().numpy()
        lengths.extend(features['attention_mask'].sum(-1).tolist())
        batches.append({'prefix':stem,'keys':keys,'constants':constants,'start':start,
                        'count':len(features['input_ids'])})
    return {'tokenization':policy,'token_batches':batches,'token_lengths':lengths,'truncated_inputs':0}


def enable_zero_truncation(model):
    """Guard native trainer/evaluation calls, including newly augmented texts."""
    if getattr(model,'_zero_truncation_enabled',False):
        return model
    policy = tokenization_policy(model)
    if hasattr(model,'preprocess') and 'processing_kwargs' in inspect.signature(model[0].preprocess).parameters:
        original = model.preprocess
        def guarded(inputs,*args,**kwargs):
            overrides = dict(kwargs.pop('processing_kwargs',None) or {})
            overrides['text'] = {**overrides.get('text',{}),'truncation':False,'max_length':None}
            features = original(inputs,*args,processing_kwargs=overrides,**kwargs)
            lengths = features['attention_mask'].sum(-1)
            if (lengths > policy['input_token_limit']).any():
                raise ValueError('zero truncation required: training/evaluation input exceeds checkpoint token limit')
            return features
        model.preprocess = guarded
    else:
        original = model.tokenize
        def guarded(inputs,*args,**kwargs):
            # prepare_text_features calls native model.tokenize on the older
            # branch; retain that original method while entering the helper.
            model.tokenize = original
            try:
                return prepare_text_features(model,inputs,policy=policy)
            finally:
                model.tokenize = guarded
        model.tokenize = guarded
    model._zero_truncation_enabled = True
    return model
