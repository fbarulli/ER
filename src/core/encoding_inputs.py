"""Checkpoint-native text preparation shared by text and hybrid encoders.

No token truncation or embedding-dimension truncation is permitted. Model
forward passes consume these features directly, without a second tokenizer.
"""
from __future__ import annotations

import hashlib
import json
import inspect
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    import torch


class PreparedTokenInputs(BaseModel):
    """Frozen native tokens and their ordered row provenance for an export."""

    plan: Mapping
    arrays: Mapping
    row_count: int = Field(strict=True, ge=1)

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra='forbid')

    @model_validator(mode='after')
    def validate_inputs(self):
        self.validate_contract()
        return self

    def validate_contract(self):
        if self.row_count < 1 or self.plan.get('truncated_inputs') != 0:
            raise ValueError('prepared tokens require a nonempty untruncated population')
        policy = self.plan['tokenization']
        if policy.get('truncation') is not False or policy.get('truncate_dim') is not None:
            raise ValueError('prepared token policy requires zero truncation')
        count, lengths = 0, []
        prefixes = set()
        for batch in self.plan['token_batches']:
            if batch['start'] != count or batch['count'] < 1 or batch['prefix'] in prefixes:
                raise ValueError('prepared token row order or batch provenance differs')
            prefixes.add(batch['prefix'])
            features = load_token_features(self.arrays, batch, 'cpu')
            batch_lengths = features['attention_mask'].sum(-1).tolist()
            if any(length > policy['input_token_limit'] for length in batch_lengths):
                raise ValueError('prepared tokens exceed native token limit')
            lengths.extend(batch_lengths)
            count += batch['count']
        if count != self.row_count or lengths != self.plan.get('token_lengths'):
            raise ValueError('prepared token population or recorded lengths differ')
        return count


def tokenization_policy(model):
    tokenizer = model.tokenizer
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None:
        raise ValueError('zero-truncation preparation requires a serialized fast tokenizer')
    backend_spec = json.loads(backend.to_str())
    # Padding/truncation in the backend are mutable per-call batching settings.
    # Bind vocabulary/normalization/postprocessing, and pin batching separately.
    backend_spec.pop('padding', None)
    backend_spec.pop('truncation', None)
    serialized = json.dumps(backend_spec,sort_keys=True).encode()
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
    formatting = dict(module.get_config_dict()) if hasattr(module,'get_config_dict') else {}
    # This mutable legacy truncation knob is superseded by the native window.
    formatting.pop('max_seq_length',None)
    return {'tokenizer_sha256':hashlib.sha256(serialized).hexdigest(),
            'input_token_limit':int(limit), 'prompt':prompt, 'truncation':False,
            'truncate_dim':None, 'padding':'longest',
            'special_tokens':tokenizer.special_tokens_map,
            'padding_side':getattr(tokenizer,'padding_side','right'),
            'do_lower_case':getattr(module,'do_lower_case',False),
            'formatting':formatting,
            'prompts':dict(getattr(model,'prompts',{}))}


def _validate_token_features(features,count):
    import torch
    ids, mask = features.get('input_ids'), features.get('attention_mask')
    if ids is None or mask is None:
        raise ValueError('encoder must expose complete token IDs and attention masks')
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != count:
        raise ValueError('prepared tokens must retain every input row with aligned masks')
    if ids.dtype != torch.long or mask.dtype not in (torch.long,torch.bool):
        raise ValueError('prepared token IDs must be int64 and masks int64 or bool')
    if not ((mask == 0) | (mask == 1)).all() or not (mask.sum(-1) > 0).all():
        raise ValueError('prepared attention masks must be binary and nonempty')
    types = features.get('token_type_ids')
    if types is not None and (types.dtype != torch.long or types.shape != ids.shape):
        raise ValueError('prepared token_type_ids must be aligned int64 tensors')


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
    _validate_token_features(features,len(texts))
    lengths = features['attention_mask'].sum(-1).tolist()
    over = [(n,int(length)) for n,length in enumerate(lengths) if length > policy['input_token_limit']]
    if over:
        raise ValueError(f'zero truncation required: inputs exceed checkpoint token limit {policy["input_token_limit"]}: {over[:10]}')
    return features


def prepare_token_batches(model, texts, arrays, *, batch_size, prefix='text'):
    """One shared policy for ordinary embeddings and text/hybrid ablations."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('token batch_size must be a positive integer')
    if not texts:
        raise ValueError('token preparation requires a nonempty population')
    import torch
    policy = tokenization_policy(model)
    import time
    started = last_progress = time.monotonic()
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
        if start+batch_size >= len(texts) or time.monotonic()-last_progress >= 10:
            print(f'[tokens/local] prepared={min(start+batch_size,len(texts))}/{len(texts)} truncated=0 elapsed={time.monotonic()-started:.1f}s',flush=True)
            last_progress = time.monotonic()
    return {'tokenization':policy,'token_batches':batches,'token_lengths':lengths,'truncated_inputs':0}


def load_token_features(
    arrays: Mapping[str, Any], batch: Mapping[str, Any], device: str | torch.device,
) -> dict[str, Any]:
    """Load frozen native features without retokenizing or coercing token dtypes."""
    import torch
    features = {key:torch.as_tensor(arrays[batch['prefix']+'/'+key],device='cpu')
                for key in batch['keys']}
    constants = batch.get('constants',{})
    if set(constants) & set(features):
        raise ValueError('prepared token constants collide with tensor features')
    features.update(constants)
    # Frozen inputs are checked before upload: Boolean mask reductions on CUDA
    # would otherwise force device synchronization for every encoder batch.
    _validate_token_features(features,batch['count'])
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()}


def enable_zero_truncation(model):
    """Guard native trainer/evaluation calls, including newly augmented texts."""
    if getattr(model,'_zero_truncation_enabled',False):
        return model
    policy = tokenization_policy(model)
    native_preprocess = getattr(model[0],'preprocess',None)
    if hasattr(model,'preprocess') and native_preprocess is not None and 'processing_kwargs' in inspect.signature(native_preprocess).parameters:
        original = model.preprocess
        def guarded(inputs,*args,**kwargs):
            overrides = dict(kwargs.pop('processing_kwargs',None) or {})
            overrides['text'] = {**overrides.get('text',{}),'truncation':False,'max_length':None}
            features = original(inputs,*args,processing_kwargs=overrides,**kwargs)
            _validate_token_features(features,len(inputs))
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
                # tokenize callers (including encode) already supply prompts;
                # only standalone preparation applies the default prompt.
                return prepare_text_features(model,inputs,policy={**policy,'prompt':None})
            finally:
                model.tokenize = guarded
        model.tokenize = guarded
    model._zero_truncation_enabled = True
    return model


def enable_cross_encoder_zero_truncation(model):
    """Reject overlong complete pairs before CrossEncoder.predict tokenizes them."""
    if getattr(model,'_zero_truncation_enabled',False):
        return model
    tokenizer = model.tokenizer
    config = getattr(getattr(model,'model',None),'config',None)
    limit = getattr(config,'max_position_embeddings',None) or getattr(model,'max_length',None)
    if not isinstance(limit,int) or limit < 1:
        raise ValueError('cross encoder needs an explicit supported token window')
    if hasattr(model,'max_length'):
        model.max_length = limit
    original = model.predict
    def guarded(pairs,*args,**kwargs):
        pairs = list(pairs)
        for n,pair in enumerate(pairs):
            if not isinstance(pair,(list,tuple)) or len(pair) != 2:
                raise ValueError('cross encoder requires complete text pairs')
            encoded = tokenizer(pair[0],pair[1],truncation=False,add_special_tokens=True)
            if len(encoded['input_ids']) > limit:
                raise ValueError(f'zero truncation required: cross-encoder pair {n} exceeds {limit} tokens')
        return original(pairs,*args,**kwargs)
    model.predict = guarded
    model._zero_truncation_enabled = True
    return model
