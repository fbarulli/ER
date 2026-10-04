"""Bound CPU preparation plans for GPU workers; no encoder forward passes."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from core.common import SEED, load_config, runtime, resolve_model


def training_config(*, epochs=None, lr=None):
    from training.training import ES_PATIENCE, ES_THRESHOLD
    cfg = load_config()
    return {
        'architecture': runtime('architecture'),
        'epochs': int(epochs if epochs is not None else cfg['training']['epochs']),
        'lr': float(lr if lr is not None else cfg['training']['lr']),
        **{key: runtime(key) for key in ('warmup_ratio','weight_decay','projection_dropout','label_smoothing',
                                         'lr_scheduler','max_grad_norm')},
        'random_easy_enabled': bool(runtime('random_easy_negatives')['enabled']),
        'random_easy_ratio_to_hard': float(runtime('random_easy_negatives')['ratio_to_hard']),
        'random_easy_candidate_pool_size': int(runtime('random_easy_negatives')['candidate_pool_size']),
        'patience': ES_PATIENCE, 'es_threshold': ES_THRESHOLD,
        'uniformity_weight': float(cfg['training']['uniformity_regularization']['weight'])
                             if cfg['training']['uniformity_regularization']['enabled'] else 0.,
        'late_epoch_decay_enabled': bool(cfg['training']['late_epoch_lr_decay']['enabled']),
        'late_epoch_decay_start_fraction': float(cfg['training']['late_epoch_lr_decay']['start_epoch_fraction']),
        'late_epoch_decay_multiplier': float(cfg['training']['late_epoch_lr_decay']['multiplier']),
    }


def data_digest(bundle):
    digest = hashlib.sha256()
    for key in ('payload','row_bc','country','mask_audit','hard_negative_mask_audit','holdout_populations'):
        value = bundle.get(key)
        if isinstance(value,np.ndarray): value=value.tolist()
        digest.update(json.dumps({key:value},sort_keys=True,ensure_ascii=False,default=str).encode())
    digest.update(json.dumps(bundle['df'].to_dict(orient='list'),sort_keys=True,ensure_ascii=False,default=str).encode())
    for key in ('pos','hp_pairs','neg','train_neg','structured_features','emb0'):
        array=np.asarray(bundle[key])
        digest.update(key.encode());digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    for key in ('neg_sources','train_neg_sources'):
        digest.update(json.dumps(np.asarray(bundle[key]).tolist(),ensure_ascii=False).encode())
    for key in ('labeled_pairs_csv','canonical_records_csv','gate_results_csv'):
        digest.update(bundle[key])
    return digest.hexdigest()


def plan_identity(bundle,*,loss,train_frac,sample,seed=SEED):
    if not 0 < train_frac <= 1:
        raise ValueError('local training plan requires 0 < train_frac <= 1')
    return {'loss':loss,'train_frac':float(train_frac),'sample':bool(sample),'seed':int(seed),
            'config_sha256':hashlib.sha256(json.dumps(load_config(),sort_keys=True,default=str).encode()).hexdigest(),
            'data_sha256':data_digest(bundle)}


def prepare_run_plan(bundle,*,loss=None,train_frac=1.,sample=False,seed=SEED):
    from training.prepared_bundle import prepared_holdout
    from training.training import prepare_fixed_training_inputs
    if np.asarray(bundle['emb0']).size:
        raise ValueError('initial embeddings have no checkpoint producer attestation; prepare verified GPU embeddings before mining')
    cfg=load_config();loss=loss or cfg['training']['loss']
    train_bc,dev_bc,test_bc=prepared_holdout(bundle,cfg['split'],seed=seed)
    data=tuple(bundle[key] for key in ('df','payload','structured_features','row_bc','country','pos','hp_pairs','emb0'))
    fixed=prepare_fixed_training_inputs(
        training_config(),loss=loss,model_id=resolve_model(cfg['training']['base_model']),
        use_hp=bool(cfg['training']['hard_positives']) and len(bundle['hp_pairs'])>0,
        band=tuple(float(x) for x in cfg['mining']['ann']['band'].split('-')),data=data,seed=seed,
        folds_override=test_bc,dev_override=dev_bc,dev_fraction=float(cfg['training']['dev_fraction']),
        neg_pairs=bundle['neg'],train_neg_pairs=bundle['train_neg'],neg_pair_sources=bundle['neg_sources'],
        train_neg_pair_sources=bundle['train_neg_sources'],mask_audit=bundle['mask_audit'],
        hard_negative_mask_audit=bundle['hard_negative_mask_audit'],
        train_frac=train_frac if train_frac<1 else None,sample=sample,
    )
    if fixed['skipped'] or not fixed['folds']:
        raise ValueError(f'local training plan is not ready: {fixed["skipped"]}')
    from training.attrition import build_attrition_ledger
    for fold in fixed['folds']:
        fold['objective']['attrition_ledger'] = build_attrition_ledger(bundle, fold)
    return {'version':1,'identity':plan_identity(bundle,loss=loss,train_frac=train_frac,sample=sample,seed=seed),
            'holdout':{'train':train_bc,'dev':dev_bc,'test':test_bc},'inputs':fixed}


def validate_run_plan(bundle,plan,*,loss,train_frac,sample,seed=SEED):
    expected=plan_identity(bundle,loss=loss,train_frac=train_frac,sample=sample,seed=seed)
    identity = dict(plan.get('identity') or {})
    if sample and identity.get('sample') is True:
        # A lifecycle smoke consumes its frozen objective rows and device
        # batches. Unrelated current config edits need not invalidate them.
        expected.pop('config_sha256')
        identity.pop('config_sha256', None)
    if plan.get('version')!=1 or identity!=expected:
        raise ValueError('prepared training row plan differs from loss/train_frac/sample/config/seed/data; rebuild locally')
    if plan['inputs']['skipped'] or not plan['inputs']['folds']:
        raise ValueError('prepared training row plan has failed/skipped folds')
    return plan


def validate_epoch_batches(plan, *, epochs, batch_sizes):
    """Check frozen CPU sampler output before provisioning the GPU worker."""
    for fold in plan['inputs']['folds']:
        objective = fold['objective']
        dataset = objective['dataset']
        sizes = {len(values) for values in dataset.values()}
        if len(sizes) != 1:
            raise ValueError('prepared objective columns differ in length; rebuild locally')
        rows = sizes.pop()
        if rows == 0:
            raise ValueError('prepared objective has no training rows; rebuild locally')
        for device, batch_size in batch_sizes.items():
            sampler = objective.get('sampler', {}).get(device, {})
            batches_by_epoch = sampler.get('epochs', [])
            if sampler.get('batch_size') != batch_size or len(batches_by_epoch) < epochs:
                raise ValueError(f'prepared {device} batch size/epochs differ; rebuild locally')
            for batches in batches_by_epoch[:epochs]:
                if any(not batch or len(batch) > batch_size for batch in batches):
                    raise ValueError(f'prepared {device} batch shape invalid; rebuild locally')
                indices = [index for batch in batches for index in batch]
                if any(type(index) is not int for index in indices) or sorted(indices) != list(range(rows)):
                    raise ValueError(f'prepared {device} epoch must cover every objective row exactly once; rebuild locally')
