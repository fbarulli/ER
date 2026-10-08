"""Bound CPU preparation plans for GPU workers; no encoder forward passes."""
from __future__ import annotations
from core.portable_archive import ByteCount
import json
import math
from pathlib import Path
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


# The anti-collapse path in training.losses / training.training needs exactly
# these knobs.  `uniformity_regularization.temperature` / `min_batch_size` are
# consumed by _tracking_contrastive_loss._uniformity_penalty whenever the
# regularizer is active, and the guardrail thresholds drive
# training.uniformity.collapse_diagnostics.  This list is deliberately small:
# unrelated config drift must never be gated (owner order 2026-10-07).
_COLLAPSE_GUARDRAIL_REQUIRED_KEYS = (
    'unrelated_pairs', 'seed', 'max_token_frequency',
    'operating_threshold', 'crossing_rate_ceiling',
    'median_penalty_start', 'p90_penalty_start', 'cosine_std_floor',
)


def _is_finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def validate_collapse_regulation(config):
    """Fail loud unless the loss functions can regulate collapse.

    Owner ruling 2026-10-07: the only real test that matters is if the loss
    functions have all they need to regulate collapse.  We check ONLY the
    uniformity regularizer and the collapse guardrail; every unrelated config
    key (laya/kaggle/paths/script lanes) is ignored on purpose, because the
    plan's data/row identity is already bound by ``data_size``.

    Pure and cheap by construction: O(number of collapse knobs) dict reads of
    an already-loaded config object.  No ``data_size``, no array work, no
    model/dataset load, no full-config serialisation, no config reload.
    """
    config = config or {}
    training = config.get('training') or {}
    missing: list[str] = []

    uniformity = training.get('uniformity_regularization')
    if uniformity is None:
        missing.append('training.uniformity_regularization')
    else:
        enabled = uniformity.get('enabled')
        weight = uniformity.get('weight')
        if enabled is None:
            missing.append('training.uniformity_regularization.enabled')
        regularizer_active = bool(enabled) or (
            _is_finite_number(weight) and float(weight) > 0
        )
        if regularizer_active:
            if not _is_finite_number(weight) or float(weight) < 0:
                missing.append('training.uniformity_regularization.weight')
            temperature = uniformity.get('temperature')
            if not _is_finite_number(temperature) or float(temperature) <= 0:
                missing.append('training.uniformity_regularization.temperature')
            min_batch_size = uniformity.get('min_batch_size')
            if (not isinstance(min_batch_size, int)
                    or isinstance(min_batch_size, bool)
                    or int(min_batch_size) < 2):
                missing.append(
                    'training.uniformity_regularization.min_batch_size')

    guardrail = config.get('collapse_guardrail')
    if guardrail is None:
        missing.append('collapse_guardrail')
    else:
        enabled = guardrail.get('enabled')
        if enabled is None:
            missing.append('collapse_guardrail.enabled')
        if enabled:
            profile = guardrail.get('profile')
            profiles = config.get('collapse_guardrail_profiles') or {}
            if not isinstance(profile, str) or not profile:
                missing.append('collapse_guardrail.profile')
            elif profile not in profiles:
                missing.append(
                    f'collapse_guardrail.profile (unresolvable: {profile!r} '
                    f'not in {sorted(profiles)})')
            for key in _COLLAPSE_GUARDRAIL_REQUIRED_KEYS:
                if not _is_finite_number(guardrail.get(key)):
                    missing.append(f'collapse_guardrail.{key}')

    if missing:
        raise ValueError(
            'collapse regulation cannot run: the loss functions are missing '
            'required anti-collapse knob(s): ' + ', '.join(missing)
            + ' (a plan that cannot regulate collapse is rejected; '
            'owner order 2026-10-07)'
        )
    return config


def data_size(bundle) -> int:
    size = ByteCount()
    for key in ('payload','row_bc','country','mask_audit','hard_negative_mask_audit','holdout_populations'):
        value = bundle.get(key)
        if isinstance(value,np.ndarray): value=value.tolist()
        size.update(json.dumps({key:value},sort_keys=True,ensure_ascii=False,default=str).encode())
    size.update(json.dumps(bundle['df'].to_dict(orient='list'),sort_keys=True,ensure_ascii=False,default=str).encode())
    for key in ('pos','hp_pairs','neg','train_neg','structured_features','emb0'):
        array=np.asarray(bundle[key])
        size.update(key.encode());size.update(str(array.dtype).encode());size.update(str(array.shape).encode())
        size.update(array.tobytes())
    for key in ('neg_sources','train_neg_sources'):
        size.update(json.dumps(np.asarray(bundle[key]).tolist(),ensure_ascii=False).encode())
    for key in ('labeled_pairs_csv','canonical_records_csv','gate_results_csv'):
        size.update(bundle[key])
    return size.total


def plan_identity(bundle,*,loss,train_frac,sample,seed=SEED):
    if not 0 < train_frac <= 1:
        raise ValueError('local training plan requires 0 < train_frac <= 1')
    return {'loss':loss,'train_frac':float(train_frac),'sample':bool(sample),'seed':int(seed),
            'data_size':data_size(bundle)}


#: The frozen input files a prepared bundle carries inline. The bundle member
#: is the ``*_csv`` byte payload; the local file registry key is the bare name
#: the plan and the loss functions read (``F['labeled_pairs']`` etc.).
_FROZEN_INPUT_MEMBERS = {'labeled_pairs': 'labeled_pairs_csv',
                         'canonical_records': 'canonical_records_csv',
                         'gate_results': 'gate_results_csv'}


def _materialize_frozen_inputs(bundle, *, results=None):
    """Write the bundle-carried frozen CSVs and bind the local file registry.

    THE single implementation: the plan path (``prepare_run_plan``) and the
    prepared trainer (``training.train_prepared._main``) both call this, so the
    ``*_csv`` member names, the return shape and the destination directory are
    declared exactly once.

    A prepared bundle ships the exact frozen bytes it trained from, so this
    materializes them instead of re-reading whatever the live checkout happens
    to hold. Returns the registry key -> written path mapping. A bundle that
    carries none of these members is left untouched (its plan is byte-identical
    to the pre-bundle behavior).
    """
    from core.common import F, RESULTS
    root = RESULTS if results is None else results
    materialized = {}
    for key, member in _FROZEN_INPUT_MEMBERS.items():
        if member not in bundle:
            continue
        destination = root / '_prepared_inputs' / Path(F[key]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(bundle[member])
        F[key] = destination
        materialized[key] = destination
        print(f"[prepared-bundle] materialized {key}={destination} "
              f"bytes={len(bundle[member]):,}", flush=True)
    return materialized


def prepare_run_plan(bundle,*,loss=None,train_frac=1.,sample=False,seed=SEED):
    from training.prepared_bundle import prepared_holdout
    from training.training import prepare_fixed_training_inputs
    if np.asarray(bundle['emb0']).size:
        raise ValueError('initial embeddings have no checkpoint producer attestation; prepare verified GPU embeddings before mining')
    _materialize_frozen_inputs(bundle)
    cfg=load_config();loss=loss or cfg['training']['loss']
    validate_collapse_regulation(cfg)
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
    # Bind only the identity fields we still own. A frozen plan built before
    # the whole-config hash was retired still carries a legacy
    # `config_size` key; ignoring it is exactly what lets the existing
    # bundle validate without a rebuild. `data_size` is the real binding.
    bound = {key: identity.get(key) for key in expected}
    if plan.get('version')!=1 or bound!=expected:
        raise ValueError('prepared training row plan differs from loss/train_frac/sample/seed/data; rebuild locally')
    validate_collapse_regulation(load_config())
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
