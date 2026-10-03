"""Local-only tokenization, graph tensorization/topology and execution planning."""
from pathlib import Path
import numpy as np
import torch
from graph_tracks.data import GraphBatch, RELATIONS, tensorize, file_hash
from graph_tracks.pooling import topology


from graph_tracks.prepared_inputs import save_batch, load_batch


def prepare_inputs(request, output, *, token_cache=None):
    """No model forward here: CPU text tokenization and frozen-vocabulary topology."""
    from model_tracks.ablation import resolve, digest
    arrays = {}
    lookup = {key:n for n,key in enumerate(request['ids'])}
    arrays['pair_indices'] = np.asarray([[lookup[p['sku_id1']],lookup[p['sku_id2']]] for p in request['pairs']],dtype=np.int64)
    # Candidate order and endpoint indices are frozen before vectors exist.
    plan = {'schema':'er-ablation-prepared-inputs-v1','token_batches':[], 'graph_batches':{},
            'candidate_ids':request['ids'], 'pair_indices':arrays['pair_indices'].tolist(),
            'known_positive_pairs':[n for n,p in enumerate(request['pairs']) if p['label']=='1'],
            'slice_groups':{}, 'variant_jobs':[], 'jobs':[]}
    axes = request['settings']['slice_columns']
    for n,pair in enumerate(request['pairs']):
        key = digest({axis:pair.get(axis) for axis in axes})
        group = plan['slice_groups'].setdefault(key,{'values':{axis:pair.get(axis) for axis in axes},'pair_indices':[]})
        group['pair_indices'].append(n)
    batch_size = request['settings']['batch_size']
    if request['track'] != 'gnn_only':
        checkpoint = request['checkpoint'] if request['track']=='text' else request['text_checkpoint']
        print('[ablation/local] using frozen native tokenizer; no encoding',flush=True)
        from model_tracks.text_export import prepare_tokens
        plan.update(prepare_tokens(resolve(checkpoint),request['texts'],arrays,batch_size=batch_size,cache=token_cache))
    payload = None
    if request['track'] != 'text':
        payload = torch.load(resolve(request['checkpoint']),map_location='cpu',weights_only=False)
        if payload.get('schema') != 'er-graph-checkpoint-v1' or payload['manifest']['track'] != request['track']:
            raise ValueError('graph checkpoint schema/track mismatch')
        if any(r['split'] != 'train' for r in payload['support_records']):
            raise ValueError('graph context must contain training listings only')
        if request['track'] == 'hybrid':
            from model_tracks.ablation import checkpoint_identity
            from core.model_input import model_input_composition
            metadata = payload['manifest']['text_metadata']
            if metadata['checkpoint_sha256'] != checkpoint_identity(resolve(request['text_checkpoint'])) or metadata['composition'] != model_input_composition().model_dump(mode='json'):
                raise ValueError('hybrid text checkpoint/composition differs from training')
        vocabulary = payload['vocabulary']
        plan['vocabulary'] = vocabulary
        save_batch(arrays,'support',tensorize(payload['support_records'],vocabulary,'cpu'),vocabulary)
    if request.get('candidate_ids'):
        arrays['candidate_text_indices'] = np.asarray(request['candidate_text_indices'],dtype=np.int64)
        plan['candidate_ids'] = request['candidate_ids']
        if payload:
            plan['candidate_batches'] = []
            for start in range(0,len(request['candidate_records']),batch_size):
                prefix = f'candidate/{start}'
                save_batch(arrays,prefix,tensorize(request['candidate_records'][start:start+batch_size],vocabulary,'cpu'),vocabulary)
                plan['candidate_batches'].append(prefix)
    job_lookup = {}
    for variant in request['variants']:
        key = digest({'text':variant['text_indices'],'graph':variant['records']})
        if key not in job_lookup:
            job_lookup[key] = len(plan['jobs'])
            graph_key = digest(variant['records'])
            if payload and graph_key not in plan['graph_batches']:
                prefixes = []
                for start in range(0,len(variant['records']),batch_size):
                    prefix = f'graph/{graph_key}/{start}'
                    save_batch(arrays,prefix,tensorize(variant['records'][start:start+batch_size],vocabulary,'cpu'),vocabulary)
                    prefixes.append(prefix)
                plan['graph_batches'][graph_key] = prefixes
            text_key = 'indices/'+key
            arrays[text_key] = np.asarray(variant['text_indices'],dtype=np.int64)
            plan['jobs'].append({'text_indices_key':text_key,'graph_key':graph_key})
        plan['variant_jobs'].append(job_lookup[key])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as handle:
        np.savez_compressed(handle,**arrays)
    plan['sha256'] = file_hash(output)
    print(f'[ablation/local] prepared token batches={len(plan["token_batches"])} unique inference jobs={len(plan["jobs"])}',flush=True)
    return plan
