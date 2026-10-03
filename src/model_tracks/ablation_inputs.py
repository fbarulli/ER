"""Local-only tokenization, graph tensorization/topology and execution planning."""
from pathlib import Path
import numpy as np
import torch
from graph_tracks.data import GraphBatch, RELATIONS, tensorize, file_hash
from graph_tracks.pooling import topology


def save_batch(arrays, prefix, batch, vocabulary):
    arrays[prefix+'/numeric'] = batch.numeric.numpy()
    for relation in RELATIONS:
        listing, value = batch.edges[relation]
        arrays[prefix+'/'+relation+'/listing'] = listing.numpy()
        arrays[prefix+'/'+relation+'/value'] = value.numpy()
        for attribute in (False, True):
            count = len(vocabulary[relation])+1 if attribute else len(batch.numeric)
            source, target, sizes = topology(batch, relation, attribute=attribute, count=count)
            stem = prefix+'/'+relation+('/attribute' if attribute else '/listing_pool')
            arrays[stem+'/source'] = source.numpy()
            arrays[stem+'/target'] = target.numpy()
            arrays[stem+'/sizes'] = sizes.numpy()


def load_batch(arrays, prefix, device, vocabulary):
    numeric = torch.as_tensor(arrays[prefix+'/numeric'],device=device)
    edges = {r:(torch.as_tensor(arrays[prefix+'/'+r+'/listing'],device=device),
                torch.as_tensor(arrays[prefix+'/'+r+'/value'],device=device)) for r in RELATIONS}
    batch = GraphBatch(numeric,edges)
    batch._pool_topology = {}
    for relation,(listing,value) in edges.items():
        signature = (id(listing),listing._version,id(value),value._version)
        for attribute in (False, True):
            count = len(vocabulary[relation])+1 if attribute else len(numeric)
            stem = prefix+'/'+relation+('/attribute' if attribute else '/listing_pool')
            tensors = [torch.as_tensor(arrays[stem+'/'+key],device=device) for key in ('source','target','sizes')]
            batch._pool_topology[(relation,attribute,count,torch.float32)] = (signature,*tensors,listing,value)
    return batch


def prepare_inputs(request, output):
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
        from sentence_transformers import SentenceTransformer
        checkpoint = request['checkpoint'] if request['track']=='text' else request['text_checkpoint']
        print('[ablation/local] loading tokenizer from frozen checkpoint; no encoding',flush=True)
        model = SentenceTransformer(str(resolve(checkpoint)),device='cpu',local_files_only=True)
        from core.encoding_inputs import prepare_token_batches
        plan.update(prepare_token_batches(model,request['texts'],arrays,batch_size=batch_size))
        del model
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
