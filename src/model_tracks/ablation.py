"""Local preparation/reporting and a GPU-only frozen-checkpoint worker.

Intervention removes a declared attribute cell, then reuses the model composer.
Title/brand evidence and the checkpoint's training graph context remain fixed.
No optimization, threshold fitting, synthetic labels or implicit cache reuse.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field
from core.common import TRAIN_ROOT
from graph_tracks.data import file_hash, load_records, RELATIONS, NUMERIC
from graph_tracks.text_cache import checkpoint_hash, composition_fingerprint


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sample_pairs: int = Field(default=100, ge=1)
    seed: int = 1729
    split: str = 'dev'
    batch_size: int = Field(default=256, ge=1)
    accelerator: str = 'T4'
    attributes: list[str] = Field(default_factory=list)
    output_dir: str = 'results/attribute_ablation'
    report_path: str = 'results/attribute_ablation/report.json'
    retrieval_ks: list[int] = Field(default_factory=lambda: [1, 5, 10])
    graph_fields: dict[str, list[str]] = Field(default_factory=dict)
    slice_columns: list[str] = Field(default_factory=list)


def settings(path=None):
    return Settings.model_validate(yaml.safe_load((path or TRAIN_ROOT/'config/attribute_ablation.yaml').read_text()))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else TRAIN_ROOT/path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_name(path):
    path = Path(path).resolve()
    return path.relative_to(TRAIN_ROOT).as_posix() if path.is_relative_to(TRAIN_ROOT) else str(path)


def checkpoint_identity(path):
    return checkpoint_hash(path) if path.is_dir() else file_hash(path)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def declaration_removed(row, attribute):
    from core.text import normalized_attribute_text
    result = dict(row)
    result['attribute'] = ';'.join(part for part in str(row.get('attribute', '')).split(';')
        if ':' not in part or normalized_attribute_text(part.split(':', 1)[0]) != attribute)
    return result


def graph_removed(record, fields):
    result = copy.deepcopy(record)
    for field in fields:
        channel, key = field.split('.', 1)
        allowed = RELATIONS if channel == 'attribute' else NUMERIC if channel == 'numeric' else ()
        if key not in allowed:
            raise ValueError(f'unsupported graph field: {field}')
        result[channel].pop(key, None)
    return result


def sample_pairs(frame, cfg):
    required = {'sku_id1', 'sku_id2', 'label', 'split'}
    if not required.issubset(frame.columns):
        raise ValueError(f'pairs require {sorted(required)}')
    if cfg.split not in {'dev', 'test'}:
        raise ValueError('ablation requires an explicit held-out dev or test split')
    frame = frame[frame.split == cfg.split].copy()
    if frame.empty or not frame.label.isin(['0', '1']).all():
        raise ValueError('selected split needs binary labeled pairs')
    # Round-robin across observed joint strata, deterministic within each.
    # An all-empty column (empty CSV cells read as '' with keep_default_na=False)
    # is a degenerate single stratum, not a real axis: exclude it like an
    # absent column so prepare() reports it in missing_axes instead.
    axes = ['label'] + [x for x in cfg.slice_columns if x in frame and x != 'lineage_id'
                       and not x.startswith(('gate_', 'jev_'))
                       and not (frame[x].dtype == object and frame[x].eq('').all())]
    rng = np.random.default_rng(cfg.seed)
    groups = [list(rng.permutation(group.index)) for _, group in frame.groupby(axes, sort=True, dropna=False)]
    chosen = []
    while groups and len(chosen) < cfg.sample_pairs:
        for group in groups:
            if group and len(chosen) < cfg.sample_pairs:
                chosen.append(group.pop())
        groups = [group for group in groups if group]
    return frame.loc[chosen].to_dict('records')


def validate_sources(request):
    for path, expected in request['sources'].items():
        source = resolve(path)
        if not source.exists() or checkpoint_identity(source) != expected:
            raise ValueError(f'ablation source changed: {path}')
    if composition_fingerprint() != request['composition']:
        raise ValueError('ablation composition changed; prepare again locally')
    if file_hash(Path(__file__)) != request['implementation_sha256']:
        raise ValueError('ablation implementation changed; prepare again locally')


def prepare(catalog, pairs, checkpoint, *, track='text', listings=None, text_checkpoint=None, config=None):
    from core.attribute_universe import attribute_registry
    from core.model_input import build_sku_text, model_input_info
    from core.sku_identity import row_identity
    from core.attribute_conflicts import canonical_attribute_info
    from core.attribute_decision import engine
    cfg = settings(config)
    if track not in {'text', 'gnn_only', 'hybrid'}:
        raise ValueError('unknown track')
    if (track != 'text') != bool(listings) or (track == 'hybrid') != bool(text_checkpoint):
        raise ValueError('graph tracks require listings; only hybrid requires text_checkpoint')
    inputs = [catalog, pairs, checkpoint] + ([listings] if listings else []) + ([text_checkpoint] if text_checkpoint else [])
    config = config or TRAIN_ROOT/'config/attribute_ablation.yaml'
    inputs += [config, TRAIN_ROOT/'src/core/encoding_inputs.py',TRAIN_ROOT/'src/model_tracks/ablation_inputs.py',
               TRAIN_ROOT/'src/graph_tracks/infer.py',TRAIN_ROOT/'src/graph_tracks/pooling.py']
    sources = {source_name(p): checkpoint_identity(Path(p)) for p in inputs}
    chosen = sample_pairs(pd.read_csv(pairs, dtype=str, keep_default_na=False), cfg)
    # A present-but-empty slice column reads as '' (not None); normalize
    # all-empty axes to None so they are reported in missing_axes and the
    # report rows carry null instead of a silent empty-string stratum.
    for axis in cfg.slice_columns:
        if chosen and all(p.get(axis) is None or p.get(axis) == '' for p in chosen):
            for p in chosen:
                p[axis] = None
    frame = pd.read_csv(catalog, dtype=str, keep_default_na=False)
    if 'sku_id' not in frame or frame.sku_id.duplicated().any() or (frame.sku_id == '').any():
        raise ValueError('catalog requires unique nonempty sku_id')
    rows = frame.set_index('sku_id', drop=False).to_dict('index')
    ids = sorted({p[k] for p in chosen for k in ('sku_id1', 'sku_id2')})
    if set(ids)-rows.keys():
        raise ValueError('pair endpoint absent from catalog')
    attributes = cfg.attributes or sorted(attribute_registry())
    if len(set(attributes)) != len(attributes) or set(attributes)-attribute_registry().keys():
        raise ValueError('attributes must be unique registry keys')
    records = {r['sku_id']: r for r in load_records(listings)} if listings else {}
    if listings and any(i not in records or records[i]['split'] != cfg.split for i in ids):
        raise ValueError('graph endpoints must belong to the selected held-out split')
    for pair in chosen:
        a, b = (rows[pair[k]] for k in ('sku_id1', 'sku_id2'))
        pair['gtin1'], pair['gtin2'] = a.get('gtin'), b.get('gtin')
        pair['current_attribute_evidence'] = engine().evaluate(canonical_attribute_info(a),
            canonical_attribute_info(b), left_raw=a, right_raw=b).as_dict()
        for axis in cfg.slice_columns:
            pair.setdefault(axis, None)
    texts, lookup = [], {}
    def intern(text):
        if text not in lookup:
            lookup[text] = len(texts)
            texts.append(text)
        return lookup[text]
    def compose(row):
        return build_sku_text(pd.Series(row), model_input_info(row_identity(row).as_mapping()))
    started = time.monotonic()
    print(f'[ablation/local] composing baseline endpoints={len(ids)} attributes={len(attributes)}',flush=True)
    baseline_text = []
    if track != 'gnn_only':
        for n,i in enumerate(ids):
            baseline_text.append(intern(compose(rows[i])))
            if (n+1) % 25 == 0 or n+1 == len(ids):
                print(f'[ablation/local] baseline={n+1}/{len(ids)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    baseline_records = [records[i] for i in ids] if listings else []
    variants = [{'attribute':None, 'channel':'baseline', 'text_indices':baseline_text,
                 'records':baseline_records, 'changed_listings':0}]
    for attr_index, attribute in enumerate(attributes,1):
        altered_text = []
        if baseline_text:
            for n,i in enumerate(ids):
                changed_row = declaration_removed(rows[i],attribute)
                altered_text.append(baseline_text[n] if changed_row['attribute'] == rows[i].get('attribute','') else intern(compose(changed_row)))
        altered_records = [graph_removed(r, cfg.graph_fields.get(attribute, [])) for r in baseline_records]
        for channel in (['text'] if track == 'text' else ['graph'] if track == 'gnn_only' else ['text', 'graph', 'both']):
            ti = altered_text if channel in {'text', 'both'} else baseline_text
            gr = altered_records if channel in {'graph', 'both'} else baseline_records
            changed = sum((bool(ti) and ti[n] != baseline_text[n]) or
                          (bool(gr) and gr[n] != baseline_records[n]) for n in range(len(ids)))
            variants.append({'attribute':attribute, 'channel':channel, 'text_indices':ti,
                             'records':gr, 'changed_listings':changed})
        print(f'[ablation/local] attribute={attr_index}/{len(attributes)} {attribute} unique_texts={len(texts)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    request = {'schema':'er-attribute-ablation-v2','track':track, 'settings':cfg.model_dump(),
        'sources':sources, 'composition':composition_fingerprint(), 'implementation_sha256':file_hash(Path(__file__)),
        'checkpoint':source_name(checkpoint), 'text_checkpoint':source_name(text_checkpoint) if text_checkpoint else None,
        'ids':ids, 'texts':texts, 'pairs':chosen, 'variants':variants,
        'intervention':'declared attribute removed; title/brand and training graph context fixed',
        'retrieval_scope':'fixed sampled endpoint catalog; incomplete known-positive truth',
        'missing_axes':[a for a in cfg.slice_columns if all(p.get(a) is None or p.get(a) == '' for p in chosen)]}
    from model_tracks.ablation_inputs import prepare_inputs
    resolve(cfg.output_dir).mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=resolve(cfg.output_dir)) as tmp:
        prepared = Path(tmp)/'prepared_inputs.npz'
        request['prepared_inputs'] = prepare_inputs(request,prepared)
        validate_sources(request)
        output = resolve(cfg.output_dir)/digest(request)[:24]
        output.mkdir(parents=True, exist_ok=True)
        destination = output/prepared.name
        if destination.exists():
            if file_hash(destination) != request['prepared_inputs']['sha256']:
                raise ValueError('prepared tensors differ')
        else:
            prepared.replace(destination)
        path = output/'request.json'
        if path.exists() and json.loads(path.read_text()) != request:
            raise ValueError('existing request differs')
        write(path,request)
    print(f'[ablation/local] pairs={len(chosen)} endpoints={len(ids)} unique_texts={len(texts)} '
          f'variants={len(variants)-1} changed={sum(v["changed_listings"] > 0 for v in variants[1:])}', flush=True)
    return path


def load_prepared(request_path, request):
    plan = request.get('prepared_inputs')
    if request.get('schema') != 'er-attribute-ablation-v2' or not plan:
        raise ValueError('locally prepared model inputs required; prepare again')
    path = request_path.parent/'prepared_inputs.npz'
    if file_hash(path) != plan['sha256']:
        raise ValueError('prepared input checksum mismatch')
    return np.load(path,allow_pickle=False)


def encode(request_path, output, *, device='cuda'):
    """Colab inference only; all interventions and texts arrive prepared."""
    import torch
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('ablation encoding requires CUDA')
    if output.exists():
        raise FileExistsError(output)
    request = json.loads(request_path.read_text())
    # Sources are relocated by the launcher but expected hashes stay frozen.
    validate_sources(request)
    from model_tracks.ablation_inputs import load_batch
    from core.encoding_inputs import tokenization_policy
    arrays = load_prepared(request_path,request)
    plan = request['prepared_inputs']
    track = request['track']
    text_vectors = None
    if track != 'gnn_only':
        from sentence_transformers import SentenceTransformer
        checkpoint = request['checkpoint'] if track == 'text' else request['text_checkpoint']
        model = SentenceTransformer(str(resolve(checkpoint)),device=device,local_files_only=True)
        model.eval()
        if tokenization_policy(model) != plan['tokenization']:
            raise ValueError('worker tokenizer/checkpoint policy differs from local preparation')
        chunks = []
        with torch.no_grad():
            for n,batch in enumerate(plan['token_batches'],1):
                features = {key:torch.as_tensor(arrays[batch['prefix']+'/'+key],device=device) for key in batch['keys']}
                features.update(batch['constants'])
                vectors = model(features)['sentence_embedding']
                chunks.append(torch.nn.functional.normalize(vectors,p=2,dim=1).cpu().numpy())
                print(f'[ablation/gpu] prepared text batch={n}/{len(plan["token_batches"])}; truncated=0',flush=True)
        text_vectors = np.concatenate(chunks)
        del model
    encoder = None
    graph_batches = {}
    if track != 'text':
        from graph_tracks.infer import GraphEncoder
        vocabulary = plan['vocabulary']
        support = load_batch(arrays,'support',device,vocabulary)
        encoder = GraphEncoder(resolve(request['checkpoint']),device,prepared_support=support)
        if encoder.vocabulary != vocabulary:
            raise ValueError('prepared vocabulary differs from checkpoint')
        graph_batches = {key:[load_batch(arrays,prefix,device,vocabulary) for prefix in prefixes]
                         for key,prefixes in plan['graph_batches'].items()}
    indices = arrays['pair_indices']
    results = []
    for n,job in enumerate(plan['jobs'],1):
        text = text_vectors[arrays[job['text_indices_key']]] if text_vectors is not None else None
        vec = text if encoder is None else encoder.encode_prepared(graph_batches[job['graph_key']],text)
        if encoder is None:
            score = (vec[indices[:,0]]*vec[indices[:,1]]).sum(-1)
        else:
            with torch.no_grad():
                score = encoder.scorer(torch.as_tensor(vec,device=device),torch.as_tensor(indices,device=device),
                    None if text is None else torch.as_tensor(text,device=device)).sigmoid().cpu().numpy()
        results.append((vec,score))
        print(f'[ablation/gpu] prepared inference job={n}/{len(plan["jobs"])}',flush=True)
    vectors = [results[job][0] for job in plan['variant_jobs']]
    scores = [results[job][1] for job in plan['variant_jobs']]
    arrays.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.savez_compressed(handle, vectors=np.asarray(vectors), scores=np.asarray(scores),
                            request_sha256=file_hash(request_path))
    output.with_suffix('.sha256').write_text(file_hash(output))


def frozen_threshold(source, value):
    path = resolve(source)
    if not path.is_file():
        raise ValueError('threshold source must be an existing saved report')
    before = file_hash(path)
    track = None
    checkpoint = None
    if path.suffix == '.csv':
        frame = pd.read_csv(path)
        values = frame['threshold'].tolist() if 'threshold' in frame else []
        if 'threshold' in frame:
            hits = frame[frame['threshold'].apply(
                lambda x: isinstance(x,(int,float)) and np.isfinite(x) and float(x) == value)]
            for column,key in (('model','track'),('checkpoint','checkpoint')):
                if column in frame:
                    found = sorted({str(v) for v in hits[column].tolist() if isinstance(v,str) and v})
                    if len(found) > 1:
                        raise ValueError(f'threshold source attests conflicting {column} values')
                    if found:
                        if key == 'track':
                            track = found[0]
                        else:
                            checkpoint = found[0]
    else:
        document = json.loads(path.read_text())
        values = []
        attested = {'track':set(), 'checkpoint':set()}
        def walk(obj):
            if isinstance(obj, dict):
                if isinstance(obj.get('threshold'),(int,float)) and np.isfinite(obj['threshold']):
                    values.append(obj['threshold'])
                    for column,key in (('model','track'),('track','track'),('checkpoint','checkpoint')):
                        witness = obj.get(column)
                        if isinstance(witness,str) and witness:
                            attested[key].add(witness)
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)
        walk(document)
        # manifests pin identity at the top level while thresholds nest in summaries
        for column,key in (('model','track'),('track','track'),('checkpoint','checkpoint')):
            witness = document.get(column) if isinstance(document,dict) else None
            if isinstance(witness,str) and witness:
                attested[key].add(witness)
        for key,seen in attested.items():
            if len(seen) > 1:
                raise ValueError(f'threshold source attests conflicting {key} values')
            if seen:
                if key == 'track':
                    track = next(iter(seen))
                else:
                    checkpoint = next(iter(seen))
    if not any(isinstance(x, (int,float)) and np.isfinite(x) and float(x) == value for x in values):
        raise ValueError('threshold differs from the saved baseline report')
    if before != file_hash(path):
        raise ValueError('threshold report changed while reading')
    return {'path':source_name(path),'sha256':before,'selection':'saved baseline; never refitted during ablation',
            'track':track, 'checkpoint':checkpoint}


def report(request_path, result, threshold, *, threshold_source, config=None, save=True):
    """Paired local comparisons at a supplied, already selected threshold."""
    if not np.isfinite(threshold) or not threshold_source:
        raise ValueError('frozen threshold and its source are required')
    threshold_provenance = frozen_threshold(threshold_source, threshold)
    request = json.loads(request_path.read_text())
    validate_sources(request)
    attested_track = threshold_provenance.get('track')
    if attested_track is not None and attested_track != request['track']:
        raise ValueError('threshold source track differs from the ablated track')
    attested_checkpoint = threshold_provenance.get('checkpoint')
    if attested_checkpoint is not None and request.get('checkpoint'):
        named = Path(request['checkpoint']).name
        if Path(attested_checkpoint).name != named:
            raise ValueError('threshold source checkpoint differs from the ablated checkpoint')
        expected = request.get('sources',{}).get(request['checkpoint'])
        if expected:
            located = Path(attested_checkpoint) if Path(attested_checkpoint).is_file() else None
            if located is None:
                for parent in resolve(threshold_source).parents[:3]:
                    located = next((hit for hit in parent.rglob(named) if hit.is_file()), None)
                    if located is not None:
                        break
            if located is not None and checkpoint_identity(located) != expected:
                raise ValueError('threshold source checkpoint identity differs from the ablated checkpoint')
    if 'prepared_inputs' in request:
        load_prepared(request_path,request).close()
    with np.load(result, allow_pickle=False) as data:
        if str(data['request_sha256'].item()) != file_hash(request_path):
            raise ValueError('ablation result belongs to another request')
        vectors, scores = data['vectors'], data['scores']
    nv, ni, npairs = len(request['variants']),len(request['ids']),len(request['pairs'])
    if vectors.ndim != 3 or vectors.shape[:2] != (nv, ni) or scores.shape != (nv, npairs):
        raise ValueError('ablation result shape mismatch')
    if not np.isfinite(vectors).all() or not np.isfinite(scores).all() or not np.allclose(np.linalg.norm(vectors, axis=-1),1,atol=1e-4):
        raise ValueError('ablation vectors must be finite and normalized')
    cfg = Settings.model_validate(request['settings'])
    if not cfg.retrieval_ks or any(k < 1 for k in cfg.retrieval_ks):
        raise ValueError('retrieval ks must be positive')
    id_lookup = {i:n for n,i in enumerate(request['ids'])}
    def ranks(vec):
        # Exact fixed-catalog ranks; stable ID ordering breaks ties deterministically.
        similarity = vec @ vec.T
        np.fill_diagonal(similarity, -np.inf)
        order = np.argsort(-similarity, axis=1, kind='stable')
        inverse = np.argsort(order, axis=1, kind='stable')+1
        return [[int(inverse[id_lookup[p['sku_id1']],id_lookup[p['sku_id2']]]),
                 int(inverse[id_lookup[p['sku_id2']],id_lookup[p['sku_id1']]])] for p in request['pairs']]
    baseline_ranks = ranks(vectors[0])
    rows = []
    for n, variant in enumerate(request['variants'][1:], 1):
        if not variant['changed_listings'] and (not np.array_equal(vectors[n],vectors[0]) or not np.array_equal(scores[n],scores[0])):
            raise ValueError('no-op ablation changed model output')
        rank = ranks(vectors[n])
        for p, pair in enumerate(request['pairs']):
            endpoints = [id_lookup[pair[k]] for k in ('sku_id1','sku_id2')]
            # The baseline variant ablates no attribute: carry the pair's
            # full evidence map instead of a meaningless {None: None}
            # (JSON "null" key); variant rows keep the ablated attribute's entry.
            evidence = pair.get('current_attribute_evidence',{})
            if variant['attribute'] is not None:
                evidence = {variant['attribute']:evidence.get(variant['attribute'])}
            rows.append({**pair, 'current_attribute_evidence':evidence,
                'attribute':variant['attribute'],'channel':variant['channel'],
                'endpoint_input_changed':[(bool(variant.get('text_indices')) and variant['text_indices'][i] != request['variants'][0]['text_indices'][i]) or
                    (bool(variant.get('records')) and variant['records'][i] != request['variants'][0]['records'][i]) for i in endpoints],
                'changed_listings':variant['changed_listings'], 'baseline_score':float(scores[0,p]),
                'ablated_score':float(scores[n,p]), 'score_delta':float(scores[n,p]-scores[0,p]),
                'decision_flip':bool((scores[n,p]>=threshold)!=(scores[0,p]>=threshold)),
                'embedding_cosine_delta':[float(1-np.dot(vectors[0,i],vectors[n,i])) for i in endpoints],
                'baseline_ranks':baseline_ranks[p], 'ablated_ranks':rank[p],
                'known_positive_recall_change':{str(k):[(int(rank[p][e]<=k)-int(baseline_ranks[p][e]<=k))
                    if pair['label']=='1' else None for e in (0,1)] for k in cfg.retrieval_ks}})
    output = {'schema':'er-attribute-ablation-report-v1', 'track':request['track'],
        'request_path':source_name(request_path), 'request_sha256':file_hash(request_path),
        'result_path':source_name(result),'result_sha256':file_hash(result),
        'sources':request['sources'],'composition':request['composition'],
        'implementation_sha256':request['implementation_sha256'], 'threshold':threshold,
        'threshold_source':str(threshold_source), 'threshold_provenance':threshold_provenance, 'split':cfg.split, 'sample_pairs':npairs,
        'intervention':request['intervention'],'retrieval_scope':request['retrieval_scope'],
        'missing_axes':request['missing_axes'], 'rows':rows}
    validate_sources(request)
    if frozen_threshold(threshold_source, threshold) != threshold_provenance:
        raise ValueError('threshold report changed during comparison')
    if not save:
        return output
    return save_report(request_path, output, config=config)


def save_report(request_path, output, *, config=None):
    """Persist an already-computed report; returns the dashboard pointer path."""
    path = resolve(settings(config).report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, output)
    # Keep each checkpoint's report alongside its inputs; dashboard pointer is latest.
    saved = request_path.parent/'report.json'
    if saved != path:
        write(saved, output)
    rows, threshold = output['rows'], output['threshold']
    print(f'[ablation/local] report={path} rows={len(rows)} threshold frozen={threshold}', flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for key in ('catalog','pairs','checkpoint'):
        prep.add_argument('--'+key,type=Path,required=True)
    prep.add_argument('--track', choices=('text','gnn_only','hybrid'),default='text')
    for key in ('listings','text-checkpoint','config'):
        prep.add_argument('--'+key,type=Path)
    worker = sub.add_parser('encode')
    worker.add_argument('--request',type=Path,required=True)
    worker.add_argument('--output',type=Path,required=True)
    post = sub.add_parser('report')
    post.add_argument('--request',type=Path,required=True)
    post.add_argument('--result',type=Path,required=True)
    post.add_argument('--threshold',type=float,required=True)
    post.add_argument('--threshold-source',required=True)
    post.add_argument('--config',type=Path)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'prepare':
        print(prepare(**args))
    elif action == 'encode':
        encode(args['request'],args['output'])
    else:
        args['request_path'] = args.pop('request')
        report(**args)


if __name__ == '__main__':
    main()
