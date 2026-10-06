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
import contextvars
import functools
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from core.common import TRAIN_ROOT, retrieval_ks
from graph_tracks.data import file_hash, load_records, RELATIONS, NUMERIC
from graph_tracks.text_cache import checkpoint_hash, composition_fingerprint


def _default_retrieval_ks() -> tuple[int, ...]:
    """Inherit evaluation.retrieval_ks unless the lane declares an override."""
    return retrieval_ks()


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_default=True)
    sample_pairs: int = Field(default=100, ge=1)
    coverage: str = 'sampled'
    uniform_channels: bool = False
    seed: int = 1729
    split: str = 'dev'
    batch_size: int = Field(default=256, ge=1)
    accelerator: str = 'T4'
    attributes: list[str] = Field(default_factory=list)
    output_dir: str = 'results/attribute_ablation'
    report_path: str = 'results/attribute_ablation/report.json'
    retrieval_catalog: str = 'full'
    hnsw_m: int = Field(default=16,ge=1)
    hnsw_ef_construction: int = Field(default=200,ge=1)
    hnsw_ef_search: int = Field(default=100,ge=1)
    retrieval_ks: list[StrictInt] = Field(
        default_factory=lambda: list(_default_retrieval_ks()), min_length=1
    )
    graph_fields: dict[str, list[str]] = Field(default_factory=dict)
    slice_columns: list[str] = Field(default_factory=list)


    @model_validator(mode='after')
    def check_retrieval(self):
        if self.coverage not in {'sampled', 'all'}:
            raise ValueError('coverage must be sampled or all')
        if any(k < 1 for k in self.retrieval_ks) or len(set(self.retrieval_ks)) != len(self.retrieval_ks):
            raise ValueError('retrieval_ks must contain unique positive integers')
        return self


def settings(path=None):
    return Settings.model_validate(yaml.safe_load((path or TRAIN_ROOT/'config/attribute_ablation.yaml').read_text()))


_PORTABLE_CONTEXT = contextvars.ContextVar('ablation_portable_context',default=None)


@contextmanager
def request_context(request_path):
    request = json.loads(request_path.read_text())
    context = None
    if request.get('portable_setup'):
        anchor = Path(request['portable_setup'])
        if anchor.is_absolute() or '..' in anchor.parts:
            raise ValueError('unsafe portable setup anchor')
        # Bound jobs always live at <suite>/<track>/ablation/request.json.
        suite = request_path.parent.parent.parent
        setup = suite/'local_inputs'/request['portable_setup']
        if not setup.exists():
            setup = TRAIN_ROOT/request['portable_setup']
        context = {'@setup':setup,'@suite':suite}
    token = _PORTABLE_CONTEXT.set(context)
    try:
        yield
    finally:
        _PORTABLE_CONTEXT.reset(token)


def scoped_request(function):
    @functools.wraps(function)
    def wrapped(request_path,*args,**kwargs):
        with request_context(request_path):
            return function(request_path,*args,**kwargs)
    return wrapped


def resolve(path):
    path = Path(path)
    if path.parts and path.parts[0] in {'@setup','@suite'}:
        context = _PORTABLE_CONTEXT.get()
        if context is None:
            raise ValueError('portable ablation source requires verified request context')
        root = context[path.parts[0]]
        result = root.joinpath(*path.parts[1:]).resolve()
        if not result.is_relative_to(root.resolve()):
            raise ValueError('unsafe portable ablation source')
        return result
    return path if path.is_absolute() else TRAIN_ROOT/path


def digest(value):
    # STREAMED, never materialized. `json.dumps` builds the whole document as
    # one contiguous string before hashing; an exhaustive-cohort request is
    # ~1 GB of JSON (732 MB measured on the 2026-10-06 text track) and the
    # gnn_only digest ran while the text track's token batches were still
    # resident, so the kernel OOM-killed the run. iterencode is the SAME
    # encoder with the SAME kwargs, so the emitted bytes — and therefore every
    # cohort_sha256 / content-addressed directory name derived from them — are
    # byte-identical to the previous implementation; only peak memory drops.
    hasher = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=False).iterencode(value):
        hasher.update(chunk.encode())
    return hasher.hexdigest()


def source_name(path):
    path = Path(path).resolve()
    return path.relative_to(TRAIN_ROOT).as_posix() if path.is_relative_to(TRAIN_ROOT) else str(path)


def checkpoint_identity(path):
    return checkpoint_hash(path) if path.is_dir() else file_hash(path)


def write(path, value):
    # STREAMED for the same reason as digest(): a request of this size must
    # never exist as one in-memory string. json.dump writes incrementally
    # through the encoder's iterencode, so the bytes on disk are unchanged.
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, indent=2,
                  allow_nan=False)
        handle.write('\n')


def _field_surfaces(text: str) -> dict[str, set[str]]:
    """Declared attribute -> set of raw ';'-parts, one pass per endpoint."""
    from core.text import normalized_attribute_text

    surfaces: dict[str, set[str]] = {}
    for part in str(text).split(';'):
        if ':' not in part:
            surfaces.setdefault('', set()).add(part)
            continue
        field, cell = part.split(':', 1)
        surfaces.setdefault(normalized_attribute_text(field), set()).add(field + ':' + cell)
    return surfaces


def declaration_removed(row, attribute):
    from core.text import normalized_attribute_text
    from training.masking import field_of
    result = dict(row)
    if result.get('frozen_payload'):
        fields = {'volume': {'volume'}, 'count per unit': {'pack'},
            'flavour': {'flavor'}, 'carbonization': {'carbonation'},
            'sweetener': {'sweetener', 'sweetener_type', 'sweetening'},
            'pack type': {'package_type'}, 'pack material type': {'package_material'},
            'juice content': {'juice_content'}}.get(attribute, set())
        result['frozen_payload'] = ' '.join(token for token in result['frozen_payload'].split()
                                             if field_of(token) not in fields)
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
    if cfg.coverage == 'all':
        if frame.empty or not frame.label.isin(['0', '1']).all():
            raise ValueError('full ablation requires binary labeled pairs')
        return frame.to_dict('records')
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


def prepare(catalog, pairs, checkpoint, *, track='text', listings=None, text_checkpoint=None, config=None, checkpoint_role='selected',composer=None,token_cache=None):
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
               TRAIN_ROOT/'src/graph_tracks/infer.py',TRAIN_ROOT/'src/graph_tracks/pooling.py',
               TRAIN_ROOT/'src/model_tracks/ablation_retrieval.py',TRAIN_ROOT/'src/training/hnsw_index.py']
    sources = {source_name(p): checkpoint_identity(Path(p)) for p in inputs}
    chosen = sample_pairs(pd.read_csv(pairs, dtype=str, keep_default_na=False), cfg)
    # A present-but-empty slice column reads as '' (not None); normalize
    # all-empty axes to None so they are reported in missing_axes and the
    # report rows carry null instead of a silent empty-string stratum.
    for axis in cfg.slice_columns:
        if chosen and all(p.get(axis) is None or p.get(axis) == '' for p in chosen):
            for p in chosen:
                p[axis] = None
    from core.text import normalized_attribute_text
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
    if listings and any(i not in records or (cfg.coverage != 'all' and records[i]['split'] != cfg.split) for i in ids):
        raise ValueError('graph endpoints must belong to the selected held-out split')
    for pair in chosen:
        a, b = (rows[pair[k]] for k in ('sku_id1', 'sku_id2'))
        pair['gtin1'], pair['gtin2'] = a.get('gtin'), b.get('gtin')
        pair['current_attribute_evidence'] = ({} if a.get('frozen_payload') or b.get('frozen_payload') else
            engine().evaluate(canonical_attribute_info(a), canonical_attribute_info(b), left_raw=a, right_raw=b).as_dict())
        if a.get('frozen_payload') or b.get('frozen_payload'):
            pair['evidence_scope'] = 'frozen payload; raw-row evidence unavailable'
        for axis in cfg.slice_columns:
            pair.setdefault(axis, None)
    texts, lookup = [], {}
    def intern(text):
        if text not in lookup:
            lookup[text] = len(texts)
            texts.append(text)
        return lookup[text]
    def compose(row):
        if row.get('frozen_payload'):
            return row['frozen_payload']
        if composer is not None:
            return composer(row)
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
    endpoint_surfaces = {i: _field_surfaces(rows[i].get('attribute', '')) for i in ids}
    for attr_index, attribute in enumerate(attributes,1):
        altered_text = []
        if baseline_text:
            for n,i in enumerate(ids):
                row_surfaces = endpoint_surfaces[i]
                frozen = rows[i].get('frozen_payload')
                if attribute in row_surfaces or frozen:
                    kept_parts = [
                        part for part in str(rows[i].get('attribute', '')).split(';')
                        if part not in row_surfaces.get(attribute, ())
                    ]
                    kept_attribute = ';'.join(kept_parts)
                    changed_row = ({**rows[i], 'attribute': kept_attribute}
                        if not frozen else declaration_removed(rows[i], attribute))
                    altered_text.append(baseline_text[n] if changed_row == rows[i] else intern(compose(changed_row)))
                else:
                    altered_text.append(baseline_text[n])
        altered_records = [graph_removed(r, cfg.graph_fields.get(attribute, [])) for r in baseline_records]
        for channel in (['text', 'graph', 'both'] if cfg.uniform_channels else
                        ['text'] if track == 'text' else ['graph'] if track == 'gnn_only' else ['text', 'graph', 'both']):
            ti = altered_text if channel in {'text', 'both'} else baseline_text
            gr = altered_records if channel in {'graph', 'both'} else baseline_records
            changed = sum((bool(ti) and ti[n] != baseline_text[n]) or
                          (bool(gr) and gr[n] != baseline_records[n]) for n in range(len(ids)))
            variants.append({'attribute':attribute, 'channel':channel, 'text_indices':ti,
                             'records':gr, 'changed_listings':changed})
        print(f'[ablation/local] attribute={attr_index}/{len(attributes)} {attribute} unique_texts={len(texts)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    if cfg.retrieval_catalog not in {'full','sampled'}:
        raise ValueError('retrieval_catalog must be full or sampled')
    candidate_ids = sorted(rows) if cfg.retrieval_catalog == 'full' else ids
    candidate_text = []
    if track != 'gnn_only':
        baseline_lookup = dict(zip(ids,baseline_text))
        last_progress = time.monotonic()
        for n,i in enumerate(candidate_ids,1):
            candidate_text.append(baseline_lookup[i] if i in baseline_lookup else intern(compose(rows[i])))
            if n == len(candidate_ids) or time.monotonic()-last_progress >= 10:
                print(f'[ablation/local] candidate texts={n}/{len(candidate_ids)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                last_progress = time.monotonic()
    candidate_records = [records[i] for i in candidate_ids] if listings else []
    request = {'schema':'er-attribute-ablation-v2','track':track,'checkpoint_role':checkpoint_role, 'settings':cfg.model_dump(),
        'sources':sources, 'composition':composition_fingerprint(), 'implementation_sha256':file_hash(Path(__file__)),
        'checkpoint':source_name(checkpoint), 'text_checkpoint':source_name(text_checkpoint) if text_checkpoint else None,
        'candidate_ids':candidate_ids,'candidate_text_indices':candidate_text,'candidate_records':candidate_records,
        'ids':ids, 'texts':texts, 'pairs':chosen, 'variants':variants,
        'cohort_sha256':digest(chosen),
        'coverage':{'mode':cfg.coverage, 'pair_rows':len(chosen),
            'by_scope':pd.Series([p.get('evaluation_scope', p['split']) for p in chosen]).value_counts().to_dict(),
            'by_label':pd.Series([p['label'] for p in chosen]).value_counts().to_dict(),
            'by_population':pd.Series([p.get('population') or 'real' for p in chosen]).value_counts().to_dict(),
            'attributes':attributes, 'uniform_channels':cfg.uniform_channels},
        'intervention':'declared attribute removed; title/brand and training graph context fixed',
        'retrieval_scope':f'fixed {cfg.retrieval_catalog} catalog; query-only interventions; incomplete known-positive truth',
        'missing_axes':[a for a in cfg.slice_columns if all(p.get(a) is None or p.get(a) == '' for p in chosen)]}
    from model_tracks.ablation_inputs import prepare_inputs
    resolve(cfg.output_dir).mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=resolve(cfg.output_dir)) as tmp:
        prepared = Path(tmp)/'prepared_inputs.npz'
        request['prepared_inputs'] = prepare_inputs(request,prepared,token_cache=token_cache)
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


@scoped_request
def encode(request_path, output, *, device='cuda',saved_text=None,text_model=None,saved_candidates=None,graph_encoder=None):
    """Colab inference only; all interventions and texts arrive prepared."""
    import torch
    from model_tracks.embedding_forward import validate_embedding_device
    device = validate_embedding_device(device)
    if output.exists():
        raise FileExistsError(output)
    request = json.loads(request_path.read_text())
    # Sources are relocated by the launcher but expected hashes stay frozen.
    validate_sources(request)
    from model_tracks.ablation_inputs import load_batch
    from core.encoding_inputs import tokenization_policy, load_token_features
    arrays = load_prepared(request_path,request)
    plan = request['prepared_inputs']
    track = request['track']
    text_vectors = None
    if track != 'gnn_only':
        from sentence_transformers import SentenceTransformer
        checkpoint = request['checkpoint'] if track == 'text' else request['text_checkpoint']
        model = text_model
        expected_checkpoint = checkpoint_identity(resolve(checkpoint))
        if model is not None and getattr(model,'_er_checkpoint_sha256',None) != expected_checkpoint:
            raise ValueError('shared text model checkpoint differs from frozen request')
        if model is not None and model.device.type != device:
            raise ValueError('shared text model device differs from frozen request')
        if model is None:
            model = SentenceTransformer(str(resolve(checkpoint)),device=device,local_files_only=True)
        model.eval()
        if tokenization_policy(model) != plan['tokenization']:
            raise ValueError('worker tokenizer/checkpoint policy differs from local preparation')
        covered = np.zeros(len(request['texts']),dtype=bool)
        seeded = None
        if saved_text is not None:
            from graph_tracks.data import load_text_cache
            from graph_tracks.text_cache import texts_hash
            from core.model_input import model_input_composition
            with np.load(saved_text,allow_pickle=False) as cache:
                saved_ids = cache['ids'].astype(str).tolist()
            candidates,metadata = load_text_cache(saved_text,request['candidate_ids'])
            mapping = dict(zip(request['candidate_ids'],request['candidate_text_indices']))
            if set(saved_ids) != set(mapping):
                raise ValueError('baseline text export catalog differs from prepared ablation')
            if metadata.get('checkpoint_sha256') != expected_checkpoint or metadata.get('tokenization') != plan['tokenization'] or metadata.get('composition') != model_input_composition().model_dump(mode='json') or metadata.get('text_sha256') != texts_hash([request['texts'][mapping[key]] for key in saved_ids]):
                raise ValueError('baseline text export differs from prepared native text/checkpoint')
            seeded = np.empty((len(request['texts']),candidates.shape[-1]),dtype=np.float32)
            for row,index in enumerate(request['candidate_text_indices']):
                if covered[index] and not np.allclose(seeded[index],candidates[row],atol=1e-5):
                    raise ValueError('identical baseline texts have inconsistent exported vectors')
                seeded[index] = candidates[row]
                covered[index] = True
        chunks = []
        with torch.no_grad():
            for n,batch in enumerate(plan['token_batches'],1):
                features = load_token_features(arrays,batch,device)
                selected = np.flatnonzero(~covered[batch['start']:batch['start']+batch['count']])
                if not len(selected):
                    continue
                features = {key:value[torch.as_tensor(selected,device=device)] if isinstance(value,torch.Tensor) and value.ndim and len(value)==batch['count'] else value for key,value in features.items()}
                vectors = model(features)['sentence_embedding']
                vectors = torch.nn.functional.normalize(vectors,p=2,dim=1).cpu().numpy().astype(np.float32)
                if seeded is None:
                    seeded = np.empty((len(request['texts']),vectors.shape[-1]),dtype=np.float32)
                positions = batch['start']+selected
                seeded[positions] = vectors
                covered[positions] = True
                print(f'[ablation/gpu] changed native texts={len(selected)} batch={n}/{len(plan["token_batches"])}; truncated=0',flush=True)
        if not covered.all():
            raise ValueError('prepared ablation text vectors miss native inputs')
        text_vectors = seeded
        del model
    encoder = None
    graph_batches = {}
    if track != 'text':
        from graph_tracks.infer import GraphEncoder
        vocabulary = plan['vocabulary']
        encoder = graph_encoder
        if encoder is None:
            support = load_batch(arrays,'support',device,vocabulary)
            encoder = GraphEncoder(resolve(request['checkpoint']),device,prepared_support=support)
        elif encoder.checkpoint_sha256 != file_hash(resolve(request['checkpoint'])) or encoder.device != device:
            raise ValueError('shared graph encoder differs from frozen checkpoint/device')
        if encoder.vocabulary != vocabulary:
            raise ValueError('prepared vocabulary differs from checkpoint')
        graph_batches = {key:[load_batch(arrays,prefix,device,vocabulary) for prefix in prefixes]
                         for key,prefixes in plan['graph_batches'].items()}
    candidate_vectors = saved_candidates
    if candidate_vectors is not None and (candidate_vectors.dtype != np.float32 or candidate_vectors.shape[0] != len(request['candidate_ids']) or not np.isfinite(candidate_vectors).all() or not np.allclose(np.linalg.norm(candidate_vectors,axis=1),1,atol=1e-4)):
        raise ValueError('saved graph candidate vector contract mismatch')
    if candidate_vectors is None and request.get('candidate_ids'):
        candidate_text = text_vectors[arrays['candidate_text_indices']] if text_vectors is not None else None
        candidate_vectors = candidate_text if encoder is None else encoder.encode_prepared(
            [load_batch(arrays,prefix,device,plan['vocabulary']) for prefix in plan['candidate_batches']],candidate_text)
    indices = arrays['pair_indices']
    results = []
    for n,job in enumerate(plan['jobs'],1):
        text = text_vectors[arrays[job['text_indices_key']]] if text_vectors is not None else None
        if encoder is None:
            vec = text
        elif saved_candidates is not None and n-1 == plan['variant_jobs'][0]:
            candidate_lookup = {key:i for i,key in enumerate(request['candidate_ids'])}
            vec = saved_candidates[[candidate_lookup[key] for key in request['ids']]]
        else:
            vec = encoder.encode_prepared(graph_batches[job['graph_key']],text)
        if encoder is None:
            score = (vec[indices[:,0]]*vec[indices[:,1]]).sum(-1)
        else:
            with torch.no_grad():
                score = encoder.scorer(torch.as_tensor(vec,device=device),torch.as_tensor(indices,device=device),
                    None if text is None else torch.as_tensor(text,device=device)).sigmoid().cpu().numpy()
        results.append((vec,score))
        print(f'[ablation/{device}] prepared inference job={n}/{len(plan["jobs"])}',flush=True)
    vectors = [results[job][0] for job in plan['variant_jobs']]
    scores = [results[job][1] for job in plan['variant_jobs']]
    arrays.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.savez_compressed(handle, vectors=np.asarray(vectors,dtype=np.float32), scores=np.asarray(scores,dtype=np.float32),
                            request_sha256=file_hash(request_path),embedding_dtype='float32',
                            **({'candidate_vectors':np.asarray(candidate_vectors,dtype=np.float32)} if candidate_vectors is not None else {}))
    output.with_suffix('.sha256').write_text(file_hash(output))


def frozen_threshold(source, value):
    path = resolve(source)
    if not path.is_file():
        raise ValueError('threshold source must be an existing saved report')
    before = file_hash(path)
    track = None
    checkpoint = None
    claimed_sha256 = None
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
        if isinstance(document,dict):
            claimed_sha256 = document.get('checkpoint_sha256') or document.get('vectors_metadata',{}).get('checkpoint_sha256')
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
            'track':track, 'checkpoint':checkpoint, 'checkpoint_sha256':claimed_sha256}


def verify_threshold_binding(request, provenance):
    if provenance.get('track') != request['track']:
        raise ValueError('threshold source track differs or is missing')
    checkpoint = request.get('checkpoint')
    expected = request.get('sources',{}).get(checkpoint)
    if not checkpoint or not expected:
        raise ValueError('threshold requires a checkpoint identity in the request')
    claimed = provenance.get('checkpoint_sha256')
    if claimed:
        if claimed != expected:
            raise ValueError('threshold source checkpoint identity differs')
    else:
        named = provenance.get('checkpoint')
        if not named:
            raise ValueError('threshold source checkpoint identity is missing')
        if Path(named).name != Path(checkpoint).name:
            raise ValueError('threshold source checkpoint differs')
        candidates = [resolve(named),resolve(provenance['path']).parent/named,
                      resolve(provenance['path']).parent/Path(named).name]
        located = next((path for path in candidates if path.exists()),None)
        if located is None or checkpoint_identity(located) != expected:
            raise ValueError('threshold source checkpoint identity differs or cannot be verified')
    return {'track':request['track'],'checkpoint_sha256':expected,'verified':True}


@scoped_request
def validate_vectors(request_path, result):
    """Cheap integrity validation before closing the GPU; no metrics or ANN."""
    request = json.loads(request_path.read_text())
    validate_sources(request)
    if 'prepared_inputs' in request:
        load_prepared(request_path,request).close()
    with np.load(result,allow_pickle=False) as data:
        if str(data['request_sha256'].item()) != file_hash(request_path):
            raise ValueError('ablation result belongs to another request')
        vectors, scores = data['vectors'],data['scores']
        candidates = data['candidate_vectors'] if 'candidate_vectors' in data else None
    expected = (len(request['variants']),len(request['ids']))
    if vectors.dtype != np.float32 or scores.dtype != np.float32 or (candidates is not None and candidates.dtype != np.float32):
        raise ValueError('ablation exported vectors/scores require persisted float32')
    if vectors.ndim != 3 or vectors.shape[:2] != expected or scores.shape != (expected[0],len(request['pairs'])):
        raise ValueError('ablation result shape mismatch')
    for matrix in (vectors,candidates):
        if matrix is not None and (not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix,axis=-1),1,atol=1e-4)):
            raise ValueError('ablation vectors must be finite and normalized')
    if not np.isfinite(scores).all():
        raise ValueError('ablation scores must be finite')
    if request.get('candidate_ids') and (candidates is None or candidates.ndim != 2 or candidates.shape != (len(request['candidate_ids']),vectors.shape[-1])):
        raise ValueError('full catalog candidate vectors missing or shape mismatch')
    return request,vectors,scores,candidates


@scoped_request
def report(request_path, result, threshold, *, threshold_source, config=None, save=True):
    """Paired local comparisons at a supplied, already selected threshold."""
    if not np.isfinite(threshold) or not threshold_source:
        raise ValueError('frozen threshold and its source are required')
    threshold_provenance = frozen_threshold(threshold_source, threshold)
    request,vectors,scores,candidate_vectors = validate_vectors(request_path,result)
    threshold_binding = verify_threshold_binding(request,threshold_provenance)
    nv, ni, npairs = len(request['variants']),len(request['ids']),len(request['pairs'])
    cfg = Settings.model_validate(request['settings'])
    if not cfg.retrieval_ks or any(k < 1 for k in cfg.retrieval_ks):
        raise ValueError('retrieval ks must be positive')
    id_lookup = {i:n for n,i in enumerate(request['ids'])}
    from model_tracks.ablation_retrieval import RetrievalComparison
    candidate_ids = request.get('candidate_ids',request['ids'])
    if request.get('candidate_ids') and candidate_vectors is None:
        raise ValueError('full catalog candidate vectors missing')
    candidates = vectors[0] if candidate_vectors is None else candidate_vectors
    retrieval = RetrievalComparison(candidate_ids,candidates,request,request_path,cfg)
    def ranks(vec):
        return retrieval.ranks(vec)
    baseline_ranks = ranks(vectors[0])
    ann_baseline = retrieval.ann_hits(vectors[0])
    comparison_cache = {hashlib.sha256(vectors[0].tobytes()).hexdigest():(baseline_ranks,ann_baseline)}
    rows = []
    for n, variant in enumerate(request['variants'][1:], 1):
        if not variant['changed_listings'] and (not np.array_equal(vectors[n],vectors[0]) or not np.array_equal(scores[n],scores[0])):
            raise ValueError('no-op ablation changed model output')
        key = hashlib.sha256(vectors[n].tobytes()).hexdigest()
        if key not in comparison_cache:
            comparison_cache[key] = (ranks(vectors[n]),retrieval.ann_hits(vectors[n]))
        rank,ann_ablated = comparison_cache[key]
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
                'baseline_error':bool((scores[0,p]>=threshold) != (pair['label']=='1')),
                'ablated_error':bool((scores[n,p]>=threshold) != (pair['label']=='1')),
                'embedding_cosine_delta':[float(1-np.dot(vectors[0,i],vectors[n,i])) for i in endpoints],
                'baseline_ranks':baseline_ranks[p], 'ablated_ranks':rank[p],
                'ann_baseline_hits':ann_baseline[p], 'ann_ablated_hits':ann_ablated[p],
                'known_positive_recall_change':{str(k):[(int(rank[p][e]<=k)-int(baseline_ranks[p][e]<=k))
                    if pair['label']=='1' else None for e in (0,1)] for k in cfg.retrieval_ks}})
    retrieval.close()
    output = {'schema':'er-attribute-ablation-report-v1', 'track':request['track'],'checkpoint_role':request.get('checkpoint_role','selected'),
        'request_path':source_name(request_path), 'request_sha256':file_hash(request_path),
        'result_path':source_name(result),'result_sha256':file_hash(result),
        'sources':request['sources'],'composition':request['composition'],
        'implementation_sha256':request['implementation_sha256'], 'embedding_dtype':'float32', 'threshold':threshold,
        'threshold_source':str(threshold_source), 'threshold_provenance':threshold_provenance, 'threshold_binding':threshold_binding, 'split':cfg.split, 'sample_pairs':npairs,
        'intervention':request['intervention'],'retrieval_scope':request['retrieval_scope'],
        'missing_axes':request['missing_axes'], 'retrieval_catalog_count':len(candidate_ids),
        'cohort_sha256':request.get('cohort_sha256'), 'coverage':request.get('coverage'),
        'retrieval_intervention':'query only; fixed candidates', 'rows':rows}
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
