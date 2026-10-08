"""Frozen local native tokens; selected text checkpoint GPU forward only."""
import json
from pathlib import Path
import numpy as np
from graph_tracks.data import file_hash, load_records, load_text_cache
from graph_tracks.text_cache import checkpoint_hash, compose_texts, composition_fingerprint, texts_hash
from model_tracks.embedding_forward import PreparedEmbeddingForward


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def prepare_tokens(checkpoint,texts,arrays,*,batch_size,cache=None):
    from core.common import training_cfg
    from sentence_transformers import SentenceTransformer
    from core.encoding_inputs import prepare_token_batches,tokenization_policy
    from model_tracks.ablation import digest
    cache = cache if cache is not None else {}
    model_key = ('model',checkpoint_hash(checkpoint))
    if model_key not in cache:
        cache[model_key] = SentenceTransformer(str(checkpoint),device='cpu',local_files_only=True)
    model = cache[model_key]
    key = ('tokens',digest({'texts':texts,'batch_size':batch_size,'policy':tokenization_policy(model)}))
    cached = cache.get(key)
    if cached is None:
        # Evict previous populations before allocating the next token set. Keep
        # the native model, but never retain tokens for every ablation at once.
        for old_key in list(cache):
            if old_key[0] == 'tokens':
                del cache[old_key]
        frozen = {}
        plan = prepare_token_batches(model,texts,frozen,batch_size=batch_size)
        # Large token sets are already persisted by callers. Retain only small
        # sets for immediate cross-track reuse within the configured array-byte budget.
        if sum(value.nbytes for value in frozen.values()) <= training_cfg().packaging.token_cache_bytes:
            cache[key] = (plan, frozen)
    else:
        plan, frozen = cached
    arrays.update(frozen)
    return plan


def prepare(setup, checkpoint, *, batch_size=None,composer=None,token_cache=None):
    from core.common import runtime
    batch_size = runtime("batch_size_embed") if batch_size is None else batch_size
    from core.model_input import model_input_composition
    layout = _setup_layout()
    ids,texts = compose_texts(setup/layout.catalog,composer=composer)
    if set(ids) != {r['sku_id'] for r in load_records(setup/layout.prepared_dir/layout.listings)}:
        raise ValueError('text export catalog/listing population differs')
    arrays = {}
    plan = prepare_tokens(checkpoint,texts,arrays,batch_size=batch_size,cache=token_cache)
    tokens = setup/'prepared_text.npz'
    with tokens.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
    request = {'schema':'er-text-export-v1','ids':ids,'plan':plan,'tokens_sha256':file_hash(tokens),
        'catalog_sha256':file_hash(setup/layout.catalog),
        'listings_sha256':file_hash(setup/layout.prepared_dir/layout.listings),
        'pairs_sha256':file_hash(setup/layout.prepared_dir/'pairs.csv'),
        'text_sha256':texts_hash(texts),'composition':model_input_composition().model_dump(mode='json'),
        'composition_implementation_sha256':composition_fingerprint(),
        'export_implementation_sha256':file_hash(Path(__file__)),
        'token_implementation_sha256':file_hash(__import__('core.encoding_inputs',fromlist=['x']).__file__)}
    (setup/layout.text_export_request).write_text(json.dumps(request,sort_keys=True))
    return setup/layout.text_export_request


def forward(output,setup,*,device,return_model=False):
    from training.validation_inference import resolve_best_checkpoint
    layout = _setup_layout()
    request_path = setup/layout.text_export_request
    request = json.loads(request_path.read_text())
    tokens = setup/'prepared_text.npz'
    if file_hash(tokens) != request['tokens_sha256']:
        raise ValueError('text export tokens changed')
    for key,path in [('catalog_sha256',setup/layout.catalog),('listings_sha256',setup/layout.prepared_dir/layout.listings),('pairs_sha256',setup/layout.prepared_dir/'pairs.csv')]:
        if file_hash(path) != request[key]:
            raise ValueError('text export source changed: '+key)
    checkpoint,_ = resolve_best_checkpoint(output)
    contract = PreparedEmbeddingForward(device=device, checkpoint=checkpoint,
        request_path=request_path, tokens_path=tokens, plan=request['plan'],
        row_count=len(request['ids']), tokens_sha256=request['tokens_sha256'])
    vectors, model, checkpoint_sha256, _request_sha256 = contract.forward()
    metadata = {key:request[key] for key in ('catalog_sha256','listings_sha256','pairs_sha256','text_sha256','composition','composition_implementation_sha256','export_implementation_sha256','token_implementation_sha256')}
    metadata.update(checkpoint_sha256=checkpoint_sha256,
        tokenization=request['plan']['tokenization'],export_location=contract.export_location,
        truncated_inputs=0,embedding_dtype=contract.embedding_dtype,
        performance=model._er_forward_performance)
    path = output/'text__vectors.npz'
    contract.write(path, request['ids'], vectors, metadata, lambda candidate: validate(candidate, checkpoint, setup))
    model._er_checkpoint_sha256 = metadata['checkpoint_sha256']
    return (path,model) if return_model else path


def validate(path,checkpoint,setup):
    """Identity/compatibility of one saved GPU text export.

    NO FRESHNESS COMPARISON (owner directive 2026-10-08, repo-wide): the export
    request is NOT re-hashed and compared against the value recorded in the
    export metadata. The checkpoint, catalog, listings, and pairs identities are
    checked so the vectors provably belong to this suite's prepared inputs; the
    bundle's integrity is its digest at the boundary.
    """
    layout = _setup_layout()
    ids = [r['sku_id'] for r in load_records(setup/layout.prepared_dir/layout.listings)]
    vectors,metadata = load_text_cache(path,ids)
    for key,expected in [('checkpoint_sha256',checkpoint_hash(checkpoint)),('catalog_sha256',file_hash(setup/layout.catalog)),
            ('listings_sha256',file_hash(setup/layout.prepared_dir/layout.listings)),('pairs_sha256',file_hash(setup/layout.prepared_dir/'pairs.csv'))]:
        if metadata.get(key) != expected:
            raise ValueError('saved GPU text export mismatch: '+key)
    if metadata.get('export_location') not in {'Colab GPU', 'Colab CPU'} or metadata.get('truncated_inputs') != 0 or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=PreparedEmbeddingForward.normalization_atol):
        raise ValueError('saved GPU text export contract mismatch')
    return vectors,metadata
