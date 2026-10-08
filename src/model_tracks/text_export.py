"""Frozen local native tokens; selected text checkpoint GPU forward only."""
import json
from pathlib import Path
import numpy as np
from graph_tracks.data import file_size, load_records, load_text_cache
from graph_tracks.text_cache import checkpoint_size, compose_texts, composition_fingerprint, texts_size
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
    model_key = ('model',checkpoint_size(checkpoint))
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
    request = {'schema':'er-text-export-v1','ids':ids,'plan':plan,'tokens_size':file_size(tokens),
        'catalog_size':file_size(setup/layout.catalog),
        'listings_size':file_size(setup/layout.prepared_dir/layout.listings),
        'pairs_size':file_size(setup/layout.prepared_dir/'pairs.csv'),
        'text_size':texts_size(texts),'composition':model_input_composition().model_dump(mode='json'),
        'composition_implementation_size':composition_fingerprint(),
        'export_implementation_size':file_size(Path(__file__)),
        'token_implementation_size':file_size(__import__('core.encoding_inputs',fromlist=['x']).__file__)}
    (setup/layout.text_export_request).write_text(json.dumps(request,sort_keys=True))
    return setup/layout.text_export_request


def _export_request(setup):
    """The prepared suite's frozen text export request, document and path."""
    layout = _setup_layout()
    path = setup/layout.text_export_request
    return path, json.loads(path.read_text())


def _export_metadata(request, checkpoint_size, contract, model):
    """The identities recorded INTO one saved text export (never compared)."""
    metadata = {key:request[key] for key in ('catalog_size','listings_size','pairs_size','text_size','composition','composition_implementation_size','export_implementation_size','token_implementation_size')}
    metadata.update(checkpoint_size=checkpoint_size,
        tokenization=request['plan']['tokenization'],export_location=contract.export_location,
        truncated_inputs=0,embedding_dtype=contract.embedding_dtype,
        performance=model._er_forward_performance)
    return metadata


def forward(output,setup,*,device,return_model=False):
    """Forward the prepared native tokens for one selected text checkpoint.

    NO FRESHNESS COMPARISONS (owner directive 2026-10-08, repo-wide): the
    prepared tokens and the catalog/listings/pairs are NOT re-measured and
    compared against the sizes the export request recorded. The request and
    the tokens it names are the prepared suite's own frozen pair shipped inside
    one bundle (whose integrity is its size checks at the boundary), and the written
    export's identities are checked by :func:`validate` on write.
    """
    from training.validation_inference import resolve_best_checkpoint
    request_path, request = _export_request(setup)
    checkpoint,_ = resolve_best_checkpoint(output)
    contract = PreparedEmbeddingForward(device=device, checkpoint=checkpoint,
        request_path=request_path, tokens_path=setup/'prepared_text.npz', plan=request['plan'],
        row_count=len(request['ids']), tokens_size=request['tokens_size'])
    vectors, model, checkpoint_size, _ = contract.forward()
    path = output/'text__vectors.npz'
    contract.write(path, request['ids'], vectors,
                   _export_metadata(request, checkpoint_size, contract, model),
                   lambda candidate: validate(candidate, checkpoint, setup))
    model._er_checkpoint_size = checkpoint_size
    return (path,model) if return_model else path


def validate(path,checkpoint,setup):
    """Identity/compatibility of one saved GPU text export.

    NO FRESHNESS COMPARISON (owner directive 2026-10-08, repo-wide): the export
    request is NOT re-measured and compared against the value recorded in the
    export metadata. The checkpoint, catalog, listings, and pairs identities are
    checked so the vectors provably belong to this suite's prepared inputs; the
    bundle's integrity is its size checks at the boundary.
    """
    layout = _setup_layout()
    ids = [r['sku_id'] for r in load_records(setup/layout.prepared_dir/layout.listings)]
    vectors,metadata = load_text_cache(path,ids)
    for key,expected in [('checkpoint_size',checkpoint_size(checkpoint)),('catalog_size',file_size(setup/layout.catalog)),
            ('listings_size',file_size(setup/layout.prepared_dir/layout.listings)),('pairs_size',file_size(setup/layout.prepared_dir/'pairs.csv'))]:
        if metadata.get(key) != expected:
            raise ValueError('saved GPU text export mismatch: '+key)
    if metadata.get('export_location') not in {'Colab GPU', 'Colab CPU'} or metadata.get('truncated_inputs') != 0 or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=PreparedEmbeddingForward.normalization_atol):
        raise ValueError('saved GPU text export contract mismatch')
    return vectors,metadata
