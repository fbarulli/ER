"""Prepared-only frozen baseline prerequisite for the single suite GPU session."""
from core.portable_archive import ByteCount
import json
from pathlib import Path
import numpy as np
from graph_tracks.data import file_size,load_records
from graph_tracks.text_cache import compose_texts,texts_size
from training.prepare_embeddings import input_identity,validate_result
from model_tracks.embedding_forward import PreparedEmbeddingForward, validate_embedding_device


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def prepare(setup,checkpoint,*,composer=None):
    layout = _setup_layout()
    metadata = input_identity(setup,checkpoint)
    ids,texts = compose_texts(setup/layout.catalog,composer=composer)
    export = json.loads((setup/layout.text_export_request).read_text())
    if ids != export['ids'] or texts_size(texts) != export['text_size']:
        raise ValueError('baseline and selected text export population differ')
    request = {'schema':'er-embedding-request-v2','ids':ids,'texts':texts,
        'metadata':{**metadata,'text_size':texts_size(texts)},
        'prepared_text':{**export['plan'],'size':export['tokens_size']}}
    path = setup/layout.embedding_request
    # Avoid a second full-size JSON string plus UTF-8 copy alongside the texts.
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=setup,
                                     prefix=path.name + '.', suffix='.partial',
                                     delete=False) as handle:
        candidate = Path(handle.name)
        try:
            json.dump(request, handle, ensure_ascii=False, sort_keys=True)
            handle.close()
            cache = setup / layout.shared_embeddings
            if cache.exists():
                validate_result(cache, request)
            candidate.replace(path)
        finally:
            candidate.unlink(missing_ok=True)
    return path


def validate_pending(setup,checkpoint,*,native_model=None):
    """The pre-forward contract of a prepared but unencoded baseline.

    NO FRESHNESS COMPARISON (owner directive 2026-10-08, repo-wide): the
    request's recorded input sizes are not re-derived and compared against
    the current tree. What this checks is what the forward pass requires: a v2
    request whose recorded text size matches its own texts, aligned and unique
    IDs covering the prepared listings, and a frozen native-token plan whose
    tokenizer is the configured checkpoint's.
    """
    from core.encoding_inputs import PreparedTokenInputs,tokenization_policy
    if native_model is None:
        from sentence_transformers import SentenceTransformer
        native_model = SentenceTransformer(str(checkpoint),device='cpu',local_files_only=True)
    layout = _setup_layout()
    path = setup/layout.embedding_request
    request = json.loads(path.read_text())
    expected = input_identity(setup,checkpoint)
    if request.get('schema') != 'er-embedding-request-v2' or request.get('metadata',{}).get('text_size') != texts_size(request['texts']):
        raise ValueError('pending baseline request corrupt')
    if len(request['ids']) != len(request['texts']) or len(set(request['ids'])) != len(request['ids']):
        raise ValueError('pending baseline ID/text alignment differs')
    if set(request['ids']) != {r['sku_id'] for r in load_records(setup/layout.prepared_dir/layout.listings)}:
        raise ValueError('pending baseline listing population differs')
    tokens = setup/'prepared_text.npz'
    plan = request['prepared_text']
    if tokenization_policy(native_model) != plan['tokenization']:
        raise ValueError('pending baseline tokenizer differs from configured native checkpoint')
    if file_size(tokens) != plan['size'] or plan.get('truncated_inputs') != 0:
        raise ValueError('pending baseline native tokens changed')
    with np.load(tokens,allow_pickle=False) as arrays:
        prepared = PreparedTokenInputs(plan=plan, arrays=arrays, row_count=len(request['ids']))
        count = prepared.row_count
    return {'status':'prepared GPU pending','rows':count,'checkpoint_size':expected['checkpoint_size'],
        'token_size':file_size(tokens)}


def forward(setup,checkpoint,*,device,return_model=False):
    """GPU supervisor runs once before the trained workers start; no CPU composition.

    NO FRESHNESS COMPARISONS (owner directive 2026-10-08, repo-wide): the
    embedding request is never re-measured to decide whether an existing export
    still applies. An export whose recorded metadata does not match the request
    is rebuilt below, and the request/result contract is checked on identity
    only.
    """
    layout = _setup_layout()
    request_path = setup/layout.embedding_request
    request = json.loads(request_path.read_text())
    output = setup/layout.shared_embeddings
    validate_embedding_device(device)
    from sentence_transformers import SentenceTransformer
    from graph_tracks.text_cache import checkpoint_size
    model = SentenceTransformer(str(checkpoint),device=device,local_files_only=True)
    model._er_checkpoint_size = checkpoint_size(checkpoint)
    validate_pending(setup,checkpoint,native_model=model)
    if output.exists():
        validate_result(output,request)
        return (output,model) if return_model else output
    plan = request['prepared_text']
    contract = PreparedEmbeddingForward(device=device, checkpoint=checkpoint,
        request_path=request_path, tokens_path=setup/'prepared_text.npz', plan=plan,
        row_count=len(request['ids']), tokens_size=plan['size'])
    vectors, model, _, _request_size = contract.forward(model=model)
    metadata = {**request['metadata'],
        'embedding_dtype':contract.embedding_dtype,'tokenization':plan['tokenization']}
    path = contract.write(output, request['ids'], vectors, metadata,
        lambda candidate: validate_result(candidate,request))
    return (path,model) if return_model else path
