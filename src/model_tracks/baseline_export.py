"""Prepared-only frozen baseline prerequisite for the single suite GPU session."""
import hashlib
import json
from pathlib import Path
import numpy as np
from graph_tracks.data import file_hash,load_records
from graph_tracks.text_cache import compose_texts,texts_hash
from training.prepare_embeddings import input_identity,validate_result
from model_tracks.embedding_forward import PreparedEmbeddingForward, validate_embedding_device


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import training_cfg
    return training_cfg().preparation.graph_setup


def prepare(setup,checkpoint,*,composer=None):
    layout = _setup_layout()
    metadata = input_identity(setup,checkpoint)
    ids,texts = compose_texts(setup/layout.catalog,composer=composer)
    export = json.loads((setup/layout.text_export_request).read_text())
    if ids != export['ids'] or texts_hash(texts) != export['text_sha256']:
        raise ValueError('baseline and selected text export population differ')
    request = {'schema':'er-embedding-request-v2','ids':ids,'texts':texts,
        'metadata':{**metadata,'text_sha256':texts_hash(texts)},
        'prepared_text':{**export['plan'],'sha256':export['tokens_sha256']}}
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
                validate_result(cache, request, request_sha256=file_hash(candidate))
            candidate.replace(path)
        finally:
            candidate.unlink(missing_ok=True)
    return path


def validate_pending(setup,checkpoint,*,native_model=None):
    from core.encoding_inputs import PreparedTokenInputs,tokenization_policy
    if native_model is None:
        from sentence_transformers import SentenceTransformer
        native_model = SentenceTransformer(str(checkpoint),device='cpu',local_files_only=True)
    layout = _setup_layout()
    path = setup/layout.embedding_request
    request = json.loads(path.read_text())
    expected = input_identity(setup,checkpoint)
    expected.pop('composition_implementation_sha256',None)
    if request.get('schema') != 'er-embedding-request-v2' or request.get('metadata',{}).get('text_sha256') != texts_hash(request['texts']):
        raise ValueError('pending baseline request corrupt')
    for key,value in expected.items():
        if request['metadata'].get(key) != value:
            raise ValueError('pending baseline source changed: '+key)
    if len(request['ids']) != len(request['texts']) or len(set(request['ids'])) != len(request['ids']):
        raise ValueError('pending baseline ID/text alignment differs')
    if set(request['ids']) != {r['sku_id'] for r in load_records(setup/layout.prepared_dir/'listings.json')}:
        raise ValueError('pending baseline listing population differs')
    tokens = setup/'prepared_text.npz'
    plan = request['prepared_text']
    if tokenization_policy(native_model) != plan['tokenization']:
        raise ValueError('pending baseline tokenizer differs from configured native checkpoint')
    if file_hash(tokens) != plan['sha256'] or plan.get('truncated_inputs') != 0:
        raise ValueError('pending baseline native tokens changed')
    with np.load(tokens,allow_pickle=False) as arrays:
        prepared = PreparedTokenInputs(plan=plan, arrays=arrays, row_count=len(request['ids']))
        count = prepared.row_count
    return {'status':'prepared GPU pending','rows':count,'checkpoint_sha256':expected['checkpoint_sha256'],
        'request_sha256':file_hash(path),'token_sha256':file_hash(tokens)}


def forward(setup,checkpoint,*,device,return_model=False):
    """GPU supervisor runs once before the trained workers start; no CPU composition."""
    layout = _setup_layout()
    request_path = setup/layout.embedding_request
    request_sha256 = file_hash(request_path)
    request = json.loads(request_path.read_text())
    output = setup/layout.shared_embeddings
    validate_embedding_device(device)
    from sentence_transformers import SentenceTransformer
    from graph_tracks.text_cache import checkpoint_hash
    model = SentenceTransformer(str(checkpoint),device=device,local_files_only=True)
    model._er_checkpoint_sha256 = checkpoint_hash(checkpoint)
    validate_pending(setup,checkpoint,native_model=model)
    if output.exists():
        validate_result(output,request,request_sha256=request_sha256)
        return (output,model) if return_model else output
    plan = request['prepared_text']
    contract = PreparedEmbeddingForward(device=device, checkpoint=checkpoint,
        request_path=request_path, tokens_path=setup/'prepared_text.npz', plan=plan,
        row_count=len(request['ids']), tokens_sha256=plan['sha256'])
    vectors, model, _, request_sha256 = contract.forward(model=model)
    current = input_identity(setup, checkpoint)
    current.pop('composition_implementation_sha256',None)
    for key, value in current.items():
        if request['metadata'].get(key) != value:
            raise ValueError('baseline source changed during encoding: ' + key)
    metadata = {**request['metadata'],'request_sha256':request_sha256,
        'embedding_dtype':contract.embedding_dtype,'tokenization':plan['tokenization']}
    path = contract.write(output, request['ids'], vectors, metadata,
        lambda candidate: validate_result(candidate,request,request_sha256=request_sha256))
    return (path,model) if return_model else path
