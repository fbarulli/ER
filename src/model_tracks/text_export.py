"""Frozen local native tokens; selected text checkpoint GPU forward only."""
import json
from pathlib import Path
import numpy as np
from graph_tracks.data import file_hash, load_records, load_text_cache
from graph_tracks.text_cache import checkpoint_hash, compose_texts, composition_fingerprint, texts_hash


def prepare_tokens(checkpoint,texts,arrays,*,batch_size,cache=None):
    from sentence_transformers import SentenceTransformer
    from core.encoding_inputs import prepare_token_batches,tokenization_policy
    from model_tracks.ablation import digest
    cache = cache if cache is not None else {}
    model_key = ('model',checkpoint_hash(checkpoint))
    if model_key not in cache:
        cache[model_key] = SentenceTransformer(str(checkpoint),device='cpu',local_files_only=True)
    model = cache[model_key]
    key = ('tokens',digest({'texts':texts,'batch_size':batch_size,'policy':tokenization_policy(model)}))
    if key not in cache:
        frozen = {}
        plan = prepare_token_batches(model,texts,frozen,batch_size=batch_size)
        cache[key] = (plan,frozen)
    plan,frozen = cache[key]
    arrays.update(frozen)
    return plan


def prepare(setup, checkpoint, *, batch_size=256,composer=None,token_cache=None):
    from sentence_transformers import SentenceTransformer
    from core.encoding_inputs import prepare_token_batches
    from core.model_input import model_input_composition
    ids,texts = compose_texts(setup/'eligible_catalog.csv',composer=composer)
    if set(ids) != {r['sku_id'] for r in load_records(setup/'prepared/listings.json')}:
        raise ValueError('text export catalog/listing population differs')
    arrays = {}
    plan = prepare_tokens(checkpoint,texts,arrays,batch_size=batch_size,cache=token_cache)
    tokens = setup/'prepared_text.npz'
    with tokens.open('wb') as handle:
        np.savez_compressed(handle,**arrays)
    request = {'schema':'er-text-export-v1','ids':ids,'plan':plan,'tokens_sha256':file_hash(tokens),
        'catalog_sha256':file_hash(setup/'eligible_catalog.csv'),
        'listings_sha256':file_hash(setup/'prepared/listings.json'),
        'pairs_sha256':file_hash(setup/'prepared/pairs.csv'),
        'text_sha256':texts_hash(texts),'composition':model_input_composition().model_dump(mode='json'),
        'composition_implementation_sha256':composition_fingerprint(),
        'export_implementation_sha256':file_hash(Path(__file__)),
        'token_implementation_sha256':file_hash(__import__('core.encoding_inputs',fromlist=['x']).__file__)}
    (setup/'text_export_request.json').write_text(json.dumps(request,sort_keys=True))
    return setup/'text_export_request.json'


def forward(output,setup,*,return_model=False,device="cuda"):
    import torch
    from sentence_transformers import SentenceTransformer
    from core.encoding_inputs import PreparedTokenInputs, tokenization_policy, load_token_features
    from training.validation_inference import resolve_best_checkpoint
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError('text export requires CUDA')
    request_path = setup/'text_export_request.json'
    request_sha256 = file_hash(request_path)
    request = json.loads(request_path.read_text())
    tokens = setup/'prepared_text.npz'
    if file_hash(tokens) != request['tokens_sha256']:
        raise ValueError('text export tokens changed')
    for key,path in [('catalog_sha256',setup/'eligible_catalog.csv'),('listings_sha256',setup/'prepared/listings.json'),('pairs_sha256',setup/'prepared/pairs.csv')]:
        if file_hash(path) != request[key]:
            raise ValueError('text export source changed: '+key)
    checkpoint,_ = resolve_best_checkpoint(output)
    checkpoint_sha256 = checkpoint_hash(checkpoint)
    model = SentenceTransformer(str(checkpoint),device=device,local_files_only=True)
    model.eval()
    if tokenization_policy(model) != request['plan']['tokenization']:
        raise ValueError('selected checkpoint native tokenizer differs from prepared export')
    chunks = []
    with np.load(tokens,allow_pickle=False) as arrays,torch.no_grad():
        PreparedTokenInputs(plan=request['plan'], arrays=arrays, row_count=len(request['ids']))
        for batch in request['plan']['token_batches']:
            # Native features were frozen locally; never tokenize on the GPU.
            features = load_token_features(arrays,batch,device)
            vectors = model(features)['sentence_embedding']
            chunks.append(torch.nn.functional.normalize(vectors,p=2,dim=1).cpu().numpy().astype(np.float32))
    vectors = np.concatenate(chunks)
    metadata = {key:request[key] for key in ('catalog_sha256','listings_sha256','pairs_sha256','text_sha256','composition','composition_implementation_sha256','export_implementation_sha256','token_implementation_sha256')}
    if file_hash(request_path) != request_sha256:
        raise ValueError('text export request changed during encoding')
    if checkpoint_hash(checkpoint) != checkpoint_sha256 or file_hash(tokens) != request['tokens_sha256']:
        raise ValueError('text export checkpoint or tokens changed during encoding')
    metadata.update(checkpoint_sha256=checkpoint_sha256,request_sha256=request_sha256,
        tokenization=request['plan']['tokenization'],export_location='Colab GPU' if device == 'cuda' else 'Colab CPU',truncated_inputs=0,embedding_dtype='float32')
    path = output/'text__vectors.npz'
    candidate = path.with_suffix('.npz.partial')
    with candidate.open('wb') as handle:
        np.savez_compressed(handle,ids=np.asarray(request['ids'],dtype=str),embeddings=vectors,metadata=json.dumps(metadata,sort_keys=True))
    validate(candidate, checkpoint, setup)
    candidate.replace(path)
    model._er_checkpoint_sha256 = metadata['checkpoint_sha256']
    return (path,model) if return_model else path


def validate(path,checkpoint,setup):
    ids = [r['sku_id'] for r in load_records(setup/'prepared/listings.json')]
    vectors,metadata = load_text_cache(path,ids)
    for key,expected in [('checkpoint_sha256',checkpoint_hash(checkpoint)),('catalog_sha256',file_hash(setup/'eligible_catalog.csv')),
            ('listings_sha256',file_hash(setup/'prepared/listings.json')),('pairs_sha256',file_hash(setup/'prepared/pairs.csv')),
            ('request_sha256',file_hash(setup/'text_export_request.json')),
            ('composition_implementation_sha256',composition_fingerprint()),
            ('export_implementation_sha256',file_hash(Path(__file__))),
            ('token_implementation_sha256',file_hash(__import__('core.encoding_inputs',fromlist=['x']).__file__))]:
        if metadata.get(key) != expected:
            raise ValueError('saved GPU text export mismatch: '+key)
    if metadata.get('export_location') not in {'Colab GPU', 'Colab CPU'} or metadata.get('truncated_inputs') != 0 or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
        raise ValueError('saved GPU text export contract mismatch')
    return vectors,metadata
