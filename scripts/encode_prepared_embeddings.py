"""Colab worker: encode locally prepared text, never compose or reuse a cache."""
import argparse
import hashlib
import json
from pathlib import Path


def checkpoint_hash(path):
    digest = hashlib.sha256()
    for file in sorted(path.rglob('*')):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def main():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing CPU fallback')
    if args.output.exists():
        raise FileExistsError('Worker never reuses an existing output')
    raw = args.request.read_bytes()
    request = json.loads(raw)
    if request['schema'] != 'er-embedding-request-v2' or len(request['ids']) != len(request['texts']):
        raise ValueError('Invalid prepared text request')
    # Transport integrity only: semantic validation and cache decisions are local.
    if checkpoint_hash(args.checkpoint) != request['metadata']['checkpoint_sha256']:
        raise ValueError('Uploaded checkpoint differs from local request')
    print(f'[embeddings/{args.device}] loading checkpoint; prepared texts={len(request["texts"]):,}', flush=True)
    model = SentenceTransformer(str(args.checkpoint), device=args.device, local_files_only=True)
    model.eval()
    try:
        from encoding_inputs import tokenization_policy, load_token_features
    except ImportError:
        from core.encoding_inputs import tokenization_policy, load_token_features
    plan = request.get('prepared_text')
    tokens = args.request.parent/'prepared_text.npz'
    if not plan or hashlib.sha256(tokens.read_bytes()).hexdigest() != plan['sha256']:
        raise ValueError('Locally prepared tokens required; missing or corrupt token archive')
    if tokenization_policy(model) != plan['tokenization']:
        raise ValueError('Worker tokenizer policy differs from local preparation')
    print(f'[embeddings/{args.device}] encoding prepared batches; truncated=0',flush=True)
    chunks = []
    with np.load(tokens,allow_pickle=False) as data, torch.no_grad():
        for n,batch in enumerate(plan['token_batches'],1):
            features = load_token_features(data,batch,args.device)
            vector = model(features)['sentence_embedding']
            chunks.append(torch.nn.functional.normalize(vector,p=2,dim=1).cpu().numpy().astype(np.float32))
            print(f'[embeddings/{args.device}] batch={n}/{len(plan["token_batches"])}',flush=True)
    vectors = np.concatenate(chunks)
    if len(vectors) != len(request['ids']):
        raise ValueError('Prepared token population differs from request IDs')
    metadata = {**request['metadata'], 'request_sha256': hashlib.sha256(raw).hexdigest(), 'embedding_dtype':'float32'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as handle:
        np.savez_compressed(handle, ids=np.asarray(request['ids'], dtype=str),
                            embeddings=vectors, metadata=json.dumps(metadata, sort_keys=True))
    args.output.with_suffix('.sha256').write_text(hashlib.sha256(args.output.read_bytes()).hexdigest())
    print(f'[embeddings/{args.device}] encoded shape={vectors.shape}', flush=True)


if __name__ == '__main__':
    main()
