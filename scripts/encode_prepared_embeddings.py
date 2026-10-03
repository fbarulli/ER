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
    args = parser.parse_args()
    if not torch.cuda.is_available():
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
    print(f'[embeddings/gpu] loading checkpoint; prepared texts={len(request["texts"]):,}', flush=True)
    model = SentenceTransformer(str(args.checkpoint), device='cuda', local_files_only=True)
    model.eval()
    print(f'[embeddings/gpu] encoding batch_size={args.batch_size}', flush=True)
    vectors = model.encode(request['texts'], batch_size=args.batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    metadata = {**request['metadata'], 'request_sha256': hashlib.sha256(raw).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as handle:
        np.savez_compressed(handle, ids=np.asarray(request['ids'], dtype=str),
                            embeddings=vectors, metadata=json.dumps(metadata, sort_keys=True))
    args.output.with_suffix('.sha256').write_text(hashlib.sha256(args.output.read_bytes()).hexdigest())
    print(f'[embeddings/gpu] encoded shape={vectors.shape}', flush=True)


if __name__ == '__main__':
    main()
