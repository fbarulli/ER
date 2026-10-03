"""Frozen local MiniLM cache using the existing model-input composition SSOT."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from graph_tracks.data import file_hash

def composition_fingerprint():
    """Fingerprint the local composition code and all shipped parser config."""
    from core.common import TRAIN_ROOT
    files = list((TRAIN_ROOT / 'src/core').glob('*.py'))
    files += [TRAIN_ROOT / 'src/graph_tracks/text_cache.py']
    files += [TRAIN_ROOT / 'config' / name for name in (
        'paths.yaml', 'training.yaml', 'identity_dimensions.yaml',
        'identity_reviews.json', 'vocabulary.json')]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(TRAIN_ROOT).as_posix().encode())
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def compose_texts(catalog: Path):
    from core.model_input import build_sku_text, model_input_info
    from core.sku_identity import row_identity
    frame = pd.read_csv(catalog, dtype=str, keep_default_na=False, low_memory=False)
    if 'sku_id' not in frame or frame.sku_id.duplicated().any() or (frame.sku_id == '').any():
        raise ValueError('catalog requires unique nonempty sku_id')
    texts = []
    started = last_progress = time.monotonic()
    print(f'[embeddings/local] composing {len(frame):,} texts on CPU', flush=True)
    for index, (_, row) in enumerate(frame.iterrows(), 1):
        texts.append(build_sku_text(row, model_input_info(row_identity(row).as_mapping())))
        if index == len(frame) or time.monotonic() - last_progress >= 10:
            print(f'[embeddings/local] texts={index:,}/{len(frame):,} elapsed={time.monotonic()-started:.1f}s', flush=True)
            last_progress = time.monotonic()
    return frame.sku_id.tolist(), texts


def texts_hash(texts):
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def checkpoint_hash(path: Path) -> str:
    if not path.is_dir():
        raise ValueError('checkpoint must be a local directory; remote revisions are not pinned here')
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError('empty checkpoint')
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        digest.update(bytes.fromhex(file_hash(file)))
    return digest.hexdigest()


def create_cache(catalog: Path, checkpoint: Path, output: Path, *, batch_size=64, device='cpu', input_metadata=None):
    started = time.monotonic()
    def progress(message):
        print(f'[embeddings] {message} elapsed={time.monotonic() - started:.1f}s', flush=True)
    progress('loading catalog and text dependencies')
    from core.model_input import model_input_composition
    from sentence_transformers import SentenceTransformer
    if output.exists():
        raise FileExistsError(output)
    source_hash = file_hash(catalog)
    implementation = composition_fingerprint()
    fingerprint = checkpoint_hash(checkpoint)
    ids, texts = compose_texts(catalog)
    progress(f'loading MiniLM device={device}')
    model = SentenceTransformer(str(checkpoint), device=device, local_files_only=True)
    model.eval()
    from core.encoding_inputs import enable_zero_truncation
    enable_zero_truncation(model)
    progress(f'encoding rows={len(texts):,} batch_size={batch_size}')
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    progress(f'encoding complete shape={vectors.shape}; writing cache')
    from core.identity_policy import POLICY_PATH
    from core.common import TRAIN_ROOT
    metadata = {'checkpoint_sha256': fingerprint,
                'identity_policy_sha256': file_hash(POLICY_PATH),
                'identity_dimensions_sha256': file_hash(TRAIN_ROOT / 'config' / 'identity_dimensions.yaml'),
                'composition': model_input_composition().model_dump(mode='json'),
                'catalog_sha256': source_hash,
                'composition_implementation_sha256': implementation,
                'text_sha256': texts_hash(texts)}
    if input_metadata is not None:
        if any(input_metadata.get(key) != value for key, value in metadata.items()):
            raise ValueError('Prepared text metadata differs from actual encoder inputs')
        metadata = {**input_metadata, **metadata}
    if source_hash != file_hash(catalog) or implementation != composition_fingerprint() or fingerprint != checkpoint_hash(checkpoint):
        raise ValueError('Embedding inputs changed during generation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        np.savez_compressed(handle, ids=np.asarray(ids, dtype=str),
                            embeddings=vectors, metadata=json.dumps(metadata, sort_keys=True))
    progress(f'cache written path={output} bytes={output.stat().st_size:,}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('catalog', 'checkpoint', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    create_cache(args.catalog, args.checkpoint, args.output, batch_size=args.batch_size, device=args.device)

if __name__ == '__main__':
    main()
