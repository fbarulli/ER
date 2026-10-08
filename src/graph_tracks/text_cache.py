"""Frozen local MiniLM cache using the existing model-input composition SSOT."""
from __future__ import annotations
import argparse
from core.portable_archive import ByteCount
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from graph_tracks.data import file_size

_CONTENT_SIZE_MEMO: dict[tuple, str] = {}

def composition_fingerprint():
    """Fingerprint the local composition code and all shipped parser config.

    Same stat-signature memo as checkpoint_size (mtime+size identify
    content). The fingerprinted files are editable source/config, so any
    write to them changes mtime_ns and forces a re-measure; only repeated
    READS within a process reuse the size.
    """
    from core.common import TRAIN_ROOT, training_cfg
    files = list((TRAIN_ROOT / 'src/core').rglob('*.py'))
    files += list((TRAIN_ROOT / 'src/ner').rglob('*.py'))
    files += [TRAIN_ROOT / 'src/graph_tracks/text_cache.py', TRAIN_ROOT / 'src/pipeline.py']
    # The pinned config set is declared ONCE (config/training.yaml packaging
    # block, the same list the runtime snapshot ships); the cache key lists
    # exactly those files, so a new pinned config invalidates the cache too.
    files += [TRAIN_ROOT / 'config' / name
              for name in training_cfg().packaging.snapshot_pinned_configs]
    files.sort()
    tracked = []
    for path in files:
        info = path.stat()
        tracked.append((path.relative_to(TRAIN_ROOT).as_posix(), info.st_mtime_ns, info.st_size))
    signature = ('composition_fingerprint', tuple(tracked))
    if len(_CONTENT_SIZE_MEMO) > 64:
        _CONTENT_SIZE_MEMO.clear()
    memoized = _CONTENT_SIZE_MEMO.get(signature)
    if memoized is not None:
        return memoized
    size = ByteCount()
    for path in files:
        size.update(path.relative_to(TRAIN_ROOT).as_posix().encode())
        size.update(str(file_size(path)).encode())
    _CONTENT_SIZE_MEMO[signature] = size.total
    return _CONTENT_SIZE_MEMO[signature]


def compose_texts(catalog: Path, *, composer=None):
    from core.model_input import build_sku_text, model_input_info
    from core.sku_identity import row_identity
    from model_tracks.training_data import frozen_endpoint_text
    frame = pd.read_csv(catalog, dtype=str, keep_default_na=False, low_memory=False)
    if 'sku_id' not in frame or frame.sku_id.duplicated().any() or (frame.sku_id == '').any():
        raise ValueError('catalog requires unique nonempty sku_id')
    texts = []
    started = last_progress = time.monotonic()
    print(f'[embeddings/local] composing {len(frame):,} texts on CPU', flush=True)
    for index, (_, row) in enumerate(frame.iterrows(), 1):
        # Virtualness decides, not cell emptiness: the catalog writes one
        # shared column, so every listing also carries an empty cell.
        frozen = frozen_endpoint_text(row.sku_id, row.get('frozen_payload'),
                                      column_present='frozen_payload' in row.index)
        texts.append(frozen if frozen is not None else composer(row.to_dict()) if composer is not None else
                     build_sku_text(row, model_input_info(row_identity(row).as_mapping())))
        if index == len(frame) or time.monotonic() - last_progress >= 10:
            print(f'[embeddings/local] texts={index:,}/{len(frame):,} elapsed={time.monotonic()-started:.1f}s', flush=True)
            last_progress = time.monotonic()
    return frame.sku_id.tolist(), texts


def texts_size(texts):
    return ByteCount(json.dumps(texts, ensure_ascii=False).encode()).total


def checkpoint_size(path: Path, *, use_memo: bool = True) -> str:
    if not path.is_dir():
        raise ValueError('checkpoint must be a local directory; remote revisions are not pinned here')
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError('empty checkpoint')
    # stat-signature caching assumes checkpoint files are immutable once
    # published (mtime+size identify content). Provenance gates pass
    # use_memo=False: a re-measure must never be a stat comparison.
    tracked = []
    for file in files:
        info = file.stat()
        tracked.append((file.relative_to(path).as_posix(), info.st_mtime_ns, info.st_size))
    signature = ('checkpoint_size', str(path.resolve()), tuple(tracked))
    if len(_CONTENT_SIZE_MEMO) > 64:
        _CONTENT_SIZE_MEMO.clear()
    memoized = _CONTENT_SIZE_MEMO.get(signature) if use_memo else None
    if memoized is not None:
        return memoized
    size = ByteCount()
    for file in files:
        size.update(str(file.relative_to(path)).encode())
        size.update(str(file_size(file)).encode())
    _CONTENT_SIZE_MEMO[signature] = size.total
    return _CONTENT_SIZE_MEMO[signature]


def create_cache(catalog: Path, checkpoint: Path, output: Path, *, batch_size=64, device='cpu', input_metadata=None):
    started = time.monotonic()
    def progress(message):
        print(f'[embeddings] {message} elapsed={time.monotonic() - started:.1f}s', flush=True)
    progress('loading catalog and text dependencies')
    from core.model_input import model_input_composition
    from sentence_transformers import SentenceTransformer
    if output.exists():
        raise FileExistsError(output)
    source_size = file_size(catalog)
    implementation = composition_fingerprint()
    checkpoint_size_value = checkpoint_size(checkpoint)
    ids, texts = compose_texts(catalog)
    progress(f'loading MiniLM device={device}')
    model = SentenceTransformer(str(checkpoint), device=device, local_files_only=True)
    model.eval()
    from core.encoding_inputs import enable_zero_truncation
    enable_zero_truncation(model)
    progress(f'encoding rows={len(texts):,} batch_size={batch_size}')
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    progress(f'encoding complete shape={vectors.shape}; writing cache')
    from core.identity_policy import POLICY_PATH
    from core.common import F
    metadata = {'checkpoint_size': checkpoint_size_value,
                'embedding_dtype': 'float32',
                'identity_policy_size': file_size(POLICY_PATH),
                'identity_dimensions_size': file_size(F['identity_dimensions']),
                'composition': model_input_composition().model_dump(mode='json'),
                'catalog_size': source_size,
                'composition_implementation_size': implementation,
                'text_size': texts_size(texts)}
    if input_metadata is not None:
        if any(input_metadata.get(key) != value for key, value in metadata.items()):
            raise ValueError('Prepared text metadata differs from actual encoder inputs')
        metadata = {**input_metadata, **metadata}
    if source_size != file_size(catalog) or checkpoint_size_value != checkpoint_size(checkpoint):
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
