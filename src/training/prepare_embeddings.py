"""Standalone frozen MiniLM embedding job for the hybrid training track."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

from graph_tracks.data import file_hash, load_records, load_text_cache
from graph_tracks.text_cache import checkpoint_hash, create_cache, compose_texts, composition_fingerprint, texts_hash


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import training_cfg
    return training_cfg().preparation.graph_setup


def input_identity(setup: Path, checkpoint: Path) -> dict:
    from core.common import TRAIN_ROOT
    from core.identity_policy import POLICY_PATH
    from core.model_input import model_input_composition
    layout = _setup_layout()
    manifest = json.loads((setup / layout.prepared_dir / 'input_manifest.json').read_text())
    baseline = json.loads((setup / layout.manifest).read_text())
    expected = {
        'catalog_sha256': file_hash(setup / layout.catalog),
        'identity_policy_sha256': file_hash(POLICY_PATH),
        'identity_dimensions_sha256': file_hash(TRAIN_ROOT / 'config/identity_dimensions.yaml'),
        'checkpoint_sha256': checkpoint_hash(checkpoint),
        'composition': model_input_composition().model_dump(mode='json'),
        'composition_implementation_sha256': composition_fingerprint(),
        # create_cache records the dtype it wrote, so the identity contract has to
        # name it too; otherwise the request omits the key and the cache builder
        # compares None against 'float32' and refuses its own output.
        'embedding_dtype': 'float32',
        'input_manifest_sha256': file_hash(setup / layout.prepared_dir / 'input_manifest.json'),
        'listings_sha256': file_hash(setup / layout.prepared_dir / 'listings.json'),
    }
    for key in ('catalog_sha256', 'identity_policy_sha256', 'identity_dimensions_sha256'):
        if expected[key] != manifest[key]:
            raise ValueError(f'Prepared inputs are stale: {key}')
    if manifest.get('listings_sha256') != expected['listings_sha256']:
        raise ValueError('Prepared inputs are stale: listings_sha256')
    if expected['checkpoint_sha256'] != baseline['text_checkpoint_sha256']:
        raise ValueError('Embedding checkpoint differs from the prepared hybrid baseline')
    return expected


def prepare_request(setup: Path, checkpoint: Path) -> dict:
    """Always compose actual text locally; no persistent text-cache shortcut."""
    expected = input_identity(setup, checkpoint)
    layout = _setup_layout()
    ids, texts = compose_texts(setup / layout.catalog)
    listing_ids = [row['sku_id'] for row in load_records(setup / layout.prepared_dir / 'listings.json')]
    if set(ids) != set(listing_ids) or len(ids) != len(listing_ids):
        raise ValueError('Catalog and prepared listings do not have identical IDs')
    drifted_inputs = input_identity(setup, checkpoint)
    drifted_inputs.pop('composition_implementation_sha256', None)
    for key, value in drifted_inputs.items():
        if expected.get(key) != value:
            raise ValueError('Embedding inputs changed during local composition')
    return {'schema': 'er-embedding-request-v2', 'ids': ids, 'texts': texts,
            'metadata': {**expected, 'text_sha256': texts_hash(texts)}}


def validate_result(path: Path, request: dict, *, request_sha256: str | None = None):
    import numpy as np
    with np.load(path, allow_pickle=False) as cache:
        if cache['ids'].astype(str).tolist() != request['ids']:
            raise ValueError('Embedding result ID order/population differs from request')
    vectors, metadata = load_text_cache(path, request['ids'])
    expected = dict(request['metadata'])
    if request_sha256 is not None:
        expected['request_sha256'] = request_sha256
    expected.pop('composition_implementation_sha256', None)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Embedding cache is stale: {key}')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Embedding result contains non-normalized vectors')
    return vectors.shape


def validate_prepared_provenance(cache: Path, metadata: dict, manifest: dict):
    """Consumers fail closed if prepared text provenance is missing or stale."""
    setup = cache.parent
    layout = _setup_layout()
    path = setup / layout.embedding_request
    if not path.is_file():
        raise ValueError('text cache lacks prepared text provenance; regenerate locally')
    request = json.loads(path.read_text())
    if request.get('schema') != 'er-embedding-request-v2':
        raise ValueError('legacy text cache request; regenerate locally')
    expected = request['metadata']
    if expected.get('text_sha256') != texts_hash(request['texts']):
        raise ValueError('prepared text content hash mismatch')
    if len(request['ids']) != len(request['texts']) or len(set(request['ids'])) != len(request['ids']):
        raise ValueError('invalid prepared text population')
    current = {
        'catalog_sha256': file_hash(setup / layout.catalog),
        'input_manifest_sha256': file_hash(setup / layout.prepared_dir / 'input_manifest.json'),
        'listings_sha256': file_hash(setup / layout.prepared_dir / 'listings.json'),
    }
    for key, value in current.items():
        if expected.get(key) != value:
            raise ValueError(f'text cache provenance is stale: {key}')
    for key in ('catalog_sha256', 'identity_policy_sha256', 'identity_dimensions_sha256', 'listings_sha256'):
        if expected.get(key) != manifest.get(key):
            raise ValueError(f'text cache/prepared provenance mismatch: {key}')
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'text cache provenance mismatch: {key}')
    if metadata.get('request_sha256') and metadata['request_sha256'] != file_hash(path):
        raise ValueError('text cache request checksum mismatch')
    validate_result(cache, request)


def prepare(setup: Path, checkpoint: Path, *, device='cuda', batch_size=256) -> dict:
    import torch
    from core.common import TRAIN_ROOT
    from core.identity_policy import POLICY_PATH
    from core.model_input import model_input_composition

    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: select a GPU runtime for the embedding job')
    setup, checkpoint = setup.resolve(), checkpoint.resolve()
    layout = _setup_layout()
    catalog = setup / layout.catalog
    output = setup / layout.shared_embeddings
    request = prepare_request(setup, checkpoint)
    expected = request['metadata']
    ids = request['ids']

    def validate(path):
        return validate_result(path, request)

    started = time.monotonic()
    if output.exists():
        shape = validate(output)
        status = 'reused'
    else:
        print(f'[embeddings] device={device} listings={len(ids):,} batch_size={batch_size}', flush=True)
        with tempfile.TemporaryDirectory(prefix='embedding-job-', dir=setup) as temporary:
            candidate = Path(temporary) / output.name
            create_cache(catalog, checkpoint, candidate, device=device, batch_size=batch_size, input_metadata=expected)
            shape = validate(candidate)
            drifted_inputs = input_identity(setup, checkpoint)
            drifted_inputs.pop('composition_implementation_sha256', None)
            for key, value in drifted_inputs.items():
                if expected.get(key) != value:
                    raise ValueError('Embedding inputs changed during generation')
            request_path = setup / (layout.embedding_request + '.tmp')
            request_path.write_text(json.dumps(request, ensure_ascii=False, sort_keys=True))
            request_path.replace(setup / layout.embedding_request)
            candidate.replace(output)
        status = 'created'
    result = {'status': status, 'output': str(output), 'rows': shape[0], 'dimensions': shape[1],
              'seconds': time.monotonic() - started, 'sha256': file_hash(output)}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    from core.common import resolve_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup-dir', type=Path, default=Path('data/track_setup'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args(argv)
    prepare(args.setup_dir, args.checkpoint or Path(resolve_model('minilm_l6')),
            device=args.device, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
