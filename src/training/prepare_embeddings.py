"""Standalone frozen MiniLM embedding job for the hybrid training track."""
from __future__ import annotations

import argparse
from core.portable_archive import ByteCount
import json
from pathlib import Path
import tempfile
import time

from graph_tracks.data import file_size, load_records, load_text_cache
from graph_tracks.text_cache import checkpoint_size, create_cache, compose_texts, composition_fingerprint, texts_size


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import prepared_setup_layout
    return prepared_setup_layout()


def input_identity(setup: Path, checkpoint: Path) -> dict:
    """The identity of one embedding request: the inputs it is composed from.

    NO FRESHNESS COMPARISONS (owner directive 2026-10-08, repo-wide): the
    recorded catalog/manifest sizes are carried as the request's IDENTITY, not
    re-derived to decide whether previously built data still applies. Prepared inputs
    are trusted as shipped inside the suite bundle, whose integrity is its
    size checks at the boundary. The one comparison left is a compatibility
    requirement, not a freshness test: the frozen native tokens were built for a
    specific text checkpoint, so encoding them with a different checkpoint is a
    contract violation.
    """
    from core.common import F, TRAIN_ROOT
    from core.identity_policy import POLICY_PATH
    from core.model_input import model_input_composition
    layout = _setup_layout()
    baseline = json.loads((setup / layout.manifest).read_text())
    expected = {
        'catalog_size': file_size(setup / layout.catalog),
        'identity_policy_size': file_size(POLICY_PATH),
        'identity_dimensions_size': file_size(F['identity_dimensions']),
        'checkpoint_size': checkpoint_size(checkpoint),
        'composition': model_input_composition().model_dump(mode='json'),
        'composition_implementation_size': composition_fingerprint(),
        # create_cache records the dtype it wrote, so the identity contract has to
        # name it too; otherwise the request omits the key and the cache builder
        # compares None against 'float32' and refuses its own output.
        'embedding_dtype': 'float32',
        'input_manifest_size': file_size(setup / layout.prepared_dir / layout.input_manifest),
        'listings_size': file_size(setup / layout.prepared_dir / layout.listings),
    }
    if expected['checkpoint_size'] != baseline['text_checkpoint_size']:
        raise ValueError('Embedding checkpoint differs from the prepared hybrid baseline')
    return expected


def prepare_request(setup: Path, checkpoint: Path) -> dict:
    """Always compose actual text locally; no persistent text-cache shortcut."""
    expected = input_identity(setup, checkpoint)
    layout = _setup_layout()
    ids, texts = compose_texts(setup / layout.catalog)
    listing_ids = [row['sku_id'] for row in load_records(setup / layout.prepared_dir / layout.listings)]
    if set(ids) != set(listing_ids) or len(ids) != len(listing_ids):
        raise ValueError('Catalog and prepared listings do not have identical IDs')
    drifted_inputs = input_identity(setup, checkpoint)
    drifted_inputs.pop('composition_implementation_size', None)
    for key, value in drifted_inputs.items():
        if expected.get(key) != value:
            raise ValueError('Embedding inputs changed during local composition')
    return {'schema': 'er-embedding-request-v2', 'ids': ids, 'texts': texts,
            'metadata': {**expected, 'text_size': texts_size(texts)}}


def validate_result(path: Path, request: dict):
    """Whether an embedding result belongs to the request that produced it.

    Identity/compatibility only: the ID population, the recorded request
    metadata, and vector normalization. There is NO recorded-size comparison — a
    recorded request size is never re-derived and compared (owner directive
    2026-10-08: freshness checks are removed repo-wide; a bundle's integrity is
    its size checks at the boundary).
    """
    import numpy as np
    with np.load(path, allow_pickle=False) as cache:
        if cache['ids'].astype(str).tolist() != request['ids']:
            raise ValueError('Embedding result ID order/population differs from request')
    vectors, metadata = load_text_cache(path, request['ids'])
    expected = dict(request['metadata'])
    expected.pop('composition_implementation_size', None)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Embedding result metadata differs from its request: {key}')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Embedding result contains non-normalized vectors')
    return vectors.shape


def validate_prepared_provenance(cache: Path, metadata: dict):
    """The cache's identity consistency with the request that produced it.

    ``metadata`` is the ALREADY loaded cache metadata (the dashboard reads it
    for its own report); the consistency pass itself re-reads it through
    ``validate_result``.

    NO FRESHNESS COMPARISONS (owner directive 2026-10-08, repo-wide): the
    prepared provenance is not re-derived and compared against the sizes
    recorded beside it — the bundle's identity is its size checks at the boundary,
    and a cache that does not match its own request is rebuilt, never refused.
    What remains is what makes the cache USABLE: a current request schema, the
    recorded text size matching the texts it carries, and a unique ID
    population aligned with those texts.
    """
    setup = cache.parent
    layout = _setup_layout()
    path = setup / layout.embedding_request
    if not path.is_file():
        raise ValueError('text cache lacks prepared text provenance; regenerate locally')
    request = json.loads(path.read_text())
    if request.get('schema') != 'er-embedding-request-v2':
        raise ValueError('legacy text cache request; regenerate locally')
    expected = request['metadata']
    if expected.get('text_size') != texts_size(request['texts']):
        raise ValueError('prepared text content size mismatch')
    if len(request['ids']) != len(request['texts']) or len(set(request['ids'])) != len(request['ids']):
        raise ValueError('invalid prepared text population')
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
    shape = None
    status = 'created'
    if output.exists():
        try:
            shape = validate(output)
            status = 'reused'
        except ValueError as exc:
            # An incompatible cache is REBUILT, never refused (owner directive
            # 2026-10-08): no gate on a recorded size may block a fresh build.
            print(f'[embeddings] existing cache does not match this request ({exc}); '
                  'rebuilding', flush=True)
            shape = None
    if shape is None:
        status = 'created'
        print(f'[embeddings] device={device} listings={len(ids):,} batch_size={batch_size}', flush=True)
        with tempfile.TemporaryDirectory(prefix='embedding-job-', dir=setup) as temporary:
            candidate = Path(temporary) / output.name
            create_cache(catalog, checkpoint, candidate, device=device, batch_size=batch_size, input_metadata=expected)
            shape = validate(candidate)
            drifted_inputs = input_identity(setup, checkpoint)
            drifted_inputs.pop('composition_implementation_size', None)
            for key, value in drifted_inputs.items():
                if expected.get(key) != value:
                    # A within-build race guard, not a freshness gate: the build
                    # just consumed inputs that changed under it, so publishing
                    # would record vectors for the wrong population.
                    raise ValueError('Embedding inputs changed during generation')
            request_path = setup / (layout.embedding_request + '.tmp')
            request_path.write_text(json.dumps(request, ensure_ascii=False, sort_keys=True))
            request_path.replace(setup / layout.embedding_request)
            candidate.replace(output)
    result = {'status': status, 'output': str(output), 'rows': shape[0], 'dimensions': shape[1],
              'seconds': time.monotonic() - started, 'size': file_size(output)}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    from core.common import resolve_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup-dir', type=Path, default=None,
                        help='prepared-setup root (default: the suite config setup_dir)')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args(argv)
    setup_dir = args.setup_dir
    if setup_dir is None:
        # ONE home for the prepared-setup root: the suite config's setup_dir,
        # resolved by the graph setup producer (no second hand-spelled copy).
        from graph_tracks.setup import default_setup_dir
        setup_dir = default_setup_dir()
    prepare(setup_dir, args.checkpoint or Path(resolve_model('minilm_l6')),
            device=args.device, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
