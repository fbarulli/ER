"""Standalone frozen MiniLM embedding job for the hybrid training track."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from graph_tracks.data import file_hash, load_records, load_text_cache
from graph_tracks.text_cache import checkpoint_hash, create_cache


def prepare(setup: Path, checkpoint: Path, *, device='cuda', batch_size=64) -> dict:
    import torch
    from core.common import TRAIN_ROOT
    from core.identity_policy import POLICY_PATH
    from core.model_input import model_input_composition

    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: select a GPU runtime for the embedding job')
    setup, checkpoint = setup.resolve(), checkpoint.resolve()
    catalog = setup / 'eligible_catalog.csv'
    output = setup / 'shared_minilm__embeddings.npz'
    manifest = json.loads((setup / 'prepared/input_manifest.json').read_text())
    baseline = json.loads((setup / 'setup_manifest.json').read_text())
    expected = {
        'catalog_sha256': file_hash(catalog),
        'identity_policy_sha256': file_hash(POLICY_PATH),
        'identity_dimensions_sha256': file_hash(TRAIN_ROOT / 'config/identity_dimensions.yaml'),
        'checkpoint_sha256': checkpoint_hash(checkpoint),
        'composition': model_input_composition().model_dump(mode='json'),
    }
    for key in ('catalog_sha256', 'identity_policy_sha256', 'identity_dimensions_sha256'):
        if expected[key] != manifest[key]:
            raise ValueError(f'Prepared inputs are stale: {key}')
    if expected['checkpoint_sha256'] != baseline['text_checkpoint_sha256']:
        raise ValueError('Embedding checkpoint differs from the prepared hybrid baseline')
    ids = [row['product_id'] for row in load_records(setup / 'prepared/listings.json')]

    def validate(path):
        vectors, metadata = load_text_cache(path, ids)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f'Embedding cache is stale: {key}')
        return vectors.shape

    started = time.monotonic()
    if output.exists():
        shape = validate(output)
        status = 'reused'
    else:
        print(f'[embeddings] device={device} listings={len(ids):,} batch_size={batch_size}', flush=True)
        with tempfile.TemporaryDirectory(prefix='embedding-job-', dir=setup) as temporary:
            candidate = Path(temporary) / output.name
            create_cache(catalog, checkpoint, candidate, device=device, batch_size=batch_size)
            shape = validate(candidate)
            if file_hash(catalog) != expected['catalog_sha256'] or checkpoint_hash(checkpoint) != expected['checkpoint_sha256']:
                raise ValueError('Embedding inputs changed during generation')
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
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args(argv)
    prepare(args.setup_dir, args.checkpoint or Path(resolve_model('minilm_l6')),
            device=args.device, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
