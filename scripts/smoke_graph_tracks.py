"""Actual local MiniLM -> prepared graph -> both workers -> W&B offline + DVC smoke.

Uses synthetic listings, so results prove lifecycle wiring, not model quality.
No shared dataset, model checkpoint, Git state or online account is modified.

The build steps are exposed as importable functions so other harnesses (e.g.
scripts/training_profile.py) can reuse the exact same synthetic inputs/config
without re-implementing them; ``main`` keeps the original CLI behaviour.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml


def build_synthetic_inputs(root: Path, *, text_checkpoint: Path) -> dict:
    """Write the synthetic catalog/splits/pairs and build prepared text + cache.

    The layout mirrors the production suite naming (``eligible_catalog.csv``,
    ``prepared/``, ``setup_manifest.json``) so the hybrid text cache is built
    through ``training.prepare_embeddings.prepare`` and carries the provenance
    the graph preflight validates. Returns the created paths.

    ``root`` is created if missing; callers that need exclusivity create it
    first (``main`` does, preserving its original ``exist_ok=False``).
    """
    from graph_tracks.prepare import prepare
    from graph_tracks.text_cache import checkpoint_hash
    from training.prepare_embeddings import prepare as prepare_embeddings
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(text_checkpoint).resolve()
    catalog, splits, pairs = root / 'eligible_catalog.csv', root / 'listing_splits.csv', root / 'shared_pairs.csv'
    rows, assignments, labels = [], [], []
    for split in ('train', 'dev', 'test'):
        for i in range(3):
            rows.append({'sku_id': f'{split}-{i}', 'sku_name_eng': 'Lemon drink 330 ml' if i < 2 else 'Orange drink 500 ml',
                         'brand': 'Example', 'attribute': 'Flavour: Lemon; Volume: 330' if i < 2 else 'Flavour: Orange; Volume: 500',
                         'gtin': ''})
            assignments.append({'sku_id': f'{split}-{i}', 'split': split})
        for i in (1, 2):
            labels.append({'sku_id1': f'{split}-0', 'sku_id2': f'{split}-{i}',
                           'label': int(i == 1), 'split': split})
    pd.DataFrame(rows).to_csv(catalog, index=False)
    pd.DataFrame(assignments).to_csv(splits, index=False)
    pd.DataFrame(labels).to_csv(pairs, index=False)
    prepared = prepare(catalog, splits, pairs, root / 'prepared')
    (root / 'setup_manifest.json').write_text(
        json.dumps({'text_checkpoint_sha256': checkpoint_hash(checkpoint)}))
    cache = Path(prepare_embeddings(root, checkpoint, device='cpu', batch_size=64)['output'])
    return {'root': root, 'catalog': catalog, 'splits': splits, 'pairs': pairs,
            'prepared': prepared, 'cache': cache}


def build_track_config(root: Path, prepared: Path, cache: Path | None, track: str, *,
                       epochs: int = 2, hidden_dim: int = 8, output_dim: int = 8,
                       dvc: dict | None = None, postprocess: bool = True,
                       include_inputs: bool = True, retrieval_ks: tuple = (1, 2),
                       wandb: dict | None = None) -> Path:
    """Write one graph-lane YAML pointing at the prepared synthetic inputs."""
    root = Path(root).resolve()
    prepared = Path(prepared)
    cfg = {'track': track, 'listings': str(prepared), 'pairs': str(prepared.parent / 'pairs.csv'),
           'input_manifest': str(prepared.parent / 'input_manifest.json'),
           'output_dir': str(root / 'runs'), 'hidden_dim': hidden_dim, 'output_dim': output_dim,
           'epochs': epochs, 'wandb': wandb or {'project': 'e-r', 'mode': 'offline'},
           'dvc': dvc or {'enabled': False, 'remote': None, 'push': False},
           'retrieval_ks': list(retrieval_ks), 'device': 'cpu',
           'postprocess': postprocess, 'include_inputs': include_inputs}
    if track == 'hybrid':
        cfg['text_cache'] = str(cache)
    config = root / f'{track}__smoke.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return config


def train_tracks(root: Path, *, text_checkpoint: Path, tracks=('gnn_only', 'hybrid'),
                 epochs: int = 2, dvc: dict | None = None, postprocess: bool = True,
                 include_inputs: bool = True, retrieval_ks: tuple = (1, 2),
                 run_tag: str = 'smoke') -> tuple[dict, dict]:
    """Build the synthetic dataset once, then train every requested track."""
    from graph_tracks.train import train
    inputs = build_synthetic_inputs(root, text_checkpoint=text_checkpoint)
    checkpoints = {}
    for track in tracks:
        config = build_track_config(inputs['root'], inputs['prepared'], inputs['cache'], track,
                                    epochs=epochs, dvc=dvc, postprocess=postprocess,
                                    include_inputs=include_inputs, retrieval_ks=retrieval_ks)
        checkpoints[track] = train(config, run_tag=run_tag)
    return inputs, checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--text-checkpoint', type=Path,
                        default=Path('artifacts/models/all-MiniLM-L6-v2'))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs, checkpoints = train_tracks(
        root, text_checkpoint=args.text_checkpoint,
        dvc={'enabled': True, 'remote': str(root / 'local_dvc_remote'), 'push': True})
    for track, checkpoint in checkpoints.items():
        print(f'[graph-smoke] {track} complete checkpoint={checkpoint}')
    print(f'[graph-smoke] all local outputs: {root}')


if __name__ == '__main__':
    main()
