"""Actual local MiniLM -> prepared graph -> both workers -> W&B offline + DVC smoke.

Uses synthetic listings, so results prove lifecycle wiring, not model quality.
No shared dataset, model checkpoint, Git state or online account is modified.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml
from graph_tracks.prepare import prepare
from graph_tracks.text_cache import create_cache
from graph_tracks.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--text-checkpoint', type=Path,
                        default=Path('artifacts/models/all-MiniLM-L6-v2'))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    catalog, splits, pairs = root / 'shared_catalog.csv', root / 'shared_splits.csv', root / 'shared_pairs.csv'
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
    prepared = prepare(catalog, splits, pairs, root / 'shared_prepared')
    cache = create_cache(catalog, args.text_checkpoint.resolve(), root / 'shared_minilm__embeddings.npz')
    for track in ('gnn_only', 'hybrid'):
        cfg = {'track': track, 'listings': str(prepared), 'pairs': str(prepared.parent / 'pairs.csv'),
               'input_manifest': str(prepared.parent / 'input_manifest.json'),
               'output_dir': str(root / 'runs'), 'hidden_dim': 8, 'output_dim': 8, 'epochs': 2,
               'wandb': {'project': 'e-r', 'mode': 'offline'},
               'dvc': {'enabled': True, 'remote': str(root / 'local_dvc_remote'), 'push': True},
               'retrieval_ks': [1, 2], 'device': 'cpu'}
        if track == 'hybrid':
            cfg['text_cache'] = str(cache)
        config = root / f'{track}__smoke.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        checkpoint = train(config, run_tag='smoke')
        print(f'[graph-smoke] {track} complete checkpoint={checkpoint}')
    print(f'[graph-smoke] all local outputs: {root}')

if __name__ == '__main__':
    main()
