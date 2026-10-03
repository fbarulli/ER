"""Prepare real catalog inputs for graph tracks without starting training.

Derives the same component split as text training. Labeled entity pairs use
the lexically first listing per entity; same-entity listings form positive
chains. Cross-split negatives are excluded and counted, never relabeled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from graph_tracks.data import census, file_hash, fit_vocabulary, load_records
from graph_tracks.prepare import prepare
from graph_tracks.text_cache import checkpoint_hash
from graph_tracks.train import write_json


def listing_contract(catalog, labels, populations):
    from training.folds import normalize_gtin
    roles = {}
    for split, values in populations.items():
        for value in values:
            key = normalize_gtin(value)
            if key in roles and roles[key] != split:
                raise ValueError('entity assigned to multiple splits')
            roles[key] = split
    frame = catalog.fillna('').copy()
    if frame.sku_id.duplicated().any() or (frame.sku_id == '').any():
        raise ValueError('catalog requires unique nonempty sku_id')
    keys = frame.gtin.map(normalize_gtin)
    retained = keys.isin(roles) & keys.ne('')
    excluded = int((~retained).sum())
    frame = frame.loc[retained].sort_values('sku_id').reset_index(drop=True)
    keys = frame.gtin.map(normalize_gtin)
    assignments = pd.DataFrame({'sku_id': frame.sku_id, 'split': keys.map(roles)})
    groups = {}
    for key, listing in zip(keys, frame.sku_id):
        groups.setdefault(key, []).append(listing)
    pair_map, skipped = {}, Counter()

    def add(a, b, label, split):
        key = tuple(sorted((a, b)))
        if a == b:
            if label == 0:
                raise ValueError('negative label within one normalized entity')
            skipped['self_positive'] += 1
            return
        value = (int(label), split)
        if key in pair_map and pair_map[key] != value:
            raise ValueError('conflicting listing-pair supervision')
        pair_map[key] = value

    from core.gtin import gtin_validity
    trusted = frame.loc[gtin_validity(frame.gtin)].sku_id
    trusted_ids = set(trusted)
    for key, listings in groups.items():
        eligible = [listing for listing in listings if listing in trusted_ids]
        skipped['untrusted_identity_chain_listings'] += len(listings) - len(eligible)
        for a, b in zip(eligible, eligible[1:]):
            add(a, b, 1, roles[key])
    for row in labels.itertuples(index=False):
        a, b = normalize_gtin(row.gtin1), normalize_gtin(row.gtin2)
        label = int(row.true_label)
        if label not in (0, 1):
            raise ValueError('invalid entity label')
        if a not in groups or b not in groups:
            skipped['missing_listing_endpoint'] += 1
            continue
        if roles[a] != roles[b]:
            if label:
                raise ValueError('positive label crosses shared split')
            skipped['cross_split_negative'] += 1
            continue
        add(groups[a][0], groups[b][0], label, roles[a])
    pairs = pd.DataFrame([
        {'sku_id1': a, 'sku_id2': b, 'label': label, 'split': split}
        for (a, b), (label, split) in sorted(pair_map.items())
    ], columns=['sku_id1', 'sku_id2', 'label', 'split'])
    return frame, assignments, pairs, {'excluded_unassigned_listings': excluded,
                                      'skipped_labels': dict(skipped)}


def setup(output: Path, checkpoint: Path) -> Path:
    from core.common import F, SEED, TRAIN_ROOT, load_dataset_deduped, training_cfg
    from core.identity_policy import POLICY_PATH
    from training.base_data import load_base_data
    from training.folds import derive_holdout
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = checkpoint.resolve()
    baseline_hash = checkpoint_hash(checkpoint)
    catalog = load_dataset_deduped().fillna('')
    data = load_base_data(catalog, payload_variant='full')
    train, dev, test = derive_holdout(data['pos'], data['row_bc'],
                                    dict(training_cfg().split), seed=SEED)
    labels = pd.read_csv(F['labeled_pairs'], dtype=str, keep_default_na=False)
    frame, assignments, pairs, accounting = listing_contract(
        catalog, labels, {'train': train, 'dev': dev, 'test': test})
    # Validate before publishing any setup artifacts.
    from graph_tracks.train import load_pairs
    load_pairs_from = [{'sku_id': r.sku_id, 'split': r.split}
                       for r in assignments.itertuples(index=False)]
    output.mkdir(parents=True)
    frame.to_csv(output / 'eligible_catalog.csv', index=False)
    assignments.to_csv(output / 'listing_splits.csv', index=False)
    pairs.to_csv(output / 'listing_pairs.csv', index=False)
    load_pairs(output / 'listing_pairs.csv', load_pairs_from)
    listings = prepare(output / 'eligible_catalog.csv', output / 'listing_splits.csv',
                       output / 'listing_pairs.csv', output / 'prepared')
    records = load_records(listings)
    write_json(output / 'graph_census.json', census(records, fit_vocabulary(records)))
    import subprocess
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=TRAIN_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    write_json(output / 'setup_manifest.json', {
        'schema': 'er-track-setup-v1', 'git_revision': revision, 'seed': SEED,
        'source_catalog_sha256': file_hash(F['dataset_deduped']),
        'labeled_pairs_sha256': file_hash(F['labeled_pairs']),
        'identity_policy_sha256': file_hash(POLICY_PATH),
        'text_checkpoint': str(checkpoint), 'text_checkpoint_sha256': baseline_hash,
        'text_checkpoint_status': 'local baseline; fine-tuning history not inferred',
        'split_protocol': 'training.folds.derive_holdout',
        'pair_protocol': 'first listing per labeled entity plus same-entity positive chains',
        'negative_policy': 'same-split labeled negatives only',
        'pair_counts': {split: {str(label): int(count) for label, count in group.label.value_counts().items()}
                        for split, group in pairs.groupby('split')},
        **accounting,
    })
    for track, template in [('gnn_only', 'gnn'), ('hybrid', 'hybrid')]:
        cfg = yaml.safe_load((TRAIN_ROOT / f'config/graph_tracks_{template}.yaml').read_text())
        cfg.update(listings=str(listings), pairs=str(listings.parent / 'pairs.csv'),
                   input_manifest=str(listings.parent / 'input_manifest.json'), report_test=False)
        if track == 'hybrid':
            cfg['text_cache'] = str(output / 'shared_minilm__embeddings.npz')
            cfg['text_checkpoint_sha256'] = baseline_hash
        (output / f'{track}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/track_setup'))
    parser.add_argument('--text-checkpoint', type=Path,
                        default=Path('artifacts/models/all-MiniLM-L6-v2'))
    args = parser.parse_args()
    print(setup(args.output, args.text_checkpoint))


if __name__ == '__main__':
    main()
