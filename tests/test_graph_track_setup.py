import pandas as pd
import pytest
from pathlib import Path
import json
import hashlib
import zipfile
import yaml

from graph_tracks.setup import listing_contract


def test_listing_split_and_negative_accounting():
    catalog = pd.DataFrame({'product_id': ['b', 'a', 'c', 'd', 'e', 'missing'],
                            'barcode': ['4006381333931', '4006381333931', '2', '3', '4', '']})
    labels = pd.DataFrame({'gtin1': ['4006381333931', '4006381333931', '3'], 'gtin2': ['2', '3', '4'],
                           'true_label': ['0', '0', '1']})
    frame, splits, pairs, counts = listing_contract(
        catalog, labels, {'train': {'4006381333931', '2'}, 'dev': {'3', '4'}, 'test': set()})
    assert set(frame.product_id) == {'a', 'b', 'c', 'd', 'e'}
    assert counts['excluded_unassigned_listings'] == 1
    assert counts['skipped_labels']['cross_split_negative'] == 1
    assert ('a', 'c', 0, 'train') in list(pairs.itertuples(index=False, name=None))
    assert ('a', 'b', 1, 'train') in list(pairs.itertuples(index=False, name=None))
    lookup = splits.set_index('product_id').split.to_dict()
    assert all(lookup[r.product_id1] == lookup[r.product_id2] == r.split
               for r in pairs.itertuples(index=False))


def test_setup_rejects_positive_split_leakage():
    catalog = pd.DataFrame({'product_id': ['a', 'b'], 'barcode': ['1', '2']})
    labels = pd.DataFrame({'gtin1': ['1'], 'gtin2': ['2'], 'true_label': ['1']})
    with pytest.raises(ValueError, match='positive label crosses'):
        listing_contract(catalog, labels, {'train': {'1'}, 'dev': {'2'}})


def test_setup_rejects_normalized_entity_split_conflict():
    catalog = pd.DataFrame({'product_id': ['a'], 'barcode': ['1']})
    with pytest.raises(ValueError, match='multiple splits'):
        listing_contract(catalog, pd.DataFrame(), {'train': {'1'}, 'dev': {'0001'}})


def test_preflight_and_portable_package_hashes(tmp_path):
    from graph_tracks.prepare import prepare
    from graph_tracks.preflight import preflight
    from graph_tracks.worker_package import package
    catalog, splits, pairs = [tmp_path / f'{s}.csv' for s in ('catalog', 'splits', 'pairs')]
    pd.DataFrame([{'product_id': f'{s}-{i}', 'title': 'Lemon drink 330 ml', 'barcode': ''}
                  for s in ('train', 'dev', 'test') for i in range(3)]).to_csv(catalog, index=False)
    pd.DataFrame([{'product_id': f'{s}-{i}', 'split': s}
                  for s in ('train', 'dev', 'test') for i in range(3)]).to_csv(splits, index=False)
    pd.DataFrame([{'product_id1': f'{s}-0', 'product_id2': f'{s}-{i}',
                   'label': int(i == 1), 'split': s}
                  for s in ('train', 'dev', 'test') for i in (1, 2)]).to_csv(pairs, index=False)
    listings = prepare(catalog, splits, pairs, tmp_path / 'prepared')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'track': 'gnn_only', 'listings': str(listings),
        'pairs': str(listings.parent / 'pairs.csv'),
        'input_manifest': str(listings.parent / 'input_manifest.json'),
        'output_dir': str(tmp_path / 'runs'), 'device': 'cpu', 'report_test': False,
        'wandb': {'mode': 'disabled'}, 'dvc': {'enabled': False}}))
    assert preflight(config)['pairs']['dev'] == {'positive': 1, 'negative': 1}
    archive_path = package(config, tmp_path / 'worker.zip')
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('data/graph_worker/gnn_only/package_manifest.json'))
        for path, expected in manifest['files_sha256'].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected
        settings = yaml.safe_load(archive.read('data/graph_worker/gnn_only/worker.yaml'))
        assert settings['device'] == 'cuda'
        assert settings['report_test'] is False
        assert not Path(settings['listings']).is_absolute()
    assert not (tmp_path / 'runs').exists()
    (listings.parent / 'pairs.csv').write_text('tampered')
    with pytest.raises(ValueError, match='pairs_sha256'):
        preflight(config)
