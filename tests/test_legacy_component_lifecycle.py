from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import types

import pandas as pd
import pytest
from cli import colab


def fixture_setup(tmp_path, *, overlapping=False):
    setup = tmp_path / 'setup'
    (setup / 'prepared').mkdir(parents=True)
    catalog = pd.DataFrame({'sku_id': ['a', 'b', 'c'],
                            'gtin': ['111', '111' if overlapping else '222', '333']})
    catalog.to_csv(setup / 'eligible_catalog.csv', index=False)
    pd.DataFrame({'sku_id': ['a', 'b', 'c'],
                  'split': ['train', 'dev', 'test']}).to_csv(setup / 'listing_splits.csv', index=False)
    digest = hashlib.sha256((setup / 'eligible_catalog.csv').read_bytes()).hexdigest()
    (setup / 'prepared/input_manifest.json').write_text(json.dumps({'catalog_sha256': digest}))
    return setup


def test_materialized_holdout_contains_only_shared_dev_test(tmp_path):
    fixture_setup(tmp_path)
    with patch.object(colab, 'TRAIN_ROOT', tmp_path), \
         patch('model_tracks.config.load_config', return_value=types.SimpleNamespace(setup_dir='setup')), \
         patch('model_tracks.preflight.preflight', return_value={}), \
         patch('core.identity_policy.reviewed_row_mask', side_effect=lambda df: pd.Series(False, index=df.index)):
        sources = colab._legacy_validation_sources()
    assert set(pd.read_csv(sources['training']).sku_id) == {'a'}
    assert set(pd.read_csv(sources['sample']).sku_id) == {'b', 'c'}
    assert set(pd.read_csv(sources['source']).sku_id) == {'a', 'b', 'c'}


def test_component_entity_overlap_fails_before_materialization(tmp_path):
    fixture_setup(tmp_path, overlapping=True)
    with patch.object(colab, 'TRAIN_ROOT', tmp_path), \
         patch('model_tracks.config.load_config', return_value=types.SimpleNamespace(setup_dir='setup')), \
         patch('model_tracks.preflight.preflight', return_value={}):
        with pytest.raises(ValueError, match='train entities overlap'):
            colab._legacy_validation_sources()
    assert not (tmp_path / 'results').exists()


def test_legacy_sample_preparation_rejected_before_building():
    with pytest.raises(ValueError, match='--tracks-config'):
        colab._build_local_training_bundles(profiles=['baseline'], model=None, sample=128)


def test_actual_prepared_bundle_must_match_shared_listing_roles(tmp_path):
    fixture_setup(tmp_path)
    with patch.object(colab, 'TRAIN_ROOT', tmp_path), \
         patch('model_tracks.config.load_config', return_value=types.SimpleNamespace(setup_dir='setup')), \
         patch('training.prepared_bundle.load_prepared_bundle', return_value=(None, {})), \
         patch('training.prepared_bundle.prepared_holdout', return_value=({'222'}, {'111'}, {'333'})):
        with pytest.raises(ValueError, match='differs from the shared component split'):
            colab._validate_legacy_bundle_partitions([Path('stale.pkl.gz')])


def test_parallel_materializers_use_independent_temporary_files(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    fixture_setup(tmp_path)
    with patch.object(colab, 'TRAIN_ROOT', tmp_path), \
         patch('model_tracks.config.load_config', return_value=types.SimpleNamespace(setup_dir='setup')), \
         patch('model_tracks.preflight.preflight', return_value={}), \
         patch('core.identity_policy.reviewed_row_mask', side_effect=lambda df: pd.Series(False, index=df.index)):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: colab._legacy_validation_sources(), range(2)))
    assert results[0] == results[1]
    assert set(pd.read_csv(results[0]['sample']).sku_id) == {'b', 'c'}
    assert not list((tmp_path / 'results').rglob('*.tmp'))
