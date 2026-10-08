"""The standalone bundlers round-trip through the shared ``Bundle`` type.

Each bundler is a *transport* around one sealed archive: it must hand back an
archive that ``core.bundle.Bundle`` can load (one boundary integrity check) and
read through its accessors. NER model tars have no Bundle-compatible role
manifest, so that gap is pinned here rather than silently bypassed.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from core.bundle import Bundle, BundleRole


def _graph_result_source(root):
    source = root / 'run'
    source.mkdir()
    (source / 'gnn_only__run_manifest.json').write_text(
        json.dumps({'schema': 'er-graph-run-v1', 'track': 'gnn_only',
                    'run_tag': 'bundle'}))
    (source / 'gnn_only__report.json').write_text('{}')
    return source


def test_graph_result_bundle_round_trips_as_result(tmp_path):
    from graph_tracks.bundle import bundle

    archive = bundle(_graph_result_source(tmp_path), tmp_path / 'gnn_only__bundle.zip')
    handle = Bundle.load(archive, BundleRole.result,
                         manifest_name='gnn_only__bundle_manifest.json')

    assert handle.role is BundleRole.result
    assert handle.run_tag() == 'bundle'
    assert 'gnn_only__report.json' in handle.members()


def _graph_worker_config(tmp_path):
    from graph_tracks.prepare import prepare

    catalog, splits, pairs = [tmp_path / f'{s}.csv' for s in ('catalog', 'splits', 'pairs')]
    pd.DataFrame([{'sku_id': f'{s}-{i}', 'sku_name_eng': 'Lemon drink 330 ml', 'gtin': ''}
                  for s in ('train', 'dev', 'test') for i in range(3)]).to_csv(catalog, index=False)
    pd.DataFrame([{'sku_id': f'{s}-{i}', 'split': s}
                  for s in ('train', 'dev', 'test') for i in range(3)]).to_csv(splits, index=False)
    pd.DataFrame([{'sku_id1': f'{s}-0', 'sku_id2': f'{s}-{i}',
                   'label': int(i == 1), 'split': s}
                  for s in ('train', 'dev', 'test') for i in (1, 2)]).to_csv(pairs, index=False)
    listings = prepare(catalog, splits, pairs, tmp_path / 'prepared')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({
        'track': 'gnn_only', 'listings': str(listings),
        'pairs': str(listings.parent / 'pairs.csv'),
        'input_manifest': str(listings.parent / 'input_manifest.json'),
        'output_dir': str(tmp_path / 'runs'), 'device': 'cpu', 'report_test': False,
        'wandb': {'mode': 'disabled'}, 'dvc': {'enabled': False}}))
    return config


def test_worker_package_round_trips_as_inputs(tmp_path):
    from graph_tracks.worker_package import package

    archive = package(_graph_worker_config(tmp_path), tmp_path / 'worker.zip')
    manifest_name = 'data/graph_worker/gnn_only/package_manifest.json'
    handle = Bundle.load(archive, BundleRole.inputs, manifest_name=manifest_name)

    assert handle.role is BundleRole.inputs
    assert handle.run_tag()
    assert any(member.endswith('worker.yaml') for member in handle.members())


def test_ner_model_tar_has_no_bundle_role_manifest(tmp_path):
    """Reported gap: BundleRole has no model/weights fit for NER model tars.

    The base-model and final-model archives are plain tars with no per-role
    manifest/inventory, so ``Bundle`` cannot consume them without adding a
    member (changing the member selection) or a role (core/schemas.py).
    """
    from core.archive_reader import tar_archive

    model = tmp_path / 'ner_model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    archive = tmp_path / 'ner_model.tar.zst'
    with tar_archive(archive, 'x') as tar:
        tar.add(model / 'config.json', arcname='ner_model/config.json')

    with pytest.raises(ValueError, match='manifest missing'):
        Bundle.load(archive, BundleRole.result,
                    manifest_name='ner_artifacts_manifest.json')
