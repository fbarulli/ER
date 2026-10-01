from pathlib import Path
from types import SimpleNamespace
import zipfile
import hashlib
import json

import yaml
import pytest
import numpy as np


def test_package_ships_mutable_inputs_and_current_config(tmp_path, monkeypatch):
    import core.common
    from model_tracks import package as packaging

    setup = tmp_path / 'prepared'
    setup.mkdir()
    for track in ('gnn_only', 'hybrid'):
        (setup / f'{track}.yaml').write_text(yaml.safe_dump({'listings': 'prepared/listings.csv'}))
    (setup / 'listings.csv').write_text('product_id\na\n')
    bundle = setup / 'text_prepared.pkl.gz'
    bundle.write_bytes(b'prepared snapshot')
    bundle.with_suffix('.gz.json').write_text('{}')
    for directory in ('graph_tracks', 'model_tracks', 'training', 'core'):
        folder = tmp_path / 'src' / directory
        folder.mkdir(parents=True)
        (folder / '__init__.py').write_text('')
    (tmp_path / 'src/pipeline.py').write_text('# current pipeline\n')
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/diet_manifest.py').write_text('# current diet gate\n')
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    snapshots = {}
    for name in ('paths.yaml', 'training.yaml', 'identity_dimensions.yaml',
                 'identity_reviews.json', 'vocabulary.json'):
        snapshots[f'config/{name}'] = f'current {name}\n'.encode()
        (config_dir / name).write_bytes(snapshots[f'config/{name}'])
    inputs = {}
    for key in ('dataset_deduped', 'labeled_pairs', 'canonical_records', 'gate_results'):
        path = tmp_path / 'data' / f'{key}.csv'
        path.parent.mkdir(exist_ok=True)
        snapshots[f'data/{key}.csv'] = f'current {key}\n'.encode()
        path.write_bytes(snapshots[f'data/{key}.csv'])
        inputs[key] = path
    settings = dict(setup_dir='prepared', text_bundle='prepared/text_prepared.pkl.gz',
                    device='cpu', report_test=False)
    cfg = SimpleNamespace(**settings, model_dump=lambda: settings.copy())
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(core.common, 'F', inputs)
    monkeypatch.setattr(packaging, 'load_config', lambda _: cfg)
    monkeypatch.setattr(packaging, 'preflight', lambda _: {'status': 'ok'})
    monkeypatch.setattr(packaging.subprocess, 'run', lambda *_, **__: SimpleNamespace(stdout='revision\n'))
    output = packaging.package(Path('suite.yaml'), tmp_path / 'package.zip')
    metadata = packaging.verify(output)
    with zipfile.ZipFile(output) as archive:
        for name, content in snapshots.items():
            assert archive.read(name) == content
            assert name in metadata['files']
        assert archive.read('src/pipeline.py') == b'# current pipeline\n'
        assert archive.read('scripts/diet_manifest.py') == b'# current diet gate\n'
        for track in ('gnn_only', 'hybrid'):
            config = yaml.safe_load(archive.read(f'data/model_tracks/shared/{track}.yaml'))
            assert config['device'] == 'cpu'
            assert config['report_test'] is False
            assert config['listings'] == 'data/model_tracks/shared/listings.csv'


def test_suite_rejects_changed_labels_with_unchanged_catalog(tmp_path, monkeypatch):
    import core.common
    from model_tracks import preflight as checks

    source = tmp_path / 'catalog.csv'
    source.write_text('unchanged catalog\n')
    labels = tmp_path / 'labels.csv'
    labels.write_text('changed supervision\n')
    setup = tmp_path / 'setup'
    setup.mkdir()
    (setup / 'setup_manifest.json').write_text(json.dumps({
        'source_catalog_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'labeled_pairs_sha256': hashlib.sha256(b'previous supervision\n').hexdigest(),
    }))
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(core.common, 'F', {'dataset_deduped': source, 'labeled_pairs': labels})
    monkeypatch.setattr(checks, 'load_config', lambda _: SimpleNamespace(setup_dir='setup'))
    with pytest.raises(ValueError, match='graph setup is stale: labeled pairs'):
        checks.preflight(tmp_path / 'suite.yaml')


@pytest.mark.parametrize('changed_composition', [False, True])
def test_manifested_hybrid_requires_active_text_composition(tmp_path, monkeypatch, changed_composition):
    import core.common
    import core.model_input
    from graph_tracks import preflight as checks

    manifest = {key: 'same-hash' for key in ('listings_sha256', 'pairs_sha256',
                'identity_policy_sha256', 'identity_dimensions_sha256', 'catalog_sha256')}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    composition = core.model_input.model_input_composition().model_dump(mode='json')
    metadata = {key: manifest[key] for key in ('catalog_sha256', 'identity_policy_sha256',
                                              'identity_dimensions_sha256')}
    metadata.update(checkpoint_sha256='checkpoint',
                    composition={'outdated': True} if changed_composition else composition)
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(checks, 'file_hash', lambda _: 'same-hash')
    monkeypatch.setattr(checks, 'load_records', lambda _: [{'product_id': 'a'}])
    monkeypatch.setattr(checks, 'load_pairs', lambda *_: {})
    monkeypatch.setattr(checks, 'load_text_cache', lambda *_: (np.zeros((1, 2)), metadata))
    cfg = SimpleNamespace(input_manifest='manifest.json', allow_unmanifested_inputs=False,
                          listings='listings.csv', pairs='pairs.csv', text_cache='cache.npz',
                          text_checkpoint_sha256='checkpoint')
    if changed_composition:
        with pytest.raises(ValueError, match='text cache composition differs from active model input'):
            checks.load_inputs(cfg)
    else:
        assert checks.load_inputs(cfg)[4] == metadata
