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
        (setup / f'{track}.yaml').write_text(yaml.safe_dump({
            'track': track, 'listings': 'prepared/listings.csv',
            'pairs': 'prepared/pairs.csv', 'output_dir': 'results/graph_tracks',
            **({'text_cache': 'prepared/shared_minilm__embeddings.npz'} if track == 'hybrid' else {})}))
    (setup / 'pairs.csv').write_text('sku_id1,sku_id2,label,split\n')
    (setup / 'text.yaml').write_text(yaml.safe_dump({'track': 'text', 'output_dir': 'results/model_tracks'}))
    (setup / 'listings.csv').write_text('sku_id\na\n')
    (setup/'shared_minilm__embeddings.npz').write_bytes(b'preflight validates this mocked cache')
    bundle = setup / 'text_prepared.pkl.gz'
    bundle.write_bytes(b'prepared snapshot')
    bundle.with_suffix('.gz.json').write_text('{}')
    for directory in ('graph_tracks', 'model_tracks', 'training', 'core', 'cli'):
        folder = tmp_path / 'src' / directory
        folder.mkdir(parents=True)
        (folder / '__init__.py').write_text('')
    (tmp_path / 'src/pipeline.py').write_text('# current pipeline\n')
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/diet_manifest.py').write_text('# current diet gate\n')
    for name in ('run_colab_ablation.py','run_colab_embeddings.py'):
        (tmp_path/'scripts'/name).write_text('# publication source\n')
    (tmp_path/'src/cli/colab.py').write_text('# shared launcher\n')
    # The semantic family registry is a required packaged input; the suite
    # checkout excludes results/, so packaging must carry it.
    semantics = tmp_path / 'results' / 'semantics'
    semantics.mkdir(parents=True)
    (semantics / 'family_registry.json').write_text('{"schema": "er-family-registry-v1"}\n')
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    snapshots = {}
    for name in ('paths.yaml', 'training.yaml', 'identity_dimensions.yaml',
                 'identity_reviews.json', 'vocabulary.json', 'text_track.yaml', 'attribute_ablation.yaml'):
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
                    device='cpu', report_test=False,text_model='minilm_l6',post_training_ablation=False,ablation_config='config/attribute_ablation.yaml')
    cfg = SimpleNamespace(**settings, model_dump=lambda: settings.copy(),
                          graph_execution_overrides=lambda: {})
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(core.common, 'F', inputs)
    monkeypatch.setattr(packaging, 'load_config', lambda _: cfg)
    monkeypatch.setattr(packaging, 'preflight', lambda _,**kwargs: {'status': 'ok'})
    monkeypatch.setattr(core.common,'resolve_model',lambda _:str(tmp_path/'baseline'))
    from graph_tracks import prepared_inputs
    from model_tracks import text_export
    from graph_tracks import text_cache
    monkeypatch.setattr(text_cache,'composition_fingerprint',lambda:'frozen implementation')
    preparations = []
    monkeypatch.setattr(prepared_inputs,'prepare_training',lambda *args,**kwargs:preparations.append('graph'))
    def prepare_text(*args,**kwargs):
        kwargs['token_cache'][('model','native')] = object()
        preparations.append('text')
    monkeypatch.setattr(text_export,'prepare',prepare_text)
    from model_tracks import baseline_export
    monkeypatch.setattr(baseline_export,'prepare',lambda *args,**kwargs:preparations.append('baseline'))
    monkeypatch.setattr(packaging.subprocess, 'run', lambda *_, **__: SimpleNamespace(stdout='revision\n'))
    # The shared supervision population is built by the real bundle loader; this
    # case covers packaging, so stub the population and keep its artifacts.
    from training import prepared_bundle as prepared_bundle_mod
    from model_tracks import training_data as shared_training_data_mod
    from model_tracks import shared_graph_data as shared_graph_data_mod
    shared_stub = SimpleNamespace(fingerprint='a' * 64, examples=[], endpoints=[],
                                  model_dump_json=lambda **kwargs: '{"schema_version": 1}\n')
    monkeypatch.setattr(prepared_bundle_mod, 'load_prepared_bundle', lambda _: (SimpleNamespace(), {}))
    monkeypatch.setattr(shared_training_data_mod, 'from_bundle', lambda *a, **k: shared_stub)
    monkeypatch.setattr(shared_graph_data_mod, 'prepare_shared_graph', lambda *a, **k: {})
    output = packaging.package(Path('suite.yaml'), tmp_path / 'package.zip')
    assert preparations == ['graph','text','baseline']
    metadata = packaging.verify(output)
    with zipfile.ZipFile(output) as archive:
        for name, content in snapshots.items():
            assert archive.read(name) == content
            assert name in metadata['files']
        assert archive.read('src/pipeline.py') == b'# current pipeline\n'
        assert archive.read('scripts/diet_manifest.py') == b'# current diet gate\n'
        text = yaml.safe_load(archive.read('data/model_tracks/shared/text.yaml'))
        assert text['report_test'] is False
        assert text['retrieval_ks'] == list(core.common.ann_retrieval_ks())
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
    import training.prepare_embeddings
    monkeypatch.setattr(training.prepare_embeddings, 'validate_prepared_provenance', lambda *_: None)
    metadata.update(checkpoint_sha256='checkpoint',
                    composition={'outdated': True} if changed_composition else composition)
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(checks, 'file_hash', lambda _: 'same-hash')
    monkeypatch.setattr(checks, 'load_records', lambda _: [{'sku_id': 'a'}])
    monkeypatch.setattr(checks, 'load_pairs', lambda *_: {})
    monkeypatch.setattr(checks, 'load_text_cache', lambda *_: (np.zeros((1, 2)), metadata))
    cfg = SimpleNamespace(device='cpu',input_manifest='manifest.json', allow_unmanifested_inputs=False,
                          listings='listings.csv', pairs='pairs.csv', text_cache='cache.npz',
                          text_checkpoint_sha256='checkpoint')
    if changed_composition:
        with pytest.raises(ValueError, match='text cache composition differs from active model input'):
            checks.load_inputs(cfg)
    else:
        assert checks.load_inputs(cfg)[4] == metadata
