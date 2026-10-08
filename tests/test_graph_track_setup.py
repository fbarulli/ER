import pandas as pd
import pytest
from pathlib import Path
import json
import hashlib
import zipfile
import yaml

from graph_tracks.setup import listing_contract


def test_listing_split_and_negative_accounting():
    catalog = pd.DataFrame({'sku_id': ['b', 'a', 'c', 'd', 'e', 'missing'],
                            'gtin': ['4006381333931', '4006381333931', '2', '3', '4', '']})
    labels = pd.DataFrame({'gtin1': ['4006381333931', '4006381333931', '3'], 'gtin2': ['2', '3', '4'],
                           'true_label': ['0', '0', '1']})
    frame, splits, pairs, counts = listing_contract(
        catalog, labels, {'train': {'4006381333931', '2'}, 'dev': {'3', '4'}, 'test': set()})
    assert set(frame.sku_id) == {'a', 'b', 'c', 'd', 'e'}
    assert counts['excluded_unassigned_listings'] == 1
    assert counts['skipped_labels']['cross_split_negative'] == 1
    assert ('a', 'c', 0, 'train') in list(pairs.itertuples(index=False, name=None))
    assert ('a', 'b', 1, 'train') in list(pairs.itertuples(index=False, name=None))
    lookup = splits.set_index('sku_id').split.to_dict()
    assert all(lookup[r.sku_id1] == lookup[r.sku_id2] == r.split
               for r in pairs.itertuples(index=False))


def test_setup_rejects_positive_split_leakage():
    catalog = pd.DataFrame({'sku_id': ['a', 'b'], 'gtin': ['1', '2']})
    labels = pd.DataFrame({'gtin1': ['1'], 'gtin2': ['2'], 'true_label': ['1']})
    with pytest.raises(ValueError, match='positive label crosses'):
        listing_contract(catalog, labels, {'train': {'1'}, 'dev': {'2'}})


def test_setup_rejects_normalized_entity_split_conflict():
    catalog = pd.DataFrame({'sku_id': ['a'], 'gtin': ['1']})
    with pytest.raises(ValueError, match='multiple splits'):
        listing_contract(catalog, pd.DataFrame(), {'train': {'1'}, 'dev': {'0001'}})


def test_listing_pairs_preserve_optional_trace_axes_in_lineage():
    catalog = pd.DataFrame({'sku_id':['a','b'],'gtin':['1','2']})
    labels = pd.DataFrame({'gtin1':['1'],'gtin2':['2'],'true_label':['0'],
                           'difficulty':['hard'],'gate evidence':['declared flavor conflict']})
    _,_,pairs,accounting = listing_contract(catalog,labels,{'train':{'1','2'}})
    assert list(pairs.columns) == ['sku_id1','sku_id2','label','split']
    origin = accounting['pair_lineage'][0]['origins'][0]
    assert origin['source_row'] == 1
    assert origin['metadata'] == {'difficulty':'hard','gate evidence':'declared flavor conflict'}
    assert 'difficulty' not in accounting['missing_axes']
    assert accounting['augmentation'].startswith('not_applicable')


def test_preflight_and_portable_package_hashes(tmp_path):
    from graph_tracks.prepare import prepare
    from graph_tracks.preflight import preflight
    from graph_tracks.worker_package import package
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


# ── ONE writer per setup artifact, ONE home per layout name ────────────────
# The audit (2026-10-08) found the prepared-setup layout names re-spelled as
# literals across the lanes, and the catalog/split/pair CSVs + the text lane
# config written by BOTH the producer (graph_tracks.setup) and two rebuilders
# (model_tracks.shared_graph_data, model_tracks.smoke_inputs). These pins hold
# the consolidation: the names come from the spec, the writes come from one
# function, and the resolved names are byte-identical to the literals they
# replaced (so no emitted artifact moved).


def test_prepared_setup_layout_names_are_the_declared_spec():
    from core.common import training_cfg

    layout = training_cfg().preparation.graph_setup
    assert (layout.catalog, layout.splits, layout.pairs, layout.pair_lineage,
            layout.manifest, layout.census, layout.text_config,
            layout.embedding_request, layout.shared_embeddings,
            layout.shared_training_data, layout.shared_training_projection,
            layout.text_training_binding, layout.text_export_request,
            layout.prepared_dir) == (
        'eligible_catalog.csv', 'listing_splits.csv', 'listing_pairs.csv',
        'pair_lineage.json', 'setup_manifest.json', 'graph_census.json',
        'text.yaml', 'embedding_inputs.json', 'shared_minilm__embeddings.npz',
        'shared_training_data.json', 'shared_training_projection.json',
        'text_training_binding.json', 'text_export_request.json', 'prepared')
    assert layout.track_config('gnn_only') == 'gnn_only.yaml'
    assert layout.track_config('cascade') == 'cascade.yaml'


def test_write_setup_frames_matches_the_inline_csv_writer(tmp_path):
    """The one writer emits the exact bytes the producer's to_csv calls did."""
    from graph_tracks.setup import write_setup_frames

    catalog = pd.DataFrame({'sku_id': ['a', 'b'], 'gtin': ['1', '2']})
    splits = pd.DataFrame({'sku_id': ['a', 'b'], 'split': ['train', 'dev']})
    pairs = pd.DataFrame([{'sku_id1': 'a', 'sku_id2': 'b', 'label': 1, 'split': 'train'}])
    write_setup_frames(tmp_path, catalog=catalog, splits=splits, pairs=pairs)
    assert (tmp_path / 'eligible_catalog.csv').read_bytes() == \
        catalog.to_csv(index=False).encode()
    assert (tmp_path / 'listing_splits.csv').read_bytes() == \
        splits.to_csv(index=False).encode()
    assert (tmp_path / 'listing_pairs.csv').read_bytes() == \
        pairs.to_csv(index=False).encode()


def test_the_setup_csv_and_text_config_have_one_writer():
    """The rebuilders call the producer's writer; none re-emits the frames."""
    import inspect

    from graph_tracks import setup as setup_module
    from model_tracks import shared_graph_data, smoke_inputs

    for module in (shared_graph_data, smoke_inputs):
        source = inspect.getsource(module)
        assert 'write_setup_frames' in source, module.__name__
        for name in ('eligible_catalog.csv', 'listing_splits.csv', 'listing_pairs.csv'):
            assert name not in source, f'{module.__name__} re-spells/re-writes {name}'
    assert 'write_text_config' in inspect.getsource(smoke_inputs)
    assert 'write_text_config' in inspect.getsource(setup_module)
    # setup.py owns the frames writer; the two rebuilders only call it.
    for module in (shared_graph_data, smoke_inputs):
        assert 'write_setup_frames(' in inspect.getsource(module)


#: Every lane the audit (2026-10-08) consolidated onto the declared setup layout
#: (``training.preparation.graph_setup``), with the accessor it reads the spec
#: through. ``handoff`` is handed the spec by its caller, so it has no accessor.
_CONSOLIDATED_UNITS = (
    ('training.prepare_embeddings', '_setup_layout'),
    ('training.prepare_tokens', '_setup_layout'),
    ('training.handoff', None),
    ('graph_tracks.preflight', '_setup_layout'),
    ('graph_tracks.train', '_setup_layout'),
    ('model_tracks.staged_ablation', '_setup_layout'),
    ('model_tracks.ablation_cohort', '_setup_layout'),
    ('model_tracks.bundle_steps', '_setup_layout'),
    ('model_tracks.text_export', '_setup_layout'),
    ('model_tracks.baseline_export', '_setup_layout'),
    ('model_tracks.baseline_ablation', '_setup_layout'),
    ('model_tracks.worker', '_setup_layout'),
)

#: The declared artifact names the audit replaced. A consolidated lane may only
#: reach them through the spec; naming one itself is a second declaration the
#: config can no longer steer.
_LAYOUT_NAMES = (
    'eligible_catalog.csv', 'setup_manifest.json', 'graph_census.json',
    'pair_lineage.json', 'embedding_inputs.json', 'shared_minilm__embeddings.npz',
    'shared_training_data.json', 'shared_training_projection.json',
    'text_training_binding.json', 'text_export_request.json',
)


def test_consolidated_lanes_read_the_declared_setup_layout():
    """Every consolidated lane resolves the ONE declared layout object.

    The audit's replacement was mechanical; this pin makes it structural: the
    accessor returns the declared spec, so a lane that starts building its own
    layout copy cannot pass here.
    """
    import importlib

    from core.common import training_cfg

    layout = training_cfg().preparation.graph_setup
    for name, accessor in _CONSOLIDATED_UNITS:
        if accessor is None:
            continue
        module = importlib.import_module(name)
        assert getattr(module, accessor)() == layout, name


def test_no_consolidated_lane_respells_a_declared_layout_name():
    """No consolidated lane names a layout artifact as its own literal.

    Same idiom as the one-writer pin above: the shipped source IS the artifact,
    so a re-spelled name is exactly the drift this consolidation removed.
    """
    import importlib
    import inspect

    for name, _ in _CONSOLIDATED_UNITS:
        source = inspect.getsource(importlib.import_module(name))
        for literal in _LAYOUT_NAMES:
            assert literal not in source, f'{name} re-spells {literal}'
    # The Colab local embedding launcher ships as a file, not as an import.
    launcher = (Path(__file__).resolve().parents[1]
                / 'scripts/run_colab_embeddings.py').read_text()
    for literal in _LAYOUT_NAMES:
        assert literal not in launcher, f'scripts/run_colab_embeddings.py re-spells {literal}'


# ── the prepared-setup ROOT is the suite config's setup_dir, in code ────────
# The prepared-setup root (config/model_tracks.yaml ``setup_dir``) is read by
# the orchestrator (training.prepare_all), the model-track preflight/worker/
# finalize surfaces and the data gate. The setup producer's standalone CLI used
# to default to a second, hand-spelled ``data_dir/track_setup`` copy, so a
# retargeted suite ``setup_dir`` silently stranded the tree it wrote.

def test_default_setup_dir_is_the_suite_config_setup_dir():
    from core.common import TRAIN_ROOT, artifact
    from graph_tracks.setup import default_setup_dir
    from model_tracks.config import load_config as load_suite

    suite = load_suite(artifact('model_tracks_config'))
    assert default_setup_dir() == (Path(TRAIN_ROOT) / suite.setup_dir).resolve()


def test_default_setup_dir_follows_a_retargeted_suite_and_falls_back(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from core import common
    from graph_tracks import setup as setup_module
    import model_tracks.config as suite_config

    config = tmp_path / 'model_tracks.yaml'
    config.write_text('setup_dir: data/elsewhere\n')
    monkeypatch.setattr(common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(common, 'artifact', lambda key: config.resolve())
    monkeypatch.setattr(suite_config, 'load_config',
                        lambda _: SimpleNamespace(setup_dir='data/elsewhere'))
    assert setup_module.default_setup_dir() == (tmp_path / 'data' / 'elsewhere').resolve()

    # an ABSENT suite config falls back to the historical literal, never a crash
    monkeypatch.setattr(common, 'artifact', lambda key: tmp_path / 'missing.yaml')
    fallback = Path(common._CFG['paths']['data_dir']) / 'track_setup'
    assert setup_module.default_setup_dir() == fallback

    # an UNDECLARED layout key is the same fallback, not a KeyError leak
    def _boom(key):
        raise KeyError(key)

    monkeypatch.setattr(common, 'artifact', _boom)
    assert setup_module.default_setup_dir() == fallback


#: The producer + consumer surfaces the prepared-layout SSOT audit repointed
#: (the same set the task verified), scanned for declared-name literals in CODE.
_DECLARED_NAME_SURFACES = (
    'graph_tracks.prepare', 'graph_tracks.setup', 'graph_tracks.train',
    'graph_tracks.preflight', 'graph_tracks.data', 'graph_tracks.config',
    'graph_tracks.worker_package', 'training.prepare_all',
    'training.prepare_embeddings', 'training.prepare_tokens', 'cli.colab',
)


def test_declared_prepared_names_are_only_reached_through_the_spec():
    """A declared graph_setup FILENAME is never a string literal in CODE.

    The producer (``graph_tracks.prepare``/``graph_tracks.setup``) and every
    consumer must reach these names through ``PreparationGraphSetupSpec``, so a
    producer and its consumer can never disagree. Docstrings/comments are not
    filesystem paths, so the scan reads STRING tokens only; ``prepared_dir`` is
    excluded because it is a directory name (and an unrelated trace step is
    legitimately called ``prepared``).
    """
    import ast
    import importlib
    import io
    import tokenize

    from core.schemas import PreparationGraphSetupSpec

    spec = PreparationGraphSetupSpec()
    names = set(spec.model_dump().values()) - {spec.track_config_suffix, spec.prepared_dir}
    offenders = []
    for module_name in _DECLARED_NAME_SURFACES:
        module = importlib.import_module(module_name)
        source = Path(module.__file__).read_text(encoding='utf-8')
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.STRING:
                continue
            try:
                value = ast.literal_eval(token.string)
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, str) and value in names:
                offenders.append(f'{module_name}:{token.start[0]} {value!r}')
    assert offenders == [], (
        'declared prepared-layout names respelled as code literals: '
        + ', '.join(offenders))
