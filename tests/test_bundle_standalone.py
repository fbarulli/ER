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


def test_standalone_result_bundle_round_trips_through_the_shared_writer(tmp_path):
    """A transport loads one Bundle and can save it back through the same writer.

    The GraphTrack bundler and the shared writer must agree on the member set:
    re-sealing a materialized bundle reproduces exactly the members that were
    loaded, so handoff code never re-implements archive writing.
    """
    from graph_tracks.bundle import bundle

    archive = bundle(_graph_result_source(tmp_path), tmp_path / 'gnn_only__bundle.zip')
    handle = Bundle.load(archive, BundleRole.result,
                         manifest_name='gnn_only__bundle_manifest.json')
    tree = handle.materialize(tmp_path / 'tree')
    resealed = Bundle.seal_archive(
        tmp_path / 'again.zip',
        {member: tree.local / member for member in tree.members()},
        role=BundleRole.result, manifest_name='gnn_only__bundle_manifest.json',
        metadata={'run_tag': handle.run_tag()})

    loaded = Bundle.load(resealed.path, BundleRole.result,
                         manifest_name='gnn_only__bundle_manifest.json')
    assert loaded.members() == handle.members()
    assert loaded.run_tag() == handle.run_tag()
    assert loaded.digest is not None


def test_worker_package_manifest_mirrors_the_bundle_inventory(tmp_path):
    """The worker package keeps BOTH inventories, and they agree.

    ``Bundle.load`` verifies the config ``bundle.files_key`` inventory that
    ``Bundle.seal_archive`` computes while writing; the historical
    ``files_sha256`` key survives only so already-written packages and the
    printed ``--verify`` instructions keep working. A divergence between the
    two would ship a manifest whose boundary check and legacy check disagree,
    so this pins them equal member-for-member.
    """
    import zipfile

    from graph_tracks.worker_package import package

    archive = package(_graph_worker_config(tmp_path), tmp_path / 'worker.zip')
    manifest_name = 'data/graph_worker/gnn_only/package_manifest.json'
    with zipfile.ZipFile(archive) as saved:
        manifest = json.loads(saved.read(manifest_name))
        members = set(saved.namelist())

    assert manifest['files_sha256'] == manifest['files']
    assert set(manifest['files']) == members - {manifest_name}
    handle = Bundle.load(archive, BundleRole.inputs, manifest_name=manifest_name)
    assert set(handle.members()) == members
    assert handle.digest


def test_worker_package_honors_the_declared_lane_output_dir(tmp_path):
    """The packaged worker carries the lane config's own ``output_dir``.

    A literal used to override the config, so a retuned lane still shipped the
    hard-coded tree. Every shipped lane declares its output tree in config
    (``results/graph_tracks``), so that value must round-trip byte-identically,
    and another declared value must be honored rather than overridden.
    """
    import zipfile

    from graph_tracks.worker_package import package

    config = _graph_worker_config(tmp_path)
    for index, declared in enumerate(('results/graph_tracks', 'results/other_lane')):
        settings = yaml.safe_load(config.read_text())
        settings['output_dir'] = declared
        config.write_text(yaml.safe_dump(settings))
        archive = package(config, tmp_path / f'worker_{index}.zip')
        with zipfile.ZipFile(archive) as saved:
            worker = yaml.safe_load(saved.read('data/graph_worker/gnn_only/worker.yaml'))
        assert worker['output_dir'] == declared
        assert worker['device'] == 'cuda'


def test_worker_package_member_root_follows_the_declared_layout(tmp_path, monkeypatch):
    """The ZIP's member root is the declared layout, never a spelled path.

    Re-pointing the layout must move the whole packaged tree: a hard-coded member
    root would keep shipping the old one wherever the config points.
    """
    import zipfile

    from core import common
    from graph_tracks.worker_package import package

    layout = common.LAYOUTS['graph_worker_package']
    monkeypatch.setitem(common.LAYOUTS, 'graph_worker_package',
                        layout.model_copy(update={'template': 'data/alternate_worker/{track}'}))
    archive = package(_graph_worker_config(tmp_path), tmp_path / 'worker.zip')
    with zipfile.ZipFile(archive) as saved:
        assert 'data/alternate_worker/gnn_only/package_manifest.json' in saved.namelist()


def test_ner_artifact_manifest_boundary_round_trips(tmp_path, monkeypatch):
    """The NER result boundary is the sidecar artifact manifest, not a Bundle.

    The three final artifacts travel as SEPARATE objects, so integrity is
    checked once per transfer against the write-last manifest: the producer
    (``ner._write_artifact_manifest``) entries must be exactly what the
    consumer (``colab_ner._read_artifact_manifest`` + ``_verify_download``)
    accepts, and a tampered transfer must fail loud.
    """
    from ner import colab_ner, ner

    results = tmp_path / 'results'
    results.mkdir()
    monkeypatch.setattr(ner, 'RESULTS_DIR', results)
    artifacts = []
    for name in colab_ner.EXPECTED_FINAL_ARTIFACTS:
        path = tmp_path / name
        path.write_bytes(f'artifact::{name}'.encode())
        artifacts.append(path)

    manifest = ner._write_artifact_manifest(artifacts)
    entries = colab_ner._read_artifact_manifest(manifest)
    for name in colab_ner.EXPECTED_FINAL_ARTIFACTS:
        assert entries[name]['sha256'] == ner._sha256_file(tmp_path / name)
        colab_ner._verify_download(tmp_path / name, entries[name]['sha256'])

    tampered_name = colab_ner.EXPECTED_FINAL_ARTIFACTS[0]
    tampered = tmp_path / tampered_name
    tampered.write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        colab_ner._verify_download(tampered, entries[tampered_name]['sha256'])


def test_ner_producer_digest_agrees_with_the_shared_primitive(tmp_path):
    """The producer owns a copy of the digest because the bare remote runtime
    has no ``core.manifest``; the consumer delegates to the shared primitive,
    so the two implementations must agree byte-for-byte."""
    from core.manifest import sha256_file
    from ner import ner

    sample = tmp_path / 'artifact.bin'
    sample.write_bytes(b'ner-artifact' * 1024)
    assert ner._sha256_file(sample) == sha256_file(sample)


def test_ner_config_expansion_is_the_pinned_copy(monkeypatch):
    """The bare NER runtime expands ``${base_dir}``/``${results_dir}`` itself.

    ``core.common`` (which owns ``TRAIN_ROOT``/``RESULTS``) is deliberately
    absent there -- see ``ner/ner.py``'s ``ModuleNotFoundError`` fallback -- so a
    shared helper would pull ``core`` into the bare runtime. The pinned copy must
    therefore spell the same two names, resolve them to the same paths the shared
    loader owns, and expand nothing else (no environment variables), exactly like
    the live sibling ``ner.colab_ner.expand_vars``.
    """
    from core.common import RESULTS, TRAIN_ROOT
    from ner import ner

    assert ner._resolve_config_value('${base_dir}/x') == f'{TRAIN_ROOT}/x'
    assert ner._resolve_config_value('${results_dir}/ner_model') == \
        f'{RESULTS}/ner_model'
    assert ner._resolve_config_value({'a': ['${results_dir}/y', 3]}) == \
        {'a': [f'{RESULTS}/y', 3]}
    # only the two declared names are substituted; everything else stays literal
    assert ner._resolve_config_value('${unknown}/z') == '${unknown}/z'
    monkeypatch.setenv('ER_NER_PROBE', 'expanded')
    assert ner._resolve_config_value('$ER_NER_PROBE') == '$ER_NER_PROBE'
