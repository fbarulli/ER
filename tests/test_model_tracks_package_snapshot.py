from pathlib import Path
from types import SimpleNamespace
import zipfile
from core.portable_archive import ByteCount
import json

import yaml
import pytest


#: ``package()`` now writes its rows into the ONE consolidated trace, so this
#: module keeps them out of the real results tree. The path is under a
#: subdirectory on purpose: the packaged member set must not see the trace.
@pytest.fixture(autouse=True)
def _hermetic_trace(tmp_path, monkeypatch):
    import core.tracing as tracing

    target = tmp_path / 'logs' / 'training_trace.csv'
    monkeypatch.setattr(tracing, 'trace_path', lambda: target)
    monkeypatch.setenv('EUROMONITOR_TRACE_RUN', 'run-test-suite-inputs')
    return target


def test_package_ships_mutable_inputs_and_current_config(tmp_path, monkeypatch):
    import core.common
    from model_tracks import package as packaging

    setup = tmp_path / 'prepared'
    setup.mkdir()
    (setup / 'gnn_only.yaml').write_text(yaml.safe_dump({
        'track': 'gnn_only', 'listings': 'prepared/listings.csv',
        'pairs': 'prepared/pairs.csv', 'output_dir': 'results/graph_tracks'}))
    (setup / 'cascade.yaml').write_text(yaml.safe_dump({
        'track': 'cascade', 'listings': 'prepared/listings.csv',
        'pairs': 'prepared/pairs.csv', 'output_dir': 'results/graph_tracks',
        'text_index': 'results/graph_tracks/text__index',
        'gnn_checkpoint': 'results/graph_tracks/gnn_only__best_checkpoint.json'}))
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
    semantics = tmp_path / 'artifacts' / 'evidence' / 'semantics'
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
    monkeypatch.setattr(core.common, 'git_revision', lambda: 'revision')
    # The shared supervision population is built by the real bundle loader; this
    # case covers packaging, so stub the population and keep its artifacts.
    from training import prepared_bundle as prepared_bundle_mod
    from model_tracks import training_data as shared_training_data_mod
    from model_tracks import shared_graph_data as shared_graph_data_mod
    shared_stub = shared_training_data_mod.SharedTrainingData.model_construct(
        source_rows=1, canonical_rows=0, payload_rows=1,
        examples=[], endpoints=[])
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
        for track in ('gnn_only', 'cascade'):
            config = yaml.safe_load(archive.read(f'data/model_tracks/shared/{track}.yaml'))
            assert config['device'] == 'cpu'
            assert config['report_test'] is False
            assert config['listings'] == 'data/model_tracks/shared/listings.csv'

    # ── the ONE consolidated trace (core.tracing, stage suite_inputs) ───────
    import core.tracing as tracing
    from core.schemas import TraceRow
    frame = tracing.read_trace(tracing.trace_path())
    tracing.assert_trace_frame(frame)
    for row in frame.to_dict('records'):
        TraceRow.model_validate(row)
    assert set(frame['stage']) == {'suite_inputs'}
    assert set(frame['run_id']) == {'run-test-suite-inputs'}

    collected_row = frame[frame['step'] == 'members.collected'].iloc[0]
    collected = tracing.detail_json(collected_row['detail'])
    assert collected['manifest_member'] == packaging.package_manifest()
    # every collected member is sealed: the manifest member is never one
    assert int(collected_row['in_count']) == int(collected_row['out_count'])
    assert int(collected_row['dropped_count']) == 0

    # the per-group census is EXACT and closes over the sealed member set
    census = {str(r['reason']): int(r['in_count'])
              for _, r in frame[frame['step'] == 'member.reason_census'].iterrows()}
    assert sum(census.values()) == collected['archive_members']
    assert census['config'] == len([name for name in snapshots if name.startswith('config/')])
    assert census['inputs_csv'] == len([name for name in snapshots if name.endswith('.csv')]) + 2
    assert collected['by_group']['source_code']['members'] >= 1
    assert collected['by_group']['source_code']['bytes'] >= len(b'# current pipeline\n')

    # the named ENTITY rows carry the REAL size of the file each member seals,
    # and every sampled member is a member of the archive's own inventory
    named = {str(r['key']): tracing.detail_json(r['detail'])
             for _, r in frame[frame['scope'] == 'entity'].iterrows()}
    assert named['src/pipeline.py']['bytes'] == len(b'# current pipeline\n')
    assert named['config/paths.yaml']['bytes'] == len(b'current paths.yaml\n')
    assert set(named) <= set(metadata['files'])
    budget = tracing.detail_json(frame[frame['step'] == 'member.sample_budget'].iloc[0]['detail'])
    assert budget['population'] == collected['archive_members']
    assert budget['sampled'] == len(named)

    # the SEAL: the transport token the boundary re-checks, and the archive's
    # real size on disk — the trace states what the transport will carry
    sealed = tracing.detail_json(frame[frame['step'] == 'archive.sealed'].iloc[0]['detail'])
    assert sealed['size'] == ByteCount(output.read_bytes()).total
    assert sealed['bytes'] == output.stat().st_size
    assert sealed['members'] == collected['archive_members']
    assert sealed['role'] == 'inputs'
    assert sealed['schema'] == 'er-model-tracks-package-v1'
    assert sealed['revision'] == 'revision'
    assert sealed['source_bytes'] == collected['member_bytes']
    # the inlined portable configs are members too, and the seal names them
    assert len(sealed['inline_configs']) == 4
    assert set(sealed['inline_configs']) <= set(metadata['files'])


def test_suite_preflight_has_no_recorded_digest_freshness_gate(tmp_path, monkeypatch):
    """ZERO freshness checks (owner directive 2026-10-08, repo-wide).

    The suite preflight used to re-derive the source-catalog / labeled-pairs
    digests and compare them against the setup manifest's recorded values,
    raising 'graph setup is stale: ...'. That comparison is gone: the recorded
    digests are identity metadata carried inside the bundle, and the prepared
    suite is trusted as shipped. The mismatching fixture here therefore walks
    straight past the digest comparison to the next structural step (loading the
    prepared lane config, which this fixture does not provide).
    """
    import core.common
    from model_tracks import preflight as checks

    source = tmp_path / 'catalog.csv'
    source.write_text('unchanged catalog\n')
    labels = tmp_path / 'labels.csv'
    labels.write_text('changed supervision\n')
    setup = tmp_path / 'setup'
    setup.mkdir()
    (setup / 'setup_manifest.json').write_text(json.dumps({
        'source_catalog_size': ByteCount(source.read_bytes()).total,
        'labeled_pairs_size': ByteCount(b'previous supervision\n').total,
    }))
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(core.common, 'F', {'dataset_deduped': source, 'labeled_pairs': labels})
    monkeypatch.setattr(checks, 'load_config', lambda _: SimpleNamespace(setup_dir='setup'))
    with pytest.raises(FileNotFoundError, match='gnn_only.yaml'):
        checks.preflight(tmp_path / 'suite.yaml')


# Retired: the manifested text-cache composition gate exercised the fused
# hybrid text_cache path. No live track declares text_cache any more -- the
# cascade forbids it at config load (covered by
# test_cascade_wiring.py::test_cascade_lane_config_forbids_the_fused_text_cache),
# and gnn_only forbids it too -- so the case was deleted rather than ported.
