from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize('report_test', [False, True])
def test_text_worker_passes_suite_controls(tmp_path, monkeypatch, report_test):
    from model_tracks import worker
    import core.common
    import training.prepared_bundle
    import model_tracks.text_report

    (tmp_path / 'setup_manifest.json').write_text('{}')
    monkeypatch.setenv('EUROMONITOR_RESULTS_DIR', str(tmp_path / 'out'))
    monkeypatch.setenv('ER_TRACK_BARRIER', str(tmp_path))
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(worker, 'load_config', lambda _: SimpleNamespace(
        setup_dir='.', text_bundle='bundle', text_model='minilm_l6', epochs=1,
        device='cpu', report_test=report_test))
    monkeypatch.setattr(training.prepared_bundle, 'load_prepared_bundle',
                        lambda _: (SimpleNamespace(payload_variant='full'), {}))
    monkeypatch.setattr(worker, 'wait_for_start', lambda *_: None)
    commands = []
    monkeypatch.setattr(worker.subprocess, 'run', lambda command, **_: commands.append(command))
    monkeypatch.setattr(model_tracks.text_report, 'complete',
                        lambda output, *_, **__: (output / 'text__report.json').write_text('{}'))
    monkeypatch.delenv('ER_INCREMENTAL_DVC', raising=False)
    worker.run(tmp_path / 'suite.yaml', 'text', 'test')
    command = commands[0]
    assert command[command.index('--device') + 1] == 'cpu'
    assert ('--report-test' if report_test else '--no-report-test') in command
    events = [json.loads(line) for line in (tmp_path / 'out/worker_events.jsonl').read_text().splitlines()]
    publication = [row['status'] for row in events if row['phase'] == 'publication']
    assert publication == ['skipped']


@pytest.mark.parametrize('has_embeddings', [False, True])
@pytest.mark.parametrize('parent_cache', ['present', 'absent', 'checkpoint_changed'])
@pytest.mark.parametrize('relative_paths', [False, True])
def test_smoke_without_copies_projects_embedding_rows(tmp_path, monkeypatch, has_embeddings, parent_cache, relative_paths):
    from model_tracks.smoke_inputs import prepare_smoke
    import training.prepared_bundle as bundles
    import graph_tracks.prepare
    import graph_tracks.data
    import graph_tracks.text_cache

    setup = tmp_path / 'setup'
    setup.mkdir()
    catalog = pd.DataFrame({'sku_id': ['a', 'b', 'c'], 'gtin': ['1', '2', '3']})
    catalog.to_csv(setup / 'eligible_catalog.csv', index=False)
    pd.DataFrame({'sku_id': ['a', 'b', 'c'], 'split': ['train'] * 3}).to_csv(setup / 'listing_splits.csv', index=False)
    pd.DataFrame(columns=['sku_id1', 'sku_id2', 'split', 'label']).to_csv(setup / 'listing_pairs.csv', index=False)
    checkpoint = tmp_path / 'checkpoint'
    (setup / 'setup_manifest.json').write_text(json.dumps(
        {'text_checkpoint': str(checkpoint), 'text_checkpoint_sha256': 'baseline'}))
    if parent_cache == 'present':
        (setup / 'shared_minilm__embeddings.npz').write_bytes(b'mocked cache')
    for track in ('gnn_only', 'hybrid'):
        (setup / f'{track}.yaml').write_text(json.dumps({'listings': str(setup / 'prepared/listings.csv')}))
    empty_pairs = np.empty((0, 2), dtype=int)
    emb = np.arange(12, dtype=float).reshape(6, 2) if has_embeddings else np.empty((0, 2))
    bundle = dict(df=catalog, payload=['a', 'b', 'c', 'A', 'B', 'C'],
                  row_bc=np.array(['1', '2', '3', '1', '2', '3']),
                  country=np.array(['US'] * 6), structured_features=np.zeros((6, 2)),
                  emb0=emb, pos=empty_pairs, hp_pairs=empty_pairs, neg=empty_pairs,
                  train_neg=empty_pairs, neg_sources=np.array([]), train_neg_sources=np.array([]),
                  mask_audit=[], hard_negative_mask_audit=[], labeled_pairs_csv=b'',
                  canonical_records_csv=b'', gate_results_csv=b'', payload_variant='full', masking_profile='test')
    monkeypatch.setattr(bundles, 'load_prepared_bundle', lambda _: (None, bundle))
    monkeypatch.setattr(bundles, 'prepared_holdout', lambda *_, **__: ({'1', '2', '3'}, set(), set()))
    captured = {}
    monkeypatch.setattr(bundles, 'write_prepared_bundle', lambda _, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(graph_tracks.prepare, 'prepare', lambda *_: None)
    monkeypatch.setattr(graph_tracks.data, 'load_text_cache', lambda *args: (np.zeros((2, 2)), {}))
    monkeypatch.setattr(graph_tracks.data, 'file_hash', lambda _: 'hash')
    cache_calls = []
    monkeypatch.setattr(graph_tracks.text_cache, 'checkpoint_hash',
                        lambda _: 'changed' if parent_cache == 'checkpoint_changed' else 'baseline')
    monkeypatch.setattr(graph_tracks.text_cache, 'create_cache',
                        lambda *args, **kwargs: cache_calls.append((args, kwargs)))
    output = tmp_path / 'smoke'
    monkeypatch.chdir(tmp_path)
    setup_arg = Path('setup') if relative_paths else setup
    output_arg = Path('smoke') if relative_paths else output
    if parent_cache == 'checkpoint_changed':
        with pytest.raises(ValueError, match='smoke baseline differs from the parent checkpoint'):
            prepare_smoke(setup_arg, output_arg, sample=2)
        assert cache_calls == []
        return
    prepare_smoke(setup_arg, output_arg, sample=2)
    import yaml
    for track in ('gnn_only', 'hybrid'):
        settings = yaml.safe_load((output / f'{track}.yaml').read_text())
        assert settings['listings'] == str(output / 'prepared/listings.csv')
    if parent_cache == 'absent':
        assert cache_calls == [((output / 'eligible_catalog.csv', checkpoint,
                                 output / 'shared_minilm__embeddings.npz'), {'device': 'cpu'})]
        assert pd.read_csv(cache_calls[0][0][0]).sku_id.tolist() == ['a', 'b']
    else:
        assert cache_calls == []
    assert captured['payload'] == ['a', 'b', 'A', 'B', 'C']
    np.testing.assert_array_equal(captured['emb0'], emb[[0, 1, 3, 4, 5]] if has_embeddings else emb)
    assert captured['mask_audit'] == []


def test_prepared_cli_accepts_cpu_and_withheld_test(monkeypatch):
    from training.train_prepared import _parse_args
    monkeypatch.setattr('sys.argv', ['train_prepared', '--bundle', 'bundle.pkl.gz',
                                    '--device', 'cpu', '--no-report-test'])
    args = _parse_args()
    assert args.device == 'cpu'
    assert args.report_test is False


def test_prepared_trainer_rejects_ignored_guardrail_override(monkeypatch):
    from training import train_prepared as trainer

    monkeypatch.setattr('sys.argv', ['train_prepared', '--bundle', 'missing.pkl.gz',
                                    '--collapse-guardrail-profile', 'different'])
    args = trainer._parse_args()
    monkeypatch.setattr(trainer, 'load_config',
                        lambda: {'collapse_guardrail': {'profile': 'configured'}})
    monkeypatch.setattr(trainer, 'set_determinism', lambda _: None)
    monkeypatch.setattr(trainer, 'load_prepared_bundle',
                        lambda _: pytest.fail('must reject override before loading bundle'))
    with pytest.raises(ValueError, match='CLI profile=.*differs from'):
        trainer._main(args, SimpleNamespace())


@pytest.mark.parametrize('report_test', [False, True])
@pytest.mark.parametrize('sample', [False, True])
def test_prepared_trainer_forwards_explicit_test_policy(tmp_path, monkeypatch, report_test, sample):
    from training import train_prepared as trainer

    monkeypatch.setattr('sys.argv', ['train_prepared', '--bundle', 'bundle.pkl.gz',
                                    '--device', 'cpu',
                                    '--report-test' if report_test else '--no-report-test'])
    args = trainer._parse_args()
    args.sample = 100 if sample else None
    manifest = SimpleNamespace(payload_variant='full', masking_profile='default',
                               sha256='test', n_df=1, n_payload=1,
                               model_dump=lambda: {})
    pairs = np.empty((0, 2), dtype=int)
    bundle = dict(df=pd.DataFrame({'gtin': ['1']}), payload=['item'],
                  structured_features=np.zeros((1, 2)), row_bc=np.array(['1']),
                  country=np.array(['US']), pos=pairs, hp_pairs=pairs,
                  emb0=np.empty((0, 2)), neg=pairs, train_neg=pairs,
                  neg_sources=[], train_neg_sources=[], mask_audit=[],
                  hard_negative_mask_audit=[], labeled_pairs_csv=b'',
                  canonical_records_csv=b'', gate_results_csv=b'')
    monkeypatch.setattr(trainer, 'load_prepared_bundle', lambda _: (manifest, bundle))
    monkeypatch.setattr(trainer, 'prepared_holdout', lambda *_, **__: ({'1'}, set(), {'test'}))
    monkeypatch.setattr(trainer, 'masking_cfg', lambda _: dict(mask_hard_negatives=False,
                        hard_negative_frac=0, hard_negative_mask_prob=None,
                        hard_negative_mask_lo=0, hard_negative_mask_hi=0))
    monkeypatch.setattr(trainer, 'RESULTS', tmp_path)
    monkeypatch.setattr(trainer, 'F', {key: tmp_path / f'{key}.csv'
                                    for key in ('labeled_pairs', 'canonical_records', 'gate_results')})
    captured = {}
    def train(*_, **kwargs):
        captured.update(kwargs)
        return [{'status': 'ok', 'calibration_status': 'unavailable' if sample else 'available',
                 'calibration_reason': 'sample has no reserved calibration' if sample else '',
                 'test_eval': 'skipped_selection_mode' if not report_test else 'evaluated'}]
    monkeypatch.setattr(trainer, 'train_one_config', train)
    trainer._main(args, SimpleNamespace())
    assert captured['selection_mode'] is (not report_test)
    assert captured['skip_test_eval'] is (not report_test)
    assert captured['on_cuda'] is False
    assert captured['folds_override'] == {'test'}
    assert captured['sample'] is sample
    saved = pd.read_csv(next(tmp_path.glob('*fold_metrics.csv')))
    assert saved.iloc[0]['status'] == 'ok'
    assert saved.iloc[0]['calibration_status'] == ('unavailable' if sample else 'available')
    if sample:
        assert saved.iloc[0]['calibration_reason'] == 'sample has no reserved calibration'
    if not report_test:
        assert saved.iloc[0]['test_eval'] == 'skipped_selection_mode'


def test_suite_preflight_rejects_changed_source_catalog_before_launch(tmp_path, monkeypatch):
    import core.common
    from model_tracks import preflight as checks

    setup = tmp_path / 'setup'
    setup.mkdir()
    (setup / 'setup_manifest.json').write_text(json.dumps({'source_catalog_sha256': 'old'}))
    source = tmp_path / 'catalog.csv'
    source.write_text('sku_id\nnew\n')
    monkeypatch.setattr(core.common, 'TRAIN_ROOT', tmp_path)
    monkeypatch.setattr(core.common, 'F', {'dataset_deduped': source})
    monkeypatch.setattr(checks, 'load_config', lambda _: SimpleNamespace(setup_dir='setup'))
    with pytest.raises(ValueError, match='graph setup is stale: source catalog'):
        checks.preflight(tmp_path / 'suite.yaml')


def test_smoke_retains_cross_population_copy_dependencies(tmp_path, monkeypatch):
    from model_tracks.smoke_inputs import prepare_smoke
    import training.prepared_bundle as bundles
    import graph_tracks.prepare
    import graph_tracks.data

    setup = tmp_path / 'setup'
    setup.mkdir()
    catalog = pd.DataFrame({'sku_id': ['a', 'b', 'c', 'd'], 'gtin': ['1', '2', '3', '4']})
    catalog.to_csv(setup / 'eligible_catalog.csv', index=False)
    catalog.assign(split='train')[['sku_id', 'split']].to_csv(setup / 'listing_splits.csv', index=False)
    pd.DataFrame(columns=['sku_id1', 'sku_id2', 'split', 'label']).to_csv(setup / 'listing_pairs.csv', index=False)
    (setup / 'setup_manifest.json').write_text('{}')
    (setup / 'shared_minilm__embeddings.npz').write_bytes(b'mocked cache')
    for track in ('gnn_only', 'hybrid'):
        (setup / f'{track}.yaml').write_text('{}')
    positive = {'anchor_payload_idx': 8, 'pair_payload_idx': 4, 'copy_payload_idx': 9}
    negative = {'anchor_payload_idx': 0, 'pair_payload_idx': 5, 'copy_payload_idx': 8}
    unrelated = {'anchor_payload_idx': 2, 'pair_payload_idx': 7, 'copy_payload_idx': 10}
    negatives = np.array([[0, 5], [8, 5], [10, 7]])
    bundle = dict(df=catalog, payload=['a', 'b', 'c', 'd', 'A', 'B', 'C', 'D',
                                    'negative copy', 'positive counterpart', 'unrelated copy'],
                  row_bc=np.array(['1', '2', '3', '4', '1', '2', '3', '4', '1', '1', '3']),
                  country=np.array(['US'] * 11), structured_features=np.zeros((11, 2)),
                  emb0=np.arange(22).reshape(11, 2), pos=np.array([[8, 9], [2, 6]]),
                  hp_pairs=np.empty((0, 2), dtype=int), neg=negatives, train_neg=negatives,
                  neg_sources=np.array(['base', 'selected', 'unrelated']),
                  train_neg_sources=np.array(['base', 'selected', 'unrelated']),
                  mask_audit=[positive], hard_negative_mask_audit=[negative, unrelated],
                  labeled_pairs_csv=b'', canonical_records_csv=b'', gate_results_csv=b'',
                  payload_variant='full', masking_profile='baseline')
    monkeypatch.setattr(bundles, 'load_prepared_bundle', lambda _: (None, bundle))
    monkeypatch.setattr(bundles, 'prepared_holdout', lambda *_, **__: ({'1', '2', '3', '4'}, set(), set()))
    captured = {}
    monkeypatch.setattr(bundles, 'write_prepared_bundle', lambda _, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(graph_tracks.prepare, 'prepare', lambda *_: None)
    monkeypatch.setattr(graph_tracks.data, 'load_text_cache', lambda *_: (np.zeros((2, 2)), {}))
    monkeypatch.setattr(graph_tracks.data, 'file_hash', lambda _: 'hash')
    prepare_smoke(setup, tmp_path / 'smoke', sample=2)
    assert captured['df'].sku_id.tolist() == ['a', 'b']
    assert captured['payload'] == ['a', 'b', 'A', 'B', 'C', 'D', 'negative copy', 'positive counterpart']
    assert captured['mask_audit'] == [{'anchor_payload_idx': 6, 'pair_payload_idx': 2, 'copy_payload_idx': 7}]
    assert captured['hard_negative_mask_audit'] == [{'anchor_payload_idx': 0, 'pair_payload_idx': 3, 'copy_payload_idx': 6}]
    np.testing.assert_array_equal(captured['pos'], [[6, 7]])
    np.testing.assert_array_equal(captured['train_neg'], [[0, 3], [6, 3]])
    np.testing.assert_array_equal(captured['train_neg_sources'], ['base', 'selected'])
    # Projection must leave the full parent's lineage unchanged.
    assert positive['anchor_payload_idx'] == 8
    assert unrelated['copy_payload_idx'] == 10
