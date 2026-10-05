"""The handoff boundary attests loss/batch correctness and loads the bundle once."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.common import SEED, runtime, training_cfg


def _frozen_plan(loss, rows, batch_sizes, epochs):
    def epoch(bs):
        return [rows[i:i + bs] for i in range(0, len(rows), bs)]

    sampler = {device: {'batch_size': bs, 'epochs': [epoch(bs)] * epochs}
               for device, bs in batch_sizes.items()}
    return {'version': 1,
            'identity': {'loss': loss, 'train_frac': 1.0, 'sample': False,
                         'seed': SEED, 'config_sha256': '0' * 64, 'data_sha256': '1' * 64},
            'holdout': {'train': [], 'dev': [], 'test': []},
            'inputs': {'skipped': False,
                       'folds': [{'objective': {'dataset': {'anchor': list(rows)},
                                                'sampler': sampler}}]}}


def _handoff_env(tmp_path, monkeypatch, plan):
    import training.prepare_all as preparation
    from model_tracks.config import SuiteConfig
    root = tmp_path
    smoke = root / 'data/prepared/smoke_200'
    smoke.mkdir(parents=True)
    smoke_file = smoke / 'pairs.csv'
    smoke_file.write_text('smoke')
    setup = root / 'setup'
    setup.mkdir()
    files = {key: root / f'{key}.csv' for key in preparation.REUSABLE_KEYS}
    for path in files.values():
        path.write_text('bytes')
    (setup / 'setup_manifest.json').write_text(json.dumps({
        'source_catalog_sha256': preparation.sha256(files['dataset_deduped']),
        'labeled_pairs_sha256': preparation.sha256(files['labeled_pairs']),
        'text_checkpoint_sha256': '0' * 64, 'smoke': False}))
    bundle = root / 'bundle.pkl.gz'
    bundle.write_bytes(b'bundle')
    Path(str(bundle) + '.json').write_text('{}')
    archive = root / 'all_tracks_inputs.tar.zst'
    archive.write_bytes(b'archive')
    monkeypatch.setattr('core.common.F', files)
    monkeypatch.setattr(preparation, 'preparation_provenance',
                        lambda *args: {'text_checkpoint': '0' * 64})
    prepared = {'canonical_records_csv': files['canonical_records'].read_bytes(),
                'gate_results_csv': files['gate_results'].read_bytes(),
                'labeled_pairs_csv': files['labeled_pairs'].read_bytes()}
    if plan is not None:
        prepared['training_plan'] = plan
        prepared['training_tokens'] = {
            'texts': ['a', 'b'],
            'variants': {'': {'rows': [
                {'input_ids': np.array([1, 2]), 'attention_mask': np.array([1, 1])},
                {'input_ids': np.array([3]), 'attention_mask': np.array([1])}]}},
            'policy': {'input_token_limit': 8, 'padding_side': 'right'}}
    header = SimpleNamespace(sha256='a' * 64, payload_variant='full',
                             masking_profile='base', model_dump=lambda **kwargs: {'sha256': 'a' * 64})
    monkeypatch.setattr('training.prepared_bundle.load_prepared_bundle',
                        lambda path, *, verify_inputs: (header, prepared))
    monkeypatch.setattr('model_tracks.package.verify',
                        lambda path: {'preflight': {'shared_training_data': {'sha256': 's' * 64}}})
    suite = SuiteConfig(setup_dir='setup', text_bundle='setup/text.pkl.gz')
    return {'root': root, 'smoke': smoke, 'smoke_file': smoke_file, 'setup': setup,
            'files': files, 'bundle': bundle, 'archive': archive, 'suite': suite,
            'provenance': {'text_checkpoint': '0' * 64}}


def _verify(env, **overrides):
    from training.handoff import verify_training_loads
    arguments = dict(root=env['root'], suite=env['suite'],
                     suite_config_path=env['root'] / 'cfg.yaml',
                     checkpoint=env['root'] / 'ck', setup_dir=env['setup'],
                     full_bundle=env['bundle'], text_bundle=env['bundle'],
                     suite_archive=env['archive'], provenance=env['provenance'],
                     smoke_dir=env['smoke'],
                     smoke_original={str(env['smoke_file']): hashlib.sha256(b'smoke').hexdigest()},
                     reusable_paths=list(env['files'].values()))
    arguments.update(overrides)
    return verify_training_loads(**arguments)


def test_handoff_attests_loss_batch_contract_and_inventories_outputs(tmp_path, monkeypatch):
    from model_tracks.config import SuiteConfig
    from training.handoff import write_handoff_report
    plan = _frozen_plan(training_cfg().training.loss, [0, 1, 2],
                        {device: int(runtime('batch_size_' + device))
                         for device in ('cpu', 'cuda')},
                        SuiteConfig(setup_dir='setup', text_bundle='setup/text.pkl.gz').epochs)
    report = _verify(_handoff_env(tmp_path, monkeypatch, plan))
    write_handoff_report(report, tmp_path / 'handoff.json')
    saved = json.loads((tmp_path / 'handoff.json').read_text())
    assert saved['status'] == 'pass' and saved['finished_at']
    attestation = saved['loss_batch_correctness']
    assert attestation['coverage'] == 'every objective row exactly once per epoch'
    assert attestation['loss'] == training_cfg().training.loss
    assert attestation['epochs'] == SuiteConfig(setup_dir='setup', text_bundle='setup/text.pkl.gz').epochs
    assert attestation['batch_sizes'] == {device: int(runtime('batch_size_' + device))
                                          for device in ('cpu', 'cuda')}
    assert {entry['input'] for entry in saved['inputs']} >= {'text_bundle', 'suite_package'}
    assert saved['bundle_header'] == {'sha256': 'a' * 64}
    assert saved['suite_package']['preflight']
    inventory = saved['final_inventory']
    assert inventory[str(tmp_path / 'bundle.pkl.gz')]['sha256'] == hashlib.sha256(b'bundle').hexdigest()
    assert set(inventory) >= {str(path) for path in tmp_path.glob('*.csv')}


def test_handoff_rejects_loss_drift(tmp_path, monkeypatch):
    plan = _frozen_plan('other_loss', [0, 1], {'cpu': 2, 'cuda': 2}, 1)
    with pytest.raises(ValueError, match='loss differs'):
        _verify(_handoff_env(tmp_path, monkeypatch, plan))


def test_handoff_rejects_mutated_smoke_inputs(tmp_path, monkeypatch):
    from model_tracks.config import SuiteConfig
    env = _handoff_env(tmp_path, monkeypatch, None)
    env['smoke_file'].write_text('mutated')
    with pytest.raises(ValueError, match='Smoke files changed'):
        _verify(env)


def test_handoff_without_frozen_plan_records_absence(tmp_path, monkeypatch):
    report = _verify(_handoff_env(tmp_path, monkeypatch, None))
    assert report.loss_batch_correctness is None
    assert 'plan' in report.checks['loss_batch_correctness']


def test_handoff_rejects_provenance_drift_stale_csvs_and_graph_manifest(tmp_path, monkeypatch):
    import training.prepare_all as preparation
    env = _handoff_env(tmp_path, monkeypatch, None)
    with pytest.raises(ValueError, match='changed during the run'):
        _verify(env, provenance={'text_checkpoint': 'f' * 64})
    env['files']['canonical_records'].write_text('mutated')
    with pytest.raises(ValueError, match='Bundle contains stale canonical_records'):
        _verify(env)
    env['files']['canonical_records'].write_text('bytes')
    manifest = env['setup'] / 'setup_manifest.json'
    document = json.loads(manifest.read_text())
    document['source_catalog_sha256'] = '9' * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='Graph inputs contain stale'):
        _verify(env)
