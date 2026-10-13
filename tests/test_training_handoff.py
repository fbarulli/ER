"""The handoff boundary attests loss/batch correctness and loads the bundle once."""
from core.portable_archive import ByteCount
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.common import SEED, runtime, training_cfg


#: The boundary now writes its rows into the ONE consolidated trace, so this
#: module keeps them out of the real results tree, exactly like the traceability
#: stage tests do.
@pytest.fixture(autouse=True)
def _hermetic_trace(tmp_path, monkeypatch):
    import core.tracing as tracing

    # Under a subdirectory on purpose: the boundary's own test globs the tmp
    # root for its declared artifacts, and the trace is not one of them.
    target = tmp_path / "logs" / "training_trace.csv"
    monkeypatch.setattr(tracing, "trace_path", lambda: target)
    monkeypatch.setenv("EUROMONITOR_TRACE_RUN", "run-test-handoff")
    return target


def _trace_rows():
    """The boundary's trace rows, every one proven against the row contract."""
    import core.tracing as tracing
    from core.schemas import TraceRow

    frame = tracing.read_trace(tracing.trace_path())
    tracing.assert_trace_frame(frame)
    for row in frame.to_dict("records"):
        TraceRow.model_validate(row)
    return frame


def _step(frame, step):
    hit = frame[frame["step"].astype(str) == step]
    assert len(hit) == 1, f"expected one {step!r} row, got {len(hit)}"
    return hit.iloc[0]


def _frozen_plan(loss, rows, batch_sizes, epochs):
    def epoch(bs):
        return [rows[i:i + bs] for i in range(0, len(rows), bs)]

    sampler = {device: {'batch_size': bs, 'epochs': [epoch(bs)] * epochs}
               for device, bs in batch_sizes.items()}
    return {'version': 1,
            'identity': {'loss': loss, 'train_frac': 1.0, 'sample': False,
                         'seed': SEED},
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
        'source_catalog_size': preparation.size(files['dataset_deduped']),
        'labeled_pairs_size': preparation.size(files['labeled_pairs']),
        'text_checkpoint_size': '0' * 64, 'smoke': False}))
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
    header = SimpleNamespace(size=1234, payload_variant='full',
                            masking_profile='base', model_dump=lambda **kwargs: {'size': 1234})
    monkeypatch.setattr('training.prepared_bundle.load_prepared_bundle',
                        lambda path: (header, prepared))
    monkeypatch.setattr('model_tracks.package.verify',
                        lambda path: {'preflight': {'shared_training_data': {'size': 's' * 64}}})
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
                     smoke_original={str(env['smoke_file']): ByteCount(b'smoke').total},
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
    assert saved['bundle_header'] == {'size': 1234}
    assert saved['suite_package']['preflight']
    inventory = saved['final_inventory']
    assert inventory[str(tmp_path / 'bundle.pkl.gz')]['size'] == ByteCount(b'bundle').total
    assert set(inventory) >= {str(path) for path in tmp_path.glob('*.csv')}


def test_handoff_rejects_loss_drift(tmp_path, monkeypatch):
    plan = _frozen_plan('other_loss', [0, 1], {'cpu': 2, 'cuda': 2}, 1)
    with pytest.raises(ValueError, match='loss differs'):
        _verify(_handoff_env(tmp_path, monkeypatch, plan))


def test_handoff_does_not_gate_on_drift(tmp_path, monkeypatch):
    """Owner directive 2026-10-08: the boundary has no freshness verdicts.

    Drifted provenance, changed bundle copies and a moved graph-setup hash are
    read back for the record, never compared to decide whether data still
    applies; the bundle digest at load is the boundary's integrity check.
    """
    env = _handoff_env(tmp_path, monkeypatch, None)
    report = _verify(env, provenance={'text_checkpoint': 'f' * 64})
    assert report.status == 'pass'
    assert 'frozen_csv_agreement' not in report.checks
    assert 'smoke_unchanged' not in report.checks
    env['files']['canonical_records'].write_text('mutated')
    _verify(env)  # a changed bundle copy is not a staleness failure
    env['files']['canonical_records'].write_text('bytes')
    manifest = env['setup'] / 'setup_manifest.json'
    document = json.loads(manifest.read_text())
    document['source_catalog_size'] = '9' * 64
    manifest.write_text(json.dumps(document))
    report = _verify(env)
    assert report.checks['graph_setup']['source_catalog_size'] == '9' * 64


def test_handoff_accepts_mutated_smoke_inputs(tmp_path, monkeypatch):
    from model_tracks.config import SuiteConfig
    env = _handoff_env(tmp_path, monkeypatch, None)
    env['smoke_file'].write_text('mutated')
    # The smoke tree is no longer re-hashed against its start-of-run snapshot.
    assert _verify(env).status == 'pass'


def test_handoff_without_frozen_plan_records_absence(tmp_path, monkeypatch):
    report = _verify(_handoff_env(tmp_path, monkeypatch, None))
    assert report.loss_batch_correctness is None
    assert 'plan' in report.checks['loss_batch_correctness']


# ── the ONE consolidated trace (core.tracing, stage verify_handoff) ─────────
def test_handoff_trace_enumerates_every_check_load_and_the_report(tmp_path, monkeypatch):
    """A passing boundary states WHAT it checked, WHAT it loaded, and the report."""
    from core.tracing import detail_json
    from model_tracks.config import SuiteConfig
    from training.handoff import CHECK_ORDER

    suite = SuiteConfig(setup_dir='setup', text_bundle='setup/text.pkl.gz')
    plan = _frozen_plan(training_cfg().training.loss, [0, 1, 2],
                        {device: int(runtime('batch_size_' + device))
                         for device in ('cpu', 'cuda')}, suite.epochs)
    report = _verify(_handoff_env(tmp_path, monkeypatch, plan))

    frame = _trace_rows()
    assert set(frame["stage"]) == {'verify_handoff'}
    assert set(frame["run_id"]) == {'run-test-handoff'}

    # every declared check is NAMED and states its verdict + readback
    for name in CHECK_ORDER:
        row = _step(frame, f'check.{name}')
        assert row["reason"] == 'passed'
        assert detail_json(row["detail"])["verdict"] == 'passed'
    assert detail_json(_step(frame, 'check.bundle_load')["detail"])["readback"]['size'] == 1234
    assert detail_json(_step(frame, 'check.inventory')["detail"])["readback"]['artifacts'] == len(
        report.final_inventory)

    # the load meter: exact totals, and one named ENTITY row per metered input
    metered = detail_json(_step(frame, 'loads.metered')["detail"])
    assert metered['inputs'] == len(report.inputs)
    assert metered['loads'] == sum(entry.loads for entry in report.inputs)
    assert metered['bytes'] == sum(entry.bytes for entry in report.inputs)
    assert metered['size_pinned'] == sum(1 for entry in report.inputs if entry.size)
    named = frame[frame["step"] == 'loads.input']
    assert set(named["scope"]) == {'entity'}
    assert set(named["key"]) == {entry.input for entry in report.inputs}
    bundle_row = named[named["key"] == 'text_bundle'].iloc[0]
    assert detail_json(bundle_row["detail"]) == {
        'path': str(tmp_path / 'bundle.pkl.gz'), 'loads': 1, 'bytes': len(b'bundle'),
        'seconds': detail_json(bundle_row["detail"])['seconds'], 'size': 1234,
    }

    # the report's own census, and it agrees with the report object
    handoff = detail_json(_step(frame, 'report.handoff')["detail"])
    assert handoff['status'] == 'pass'
    assert handoff['loads'] == len(report.inputs)
    assert handoff['inventory_artifacts'] == len(report.final_inventory)
    assert handoff['loss_batch_attested'] is True
    assert handoff['bundle_size'] == 1234
    assert handoff['total_seconds'] == report.total_seconds


def test_handoff_failure_names_the_check_that_raised_and_keeps_the_error(tmp_path, monkeypatch):
    """A failing boundary commits evidence, then re-raises the boundary's error."""
    from core.tracing import detail_json

    plan = _frozen_plan('other_loss', [0, 1], {'cpu': 2, 'cuda': 2}, 1)
    with pytest.raises(ValueError, match='loss differs'):
        _verify(_handoff_env(tmp_path, monkeypatch, plan))

    frame = _trace_rows()
    failed = _step(frame, 'boundary.failed')
    evidence = detail_json(failed["detail"])
    assert evidence['failed_check'] == 'loss_batch'
    assert evidence['error_type'] == 'ValueError'
    assert 'loss differs' in failed["reason"]
    # the checks that had already passed are stated; the failed one is not
    assert 'provenance' in evidence['checks_passed']
    assert 'bundle_load' in evidence['checks_passed']
    assert 'loss_batch' not in evidence['checks_passed']
    assert 'check.loss_batch' not in set(frame["step"])
