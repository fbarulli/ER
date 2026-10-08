import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from model_tracks.parallel import run_parallel
from model_tracks.resume import TRAINING_TRACKS


def test_all_workers_overlap_and_outputs_are_isolated(tmp_path):
    code = '''import json, os, pathlib, time
from model_tracks.parallel import wait_for_start
track=os.environ['ER_TRACK_NAME']
root=pathlib.Path(os.environ['EUROMONITOR_RESULTS_DIR'])
wait_for_start(pathlib.Path(os.environ['ER_TRACK_BARRIER']),track,timeout=10)
start=time.time()
time.sleep(.4)
(root/'result.json').write_text(json.dumps({'track':track,'start':start,'end':time.time(),'wandb':os.environ['WANDB_DIR']}))
'''
    commands={track:[sys.executable,'-c',code] for track in TRAINING_TRACKS}
    result=run_parallel(commands,tmp_path,{**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')},barrier_timeout=10)
    rows=[json.loads((tmp_path/track/'result.json').read_text()) for track in commands]
    assert max(row['start'] for row in rows)<min(row['end'] for row in rows)
    assert len({row['wandb'] for row in rows})==len(TRAINING_TRACKS)
    assert result['mode']=='parallel'


def test_failed_worker_aborts_whole_suite(tmp_path):
    code='''import os,pathlib,time
from model_tracks.parallel import wait_for_start
track=os.environ['ER_TRACK_NAME']
wait_for_start(pathlib.Path(os.environ['ER_TRACK_BARRIER']),track,timeout=10)
if track=='gnn_only': raise SystemExit(3)
time.sleep(30)
'''
    commands={track:[sys.executable,'-c',code] for track in TRAINING_TRACKS}
    with pytest.raises(RuntimeError,match='workers failed'):
        run_parallel(commands,tmp_path,{**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')},barrier_timeout=10)


def test_existing_auth_is_reused_without_copying_tokens(tmp_path,monkeypatch):
    entry=Path(__file__).resolve().parents[1]/'src/cli/colab_cli_entry.py'
    spec=importlib.util.spec_from_file_location('entry_under_test',entry)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project=tmp_path/'project'
    project.mkdir()
    upstream=tmp_path/'global_token.json'
    upstream.write_text('test token placeholder')
    monkeypatch.setattr(module,'STATE_DIR',project)
    assert module._token_config_path(str(upstream))==str(upstream)
    assert not (project/'token.json').exists()
    (project/'token.json').write_text('project placeholder')
    assert module._token_config_path(str(upstream))==str(project/'token.json')


def test_run_publication_reuses_the_verified_result_handle(tmp_path, monkeypatch):
    """The supervisor hands its verified result handle to publication.

    Publication re-loads (and re-verifies) the result archive whenever no
    boundary handle is supplied, so a multi-GB suite was otherwise integrity
    checked once per downstream step of the same process, contradicting the
    one-check-per-VM-crossing contract.
    """
    from types import SimpleNamespace as NS

    import yaml

    from core.bundle import BundleRole
    from model_tracks import local_complete
    from model_tracks import run as suite_run

    config = tmp_path / 'suite.yaml'
    config.write_text(yaml.safe_dump({
        'setup_dir': 'setup', 'text_bundle': 'bundle.pkl.gz', 'device': 'cpu',
        'publish_git': False, 'publish_dvc': True, 'profiling': False}))
    output = tmp_path / 'run'
    output.mkdir()
    monkeypatch.delenv('ER_GPU_TRAINING_ONLY', raising=False)
    monkeypatch.setenv('DVC_API_KEY', 'test-token')
    monkeypatch.setattr(suite_run, 'preflight', lambda *a, **k: {})
    monkeypatch.setattr('model_tracks.data_gate.validate',
                        lambda *a, **k: NS(tracks={}, attestation='attestation'))
    monkeypatch.setattr('model_tracks.resume.suite_identity',
                        lambda *a, **k: {'implementation': {}})
    monkeypatch.setattr('model_tracks.resume.completed_track', lambda *a, **k: True)
    monkeypatch.setattr('model_tracks.resume.selected_checkpoint_dirs',
                        lambda *a, **k: frozenset())
    monkeypatch.setattr(suite_run, 'run_parallel',
                        lambda commands, root, env, **k: {'mode': 'parallel', 'workers': []})
    monkeypatch.setattr(suite_run, '_run_postprocess_track', lambda *a, **k: 'cascade')
    captured = {}
    monkeypatch.setattr(local_complete, '_publish',
                        lambda final, settings, run_tag, **k: captured.update(k) or final)
    events = NS(attempt='attempt', last_phase=None, emit=lambda *a, **k: None)

    suite_run._run(config, output, 'run-tag', events=events)

    # The freshly sealed archive's own writer handle reaches publication; a
    # missing kwarg here is a second Bundle.load of the same bytes downstream.
    assert captured['bundle'].role is BundleRole.result
    assert captured['bundle'].digest
    assert captured['destination'] == output


def test_recovery_package_publishes_the_writer_digest_as_the_sidecar(tmp_path):
    """The recovery transport token comes from the seal, not a second hash.

    ``recovery_package`` seals through the shared writer, which already captured
    the whole-file SHA256 while writing; the sidecar is written from that digest
    so the all-epoch (largest) recovery archive is never read back to be hashed
    again by the failure-recovery script.
    """
    from core.archive_reader import archive_sidecar
    from core.bundle import Bundle, BundleRole, bundle_spec
    from model_tracks.package import recovery_package

    output = tmp_path / 'interrupted'
    (output / 'text').mkdir(parents=True)
    (output / 'suite_manifest.json').write_text(json.dumps({'run_tag': 'r'}))
    (output / 'text' / 'last.pt').write_bytes(b'checkpoint')
    archive = recovery_package(output, tmp_path / 'recovery.zip', 'r')

    sidecar = archive_sidecar(archive, bundle_spec().sha256_sidecar_suffix)
    handle = Bundle.load(archive, BundleRole.recovery)
    assert sidecar.is_file()
    assert sidecar.read_text().strip() == handle.digest


def test_run_retention_markers_follow_the_bundle_inventory_name(tmp_path, monkeypatch):
    """Retention recognizes a run root by the bundle contract's marker name."""
    from core.bundle import bundle_spec
    from model_tracks import run_retention

    monkeypatch.setattr(bundle_spec(), 'inventory_file', 'declared_inventory.json',
                        raising=False)
    declared = tmp_path / 'declared'
    (declared / 'text').mkdir(parents=True)
    (declared / 'text' / 'declared_inventory.json').write_text('{}')
    stale = tmp_path / 'stale'
    (stale / 'text').mkdir(parents=True)
    (stale / 'text' / 'track_inventory.json').write_text('{}')

    assert run_retention._run_markers()[0] == 'declared_inventory.json'
    assert run_retention._looks_like_run(declared)
    assert not run_retention._looks_like_run(stale)


def test_ablation_template_names_come_from_the_bundle_spec(tmp_path, monkeypatch):
    """The ablation template dir/member names are the spec's, never literals."""
    from core.bundle import bundle_spec
    from model_tracks import staged_ablation

    spec = bundle_spec()
    monkeypatch.setattr(spec, 'ablation_templates_dir', 'declared_templates', raising=False)
    monkeypatch.setattr(spec, 'ablation_request_file', 'declared_request.json', raising=False)
    setup = tmp_path / 'setup'
    setup.mkdir()
    staging = tmp_path / 'staging'
    staging.mkdir()
    (staging / 'prepared_inputs.npz').write_bytes(b'tensors')

    staged_ablation._copy_template(setup, 'text', staging / 'request.json', {'variants': []})
    assert (setup / 'declared_templates' / 'text' / 'declared_request.json').is_file()
    template, request = staged_ablation._read_template(setup, 'text')
    assert template == setup / 'declared_templates' / 'text'
    assert request == {'variants': []}

    (setup / 'declared_templates' / 'leftover').mkdir()
    staged_ablation._drop_staging(setup)
    assert not (setup / 'declared_templates' / 'leftover').exists()
