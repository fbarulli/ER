"""Cascade wiring: the retired hybrid track is replaced by the cascade lane.

Two contracts only:
  (a) the suite track identity (``resume.TRACKS``/``Track``) and the artifact
      namer accept ``cascade`` and no longer accept ``hybrid``;
  (b) the cascade worker branch validates the trained text-ANN + gnn-scorer
      inputs and reports through ``report_cascade`` (mocked), with no fused
      text-cache / two-weight path anywhere.
"""
from types import SimpleNamespace
from typing import get_args
import json
import os

import pytest

from graph_tracks.artifacts import name
from model_tracks import resume, worker


# ---------------------------------------------------------------------------
# (a) track identity + artifact naming
# ---------------------------------------------------------------------------
def test_track_identity_is_text_gnn_only_cascade():
    assert resume.TRACKS == ('text', 'gnn_only', 'cascade')
    assert set(get_args(resume.Track)) == {'text', 'gnn_only', 'cascade'}
    assert 'hybrid' not in resume.TRACKS
    assert 'hybrid' not in get_args(resume.Track)


def test_trained_lanes_partition_tracks_and_own_the_barrier(tmp_path):
    """Only the trained lanes reach run_parallel; the cascade runs post-hoc."""
    from model_tracks.parallel import run_parallel
    assert set(resume.TRAINING_TRACKS).isdisjoint(resume.POSTPROCESS_TRACKS)
    assert tuple(resume.TRAINING_TRACKS) + tuple(resume.POSTPROCESS_TRACKS) == resume.TRACKS
    assert resume.POSTPROCESS_TRACKS == ('cascade',)
    # The cascade trains nothing and has no start barrier, so the parallel
    # supervisor must refuse it before spawning anything.
    with pytest.raises(ValueError, match='trained tracks'):
        run_parallel({'cascade': ['true']}, tmp_path, dict(os.environ), barrier_timeout=1)


def test_artifact_namer_accepts_cascade_and_rejects_hybrid():
    assert name('cascade', 'cascade_report.json') == 'cascade__cascade_report.json'
    with pytest.raises(ValueError, match='unknown graph track'):
        name('hybrid', 'graph_model.pt')


def test_cascade_lane_config_forbids_the_fused_text_cache(tmp_path):
    from graph_tracks.config import GraphConfig
    lane = {'track': 'cascade', 'listings': 'l.json', 'pairs': 'p.csv',
            'output_dir': 'results', 'text_cache': 'fused.npz',
            'text_index': 'text__index', 'gnn_checkpoint': 'gnn__best.json'}
    with pytest.raises(Exception, match='cascade forbids the fused text_cache'):
        GraphConfig.model_validate(lane)


# ---------------------------------------------------------------------------
# (b) cascade worker branch: validate inputs, then report_cascade
# ---------------------------------------------------------------------------
def _lane(tmp_path):
    return SimpleNamespace(
        text_index=str(tmp_path / 'text__index'),
        gnn_checkpoint=str(tmp_path / 'gnn_only__best_checkpoint.json'),
        listings='listings.json', pairs=str(tmp_path / 'pairs.csv'),
        device='cpu', retrieval_ks=[1, 3])


def test_cascade_artifacts_require_text_ann_and_gnn_scorer(tmp_path):
    root = tmp_path / 'suite'
    lane = SimpleNamespace(text_index=None, gnn_checkpoint=None,
                           listings='listings.json', pairs='pairs.csv', retrieval_ks=[1, 3])
    with pytest.raises(FileNotFoundError, match='text ranker index missing'):
        worker._cascade_artifacts(root, lane)

    (root / 'text' / 'text__index').mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match='text ranker vectors missing'):
        worker._cascade_artifacts(root, lane)

    (root / 'text' / 'text__vectors.npz').write_bytes(b'text-ann')
    with pytest.raises(FileNotFoundError, match='gnn scorer vectors missing'):
        worker._cascade_artifacts(root, lane)

    (root / 'gnn_only').mkdir()
    (root / 'gnn_only' / 'gnn_only__vectors.npz').write_bytes(b'gnn-vectors')
    with pytest.raises(FileNotFoundError, match='gnn scorer checkpoint missing'):
        worker._cascade_artifacts(root, lane)

    checkpoint = root / 'gnn_only' / name('gnn_only', 'graph_model.pt')
    checkpoint.write_bytes(b'scorer')
    # Fall back to the selected-checkpoint marker written by the gnn_only lane.
    (root / 'gnn_only' / name('gnn_only', 'best_checkpoint.json')).write_text(
        json.dumps({'path': str(checkpoint)}))
    artifacts = worker._cascade_artifacts(root, lane)
    assert artifacts['text_index'] == root / 'text' / 'text__index'
    assert artifacts['gnn_checkpoint'] == checkpoint
    assert 'text_cache' not in artifacts


def test_run_sequences_cascade_after_the_parallel_trained_lanes(tmp_path, monkeypatch):
    """run._run puts only the trained lanes behind the barrier; cascade after.

    The supervisor must hand exactly ``TRAINING_TRACKS`` to ``run_parallel``
    (the MPS barrier) and launch ``POSTPROCESS_TRACKS`` only once that call has
    returned. The cascade is a trained-artifact combinator, so it must never
    appear in the barrier commands.
    """
    from types import SimpleNamespace as NS
    import yaml
    from model_tracks import run as suite_run

    config = tmp_path / 'suite.yaml'
    config.write_text(yaml.safe_dump({
        'setup_dir': 'setup', 'text_bundle': 'bundle.pkl.gz', 'device': 'cpu',
        'publish_git': False, 'publish_dvc': False, 'profiling': False}))
    output = tmp_path / 'run'
    output.mkdir()

    order, parallel, postprocess = [], {}, []
    monkeypatch.delenv('ER_GPU_TRAINING_ONLY', raising=False)
    monkeypatch.setattr(suite_run, 'preflight', lambda *a, **k: {})
    monkeypatch.setattr('model_tracks.data_gate.validate',
                        lambda *a, **k: NS(tracks={}, attestation='attestation'))
    monkeypatch.setattr('model_tracks.resume.suite_identity',
                        lambda *a, **k: {'implementation': {}})
    monkeypatch.setattr('model_tracks.resume.completed_track', lambda *a, **k: True)
    monkeypatch.setattr('model_tracks.resume.selected_checkpoint_dirs',
                        lambda *a, **k: frozenset())

    def fake_parallel(commands, root, env, *, resume=False, **kwargs):
        order.append(('parallel', tuple(commands)))
        parallel.update(commands)
        return {'mode': 'parallel', 'workers': []}

    def fake_postprocess(config_, output_, run_tag, track, env, **kwargs):
        order.append(('postprocess', track))
        postprocess.append(track)
        return track

    monkeypatch.setattr(suite_run, 'run_parallel', fake_parallel)
    monkeypatch.setattr(suite_run, '_run_postprocess_track', fake_postprocess)

    events = NS(attempt='attempt', emits=[], emit=lambda *a, **k: events.emits.append((a, k)))
    suite_run._run(config, output, 'run-tag', events=events)

    assert set(parallel) == set(resume.TRAINING_TRACKS) == {'text', 'gnn_only'}
    assert 'cascade' not in parallel
    assert postprocess == list(resume.POSTPROCESS_TRACKS) == ['cascade']
    # The barrier releases before the combinator starts; nothing else is ordered.
    assert order == [('parallel', ('text', 'gnn_only')), ('postprocess', 'cascade')]


def test_cascade_worker_trains_nothing_and_skips_the_barrier(tmp_path, monkeypatch):
    """The cascade worker branch composes only: no barrier wait, no trainer."""
    from types import SimpleNamespace as NS

    results = tmp_path / 'suite' / 'cascade'
    results.mkdir(parents=True)
    monkeypatch.setenv('EUROMONITOR_RESULTS_DIR', str(results))
    monkeypatch.setattr(worker, 'load_config', lambda _: NS(setup_dir='.'))
    seen = {}
    monkeypatch.setattr(worker, '_run_cascade',
                        lambda cfg, setup, output, events: seen.setdefault('output', output))
    monkeypatch.setattr(worker, 'wait_for_start',
                        lambda *a, **k: pytest.fail('cascade must not wait on the start barrier'))
    monkeypatch.setattr(worker.subprocess, 'run',
                        lambda *a, **k: pytest.fail('cascade must not launch a training subprocess'))

    events = NS(emits=[], emit=lambda *a, **k: events.emits.append((a, k)))
    worker._run(tmp_path / 'suite.yaml', 'cascade', 'run', resume=False, events=events)

    assert seen['output'] == results
    assert 'training' not in [phase for phase, _ in events.emits]


def test_cascade_worker_reports_both_roles_without_fusion(tmp_path, monkeypatch):
    results = tmp_path / 'suite'
    output = results / 'cascade'
    output.mkdir(parents=True)
    lane = _lane(results)
    calls = {}

    monkeypatch.setattr('graph_tracks.report.report_cascade',
                        lambda ranked, relevant, decisions, out, **kwargs: calls.update(
                            ranked=ranked, relevant=relevant, decisions=decisions,
                            out=out, kwargs=kwargs))
    monkeypatch.setattr(worker, '_cascade_lane', lambda setup: lane)
    monkeypatch.setattr(worker, '_cascade_artifacts',
                        lambda results_root, lane_: {'text_index': results / 'text__index',
                                                     'gnn_checkpoint': results / 'gnn.json'})
    monkeypatch.setattr(worker, 'load_records_from_setup', lambda setup, lane_: [])
    monkeypatch.setattr('graph_tracks.train.load_pairs', lambda path, records: {})
    monkeypatch.setattr(worker, '_cascade_roles',
                        lambda records, pairs, artifacts: ('RANKED', {'a'}, 'DECISIONS'))
    monkeypatch.setattr(worker, '_record_cascade_report_manifest',
                        lambda output, lane_, artifacts: None)
    monkeypatch.setattr('model_tracks.resume.record_completion', lambda *a, **k: None)

    events = SimpleNamespace(emits=[], emit=lambda *a, **k: events.emits.append((a, k)))
    worker._run_cascade(SimpleNamespace(), tmp_path / 'setup', output, events)

    assert calls['ranked'] == 'RANKED'
    assert calls['decisions'] == 'DECISIONS'
    assert calls['kwargs']['track'] == 'cascade'
    # No fused-weight vocabulary ever reaches the cascade report call.
    assert 'text_cosine_weight' not in repr(calls)
    assert 'graph_cosine_weight' not in repr(calls)
    phases = [a[0] for a, _ in events.emits]
    assert 'cascade' in phases
