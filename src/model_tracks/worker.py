"""Adapters for existing prepared text and graph trainers in a shared run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import shlex
import yaml

from core.run_log import RunLogger
from model_tracks.config import load_config
from model_tracks.parallel import wait_for_start

_LOG = RunLogger(__name__)


def graph_worker_settings(setup: Path, cfg, track: str, *, gpu_only: bool = False) -> dict:
    """The exact graph configuration this worker trains under.

    The pre-training data gate builds its graph inputs through this function so
    the validated configuration cannot drift from the executed one.
    """
    from graph_tracks.config import GraphConfig, load_config as load_graph_config
    settings = load_graph_config(setup / f'{track}.yaml', expected_track=track).model_dump()
    settings.update(device=cfg.device, epochs=cfg.epochs, report_test=cfg.report_test,
                    postprocess=not gpu_only, include_inputs=False)
    if cfg.dvc_enabled or gpu_only:
        # The suite publisher owns persistence; avoid a second mutable
        # local DVC snapshot while background uploads are active.
        settings['dvc'] = {**settings.get('dvc', {}), 'enabled': False}
    settings.update(cfg.graph_execution_overrides())
    return GraphConfig.model_validate(settings).model_dump()


def _cascade_lane(setup: Path):
    """The validated cascade lane config (never a trained graph lane)."""
    from graph_tracks.config import load_config as load_graph_config
    return load_graph_config(setup / 'cascade.yaml', expected_track='cascade')


def _cascade_artifacts(results: Path, lane) -> dict:
    """Locate and validate the trained text-ANN + gnn-scorer inputs.

    The cascade is a combinator that trains nothing. Its declared inputs are
    the text ranker's ANN/embedding export and the gnn_only decider's scorer
    checkpoint/embedding export. A missing artifact fails here, before any
    ranking work, and the retired fused text cache is never consulted.
    """
    from graph_tracks.artifacts import name
    text_root = results / 'text'
    gnn_root = results / 'gnn_only'
    text_index = Path(lane.text_index) if lane.text_index else text_root / name('text', 'index')
    text_vectors = text_root / name('text', 'vectors.npz')
    gnn_vectors = gnn_root / name('gnn_only', 'vectors.npz')
    gnn_checkpoint = Path(lane.gnn_checkpoint) if lane.gnn_checkpoint else None
    if gnn_checkpoint is None or not gnn_checkpoint.is_file():
        marker = gnn_root / name('gnn_only', 'best_checkpoint.json')
        if marker.is_file():
            gnn_checkpoint = Path(json.loads(marker.read_text())['path'])
    if not text_index.is_dir():
        raise FileNotFoundError(f'cascade text ranker index missing: {text_index}')
    if not text_vectors.is_file():
        raise FileNotFoundError(f'cascade text ranker vectors missing: {text_vectors}')
    if not gnn_vectors.is_file():
        raise FileNotFoundError(f'cascade gnn scorer vectors missing: {gnn_vectors}')
    if gnn_checkpoint is None or not Path(gnn_checkpoint).is_file():
        raise FileNotFoundError('cascade gnn scorer checkpoint missing')
    return {'text_index': text_index, 'text_vectors': text_vectors,
            'gnn_vectors': gnn_vectors, 'gnn_checkpoint': Path(gnn_checkpoint)}


def _load_catalog_vectors(path: Path):
    import numpy as np
    with np.load(path, allow_pickle=False) as cache:
        return [str(value) for value in cache['ids'].tolist()], cache['embeddings']


def _cascade_roles(records, pairs, artifacts) -> tuple:
    """Retrieve with the text ranker, then rerank with the gnn_only scorer.

    The two trained artifacts are consumed exactly as trained: text vectors
    feed the ANN candidate generation, and the gnn_only scorer (rebuilt from
    its checkpoint) makes the decision over the retrieved catalog geometry.
    No fused embedding or two-input score is ever computed.
    """
    import numpy as np
    import torch
    from graph_tracks.model import PairScorer
    from model_tracks.cascade import CascadeIndex, Decisions, Query, Ranked, cascade
    text_ids, text_vectors = _load_catalog_vectors(artifacts['text_vectors'])
    gnn_ids, gnn_vectors = _load_catalog_vectors(artifacts['gnn_vectors'])
    if text_ids != gnn_ids:
        raise ValueError('cascade text and gnn catalog IDs differ')
    ids = tuple(gnn_ids)
    directory = Path(artifacts['text_index']).parent / 'cascade_ann'
    index = CascadeIndex.from_arrays(
        text_vectors, torch.as_tensor(gnn_vectors), ids, directory=directory,
        checkpoint=Path(artifacts['gnn_checkpoint']), model_name='text', build_index=True)
    payload = torch.load(artifacts['gnn_checkpoint'], map_location='cpu', weights_only=False)
    scorer = PairScorer()
    scorer.load_state_dict(payload['scorer'])
    scorer.eval()
    lookup = {identifier: i for i, identifier in enumerate(ids)}
    relevant = {}
    for split in ('dev', 'test'):
        indices, labels = pairs.get(split, (np.empty((0, 2), dtype=int), np.empty(0)))
        for (left, right), label in zip(indices, labels):
            if int(label) != 1:
                continue
            relevant.setdefault(records[int(left)]['sku_id'], set()).add(records[int(right)]['sku_id'])
            relevant.setdefault(records[int(right)]['sku_id'], set()).add(records[int(left)]['sku_id'])
    k = max(1, len(ids) - 1)
    rows_ranked, rows_decided, query_ids = [], [], []
    for sku_id, truth in relevant.items():
        if sku_id not in lookup:
            continue
        row = lookup[sku_id]
        query = Query(ids=(sku_id,), text=np.asarray([text_vectors[row]], dtype=np.float32),
                      graph=torch.as_tensor(gnn_vectors[row:row + 1]))
        decisions = cascade(query, index, scorer, k)
        query_ids.append(sku_id)
        rows_ranked.append(decisions.candidate_ids[0])
        rows_decided.append(decisions)
    if not rows_ranked:
        raise ValueError('cascade found no retrieval queries with known positives')
    ranked = Ranked(query_ids=tuple(query_ids),
                    candidate_ids=np.asarray(rows_ranked, dtype=object),
                    similarities=np.full((len(rows_ranked), k), np.nan, dtype=np.float32))
    decisions = Decisions(
        query_ids=tuple(query_ids),
        candidate_ids=np.asarray([row.candidate_ids[0] for row in rows_decided], dtype=object),
        scores=np.asarray([row.scores[0] for row in rows_decided], dtype=np.float32),
        order=np.asarray([row.order[0] for row in rows_decided], dtype=np.int64),
        similarities=None)
    return ranked, [relevant[q] for q in query_ids], decisions


def load_records_from_setup(setup: Path, lane):
    from core.common import TRAIN_ROOT
    from graph_tracks.data import load_records
    return load_records((TRAIN_ROOT / lane.listings).resolve())


def _record_cascade_report_manifest(output: Path, lane, artifacts) -> None:
    """Write the shared per-track report manifest for the cascade lane.

    The cascade is a combinator, but every completed track ships one calibrated
    report manifest (the suite completion/verification contract). Its
    checkpoint identity is the trained gnn_only scorer it consumes.
    """
    from graph_tracks.artifacts import name
    from graph_tracks.data import file_hash
    from graph_tracks.report_manifest import build as build_manifest, write as write_manifest
    from core.common import TRAIN_ROOT
    listings = (TRAIN_ROOT / lane.listings).resolve()
    pairs = Path(lane.pairs)
    if not pairs.is_absolute():
        pairs = (TRAIN_ROOT / pairs).resolve()
    manifest = build_manifest(
        track='cascade', checkpoint=str(artifacts['gnn_checkpoint']),
        checkpoint_sha256=file_hash(artifacts['gnn_checkpoint']),
        listings_sha256=file_hash(listings), pairs_sha256=file_hash(pairs),
        threshold=0.5, threshold_source='dev_youden', test_reported=bool(lane.report_test),
        model_selection='dev_pr_auc', retrieval_ks=list(lane.retrieval_ks))
    write_manifest(output / name('cascade', 'report_manifest.json'), manifest)


def _run_cascade(cfg, setup: Path, output: Path, events):
    """The cascade lane: validate inputs, compose roles, report both roles."""
    from graph_tracks.report import report_cascade
    from model_tracks.resume import record_completion
    lane = _cascade_lane(setup)
    events.emit('input_validation', 'configured', device=lane.device,
                worker_config=str(setup / 'cascade.yaml'), trains_nothing=True)
    artifacts = _cascade_artifacts(output.parent, lane)
    events.emit('input_validation', 'completed', text_index=str(artifacts['text_index']),
                gnn_checkpoint=str(artifacts['gnn_checkpoint']))
    from graph_tracks.train import load_pairs
    records = load_records_from_setup(setup, lane)
    pairs = load_pairs(Path(lane.pairs), records)
    events.emit('cascade', 'started')
    ranked, relevant, decisions = _cascade_roles(records, pairs, artifacts)
    report_cascade(ranked, relevant, decisions, output, track='cascade',
                   ks=tuple(sorted(set(lane.retrieval_ks) | {1})))
    _record_cascade_report_manifest(output, lane, artifacts)
    events.emit('cascade', 'completed', queries=len(relevant))
    with _LOG.section('phase.completion', track='cascade'):
        record_completion(output, 'cascade', postprocess_complete=True)
        events.emit('completion', 'verified', inventory='track_inventory.json',
                    marker='track_complete.json')


def run(config: Path, track: str, run_tag: str, *, resume: bool = False):
    from model_tracks.telemetry import WorkerEvents
    output = Path(os.environ['EUROMONITOR_RESULTS_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    events = WorkerEvents(output, track, run_tag)
    events.emit('input_validation', 'started', config=str(config), resume_requested=resume)
    try:
        _run(config, track, run_tag, resume=resume, events=events)
    except BaseException as exc:
        events.emit('failure', 'failed', error_type=type(exc).__name__,
                    error=str(exc), failed_phase=events.last_phase, traceback=traceback.format_exc())
        raise


def _run(config: Path, track: str, run_tag: str, *, resume: bool, events):
    from core.common import TRAIN_ROOT
    cfg = load_config(config)
    gpu_only = os.environ.get('ER_GPU_TRAINING_ONLY') == '1'
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    output = Path(os.environ['EUROMONITOR_RESULTS_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    if track == 'cascade':
        # The cascade is a combinator: it trains nothing and simply consumes
        # the already-trained text ranker and gnn_only scorer. It runs after
        # both prerequisite tracks complete, so it has no start barrier.
        with _LOG.section('phase.cascade', track=track):
            _run_cascade(cfg, setup, output, events)
        return
    if track == 'text':
        from training.prepared_bundle import PreparedBundleManifest
        bundle_path = (TRAIN_ROOT / cfg.text_bundle).resolve()
        # The trainer owns full payload loading/validation after the barrier.
        # This adapter only needs the typed header to construct its command.
        manifest = PreparedBundleManifest.model_validate_json(
            bundle_path.with_suffix(bundle_path.suffix + '.json').read_text())
        events.emit('input_validation', 'configured', device=cfg.device,
                    report_test=cfg.report_test, bundle=str(cfg.text_bundle),
                    payload=manifest.payload_variant)
        command = [sys.executable, '-m', 'training.train_prepared', '--bundle', cfg.text_bundle,
                   '--shared-training-data', str(setup / 'shared_training_data.json'),
                   '--training-binding', str(setup / 'text_training_binding.json'),
                   '--model', cfg.text_model, '--epochs', str(cfg.epochs),
                   '--payload', manifest.payload_variant, '--run-tag', run_tag,
                   '--device', cfg.device,
                   '--report-test' if cfg.report_test else '--no-report-test']
        if resume and any((output / '_checkpoints').rglob('trainer_state.json')):
            command.append('--resume')
        setup_manifest = json.loads((setup / 'setup_manifest.json').read_text())
        if setup_manifest.get('smoke'):
            command.extend(['--sample', str(setup_manifest['source_listing_count'])])
    else:
        from graph_tracks.config import GraphConfig
        settings = graph_worker_settings(setup, cfg, track, gpu_only=gpu_only)
        worker_config = output / 'worker.yaml'
        worker_config.write_text(yaml.safe_dump(settings, sort_keys=False))
        events.emit('input_validation', 'configured', device=cfg.device,
                    report_test=cfg.report_test, worker_config=str(worker_config),
                    validation_owner='graph trainer load_inputs before training')
        command = [sys.executable, '-m', 'graph_tracks.train', '--config', str(worker_config),
                   '--run-tag', run_tag]
        if resume:
            from model_tracks.resume import graph_checkpoint
            checkpoint = graph_checkpoint(output, track, run_tag)
            if checkpoint:
                command.extend(['--resume', str(checkpoint)])
            else:
                # A failure before the first checkpoint has no optimizer state.
                # Preserve its manifest as evidence and restart that track.
                for folder in (output, output / f'{track}__{run_tag}'):
                    previous = folder / f'{track}__run_manifest.json'
                    if previous.exists():
                        previous.replace(previous.with_name(f'{track}__run_manifest.interrupted.json'))
    events.emit('command', 'prepared', command=shlex.join(command),
                resume_requested=resume, checkpoint_resume='--resume' in command,
                checkpoint=(command[command.index('--resume') + 1]
                            if track != 'text' and '--resume' in command else None),
                sample=(int(command[command.index('--sample') + 1]) if '--sample' in command else None))
    events.emit('barrier', 'waiting', barrier=os.environ['ER_TRACK_BARRIER'])
    wait_for_start(Path(os.environ['ER_TRACK_BARRIER']), track)
    events.emit('barrier', 'released')
    events.emit('training', 'started', includes_graph_postprocess=track != 'text')
    with _LOG.section('phase.training', track=track):
        subprocess.run(command, cwd=TRAIN_ROOT, env=os.environ.copy(), check=True)
    events.emit('training', 'completed', includes_graph_postprocess=track != 'text')
    if gpu_only or track == 'text' or cfg.post_training_ablation:
        with _LOG.section('phase.inference_export', track=track):
            events.emit('inference_export','started',device=cfg.device)
            if track == 'text':
                from model_tracks.text_export import forward
                _,selected_text_model = forward(output,setup,return_model=True,device=cfg.device)
            else:
                from graph_tracks.config import GraphConfig
                from graph_tracks.infer import forward_outputs
                from graph_tracks.artifacts import name
                selected = list(output.rglob(name(track,'best_checkpoint.json')))
                if len(selected) != 1:
                    raise ValueError('ambiguous selected graph checkpoint')
                checkpoint = Path(json.loads(selected[0].read_text())['path'])
                settings['device'] = cfg.device
                settings.update(cfg.graph_execution_overrides())
                _,selected_graph_encoder = forward_outputs(checkpoint,TRAIN_ROOT/settings['listings'],TRAIN_ROOT/settings['pairs'],
                    output/(track+'__inference'),GraphConfig.model_validate(settings),
                    text_cache=TRAIN_ROOT/settings['text_cache'] if settings.get('text_cache') else None,
                    return_encoder=True)
            events.emit('inference_export','completed',device=cfg.device)
            if cfg.post_training_ablation:
                from model_tracks.staged_ablation import forward as forward_ablation
                if track == 'text':
                    from training.validation_inference import resolve_best_checkpoint
                    checkpoint,_ = resolve_best_checkpoint(output)
                if gpu_only and not (setup/'ablation_templates'/track/'request.json').is_file():
                    # A GPU session relies on the CPU data bundle shipping the
                    # templates; a bundle built before that contract would
                    # FileNotFoundError here, so skip with a named reason.
                    # Local sessions keep the loud read.
                    events.emit('attribute_ablation_export','skipped',device=cfg.device,
                                reason='bundle shipped no ablation templates')
                else:
                    events.emit('attribute_ablation_export','started',device=cfg.device)
                    forward_ablation(output,setup,track,checkpoint,text_model=selected_text_model if track == 'text' else None,device=cfg.device,
                        graph_encoder=selected_graph_encoder if track != 'text' else None)
                    events.emit('attribute_ablation_export','completed',device=cfg.device)
            if track == 'text':
                del selected_text_model
            else:
                del selected_graph_encoder
    if track == 'text' and not gpu_only:
        with _LOG.section('phase.postprocess', track=track, report_test=cfg.report_test):
            from model_tracks.text_report import complete
            events.emit('postprocess', 'started', report_test=cfg.report_test)
            complete(output, setup, device=cfg.device, report_test=cfg.report_test)
            events.emit('postprocess', 'completed')
        from model_tracks.incremental import ArtifactPublisher
        with _LOG.section('phase.incremental_publish', track=track):
            with ArtifactPublisher(output) as publisher:
                if publisher.enabled:
                    events.emit('publication', 'started', publication_owner='ArtifactPublisher')
                    publisher.submit('postprocess', [p for p in output.iterdir()
                        if p.name.startswith('text__') or p.name == 'profiles'])
                    events.emit('publication', 'submitted')
                else:
                    events.emit('publication', 'skipped', reason='incremental publication disabled')
            if publisher.enabled:
                events.emit('publication', 'context_closed',
                            detail='publisher context finished; remote receipts are in publication metadata')
    from model_tracks.resume import record_completion
    with _LOG.section('phase.completion', track=track):
        record_completion(output, track, postprocess_complete=not gpu_only)
        events.emit('completion', 'verified', inventory='track_inventory.json',
                    marker='track_complete.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--track', choices=['text', 'gnn_only', 'cascade'], required=True)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.config, args.track, args.run_tag, resume=args.resume)


if __name__ == '__main__':
    main()
