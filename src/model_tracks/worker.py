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
from core.tracing import SCOPE_ENTITY, flush_stage_trace, stage_trace
from model_tracks.config import load_config
from model_tracks.parallel import wait_for_start

_LOG = RunLogger(__name__)


def _setup_layout():
    """The declared prepared-setup layout (training.preparation.graph_setup)."""
    from core.common import training_cfg
    return training_cfg().preparation.graph_setup


#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "worker"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout.
_TRACE = None


def trace():
    """The ONE writer for the ``worker`` stage of the current run.

    One worker process owns one track, so its rows accumulate into one stage
    commit.
    """
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's worker rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


def _spec():
    """The bundle contract from config (single source for member names)."""
    from core.bundle import _bundle_spec
    return _bundle_spec()


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


def _catalog_vectors(track_root: Path, track: str) -> tuple[Path, bool]:
    """The trained lane's catalog embedding export under its own results tree.

    A lane writes its vectors through the saved forward export (the export
    manifest is the evidence), and some flows also root them beside the track's
    own artifacts. Both are the same frozen catalog vectors, so the direct name
    wins and the manifest-scoped search is the fallback; returning the searched
    path even when nothing matched keeps the caller's failure message naming a
    real location.
    """
    from graph_tracks.artifacts import name
    direct = track_root / name(track, 'vectors.npz')
    if direct.is_file():
        return direct, True
    for manifest in sorted(track_root.rglob(name(track, 'export_manifest.json'))):
        vectors = manifest.parent / name(track, 'vectors.npz')
        if vectors.is_file():
            return vectors, True
    remaining = sorted(track_root.rglob(name(track, 'vectors.npz')))
    if remaining:
        return remaining[0], True
    return direct, False


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
    text_vectors, text_vectors_found = _catalog_vectors(text_root, 'text')
    gnn_vectors, gnn_vectors_found = _catalog_vectors(gnn_root, 'gnn_only')
    gnn_checkpoint = Path(lane.gnn_checkpoint) if lane.gnn_checkpoint else None
    if gnn_checkpoint is None or not gnn_checkpoint.is_file():
        # The bundle's role contract owns checkpoint selection: the marker
        # records the selected graph model, and the bundle locates the member
        # by name under the track (a remote machine's absolute path does not
        # survive transport).
        from core.bundle import Bundle, BundleRole
        gnn_checkpoint = Bundle.from_directory(results, BundleRole.result).checkpoint('gnn_only')
    if not text_index.is_dir():
        raise FileNotFoundError(f'cascade text ranker index missing: {text_index}')
    if not text_vectors_found:
        raise FileNotFoundError(f'cascade text ranker vectors missing: {text_vectors}')
    if not gnn_vectors_found:
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
    absent = sorted(sku_id for sku_id in relevant if sku_id not in lookup)
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
    trace().add(
        "cascade", "queries",
        in_count=len(relevant), out_count=len(rows_ranked),
        reason='a query is decidable only when its sku_id exists in the trained catalog export; '
               'a relevant id the catalog does not carry is dropped, and the dropped ids are '
               'listed at entity grain below',
        detail={'relevant_queries': len(relevant), 'decided_queries': len(rows_ranked),
                'absent_from_catalog': len(absent), 'catalog_ids': len(ids),
                'retrieval_k': k},
        source=str(artifacts['gnn_vectors']),
    )
    # The EXACT census of the dropped queries is the GROUP row; the entity rows
    # name each sku_id and why it has no decision.
    trace().add_entities(
        "cascade.query_dropped", absent,
        key_of=lambda sku_id: sku_id,
        reason_of=lambda sku_id: 'sku_absent_from_trained_catalog',
        detail_of=lambda sku_id: {'sku_id': sku_id,
                                  'catalog': str(artifacts['gnn_vectors'])},
        source=str(artifacts['gnn_vectors']),
    )
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
    checkpoint identity is the trained gnn_only scorer it consumes, and its two
    role metrics are passed explicitly from the cascade report that
    ``graph_tracks.report.report_cascade`` just wrote.
    """
    from graph_tracks.artifacts import name
    from graph_tracks.data import file_hash
    from graph_tracks.report_manifest import build as build_manifest, write as write_manifest
    from core.common import TRAIN_ROOT
    listings = (TRAIN_ROOT / lane.listings).resolve()
    pairs = Path(lane.pairs)
    if not pairs.is_absolute():
        pairs = (TRAIN_ROOT / pairs).resolve()
    roles = None
    report_path = output / name('cascade', 'cascade_report.json')
    if report_path.is_file():
        roles = json.loads(report_path.read_text(encoding='utf-8')).get('roles')
    manifest = build_manifest(
        track='cascade', checkpoint=str(artifacts['gnn_checkpoint']),
        checkpoint_sha256=file_hash(artifacts['gnn_checkpoint']),
        listings_sha256=file_hash(listings), pairs_sha256=file_hash(pairs),
        threshold=0.5, threshold_source='dev_youden', test_reported=bool(lane.report_test),
        model_selection='dev_pr_auc', retrieval_ks=list(lane.retrieval_ks), roles=roles)
    write_manifest(output / name('cascade', 'report_manifest.json'), manifest)


def _run_cascade(cfg, setup: Path, output: Path, events):
    """The cascade lane: validate inputs, compose roles, report both roles."""
    from graph_tracks.report import report_cascade
    from model_tracks.resume import record_completion
    spec = _spec()
    lane = _cascade_lane(setup)
    events.emit('input_validation', 'configured', device=lane.device,
                worker_config=str(setup / 'cascade.yaml'), trains_nothing=True)
    artifacts = _cascade_artifacts(output.parent, lane)
    events.emit('input_validation', 'completed', text_index=str(artifacts['text_index']),
                gnn_checkpoint=str(artifacts['gnn_checkpoint']))
    trace().add(
        "cascade", "inputs",
        scope=SCOPE_ENTITY, key="cascade", in_count=1, out_count=1,
        reason='the cascade trains nothing; it consumes the trained text ranker index and the '
               'trained gnn_only scorer checkpoint located under their own tracks',
        detail={'text_index': str(artifacts['text_index']),
                'text_vectors': str(artifacts['text_vectors']),
                'gnn_vectors': str(artifacts['gnn_vectors']),
                'gnn_checkpoint': str(artifacts['gnn_checkpoint'])},
        source=str(artifacts['gnn_checkpoint']),
    )
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
        events.emit('completion', 'verified', inventory=spec.inventory_file,
                    marker=spec.complete_file)
    trace().add(
        "cascade", "completed",
        # A UNIT row: the composed report is one artifact regardless of how many
        # queries fed it (the query count is stated in the detail).
        scope=SCOPE_ENTITY, key="cascade", in_count=None, out_count=1,
        reason='the cascade report and its calibrated manifest are written from the composed roles',
        detail={'output': str(output), 'queries': len(relevant),
                'retrieval_ks': list(sorted(set(lane.retrieval_ks) | {1}))},
        source=str(output),
    )


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
        trace().add(
            "failure", "aborted",
            scope=SCOPE_ENTITY, key=track,
            reason=f'the worker aborted with {type(exc).__name__}; the rows above show the last '
                   'step that ran for this track',
            detail={'track': track, 'error_type': type(exc).__name__, 'error': str(exc),
                    'failed_phase': events.last_phase},
            source=str(config),
        )
        flush_trace()
        raise
    flush_trace()


def _run(config: Path, track: str, run_tag: str, *, resume: bool, events):
    from core.common import TRAIN_ROOT
    spec = _spec()
    cfg = load_config(config)
    gpu_only = os.environ.get('ER_GPU_TRAINING_ONLY') == '1'
    setup = (TRAIN_ROOT / cfg.setup_dir).resolve()
    layout = _setup_layout()
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
                   '--shared-training-data', str(setup / layout.shared_training_data),
                   '--training-binding', str(setup / layout.text_training_binding),
                   '--model', cfg.text_model, '--epochs', str(cfg.epochs),
                   '--payload', manifest.payload_variant, '--run-tag', run_tag,
                   '--device', cfg.device,
                   '--report-test' if cfg.report_test else '--no-report-test']
        if resume and any((output / spec.checkpoint_dir).rglob(spec.trainer_state_file)):
            command.append('--resume')
        setup_manifest = json.loads((setup / layout.manifest).read_text())
        if setup_manifest.get('smoke'):
            command.extend(['--sample', str(setup_manifest['source_listing_count'])])
    else:
        from graph_tracks.config import GraphConfig
        settings = graph_worker_settings(setup, cfg, track, gpu_only=gpu_only)
        worker_config = output / spec.worker_config_file
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
    trace().add(
        "command", "prepared",
        scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
        reason='the adapter prepares the exact command the trainer runs; the trainer owns full '
               'payload loading and validation after the barrier',
        detail={'track': track, 'command': shlex.join(command), 'resume_requested': bool(resume),
                'checkpoint_resume': '--resume' in command,
                'checkpoint': (command[command.index('--resume') + 1]
                               if track != 'text' and '--resume' in command else None),
                'sample': (int(command[command.index('--sample') + 1])
                           if '--sample' in command else None),
                'device': cfg.device, 'gpu_only': gpu_only},
        source=str(config),
    )
    events.emit('barrier', 'waiting', barrier=os.environ['ER_TRACK_BARRIER'])
    wait_for_start(Path(os.environ['ER_TRACK_BARRIER']), track)
    events.emit('barrier', 'released')
    trace().add(
        "barrier", "released",
        scope=SCOPE_ENTITY, key=track,
        reason='the track released the shared start barrier, so all parallel lanes begin together',
        detail={'track': track, 'barrier': os.environ['ER_TRACK_BARRIER'],
                'device': cfg.device},
        source=str(output),
    )
    events.emit('training', 'started', includes_graph_postprocess=track != 'text')
    with _LOG.section('phase.training', track=track):
        subprocess.run(command, cwd=TRAIN_ROOT, env=os.environ.copy(), check=True)
    events.emit('training', 'completed', includes_graph_postprocess=track != 'text')
    trace().add(
        "training", "completed",
        scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
        reason='the track trainer subprocess exited successfully',
        detail={'track': track, 'run_tag': run_tag,
                'includes_graph_postprocess': track != 'text', 'device': cfg.device},
        source=str(output),
    )
    if gpu_only or track == 'text' or cfg.post_training_ablation:
        with _LOG.section('phase.inference_export', track=track):
            events.emit('inference_export','started',device=cfg.device)
            if track == 'text':
                from model_tracks.text_export import forward
                _,selected_text_model = forward(output,setup,return_model=True,device=cfg.device)
            else:
                from graph_tracks.config import GraphConfig
                from graph_tracks.infer import forward_outputs
                from core.bundle import Bundle, BundleRole
                checkpoint = Bundle.from_directory(output, BundleRole.result).checkpoint(track)
                if checkpoint is None or not checkpoint.is_file():
                    raise ValueError('selected graph checkpoint unavailable')
                settings['device'] = cfg.device
                settings.update(cfg.graph_execution_overrides())
                _,selected_graph_encoder = forward_outputs(checkpoint,TRAIN_ROOT/settings['listings'],TRAIN_ROOT/settings['pairs'],
                    output/(track+'__inference'),GraphConfig.model_validate(settings),
                    return_encoder=True)
            events.emit('inference_export','completed',device=cfg.device)
            trace().add(
                "inference_export", "completed",
                scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
                reason='the selected model scores the prepared catalog, so the ablation lane can '
                       'consume saved vectors instead of re-running inference',
                detail={'track': track, 'device': cfg.device,
                        'checkpoint': (str(checkpoint) if track != 'text' else None),
                        'selected_by': ('trainer best-metric marker' if track == 'text'
                                        else 'bundle role contract'),
                        'post_training_ablation': bool(cfg.post_training_ablation)},
                source=str(output / f'{track}__inference'),
            )
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
                    trace().add(
                        "attribute_ablation_export", "skipped",
                        scope=SCOPE_ENTITY, key=track,
                        reason='bundle_shipped_no_ablation_templates',
                        detail={'track': track, 'device': cfg.device,
                                'template': str(setup/'ablation_templates'/track/'request.json'),
                                'gpu_only': gpu_only},
                        source=str(setup/'ablation_templates'),
                    )
                else:
                    events.emit('attribute_ablation_export','started',device=cfg.device)
                    forward_ablation(output,setup,track,checkpoint,text_model=selected_text_model if track == 'text' else None,device=cfg.device,
                        graph_encoder=selected_graph_encoder if track != 'text' else None)
                    events.emit('attribute_ablation_export','completed',device=cfg.device)
                    trace().add(
                        "attribute_ablation_export", "completed",
                        scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
                        reason='the frozen template is bound to the selected checkpoint and its '
                               'vectors are encoded for the saved ablation',
                        detail={'track': track, 'device': cfg.device,
                                'checkpoint': str(checkpoint),
                                'request': str(output/'ablation/request.json')},
                        source=str(output/'ablation/request.json'),
                    )
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
            trace().add(
                "postprocess", "completed",
                scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
                reason='the text lane reported on this machine, so local completion finds the '
                       'track already postprocessed',
                detail={'track': track, 'device': cfg.device,
                        'report_test': bool(cfg.report_test)},
                source=str(output),
            )
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
        events.emit('completion', 'verified', inventory=spec.inventory_file,
                    marker=spec.complete_file)
        trace().add(
            "completion", "verified",
            scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
            reason='the track marker and its artifact inventory satisfy the completion contract',
            detail={'track': track, 'postprocess_complete': not gpu_only,
                    'inventory': spec.inventory_file, 'marker': spec.complete_file,
                    'gpu_only': gpu_only},
            source=str(output),
        )


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
