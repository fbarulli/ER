"""Single Colab supervisor for the trained lanes plus the cascade; workers own outputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from core.archive_reader import archive_sidecar
from core.bundle import bundle_spec
from core.run_log import RunLogger
from core.tracing import (
    SCOPE_ENTITY,
    flush_stage_trace,
    stage_trace,
)
from model_tracks.config import load_config
from model_tracks.parallel import (
    mps_environment,
    run_parallel,
    run_postprocess_track,
    split_tracks,
)
from model_tracks.preflight import preflight

_LOG = RunLogger(__name__)

#: The stage name this module owns in the ONE consolidated pipeline trace.
STAGE = "suite_run"

#: The module's trace writer: the shared shim's slot (``None`` until first use;
#: see :func:`core.tracing.stage_trace`), so importing this module never touches
#: the trace layout.
_TRACE = None


def trace():
    """The ONE writer for the ``suite_run`` stage of the current run.

    The supervisor emits selection, launch, completion, collection and
    publication rows for the ONE suite it owns into the same stage commit.
    """
    global _TRACE
    _TRACE = stage_trace(STAGE, _TRACE)
    return _TRACE


def flush_trace():
    """Commit this process's supervisor rows once; a no-op while empty."""
    return flush_stage_trace(_TRACE)


def run(config: Path, output: Path, run_tag: str, *, resume: bool = False) -> Path:
    import fcntl
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f'.{output.name}.suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('suite supervisor is already running') from exc
        if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
            raise ValueError('invalid run tag')
        if output.exists() and not resume:
            raise FileExistsError(output)
        result_suffix = '.' + load_config(config).result_archive_format
        if output.with_suffix(result_suffix).exists() and not resume:
            raise FileExistsError(output.with_suffix(result_suffix))
        if resume and not output.is_dir():
            raise FileNotFoundError('resume output directory does not exist')
        output.mkdir(parents=True, exist_ok=resume)
        from model_tracks.telemetry import WorkerEvents
        spec = bundle_spec()
        events = WorkerEvents(output, 'suite', run_tag, filename=spec.suite_events_file)
        events.emit('suite', 'starting', resume=resume, config=str(config), output=str(output))
        from model_tracks.resource_profile import ResourceProfile
        profile = ResourceProfile(output / 'resource_profile', load_config(config).profiling).start()
        events.resource_profile = profile
        sealed = False
        try:
            archive = _run(config, output, run_tag, resume=resume, events=events)
            events.emit('suite', 'complete', archive=str(archive))
            sealed = True
            flush_trace()
            return archive
        except BaseException as exc:
            import traceback
            events.emit('suite', 'failed', error_type=type(exc).__name__, error=str(exc),
                        failed_phase=events.last_phase, traceback=traceback.format_exc())
            trace().add(
                'run', 'failed',
                reason=f'the suite aborted with {type(exc).__name__}; the rows above show the last '
                       'step that ran',
                detail={'error_type': type(exc).__name__, 'error': str(exc),
                        'failed_phase': events.last_phase, 'resume': bool(resume)},
                source=str(config),
            )
            flush_trace()
            raise
        finally:
            profile.close()
            # Single-archive handoff: on success the suite event stream was
            # flushed into the result archive before sealing, so no second
            # `.events.jsonl` sidecar is produced (or downloaded). Only a run
            # that never sealed keeps its final event log beside the output.
            if not sealed:
                import shutil
                shutil.copyfile(events.path, output.with_suffix(spec.events_sidecar_suffix))


def _run_postprocess_track(config: Path, output: Path, run_tag: str, track: str,
                           env: dict, *, resume: bool, events) -> str:
    """Run one post-training combinator lane after the parallel barrier.

    The spawn mechanics (env, log, cwd, barrier exclusion) live in
    :func:`model_tracks.parallel.run_postprocess_track`, the one place a
    postprocess lane is launched; this supervisor only builds the command and
    records the run's rows.
    """
    command = [sys.executable, '-m', 'model_tracks.worker', '--config',
               str(config.resolve()), '--track', track, '--run-tag', f'{run_tag}-{track}']
    if resume:
        command.append('--resume')
    events.emit('worker_spawn', 'started', worker_track=track,
                log=str(output / f'{track}__worker.log'))
    run_postprocess_track(command, output, env, track, resume=resume)
    trace().add(
        'run', 'postprocess_lane',
        scope=SCOPE_ENTITY, key=track, in_count=1, out_count=1,
        reason='the postprocess combinator ran as its own worker process after the parallel phase, '
               'when the artifacts it consumes already exist',
        detail={'track': track, 'command': ' '.join(command), 'resume': bool(resume),
                'log': str(output / f'{track}__worker.log')},
        source=str(config),
    )
    return track


def _run(config: Path, output: Path, run_tag: str, *, resume: bool = False, events=None) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
        raise ValueError('invalid run tag')
    spec = bundle_spec()
    cfg = load_config(config)
    gpu_only = os.environ.get('ER_GPU_TRAINING_ONLY') == '1'
    # Structured training timings default into the run's output dir (owner
    # order 2026-10-07): the [timing] lines also stream, but timings.json/log
    # land under output/ so the result archive carries them back. An explicit
    # ER_TIMING_OUT/ER_TIMING_LOG (e.g. scripts/training_profile.py) still wins.
    os.environ.setdefault('ER_TIMING_OUT', str(output / 'logs' / 'timings.json'))
    os.environ.setdefault('ER_TIMING_LOG', str(output / 'logs' / 'timings.log'))
    if cfg.dvc_enabled and not gpu_only and not os.environ.get('DVC_API_KEY'):
        raise RuntimeError('DVC_API_KEY is required before training a publishing suite')
    events.emit('preflight', 'starting')
    with _LOG.section('phase.preflight', gpu_only=gpu_only):
        if gpu_only:
            from core.common import TRAIN_ROOT
            from core.bundle import BundleRole, manifest_name
            inputs = json.loads(
                (TRAIN_ROOT / manifest_name(BundleRole.inputs)).read_text())['preflight']
        else:
            # When this suite exports the baseline itself, the frozen embedding
            # cache is a declared pending input at preflight time: it is produced
            # by the export a few lines below.  Verifying it as missing-and-bound-to-a-checked
            # embedding request is what lets preflight run before the export instead
            # of demanding bytes that do not exist yet.
            inputs = preflight(config, allow_gpu_pending=cfg.post_training_ablation)
    # Reject an unusable parallel runtime before the baseline consumes GPU
    # time. MPS is a required capability for this suite, not a late fallback.
    if cfg.device == 'cuda':
        import shutil
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for all-track GPU run')
        if not shutil.which('nvidia-cuda-mps-control'):
            raise RuntimeError('true multi-process GPU parallelism requires NVIDIA MPS in this runtime')
        free, _ = torch.cuda.mem_get_info()
        required = sum(cfg.memory_reservations_gb.values()) + cfg.gpu_headroom_gb
        if cfg.memory_reservations_gb and required * 1024**3 > free:
            raise RuntimeError('measured combined worker memory exceeds available GPU memory')
    if gpu_only or cfg.post_training_ablation:
        from model_tracks.baseline_export import forward as forward_baseline
        from core.common import TRAIN_ROOT, resolve_model
        setup = (TRAIN_ROOT/cfg.setup_dir).resolve()
        with _LOG.section('phase.baseline_embedding', device=cfg.device):
            events.emit('baseline_embedding','started',device=cfg.device)
            baseline, baseline_model = forward_baseline(setup,Path(resolve_model(cfg.text_model)),device=cfg.device,return_model=True)
            # Keep this verified encoder through baseline ablation, then release
            # supervisor allocations before workers create their CUDA allocators.
            import torch
            import shutil
            baseline_output = output/'baseline'
            baseline_output.mkdir(exist_ok=True)
            shutil.copy2(baseline,baseline_output/baseline.name)
            events.emit('baseline_embedding','completed',device=cfg.device)
        # Ablation is not a training-session phase (owner order 2026-10-07):
        # the GPU ablation staging/forward block is removed entirely; ablation
        # lives in the CPU lanes. This deletes the ~205s
        # ablation_inputs.prepare_inputs burn the T4 box paid on 2026-10-07.
        del baseline_model
        torch.cuda.empty_cache()
    # Every data test runs here: once, on the machine that will train, after the
    # baseline export (the last producer of a gate input) and before the
    # barrier that releases any worker. One attestation then covers the exact
    # bytes the trained lanes consume, so the training path spends its
    # time training instead of re-arguing shared immutable inputs.
    from model_tracks.data_gate import validate as validate_data
    events.emit('data_gate', 'starting')
    with _LOG.section('phase.data_gate'):
        gate = validate_data(config, allow_gpu_pending=True,
                             suite_inputs=None if gpu_only else inputs)
        events.emit('data_gate', 'passed', tracks=gate.tracks, attestation=gate.attestation)
    events.emit('preflight', 'passed', inputs=inputs, device=cfg.device,
                epochs=cfg.epochs, report_test=cfg.report_test, publish=cfg.dvc_enabled)
    trace().add(
        'run', 'preflight',
        in_count=1, out_count=1,
        reason='preflight and the pre-training data gate passed on the machine that will train',
        detail={'run_tag': run_tag, 'device': cfg.device, 'gpu_only': gpu_only,
                'epochs': cfg.epochs, 'report_test': bool(cfg.report_test),
                'resume': bool(resume), 'dvc_enabled': bool(cfg.dvc_enabled),
                'post_training_ablation': bool(cfg.post_training_ablation),
                'data_gate_tracks': list(gate.tracks),
                'data_gate_attestation': str(gate.attestation)},
        source=str(config),
    )
    from model_tracks.resume import (TRACKS, TRAINING_TRACKS, POSTPROCESS_TRACKS,
                                     expected_postprocess, suite_identity, validate_suite, completed_track)
    identity = suite_identity(cfg, inputs, run_tag)
    if resume:
        validate_suite(output, identity)
        events.emit('resume', 'verified', provenance='frozen inputs, config and implementation')
    from core.common import TRAIN_ROOT
    import torch
    hardware = {'device': cfg.device, 'parallel_workers': len(TRAINING_TRACKS)}
    if cfg.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for all-track GPU run')
        free, total = torch.cuda.mem_get_info()
        hardware.update(name=torch.cuda.get_device_name(), total_bytes=total, free_bytes=free)
        required = sum(cfg.memory_reservations_gb.values()) + cfg.gpu_headroom_gb
        if cfg.memory_reservations_gb and required * 1024**3 > free:
            raise RuntimeError('measured combined worker memory exceeds available GPU memory')
    if not resume:
        from core.manifest import atomic_write_text
        from model_tracks.resume import TrainingInputBinding
        manifest = TrainingInputBinding.model_validate({
        'run_tag':run_tag, 'config':cfg.model_dump(), 'inputs':inputs, 'hardware':hardware,
        'resume_identity': identity,
        'worker_responsibility':('GPU train and checkpoint; local CPU owns postprocessing and reports'
                                 if gpu_only else 'train, checkpoint, postprocess, report own track'),
        'supervisor_responsibility':'barrier, MPS, child lifetime, collection',
        'shared_inputs':'read-only; text bundle CSVs materialized in text worker output',
        'cascade_mode':'text ranker retrieves, gnn_only scorer decides; no fused embedding'
        })
        atomic_write_text(output / spec.suite_manifest_file,
                          manifest.model_dump_json(indent=2, by_alias=True) + '\n')
    skipped = [track for track in TRACKS if resume and completed_track(
        output / track, track, postprocess_complete=expected_postprocess(track, gpu_only=gpu_only))]
    for track in skipped:
        events.emit('worker_selection', 'skipped', worker_track=track,
                    reason='completed artifacts verified against SHA256 inventory')
    trace().add(
        'run', 'worker_selection',
        in_count=len(TRACKS), out_count=len(TRACKS) - len(skipped),
        reason='a track whose completed artifacts verify against its SHA256 inventory is skipped '
               'instead of retrained',
        detail={'tracks': list(TRACKS), 'skipped': list(skipped),
                'trained_lanes': [track for track in TRAINING_TRACKS if track not in skipped],
                'postprocess_lanes': [track for track in POSTPROCESS_TRACKS if track not in skipped],
                'resume': bool(resume)},
        source='model_tracks.resume.completed_track',
    )
    trace().add_entities(
        'run.skipped_track', list(skipped),
        key_of=lambda track: track,
        reason_of=lambda track: 'completed_artifacts_verified',
        detail_of=lambda track: {'track': track, 'inventory': spec.inventory_file,
                                 'marker': spec.complete_file},
        source='model_tracks.resume.TRACKS',
    )
    # The trained lanes (text, gnn_only) run in parallel behind the start
    # barrier. The cascade trains nothing and consumes both trained artifacts,
    # so it runs sequentially after the parallel phase completes. The partition
    # is the suite taxonomy (``parallel.split_tracks``), not a per-callsite list.
    parallel_tracks, postprocess_tracks = split_tracks(
        track for track in TRACKS if track not in skipped)
    parallel_tracks, postprocess_tracks = list(parallel_tracks), list(postprocess_tracks)
    commands = {track: [sys.executable, '-m', 'model_tracks.worker', '--config', str(config.resolve()),
                        '--track', track, '--run-tag', f'{run_tag}-{track}'] + (['--resume'] if resume else [])
                for track in parallel_tracks}
    env = {**os.environ, 'PYTHONPATH':str(TRAIN_ROOT/'src'),
           'ER_SUITE_ATTEMPT': events.attempt,
           'ER_INCREMENTAL_DVC':'1' if cfg.dvc_enabled and not gpu_only else '0',
           'EUROMONITOR_DISABLE_DVC_CHECKPOINTS':'1' if gpu_only else os.environ.get('EUROMONITOR_DISABLE_DVC_CHECKPOINTS', '0'),
           'EUROMONITOR_REMOTE_TRAINING':'1' if gpu_only else os.environ.get('EUROMONITOR_REMOTE_TRAINING', '0'),
           'ER_DATA_GATE': gate.attestation,
           'ER_DATA_GATE_GPU_PENDING': '1',
           'ER_DATA_GATE_CONFIG': str(config.resolve()),
           'ER_TRAINING_PROFILE':'1' if cfg.profiling else '0'}
    with _LOG.section('phase.worker_launch', workers=len(commands) + len(postprocess_tracks), device=cfg.device):
        if not commands:
            result = {'mode': 'resume', 'workers': [], 'skipped_verified_tracks': skipped}
        elif cfg.device == 'cuda':
            share = max(1, 100 // len(commands))
            with mps_environment(output, thread_percentage=share) as mps_env:
                result = run_parallel(commands, output, {**env, **mps_env, 'PYTHONPATH':str(TRAIN_ROOT/'src')}, resume=resume)
        else:
            result = run_parallel(commands, output, env, resume=resume)
        for track in postprocess_tracks:
            result.setdefault('workers', []).append(
                _run_postprocess_track(config, output, run_tag, track, env,
                                       resume=resume, events=events))
    trace().add(
        'run', 'workers',
        in_count=len(commands) + len(postprocess_tracks),
        out_count=len(result.get('workers', [])),
        reason='the trained lanes run in parallel behind the start barrier; a postprocess '
               'combinator runs after it, when the artifacts it consumes exist',
        detail={'mode': result.get('mode'), 'parallel_lanes': len(commands),
                'postprocess_lanes': len(postprocess_tracks),
                'skipped_verified_tracks': list(skipped), 'device': cfg.device},
        source=str(output),
    )
    result['skipped_verified_tracks'] = skipped
    # Validate the current artifact generation, not just completion markers.
    with _LOG.section('phase.completion', tracks=len(TRACKS)):
        for track in TRACKS:
            if not completed_track(output / track, track,
                                   postprocess_complete=expected_postprocess(track, gpu_only=gpu_only)):
                trace().add(
                    'run', 'incomplete_track',
                    scope=SCOPE_ENTITY, key=track,
                    reason='the track does not satisfy its completion contract (markers and '
                           'artifacts verified against the SHA256 inventory)',
                    detail={'track': track, 'inventory': spec.inventory_file,
                            'marker': spec.complete_file, 'gpu_only': gpu_only},
                    source=str(output / track),
                )
                flush_trace()
                raise ValueError(f'incomplete track: {track}')
        trace().add(
            'run', 'completion_gate',
            in_count=len(TRACKS), out_count=len(TRACKS),
            reason='every track passes its completion contract before the result is collected',
            detail={'tracks': list(TRACKS), 'gpu_only': gpu_only,
                    'expected_postprocess': {track: bool(expected_postprocess(track, gpu_only=gpu_only))
                                             for track in TRACKS}},
            source='model_tracks.resume.completed_track',
        )
    if cfg.post_training_ablation and not gpu_only:
        from model_tracks.baseline_ablation import complete as complete_baseline
        from model_tracks.post_training_ablation import complete_saved
        from model_tracks.resume import record_completion
        # The frozen baseline report is optional: GPU sessions run no ablation
        # staging/forward (owner order 2026-10-07), so an archive produced there
        # ships no 'baseline/ablation' request. The local finalize
        # (bundle_steps.finalize) already skips it with a named reason; the
        # suite does the same instead of failing after every track has
        # completed and reported. The saved per-track ablation below is the part
        # that must always complete.
        baseline_request = output / 'baseline' / 'ablation' / spec.ablation_request_file
        if baseline_request.is_file():
            complete_baseline(output / 'baseline', setup, config=TRAIN_ROOT / cfg.ablation_config)
        else:
            trace().add(
                'run', 'baseline_ablation',
                reason='post-training ablation is enabled but the suite shipped no baseline '
                       'ablation; the frozen baseline report is skipped rather than refit',
                detail={'baseline': str(output / 'baseline'),
                        'request': str(baseline_request), 'present': False},
                source=str(output / 'baseline'),
            )
        complete_saved(output, cfg)
        for track in TRACKS:
            record_completion(output / track, track)
        trace().add(
            'run', 'ablation',
            in_count=len(TRACKS), out_count=len(TRACKS),
            reason='the suite completes the baseline and the saved attribute ablation before the '
                   'result is sealed, so the archive carries the frozen reports',
            detail={'baseline': str(output / 'baseline'),
                    'ablation_config': str(TRAIN_ROOT / cfg.ablation_config),
                    'tracks': list(TRACKS)},
            source=str(output / 'post_training_ablation.json'),
        )
    (output/'suite_result.json').write_text(json.dumps({'status':'ok', **result}, indent=2)+'\n')
    if getattr(events, 'resource_profile', None):
        events.resource_profile.close()
    events.emit('collection', 'starting', tracks=list(TRACKS), skipped_verified_tracks=skipped)
    archive_path = output.with_suffix('.' + cfg.result_archive_format)
    if archive_path.exists():
        if not resume:
            raise FileExistsError(archive_path)
        from model_tracks.resume import verify_suite_archive
        existing = verify_suite_archive(archive_path, output, run_tag, identity,
                                        gpu_only=gpu_only)
        archive_sha = existing.digest
        archive_sidecar(archive_path, spec.sha256_sidecar_suffix).write_text(archive_sha + '\n')
        events.emit('collection', 'verified', archive=str(archive_path), sha256=archive_sha,
                    reused=True)
        trace().add(
            'run', 'collection',
            in_count=len(existing.members()), out_count=len(existing.members()),
            reason='a sealed archive for this run already exists and verifies against the current '
                   'result tree, so it is reused instead of rewritten',
            detail={'archive': str(archive_path), 'digest': archive_sha, 'reused': True,
                    'members': len(existing.members()), 'format': cfg.result_archive_format},
            source=str(archive_path),
        )
        with _LOG.section('phase.publication', reused=True):
            if cfg.dvc_enabled and not gpu_only:
                from model_tracks.local_complete import _publish
                events.emit('publication', 'starting', archive=str(archive_path))
                # `existing` is the verified boundary handle for this archive:
                # publication consumes it instead of re-loading the bytes
                # (one integrity check per archive per VM crossing).
                _publish(archive_path, cfg, run_tag, ablation_done=cfg.post_training_ablation,
                         destination=output, bundle=existing)
                events.emit('publication', 'complete')
                trace().add(
                    'run', 'publication',
                    reason='the verified archive is published from the reused result tree',
                    detail={'archive': str(archive_path), 'published': True, 'reused_archive': True,
                            'dvc_enabled': bool(cfg.dvc_enabled)},
                    source=str(archive_path),
                )
            else:
                events.emit('publication', 'skipped', reason='publication disabled in suite config')
                trace().add(
                    'run', 'publication',
                    reason='publication is disabled in the suite config, so nothing is pushed',
                    detail={'archive': str(archive_path), 'published': False, 'reused_archive': True,
                            'dvc_enabled': bool(cfg.dvc_enabled)},
                    source=str(archive_path),
                )
        flush_trace()
        return archive_path
    from core.bundle import Bundle, BundleRole
    # The timing surfaces default into output/logs (bffadd3) and are appended
    # to by this very collection step, so the archive would hash bytes that
    # change mid-write ("archive integrity mismatch: logs/timings.log").
    # Freeze them out of the output tree first; the archived copies stay.
    for _variable in ('ER_TIMING_OUT', 'ER_TIMING_LOG'):
        _bound = os.environ.get(_variable)
        if _bound and Path(_bound).resolve().is_relative_to(output.resolve()):
            os.environ[_variable] = str(
                Path(tempfile.gettempdir()) / f'er_frozen_{Path(_bound).name}')
    publication = cfg.dvc_enabled and not gpu_only
    if not publication:
        events.emit('publication', 'skipped', reason='publication disabled in suite config')
    # The result role owns the member set (selected checkpoint only) and the
    # sealing writer; this stage neither re-derives the predicate nor re-hashes.
    result_bundle = Bundle.from_directory(output, BundleRole.result)
    files = result_bundle.collect_result_members()
    tree_files = result_bundle.members()
    trace().add(
        'run', 'collection',
        in_count=len(tree_files), out_count=len(files),
        reason='the result role decides the sealed member set: the selected checkpoint only, so '
               'every other epoch, optimizer state and resume tree is dropped',
        detail={'output': str(output), 'tree_files': len(tree_files),
                'selected_members': len(files), 'format': cfg.result_archive_format,
                'archive': str(archive_path), 'publication': publication,
                'run_tag': run_tag},
        source=str(output),
    )
    # Single-archive handoff (owner #5): the final events are emitted (and
    # fsynced by WorkerEvents) BEFORE the seal, so the one downloaded archive
    # carries the complete stream including the publication decision. The
    # post-seal transport token travels in the .sha256 sidecar.
    events.emit('collection', 'sealing', tracks=list(TRACKS), files=len(files),
                publication=publication)
    with _LOG.section('phase.archive_write', files=len(files), format=cfg.result_archive_format):
        sealed = result_bundle.seal_result(archive_path,
                                           metadata={spec.run_tag_key: run_tag},
                                           profile=cfg.profiling)
        archive_sidecar(archive_path, spec.sha256_sidecar_suffix).write_text(sealed.digest + '\n')
        events.emit('collection', 'complete', archive=str(archive_path), sha256=sealed.digest,
                    bytes=archive_path.stat().st_size)
        trace().add(
            'run', 'seal',
            in_count=len(files), out_count=len(files),
            reason='one sealed result archive is written by the role writer, which hashes each '
                   'member exactly once while writing and captures the whole-file digest there',
            detail={'archive': str(archive_path), 'digest': sealed.digest,
                    'bytes': archive_path.stat().st_size, 'members': len(files),
                    'format': cfg.result_archive_format, 'run_tag': run_tag,
                    'profile': bool(cfg.profiling)},
            source=str(archive_path),
        )
    with _LOG.section('phase.publication', reused=False):
        if publication:
            from model_tracks.local_complete import _publish
            events.emit('publication', 'starting', archive=str(archive_path))
            # `sealed` is the writer's own handle: its digest came from the
            # sealing pass, so publication reuses it rather than re-loading the
            # archive it just wrote (one integrity check per VM crossing).
            _publish(archive_path, cfg, run_tag, ablation_done=cfg.post_training_ablation,
                     destination=output, bundle=sealed)
            events.emit('publication', 'complete')
            trace().add(
                'run', 'publication',
                reason='the freshly sealed archive is published to its configured sinks',
                detail={'archive': str(archive_path), 'published': True, 'reused_archive': False,
                        'dvc_enabled': bool(cfg.dvc_enabled)},
                source=str(archive_path),
            )
        else:
            trace().add(
                'run', 'publication',
                reason='publication is disabled in the suite config, so nothing is pushed',
                detail={'archive': str(archive_path), 'published': False, 'reused_archive': False,
                        'dvc_enabled': bool(cfg.dvc_enabled)},
                source=str(archive_path),
            )
    flush_trace()
    return archive_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-tag',required=True)
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args()
    print(run(args.config,args.output,args.run_tag,resume=args.resume))


if __name__ == '__main__':
    main()
