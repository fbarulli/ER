"""Single Colab supervisor for all three model tracks; workers own outputs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys

from model_tracks.config import load_config
from model_tracks.parallel import mps_environment, run_parallel
from model_tracks.preflight import preflight


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
        if resume and not output.is_dir():
            raise FileNotFoundError('resume output directory does not exist')
        output.mkdir(parents=True, exist_ok=resume)
        from model_tracks.telemetry import WorkerEvents
        events = WorkerEvents(output, 'suite', run_tag, filename='suite_events.jsonl')
        events.emit('suite', 'starting', resume=resume, config=str(config), output=str(output))
        try:
            archive = _run(config, output, run_tag, resume=resume, events=events)
            events.emit('suite', 'complete', archive=str(archive))
            return archive
        except BaseException as exc:
            import traceback
            events.emit('suite', 'failed', error_type=type(exc).__name__, error=str(exc),
                        failed_phase=events.last_phase, traceback=traceback.format_exc())
            raise
        finally:
            # Archive contents precede collection/publication. Preserve their
            # final outcomes beside the archive without rewriting its digest.
            import shutil
            shutil.copyfile(events.path, output.with_suffix('.events.jsonl'))


def _run(config: Path, output: Path, run_tag: str, *, resume: bool = False, events=None) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
        raise ValueError('invalid run tag')
    cfg = load_config(config)
    gpu_only = os.environ.get('ER_GPU_TRAINING_ONLY') == '1'
    if cfg.dvc_enabled and not gpu_only and not os.environ.get('DVC_API_KEY'):
        raise RuntimeError('DVC_API_KEY is required before training a publishing suite')
    events.emit('preflight', 'starting')
    if gpu_only:
        from core.common import TRAIN_ROOT
        inputs = json.loads((TRAIN_ROOT / 'model_tracks_package.json').read_text())['preflight']
    else:
        inputs = preflight(config)
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
    if gpu_only:
        from model_tracks.baseline_export import forward as forward_baseline
        from core.common import resolve_model
        setup = (TRAIN_ROOT/cfg.setup_dir).resolve()
        events.emit('baseline_embedding','started',device=cfg.device)
        baseline = forward_baseline(setup,Path(resolve_model(cfg.text_model)),device=cfg.device)
        # The baseline model and features are now out of scope. Release this
        # supervisor's cached allocations before the three child workers
        # establish their independent CUDA allocators.
        import torch
        torch.cuda.empty_cache()
        import shutil
        baseline_output = output/'baseline'
        baseline_output.mkdir(exist_ok=True)
        shutil.copy2(baseline,baseline_output/baseline.name)
        events.emit('baseline_embedding','completed',device=cfg.device)
        if cfg.post_training_ablation:
            from model_tracks.baseline_ablation import forward as forward_baseline_ablation
            events.emit('baseline_ablation','started',device=cfg.device)
            forward_baseline_ablation(baseline_output,setup,Path(resolve_model(cfg.text_model)),device=cfg.device)
            torch.cuda.empty_cache()
            events.emit('baseline_ablation','completed',device=cfg.device)
    events.emit('preflight', 'passed', inputs=inputs, device=cfg.device,
                epochs=cfg.epochs, report_test=cfg.report_test, publish=cfg.dvc_enabled)
    from model_tracks.resume import TRACKS, suite_identity, validate_suite, completed_track
    identity = suite_identity(cfg, inputs, run_tag)
    if resume:
        validate_suite(output, identity)
        events.emit('resume', 'verified', provenance='frozen inputs, config and implementation')
    from core.common import TRAIN_ROOT
    from graph_tracks.data import file_hash
    import torch
    hardware = {'device': cfg.device, 'parallel_workers': 3}
    if cfg.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for all-track GPU run')
        free, total = torch.cuda.mem_get_info()
        hardware.update(name=torch.cuda.get_device_name(), total_bytes=total, free_bytes=free)
        required = sum(cfg.memory_reservations_gb.values()) + cfg.gpu_headroom_gb
        if cfg.memory_reservations_gb and required * 1024**3 > free:
            raise RuntimeError('measured combined worker memory exceeds available GPU memory')
    if not resume:
        (output / 'suite_manifest.json').write_text(json.dumps({
        'run_tag':run_tag, 'config':cfg.model_dump(), 'inputs':inputs, 'hardware':hardware,
        'resume_identity': identity,
        'worker_responsibility':('GPU train and checkpoint; local CPU owns postprocessing and reports'
                                 if gpu_only else 'train, checkpoint, postprocess, report own track'),
        'supervisor_responsibility':'barrier, MPS, child lifetime, collection',
        'shared_inputs':'read-only; text bundle CSVs materialized in text worker output',
        'hybrid_text_checkpoint':'frozen prepared baseline; no dependency on concurrent text worker'
    }, indent=2) + '\n')
    skipped = [track for track in TRACKS if resume and completed_track(output / track, track,
                                                                   postprocess_complete=not gpu_only)]
    for track in skipped:
        events.emit('worker_selection', 'skipped', worker_track=track,
                    reason='completed artifacts verified against SHA256 inventory')
    commands = {track: [sys.executable, '-m', 'model_tracks.worker', '--config', str(config.resolve()),
                        '--track', track, '--run-tag', f'{run_tag}-{track}'] + (['--resume'] if resume else [])
                for track in TRACKS if track not in skipped}
    env = {**os.environ, 'PYTHONPATH':str(TRAIN_ROOT/'src'),
           'ER_SUITE_ATTEMPT': events.attempt,
           'ER_INCREMENTAL_DVC':'1' if cfg.dvc_enabled and not gpu_only else '0',
           'EUROMONITOR_DISABLE_DVC_CHECKPOINTS':'1' if gpu_only else os.environ.get('EUROMONITOR_DISABLE_DVC_CHECKPOINTS', '0'),
           'EUROMONITOR_REMOTE_TRAINING':'1' if gpu_only else os.environ.get('EUROMONITOR_REMOTE_TRAINING', '0'),
           'ER_TRAINING_PROFILE':'1' if cfg.profiling else '0'}
    if not commands:
        result = {'mode': 'resume', 'workers': [], 'skipped_verified_tracks': skipped}
    elif cfg.device == 'cuda':
        with mps_environment(output) as mps_env:
            result = run_parallel(commands, output, {**env, **mps_env, 'PYTHONPATH':str(TRAIN_ROOT/'src')}, resume=resume)
    else:
        result = run_parallel(commands, output, env, resume=resume)
    result['skipped_verified_tracks'] = skipped
    # Every worker must finish its report before the shared run is complete.
    for track in TRACKS:
        marker = json.loads((output/track/'track_complete.json').read_text())
        if marker != {'track':track, 'status':'ok', 'postprocess_complete':not gpu_only}:
            raise ValueError(f'incomplete track: {track}')
    (output/'suite_result.json').write_text(json.dumps({'status':'ok', **result}, indent=2)+'\n')
    events.emit('collection', 'starting', tracks=list(TRACKS), skipped_verified_tracks=skipped)
    archive_path = output.with_suffix('.zip')
    if archive_path.exists():
        from core.portable_archive import verify_archive
        if not resume or verify_archive(archive_path, 'suite_bundle_manifest.json').get('run_tag') != run_tag:
            raise ValueError('existing archive belongs to a different suite')
        archive_path.with_suffix('.sha256').write_text(file_hash(archive_path) + '\n')
        events.emit('collection', 'verified', archive=str(archive_path), sha256=file_hash(archive_path),
                    reused=True)
        if cfg.dvc_enabled and not gpu_only:
            from model_tracks.publish import persist_results
            events.emit('publication', 'starting', archive=str(archive_path))
            persist_results(archive_path, run_tag)
            events.emit('publication', 'complete')
        else:
            events.emit('publication', 'skipped', reason='publication disabled in suite config')
        return archive_path
    from core.portable_archive import RESULT_ARCHIVE_EXCLUDED_DIRS
    files = {p.relative_to(output).as_posix(): p for p in output.rglob('*')
             if p.is_file() and not p.is_symlink() and not any(part in
                 RESULT_ARCHIVE_EXCLUDED_DIRS for part in p.relative_to(output).parts)
             and not ('_artifact_publications' in p.relative_to(output).parts and p.suffix != '.json')
             and not any(part.endswith('.publication') for part in p.relative_to(output).parts)
             and p.name not in {'.env','config.local'}
             and not any(part.endswith('__payload') for part in p.relative_to(output).parts)
             and not ('.dvc' in p.relative_to(output).parts and 'cache' in p.relative_to(output).parts)}
    from core.portable_archive import write_archive
    write_archive(archive_path,files,manifest_name='suite_bundle_manifest.json',metadata={'run_tag':run_tag})
    archive_path.with_suffix('.sha256').write_text(file_hash(archive_path)+'\n')
    events.emit('collection', 'complete', archive=str(archive_path), sha256=file_hash(archive_path),
                bytes=archive_path.stat().st_size)
    if cfg.dvc_enabled and not gpu_only:
        from model_tracks.publish import persist_results
        events.emit('publication', 'starting', archive=str(archive_path))
        persist_results(archive_path, run_tag)
        events.emit('publication', 'complete')
    else:
        events.emit('publication', 'skipped', reason='publication disabled in suite config')
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
