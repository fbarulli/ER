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
        return _run(config, output, run_tag, resume=resume)


def _run(config: Path, output: Path, run_tag: str, *, resume: bool = False) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
        raise ValueError('invalid run tag')
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and not output.is_dir():
        raise FileNotFoundError('resume output directory does not exist')
    cfg = load_config(config)
    if cfg.dvc_enabled and not os.environ.get('DVC_API_KEY'):
        raise RuntimeError('DVC_API_KEY is required before training a publishing suite')
    inputs = preflight(config)
    from model_tracks.resume import TRACKS, suite_identity, validate_suite, completed_track
    identity = suite_identity(cfg, inputs, run_tag)
    if resume:
        validate_suite(output, identity)
    output.mkdir(parents=True, exist_ok=resume)
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
        'worker_responsibility':'train, checkpoint, postprocess, report own track',
        'supervisor_responsibility':'barrier, MPS, child lifetime, collection',
        'shared_inputs':'read-only; text bundle CSVs materialized in text worker output',
        'hybrid_text_checkpoint':'frozen prepared baseline; no dependency on concurrent text worker'
    }, indent=2) + '\n')
    skipped = [track for track in TRACKS if resume and completed_track(output / track, track)]
    commands = {track: [sys.executable, '-m', 'model_tracks.worker', '--config', str(config.resolve()),
                        '--track', track, '--run-tag', f'{run_tag}-{track}'] + (['--resume'] if resume else [])
                for track in TRACKS if track not in skipped}
    env = {**os.environ, 'PYTHONPATH':str(TRAIN_ROOT/'src'),
           'ER_INCREMENTAL_DVC':'1' if cfg.dvc_enabled else '0',
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
        if marker != {'track':track, 'status':'ok', 'postprocess_complete':True}:
            raise ValueError(f'incomplete track: {track}')
    (output/'suite_result.json').write_text(json.dumps({'status':'ok', **result}, indent=2)+'\n')
    archive_path = output.with_suffix('.zip')
    if archive_path.exists():
        from core.portable_archive import verify_archive
        if not resume or verify_archive(archive_path, 'suite_bundle_manifest.json').get('run_tag') != run_tag:
            raise ValueError('existing archive belongs to a different suite')
        archive_path.with_suffix('.sha256').write_text(file_hash(archive_path) + '\n')
        if cfg.dvc_enabled:
            from model_tracks.publish import persist_results
            persist_results(archive_path, run_tag)
        return archive_path
    files = {p.relative_to(output).as_posix(): p for p in output.rglob('*')
             if p.is_file() and not p.is_symlink() and not any(part in
                 {'wandb', 'mlruns', 'mps_pipe', 'mps_log', '.git'} for part in p.relative_to(output).parts)
             and not ('_artifact_publications' in p.relative_to(output).parts and p.suffix != '.json')
             and not any(part.endswith('.publication') for part in p.relative_to(output).parts)
             and p.name not in {'.env','config.local'}
             and not any(part.endswith('__payload') for part in p.relative_to(output).parts)
             and not ('.dvc' in p.relative_to(output).parts and 'cache' in p.relative_to(output).parts)}
    from core.portable_archive import write_archive
    write_archive(archive_path,files,manifest_name='suite_bundle_manifest.json',metadata={'run_tag':run_tag})
    archive_path.with_suffix('.sha256').write_text(file_hash(archive_path)+'\n')
    if cfg.dvc_enabled:
        from model_tracks.publish import persist_results
        persist_results(archive_path, run_tag)
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
