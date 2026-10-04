"""One supervisor, three simultaneous model workers and a shared start barrier."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


@contextmanager
def mps_environment(root: Path):
    control = shutil.which('nvidia-cuda-mps-control')
    if not control:
        raise RuntimeError('true multi-process GPU parallelism requires NVIDIA MPS in this runtime')
    attempt = uuid.uuid4().hex
    pipes, logs = root / 'mps_pipe' / attempt, root / 'mps_log' / attempt
    pipes.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'CUDA_MPS_PIPE_DIRECTORY': str(pipes.resolve()),
           'CUDA_MPS_LOG_DIRECTORY': str(logs.resolve())}
    subprocess.run([control, '-d'], env=env, check=True, timeout=30)
    try:
        yield env
    finally:
        subprocess.run([control], input='quit\n', text=True, env=env, check=True, timeout=30)
        shutil.rmtree(pipes)  # Only this verified-stopped attempt owns these pipes.


def wait_for_start(root: Path, track: str, timeout: float = 600):
    (root / f'{track}.ready').write_text(str(os.getpid()))
    started = time.monotonic()
    while not (root / 'start').exists():
        if time.monotonic() - started > timeout:
            raise TimeoutError('all-track start barrier timed out')
        time.sleep(.1)


def run_parallel(commands: dict[str, list[str]], root: Path, env: dict,
                 *, timeout: float = 14400, barrier_timeout: float = 600,
                 resume: bool = False):
    if not commands or set(commands) - {'text', 'gnn_only', 'hybrid'}:
        raise ValueError('suite workers must be known unfinished tracks')
    # Each attempt has a fresh barrier; a restored start file cannot release
    # a resumed worker before its companions have loaded.
    barrier = root / ('barrier' if not resume else f'barrier_resume_{time.time_ns()}')
    barrier.mkdir(parents=True, exist_ok=False)
    processes, handles = {}, []
    from model_tracks.live_logs import WorkerLogs
    logs = WorkerLogs(root, commands, from_end=resume)
    from model_tracks.telemetry import WorkerEvents
    events = WorkerEvents(root, 'suite', root.name, filename='suite_events.jsonl')
    if env.get('ER_SUITE_ATTEMPT'):
        events.attempt = env['ER_SUITE_ATTEMPT']
    events.emit('workers', 'starting', tracks=list(commands), resume=resume,
                barrier=str(barrier), timeout_seconds=timeout, barrier_timeout_seconds=barrier_timeout)
    cpu_budget = os.cpu_count() or 1
    if hasattr(os, 'sched_getaffinity'):
        cpu_budget = min(cpu_budget, len(os.sched_getaffinity(0)))
    worker_threads = max(1, cpu_budget // len(commands))
    events.emit('cpu_budget', 'configured', available_cpus=cpu_budget,
                active_workers=len(commands), threads_per_worker=worker_threads)
    started = time.monotonic()
    def worker_failure(track: str) -> str:
        log_path = root / f'{track}__worker.log'
        tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-200:])
        return f'{track} worker failed (rc={processes[track].returncode})\n{tail}'
    try:
        # Spawn every worker before waiting for any worker to finish.
        for track, command in commands.items():
            log = (root / f'{track}__worker.log').open('a' if resume else 'w')
            handles.append(log)
            worker_env = {**env, 'PYTHONUNBUFFERED': '1', 'ER_TRACK_BARRIER': str(barrier.resolve()),
                          'ER_TRACK_NAME': track,
                          'EUROMONITOR_RESULTS_DIR': str((root / track).resolve()),
                          'WANDB_DIR': str((root / track / 'wandb').resolve()),
                          'WANDB_RUN_NAME': f'{root.name}-{track}',
                          'EUROMONITOR_RUN_ID': f'{root.name}-{track}',
                          'OMP_NUM_THREADS': str(worker_threads),
                          'MKL_NUM_THREADS': str(worker_threads)}
            (root / track / 'wandb').mkdir(parents=True, exist_ok=True)
            processes[track] = subprocess.Popen(command, env=worker_env, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
            events.emit('worker_spawn', 'started', worker_track=track, pid=processes[track].pid,
                        log=str(root / f'{track}__worker.log'))
        last_wait_heartbeat = time.monotonic()
        while not all((barrier / f'{track}.ready').exists() for track in commands):
            logs.drain()
            if time.monotonic() - last_wait_heartbeat >= 30:
                events.emit('barrier', 'waiting', elapsed_seconds=time.monotonic() - started,
                            ready=[track for track in commands if (barrier / f'{track}.ready').exists()],
                            waiting=[track for track in commands if not (barrier / f'{track}.ready').exists()])
                last_wait_heartbeat = time.monotonic()
            for track, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(worker_failure(track))
            if time.monotonic() - started > barrier_timeout:
                raise TimeoutError('workers did not reach the shared start barrier')
            time.sleep(.1)
        (barrier / 'start').write_text('all requested workers ready\n')
        events.emit('barrier', 'released', tracks=list(commands))
        last_heartbeat = time.monotonic()
        completed = set()
        while True:
            logs.drain()
            codes = {track: process.poll() for track, process in processes.items()}
            for track, code in codes.items():
                if code is not None and track not in completed:
                    events.emit('worker_exit', 'ok' if code == 0 else 'failed', worker_track=track,
                                pid=processes[track].pid, returncode=code)
                    completed.add(track)
            if time.monotonic() - last_heartbeat >= 30:
                events.emit('heartbeat', 'running', elapsed_seconds=time.monotonic() - started,
                            workers={track: {'pid': process.pid, 'returncode': codes[track],
                                             'log_bytes': (root / f'{track}__worker.log').stat().st_size}
                                     for track, process in processes.items()})
                last_heartbeat = time.monotonic()
            failed = {track: code for track, code in codes.items() if code not in (None, 0)}
            if failed:
                raise RuntimeError('model-track workers failed:\n' + '\n'.join(worker_failure(track) for track in failed))
            if all(code == 0 for code in codes.values()):
                return {'mode': 'parallel', 'workers': list(commands),
                        'elapsed_seconds': time.monotonic() - started}
            if time.monotonic() - started > timeout:
                raise TimeoutError('all-track training exceeded runtime limit')
            time.sleep(.2)
    except BaseException as exc:
        events.emit('workers', 'failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        # Terminate the whole owned process group, including adapter children.
        import signal
        for process in processes.values():
            if process.poll() is None:
                events.emit('worker_stop', 'terminating', pid=process.pid, signal='SIGTERM')
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                events.emit('worker_stop', 'terminating', pid=process.pid, signal='SIGKILL')
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        for handle in handles:
            handle.close()
        logs.drain(final=True)
