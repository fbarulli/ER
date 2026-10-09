"""The suite's two phases: the trained-lane parallel barrier, then the combinators.

``run_parallel`` is the barrier phase and admits ONLY the trained lanes
(``resume.TRAINING_TRACKS``). ``run_track_suite`` is the supervisor's single
entry point: it partitions any declared track set by that same taxonomy, runs
the trained lanes behind the shared start barrier, and only once that call has
returned does it run the postprocess combinators (the cascade) sequentially.
A combinator therefore never reaches the barrier, never holds up a trained
lane's release, and never runs before the artifacts it composes exist.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid

from core.bundle import bundle_spec
from core.perf_switches import perf_enabled
from core.tracing import TRACE_LANE_ENV, run_trace_env
from model_tracks.resume import POSTPROCESS_TRACKS, TRACKS, TRAINING_TRACKS


@contextmanager
def mps_environment(root: Path, *, thread_percentage: int | None = None):
    control = shutil.which('nvidia-cuda-mps-control')
    if not control:
        raise RuntimeError('true multi-process GPU parallelism requires NVIDIA MPS in this runtime')
    attempt = uuid.uuid4().hex
    # UNIX-domain socket paths have a small fixed limit. A run directory plus
    # attempt UUID exceeds it on Colab, so keep daemon/client pipes short.
    pipes = Path(tempfile.mkdtemp(prefix='er-mps-', dir='/tmp'))
    logs = root / 'mps_log' / attempt
    logs.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'CUDA_MPS_PIPE_DIRECTORY': str(pipes.resolve()),
           'CUDA_MPS_LOG_DIRECTORY': str(logs.resolve())}
    if thread_percentage and perf_enabled('parallel.mps_threads'):
        # Each concurrent worker gets an equal share of the GPU's SM threads;
        # without this every client may try to occupy the whole device.
        env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(
            max(1, min(100, int(thread_percentage))))
    startup = subprocess.run([control, '-d'], env=env, capture_output=True,
                             text=True, timeout=30)
    if startup.returncode:
        diagnostics = '\n'.join(path.read_text(errors='replace')[-4000:]
                                for path in logs.glob('*.log') if path.is_file())
        raise RuntimeError(f'MPS startup failed rc={startup.returncode}; pipes={pipes}; '
                           f'logs={logs}\n{startup.stdout}\n{startup.stderr}\n{diagnostics}')
    try:
        yield env
    finally:
        subprocess.run([control], input='quit\n', text=True, env=env, check=True, timeout=30)
        shutil.rmtree(pipes)  # Only this verified-stopped attempt owns these pipes.


def split_tracks(tracks) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition declared tracks into (trained-behind-barrier, postprocess).

    The split is the suite taxonomy, never a per-callsite list: a track that
    trains runs in the parallel barrier phase, a track that only composes
    trained artifacts runs after it. Unknown tracks fail loud rather than
    silently running in neither phase.
    """
    requested = set(tracks)
    unknown = requested - set(TRACKS)
    if unknown:
        raise ValueError(f'unknown suite track(s): {", ".join(sorted(unknown))}')
    return (tuple(track for track in TRAINING_TRACKS if track in requested),
            tuple(track for track in POSTPROCESS_TRACKS if track in requested))


def run_track_suite(commands: dict[str, list[str]], root: Path, env: dict,
                    *, resume: bool = False, multiprocess: bool = False,
                    thread_percentage: int | None = None, timeout: float = 14400,
                    barrier_timeout: float = 600) -> dict:
    """Run every declared track: trained lanes in parallel, combinators after.

    The commands may name any declared tracks. The trained lanes go through
    :func:`run_parallel` exactly as before; the postprocess combinators are
    spawned one after another only after that barrier phase has returned, so a
    combinator can never be released by (or delay) the shared start barrier and
    can never read a trained lane's artifacts before they exist.

    ``multiprocess`` scopes the MPS environment to the trained phase: the
    combinators are spawned with the plain suite environment, never with the
    MPS pipes of a training session.
    """
    if not commands:
        raise ValueError('suite run requires at least one track command')
    trained, postprocess = split_tracks(commands)
    trained_commands = {track: commands[track] for track in trained}
    combinator_commands = {track: commands[track] for track in postprocess}
    if trained_commands:
        if multiprocess:
            share = max(1, 100 // len(trained_commands)) if thread_percentage is None else thread_percentage
            with mps_environment(root, thread_percentage=share) as mps_env:
                result = run_parallel(trained_commands, root, {**env, **mps_env},
                                      resume=resume, timeout=timeout,
                                      barrier_timeout=barrier_timeout)
        else:
            result = run_parallel(trained_commands, root, env, resume=resume,
                                  timeout=timeout, barrier_timeout=barrier_timeout)
    else:
        result = {'mode': 'postprocess', 'workers': []}
    workers = list(result.get('workers') or [])
    for track, command in combinator_commands.items():
        run_postprocess_track(command, root, env, track, resume=resume)
        workers.append(track)
    result['workers'] = workers
    result['postprocess'] = list(combinator_commands)
    return result


def run_postprocess_track(command: list[str], root: Path, env: dict, track: str,
                          *, resume: bool = False) -> None:
    """Run one postprocess combinator lane after the barrier phase, sequentially.

    The lane keeps its own results root and log, but deliberately gets no
    ``ER_TRACK_BARRIER``: a combinator composes artifacts the trained lanes
    already wrote and must not wait on a start barrier that has already been
    released. It gets the RUN's trace pins like every other lane, so its rows
    join the run instead of landing in its own subtree.
    """
    if track not in POSTPROCESS_TRACKS:
        raise ValueError(f'{track} is not a declared postprocess track')
    from core.common import TRAIN_ROOT
    track_env = {**env, 'PYTHONUNBUFFERED': '1', 'ER_TRACK_NAME': track,
                 'EUROMONITOR_RESULTS_DIR': str((root / track).resolve()),
                 'WANDB_DIR': str((root / track / 'wandb').resolve()),
                 TRACE_LANE_ENV: track,
                 **run_trace_env()}
    track_env.pop('ER_TRACK_BARRIER', None)
    (root / track / 'wandb').mkdir(parents=True, exist_ok=True)
    with (root / f'{track}__worker.log').open('a' if resume else 'w') as log:
        subprocess.run(command, cwd=TRAIN_ROOT, env=track_env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


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
    if not commands or set(commands) - set(TRAINING_TRACKS):
        combinators = ', '.join(POSTPROCESS_TRACKS) or 'none declared'
        raise ValueError(
            'suite workers must be known unfinished trained tracks; trained lanes '
            'run behind the barrier and the postprocess combinators (' + combinators +
            ') run after it, through run_track_suite')
    # Each attempt has a fresh barrier; a restored start file cannot release
    # a resumed worker before its companions have loaded.
    barrier = root / ('barrier' if not resume else f'barrier_resume_{time.time_ns()}')
    barrier.mkdir(parents=True, exist_ok=False)
    processes, handles = {}, []
    from model_tracks.live_logs import WorkerLogs
    logs = WorkerLogs(root, commands, from_end=resume)
    from model_tracks.telemetry import WorkerEvents
    events = WorkerEvents(root, 'suite', root.name, filename=bundle_spec().suite_events_file)
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
        # ONE run has ONE trace: a lane owns its own RESULTS subtree (below) but
        # must append its rows to the RUN's trace, or each lane would write its
        # own logs/training_trace.csv and its per-lane run id would keep those
        # rows from ever joining the run. Resolved once for the whole suite, from
        # the layout + run identity SSOTs (core.tracing.run_trace_env).
        trace_pins = run_trace_env()
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
                          'MKL_NUM_THREADS': str(worker_threads),
                          TRACE_LANE_ENV: track,
                          **trace_pins}
            if perf_enabled('parallel.thread_pinning'):
                # BLAS backends other than OpenMP/MKL size their own pools from
                # these; pin them so three workers cannot each grab every core.
                worker_env.update(OPENBLAS_NUM_THREADS=str(worker_threads),
                                  NUMEXPR_NUM_THREADS=str(worker_threads),
                                  VECLIB_MAXIMUM_THREADS=str(worker_threads))
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
