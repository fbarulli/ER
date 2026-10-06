"""Function-level trace/timing helper for the training pipeline.

Instrumentation must land on the run's existing time logs (timings.log via
core.timing.emit_timing), including full tracebacks on failure, instead of
ad-hoc print-only logging. The emit surface is bound per process through
ER_TIMING_LOG / ER_TIMING_OUT; before anything is bound only prints happen,
so shared callers (selftests, research lanes) stay untouched.
"""
from __future__ import annotations

import functools
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


def destination() -> Path | None:
    """The existing time log currently bound for inline parent-process calls."""
    bound = os.environ.get('ER_TIMING_LOG')
    if bound is None and os.environ.get('ER_TIMING_OUT'):
        bound = Path(os.environ['ER_TIMING_OUT']).with_suffix('.log')
    return None if bound is None else Path(bound)


def send(message: str) -> None:
    """Append one line to the bound existing time log (prints before bind)."""
    path = destination()
    if path is None:
        print(message, flush=True)
        return
    from core.timing import emit_timing
    emit_timing(message, path=path)


def _label(function: Callable, prefix: str | None) -> str:
    if prefix is not None:
        return prefix + '.' + function.__name__
    return function.__module__.rsplit('.', 1)[-1] + '.' + function.__name__


def timed(function: Callable | None = None, *, prefix: str | None = None) -> Callable:
    """Trace one function as a single call-level timing unit.

    Emits started/completed lines with elapsed_seconds and, when the call
    fails, a [traceback] block whose lines land verbatim in the log. The
    label defaults to the module leaf name (training.prepare_all ->
    prepare_all.<name>, training.training -> training.<name>).
    """

    def decorate(function: Callable) -> Callable:
        label = _label(function, prefix)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            send(f'[timing] {label} state=started')
            try:
                result = function(*args, **kwargs)
            except BaseException:
                elapsed = time.perf_counter() - started
                send(f'[timing] {label} state=failed elapsed_seconds={elapsed:.3f}')
                send(_traceback_block(f'[traceback] {label}'))
                raise
            elapsed = time.perf_counter() - started
            send(f'[timing] {label} state=completed elapsed_seconds={elapsed:.3f}')
            return result

        return wrapper

    return decorate if function is None else decorate(function)


@contextmanager
def trace_step(section: str, **fields: Any):
    """Time one inline step with start/completed/failed and tracebacks."""
    detail = ''.join(f' {key}={value}' for key, value in fields.items())
    started = time.perf_counter()
    send(f'[timing] {section} state=started{detail}')
    try:
        yield
    except BaseException:
        elapsed = time.perf_counter() - started
        send(f'[timing] {section} state=failed{detail} elapsed_seconds={elapsed:.3f}')
        send(_traceback_block(f'[traceback] {section}'))
        raise
    elapsed = time.perf_counter() - started
    send(f'[timing] {section} state=completed{detail} elapsed_seconds={elapsed:.3f}')


def _traceback_block(header: str) -> str:
    return header + '\n' + traceback.format_exc()
