"""One owner for CPU preparation; in-memory artifacts never outlive the run."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
import importlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from core.run_log import RunLogger
from training.prepare_all_trace import timed, trace_step

_ACTIVE: ContextVar[TrainingPreparation | None] = ContextVar(
    'training_preparation', default=None)

_LOG = RunLogger(__name__)


def active_preparation() -> TrainingPreparation | None:
    """The preparation owning this context, if any (shared-object lookup)."""
    return _ACTIVE.get()


class _TeeWriter:
    """Mirror one redirected stage stream into the log and the live terminal.

    The log keeps the full capture; the terminal sees the same bytes so tqdm
    bars stream live instead of being buried in the stage log. isatty/fileno
    follow the terminal so tqdm enables disable=None bars and dynamic_ncols
    there (and stays disabled under pytest capture or nohup pipes). A dead
    terminal only costs the mirror: the log write must never fail because of
    it.
    """

    def __init__(self, log, terminal):
        self._log, self._terminal = log, terminal

    def write(self, text):
        """Append one write to the log, mirroring it to the terminal."""
        self._log.write(text)
        try:
            self._terminal.write(text)
        except (OSError, ValueError):
            pass
        return len(text)

    def flush(self):
        """Flush both the log and the terminal mirror."""
        self._log.flush()
        try:
            self._terminal.flush()
        except (OSError, ValueError):
            pass

    def isatty(self):
        """Report the terminal's tty status (tqdm enables live bars there)."""
        try:
            return bool(self._terminal.isatty())
        except (OSError, ValueError, AttributeError):
            return False

    def fileno(self):
        """Report the terminal's fd so capture helpers see a real fd."""
        return self._terminal.fileno()

    def writable(self):
        """Both sinks accept writes."""
        return True

    def close(self):
        """Close the log file only; the terminal belongs to its owner."""
        self._log.close()


class _StageState:
    """The saved interpreter state of one in-process stage execution."""
    __slots__ = ('argv', 'env', 'cwd', 'out', 'err')

    def __init__(self):
        """Capture everything runpy/importlib and redirectors will touch."""
        self.argv = sys.argv
        self.env = os.environ.copy()
        self.cwd = Path.cwd()
        self.out = sys.stdout
        self.err = sys.stderr

    def restore(self):
        """Put every swapped piece of interpreter state back."""
        sys.argv = self.argv
        sys.stdout, sys.stderr = self.out, self.err
        os.environ.clear()
        os.environ.update(self.env)
        os.chdir(self.cwd)


class TrainingPreparation(BaseModel):
    """Validated request plus the lifetime of all shared preparation objects.

    CSVs and manifests remain resumable checkpoints. Within an invocation the
    producer owns its artifacts; consumers share the current in-memory version.
    Run boundaries discard everything, including after failure.
    """
    model_config = ConfigDict(extra='forbid')
    run_dir: Path | None = None
    tracks_config: Path | None = None
    resume_from: Literal['dedupe', 'validation', 'full_bundle', 'suite_inputs'] = 'dedupe'
    negative_supply_run_tag: str | None = Field(
        default=None, pattern=r'^[A-Za-z0-9_-]+$')
    diagnostic: bool = False
    _datasets: dict[Path, Any] = PrivateAttr(default_factory=dict)
    _base: dict[str, Any] = PrivateAttr(default_factory=dict)
    _bundles: dict[Path, tuple[Any, dict]] = PrivateAttr(default_factory=dict)
    _aliases: dict[Path, Path] = PrivateAttr(default_factory=dict)
    _objects: dict[str, Any] = PrivateAttr(default_factory=dict)
    _token: Any = PrivateAttr(default=None)

    def __enter__(self):
        """Bind this preparation as the run's owner (one per context)."""
        if active_preparation() is not None:
            raise RuntimeError('A preparation run is already active in this context')
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *args):
        """Release every in-memory artifact at the run boundary."""
        self.clear_shared_state()
        _ACTIVE.reset(self._token)
        self._token = None

    def execute(self) -> Path:
        """Run the whole preparation as one ownership window."""
        from training.prepare_all import _prepare_all
        with self:
            _LOG.info(f'[prepare] window open: resume_from={self.resume_from}')
            return self._execute_timed()

    @timed
    def _execute_timed(self) -> Path:
        """The owned _prepare_all call, timed as one unit."""
        from training.prepare_all import _prepare_all
        return _prepare_all(**self.model_dump())

    def _drop_matching_keys(self, cache: dict, path: Path) -> list[Path]:
        """Remove exactly the cache keys under one path; return the dropped keys."""
        dropped = [key for key in cache
                   if key == path or key.is_relative_to(path)]
        for key in dropped:
            del cache[key]
        return dropped

    def invalidate(self, path: Path) -> None:
        """Forget one superseded path (and everything derived beneath it)."""
        path = path.resolve()
        self._aliases.pop(path, None)
        for cache in (self._datasets, self._bundles):
            self._drop_matching_keys(cache, path)

    def release_bundle(self, path: Path) -> None:
        """Release a completed consumer's bundle, preserving path alias identity."""
        key = self.bundle_key(path)
        self._bundles.pop(key, None)
        self._objects.pop('built_bundle:' + str(key), None)

    def bundle_key(self, path: Path) -> Path:
        """The canonical identity of a bundle path under its registered alias."""
        path = path.resolve()
        return self._aliases.get(path, path)

    def alias_bundle(self, source: Path, destination: Path) -> None:
        """Register a destination path as an alias of an existing bundle."""
        self._aliases[destination.resolve()] = self.bundle_key(source)

    def clear_shared_state(self) -> None:
        """One owner for the run-boundary cache wipe (used by exit + tests)."""
        self._datasets.clear()
        self._base.clear()
        self._bundles.clear()
        self._objects.clear()
        self._aliases.clear()

    @timed
    def run_stage(self, arguments: list[str], *, root: Path, env: dict[str, str], log):
        """Run an existing entry point in this interpreter, restoring CLI state."""
        label = ' '.join(arguments)[:72]
        with trace_step('run_stage.swap_in'):
            _LOG.info(f'[stage][start] {label}')
            state = _StageState()
            os.environ.clear()
            os.environ.update(env)
            os.chdir(root)
        started = time.time()
        try:
            code = self._invoke_entry_point(arguments, log=log)
        finally:
            with trace_step('run_stage.restore'):
                state.restore()
        elapsed = time.time() - started
        _LOG.info(f'[stage][done] {label} rc={code} elapsed_seconds={elapsed:.3f}')
        return subprocess.CompletedProcess([sys.executable, *arguments], code)

    def _invoke_entry_point(self, arguments: list[str], *, log) -> int:
        """Execute -m module or script path; return a normalizable exit code."""
        with self._stage_streams(log):
            try:
                if arguments[0] == '-m':
                    self._import_and_run(arguments)
                    return 0
                self._script_run(arguments)
                return 0
            except SystemExit as error:
                return self._normalize_exit(error)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                return 1

    def _import_and_run(self, arguments: list[str]) -> Any:
        """Run `python -m <module>` inline against the captured argv."""
        sys.argv = [arguments[1], *arguments[2:]]
        importlib.import_module(arguments[1]).main()

    def _script_run(self, arguments: list[str]) -> Any:
        """Run a script path inline as __main__ against the captured argv."""
        sys.argv = arguments
        return runpy.run_path(arguments[0], run_name='__main__')

    def _normalize_exit(self, error: SystemExit) -> int:
        """Translate one SystemExit into a normalizable integer code."""
        if isinstance(error.code, int):
            return error.code
        if error.code:
            print(error.code, file=sys.stderr)
        return 1 if error.code else 0

    @contextmanager
    def _stage_streams(self, log):
        """Swap stdout/stderr for tee mirrors, keeping terminal bars live."""
        from core.run_log import bound_timing_path
        with redirect_stdout(_TeeWriter(log, sys.stdout)), \
                redirect_stderr(_TeeWriter(log, sys.stderr)):
            yield
