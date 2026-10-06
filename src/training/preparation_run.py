"""One owner for CPU preparation; in-memory artifacts never outlive the run."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from contextvars import ContextVar
import importlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

_ACTIVE: ContextVar[TrainingPreparation | None] = ContextVar('training_preparation', default=None)


def active_preparation() -> TrainingPreparation | None:
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
        self._log.write(text)
        try:
            self._terminal.write(text)
        except (OSError, ValueError):
            pass
        return len(text)

    def flush(self):
        self._log.flush()
        try:
            self._terminal.flush()
        except (OSError, ValueError):
            pass

    def isatty(self):
        try:
            return bool(self._terminal.isatty())
        except (OSError, ValueError, AttributeError):
            return False

    def fileno(self):
        return self._terminal.fileno()

    def writable(self):
        return True

    def close(self):
        self._log.close()


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
    negative_supply_run_tag: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]+$')
    _datasets: dict[Path, Any] = PrivateAttr(default_factory=dict)
    _base: dict[str, Any] = PrivateAttr(default_factory=dict)
    _bundles: dict[Path, tuple[Any, dict]] = PrivateAttr(default_factory=dict)
    _aliases: dict[Path, Path] = PrivateAttr(default_factory=dict)
    _objects: dict[str, Any] = PrivateAttr(default_factory=dict)
    _token: Any = PrivateAttr(default=None)

    def __enter__(self):
        if active_preparation() is not None:
            raise RuntimeError('A preparation run is already active in this context')
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *args):
        self._datasets.clear()
        self._base.clear()
        self._bundles.clear()
        self._objects.clear()
        self._aliases.clear()
        _ACTIVE.reset(self._token)
        self._token = None

    def execute(self) -> Path:
        from training.prepare_all import _prepare_all
        with self:
            return _prepare_all(**self.model_dump())

    def invalidate(self, path: Path) -> None:
        path = path.resolve()
        self._aliases.pop(path, None)
        for cache in (self._datasets, self._bundles):
            for key in list(cache):
                if key == path or key.is_relative_to(path):
                    del cache[key]

    def release_bundle(self, path: Path) -> None:
        """Release a completed consumer's bundle, preserving path alias identity."""
        key = self.bundle_key(path)
        self._bundles.pop(key, None)
        self._objects.pop('built_bundle:' + str(key), None)

    def bundle_key(self, path: Path) -> Path:
        path = path.resolve()
        return self._aliases.get(path, path)

    def alias_bundle(self, source: Path, destination: Path) -> None:
        self._aliases[destination.resolve()] = self.bundle_key(source)

    def run_stage(self, arguments: list[str], *, root: Path, env: dict[str, str], log):
        """Run an existing entry point in this interpreter, restoring CLI state."""
        saved_argv, saved_env, saved_cwd = sys.argv, os.environ.copy(), Path.cwd()
        saved_out, saved_err = sys.stdout, sys.stderr
        command = [sys.executable, *arguments]
        code = 0
        try:
            os.environ.clear()
            os.environ.update(env)
            os.chdir(root)
            with redirect_stdout(_TeeWriter(log, saved_out)), \
                    redirect_stderr(_TeeWriter(log, saved_err)):
                try:
                    if arguments[0] == '-m':
                        sys.argv = [arguments[1], *arguments[2:]]
                        importlib.import_module(arguments[1]).main()
                    else:
                        sys.argv = arguments
                        runpy.run_path(arguments[0], run_name='__main__')
                except SystemExit as error:
                    code = error.code if isinstance(error.code, int) else (1 if error.code else 0)
                    if error.code and not isinstance(error.code, int):
                        print(error.code, file=sys.stderr)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    code = 1
        finally:
            sys.argv = saved_argv
            os.environ.clear()
            os.environ.update(saved_env)
            os.chdir(saved_cwd)
        return subprocess.CompletedProcess(command, code)
