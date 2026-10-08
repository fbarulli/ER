"""Central logger for the bundling pipeline.

One class owns every emission surface the pipeline uses:

  - the process logger (logging, module-shaped names)
  - the tqdm-safe console line (bars are never mangled)
  - the [timing] instrumentation (emit_timing contract, timings.log)
  - run-scoped elapsed counters for stage summaries

Pipeline modules construct one RunLogger per module and call only its methods,
so the emission format stays uniform and a run's log, timing and trace surfaces
keep their pinned formats.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.step_trace import destination, send, trace_step
from core.timing import emit_timing

_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


class RunLogger:
    """The one logging surface of the bundling pipeline.

    Responsibilities are separately-owned and individually trivial:

      - console()/file emission        -> _emit
      - timing sections                -> section()
      - progress wrap                  -> progress()
      - stage event lines              -> stage()
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._t0 = time.perf_counter()

    @classmethod
    def configure_console(cls) -> None:
        """One-time root logging setup for CLI entrypoints (idempotent)."""
        root = logging.getLogger()
        if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(handler)
        root.setLevel(logging.INFO)

    @property
    def elapsed_total(self) -> float:
        """Seconds since this logger was constructed (run-scoped summary)."""
        return round(time.perf_counter() - self._t0, 3)

    def _emit(self, message: str) -> None:
        """Tqdm-safe console write plus the bound timing log, if any."""
        send(message)

    def info(self, message: str) -> None:
        self._logger.info(message)
        self._emit(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)
        self._emit('[warning] ' + message)

    def error(self, message: str) -> None:
        self._logger.error(message)
        self._emit('[error] ' + message)

    # --- timing -----------------------------------------------------------

    @contextmanager
    def section(self, label: str, **fields: Any) -> Iterator[None]:
        """Own one [timing] section: started/completed/failed + elapsed."""
        with trace_step(label, **fields):
            yield

    def event(self, message: str, *, log_path: Path | None = None) -> None:
        """Emit one raw [timing] event line (prepare.* / stage.* style)."""
        emit_timing(message, path=log_path)

    # --- progress ---------------------------------------------------------

    def progress(self, iterable, *, desc: str, unit: str, total: int | None = None):
        """Wrap an iteration in the pipeline's tqdm bar conventions."""
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, unit=unit, total=total,
                    leave=False, disable=False, dynamic_ncols=True)

    def bar(self, *, desc: str, unit: str, total: int | None = None):
        """An explicit top-level bar the caller keeps and closes itself."""
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True,
                    disable=False)

    # --- stage events -----------------------------------------------------

    def stage(self, stage: str, state: str, *, elapsed: float | None = None,
              log_path: Path | None = None) -> None:
        """One prepare.<stage> state line in the pinned pipeline format."""
        suffix = '' if elapsed is None else f' elapsed_seconds={elapsed:.3f}'
        self.event(f'[timing] prepare.{stage} state={state}{suffix}',
                   path=log_path)


def bound_timing_path() -> Path | None:
    """The timing log bound in this process, if any (read-only lookup)."""
    bound = destination() or os.environ.get('ER_TIMING_LOG')
    return Path(bound) if bound else None


__all__ = ['RunLogger', 'bound_timing_path']
