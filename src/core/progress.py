"""src/core/progress.py — one progress reporter for the slow per-row lanes.

The identity bundle costs ~2 ms/row, so a 61k-row pass is minutes of silence.
Owner instruction (2026-09-30): long processes show progress. One helper so the
lane, the audit scripts and the tests all report the same way, and so a
non-tty (CI, nohup log) degrades to periodic lines instead of carriage returns
that fill a log file with one line per update.

The single owner class is :class:`RowLaneProgress`; the public func
:func:`tracked` (and the historic private aliases) delegate to it so the
consumers see the same surface.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, TypeVar

from core.run_log import RunLogger

_LOG = RunLogger(__name__)

T = TypeVar("T")


class RowLaneProgress:
    """One progress reporter for the slow per-row lanes.

    Single responsibility per method:
      bar            — pick the reporter: tqdm on a terminal, or the shim
      _terminal_tqdm — the live-bar reporter
      _periodic_shim — the redirected-log reporter (one line per interval)
      line_interval  — the report interval (total/20, at least 1)
    """

    REPORTS_PER_LANE = 20

    def line_interval(self, total: int | None) -> int:
        """One report line every ``total // 20`` items (the shim's cadence)."""
        return max(1, (total or 1000) // self.REPORTS_PER_LANE)

    def _terminal_tqdm(self, iterable: Iterable[T], desc: str,
                       total: int | None) -> Iterator[T]:
        from tqdm import tqdm as _tqdm_impl

        return _tqdm_impl(iterable, desc=desc, total=total, dynamic_ncols=True)

    def _periodic_shim(self, iterable: Iterable[T], desc: str,
                       total: int | None, every: int) -> Iterator[T]:
        """Print one line every ``every`` items so a redirected log stays
        readable — tqdm's default carriage-return stream is unreadable there."""
        start = time.time()
        for n, item in enumerate(iterable, 1):
            yield item
            if n % every == 0:
                rate = n / max(1e-9, time.time() - start)
                pct = f"{100.0 * n / total:5.1f}%" if total else ""
                print(f"  {desc}: {n:,}{pct} {rate:,.0f}/s", file=sys.stderr, flush=True)

    def bar(self, iterable: Iterable[T], desc: str, total: int | None = None,
            quiet: bool = False) -> Iterator[T]:
        """tqdm when it is importable and attached to a terminal, else a shim."""
        if not quiet:
            try:
                return self._terminal_tqdm(iterable, desc, total)
            except ImportError:
                pass
        return self._periodic_shim(iterable, desc, total, self.line_interval(total))


_PROGRESS = RowLaneProgress()


def _tqdm(iterable: Iterable[T], desc: str, total: int | None = None,
          quiet: bool = False) -> Iterator[T]:
    """tqdm when it is importable and attached to a terminal, else a shim."""
    return _PROGRESS.bar(iterable, desc, total, quiet)


def _shim(iterable: Iterable[T], desc: str, total: int | None,
          every: int) -> Iterator[T]:
    return _PROGRESS._periodic_shim(iterable, desc, total, every)


def tracked(iterable: Iterable[T], desc: str, total: int | None = None,
            quiet: bool = False) -> Iterator[T]:
    """Public entry point. `total` is a hint; it is corrected when len() works."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    return _PROGRESS.bar(iterable, desc, total, quiet)


__all__ = ["tracked"]
