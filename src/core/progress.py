"""src/core/progress.py — one progress reporter for the slow per-row lanes.

The identity bundle costs ~2 ms/row, so a 61k-row pass is minutes of silence.
Owner instruction (2026-09-30): long processes show progress. One helper so the
lane, the audit scripts and the tests all report the same way, and so a
non-tty (CI, nohup log) degrades to periodic lines instead of carriage returns
that fill a log file with one line per update.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def _tqdm(iterable: Iterable[T], desc: str, total: int | None = None,
          quiet: bool = False) -> Iterator[T]:
    """tqdm when it is importable and attached to a terminal, else a shim.

    The shim prints one line every `every` items so a redirected log stays
    readable — tqdm's default carriage-return stream is unreadable there.
    """
    if not quiet:
        try:
            from tqdm import tqdm as _tqdm_impl

            return _tqdm_impl(iterable, desc=desc, total=total, dynamic_ncols=True)
        except ImportError:
            pass
    return _shim(iterable, desc, total, every=max(1, (total or 1000) // 20))


def _shim(iterable: Iterable[T], desc: str, total: int | None,
          every: int) -> Iterator[T]:
    start = time.time()
    for n, item in enumerate(iterable, 1):
        yield item
        if n % every == 0:
            rate = n / max(1e-9, time.time() - start)
            pct = f"{100.0 * n / total:5.1f}%" if total else ""
            print(f"  {desc}: {n:,}{pct} {rate:,.0f}/s", file=sys.stderr, flush=True)


def tracked(iterable: Iterable[T], desc: str, total: int | None = None,
            quiet: bool = False) -> Iterator[T]:
    """Public entry point. `total` is a hint; it is corrected when len() works."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    return _tqdm(iterable, desc, total, quiet)


__all__ = ["tracked"]
