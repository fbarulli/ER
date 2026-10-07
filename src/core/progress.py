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


def _resolve_total(iterable: Iterable[T], total: int | None) -> int | None:
    """The caller's hint, corrected when the iterable declares its own length."""
    if total is not None:
        return total
    try:
        return len(iterable)  # type: ignore[arg-type]
    except TypeError:
        return None


def _shim_interval(total: int | None) -> int:
    """How many items pass between shim lines: ~20 per pass, minimum 1."""
    return max(1, (total or 1000) // 20)


def _tqdm(iterable: Iterable[T], desc: str, total: int | None = None,
          quiet: bool = False) -> Iterator[T]:
    """tqdm when it is importable and not silenced, else the shim.

    `disable` stays tqdm's own default (None): it belongs to the tqdm
    convention every downstream consumer of tracked() shares, so it is
    spelled out rather than left up to a later rewrite of the call.
    """
    if not quiet:
        try:
            from tqdm import tqdm as _tqdm_impl

            return _tqdm_impl(
                iterable, desc=desc, total=total,
                dynamic_ncols=True, disable=None,
            )
        except ImportError:
            pass
    return _shim(iterable, desc, total, _shim_interval(total))


def _shim(iterable: Iterable[T], desc: str, total: int | None,
          every: int) -> Iterator[T]:
    """Yield through, printing one stderr line every `every` items.

    The redirected-log fallback: tqdm's default carriage-return stream is
    unreadable there, so degraded reporter emits periodic lines instead.
    """
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
    return _tqdm(iterable, desc, _resolve_total(iterable, total), quiet)


__all__ = ["tracked"]
