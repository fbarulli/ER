"""Central, individually-deactivatable switches for training-speed work.

Every optimization introduced on the ``perf/training-optimization`` branch
reads its switch through :func:`perf_enabled`. This keeps the pre-optimization
behaviour reachable at runtime (for before/after measurement and for a safe
rollback) without scattering comments or reverts:

  * ``ER_PERF_LEGACY=1``      -> every optimization is off (exact old path).
  * ``ER_PERF_<NAME>=0|1``    -> override one named switch only.
  * default when unset        -> ON (the optimized path).

``<NAME>`` is upper-cased with non-alphanumerics collapsed to ``_``; e.g.
``perf_enabled("text.fast_dataloader")`` reads ``ER_PERF_TEXT_FAST_DATALOADER``.

Once an optimization is trusted in production, delete its call site and the
legacy branch; the switch then costs nothing.
"""
from __future__ import annotations

import os
import re

_LEGACY = os.environ.get('ER_PERF_LEGACY') == '1'
_TRUE = {'1', 'true', 'on', 'yes'}
_FALSE = {'0', 'false', 'off', 'no'}


def _normalise(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').upper()


def perf_enabled(name: str, *, default: bool = True) -> bool:
    """Whether optimization ``name`` is active for this process."""
    if _LEGACY:
        return False
    raw = os.environ.get('ER_PERF_' + _normalise(name))
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


def perf_int(name: str, default: int) -> int:
    """Integer tuning knob ``name``; legacy mode forces ``default``."""
    if _LEGACY:
        return default
    raw = os.environ.get('ER_PERF_' + _normalise(name))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def legacy_mode() -> bool:
    """True when every optimization is disabled globally."""
    return _LEGACY
