"""Back-compat face for training.prepare_all's instrumentation.

The implementation lives in core.step_trace; keep importing from here so
existing pointers and the history stay stable.
"""
from __future__ import annotations

from core.step_trace import destination, send, timed, trace_step

__all__ = ['destination', 'send', 'timed', 'trace_step']
