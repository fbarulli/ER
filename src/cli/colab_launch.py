"""Colab launcher session lock (split phase of cli.colab).

Single-process ownership of a Colab session: the advisory flock that refuses a
second launcher sharing one VM, its owner metadata, and the non-destructive
held probe the self-watch reads.  Split from cli/colab.py (the kaggle_lane.py
owner-module pattern) exactly like colab_runtime/colab_result_sync.

The ``cli.colab`` module stays the single surface the offline fakes patch and
the running colab identity (``sys.modules["__colab_runtime_self__"]``) stays
the only launcher module: SESSION, the CLI state dir, and the ``_timed_colab``
step decorator are all re-read through ``colab_hub.hub()`` at call time, never captured.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cli.colab_hub import hub, timed_colab



def _colab_launch_lock_path() -> Path:
    lock_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", hub().SESSION)
    return hub()._COLAB_CLI_STATE_DIR / f"launcher-{lock_name}.lock"


def _process_start_ticks(pid: int) -> int | None:
    """Return Linux's immutable process-start marker, if it is available."""
    try:
        return int((Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")).split()[21])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _read_colab_launch_owner(lock_path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _colab_launch_lock_is_held(lock_path: Path) -> bool:
    """Probe the advisory lock without altering its owner metadata."""
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


@timed_colab("step")
def acquire_colab_launch_lock():
    """Prevent independent launchers from sharing and tearing down one VM.

    Every lane intentionally uses the configured session name.  Without an
    inter-process lock, a previously interrupted local launcher can keep
    running and execute its ``finally: stop()`` while a later launch is using
    that same session.  The resulting kernel 404 is indistinguishable from a
    Colab-side failure, so refuse the second launch before it touches Colab.
    """
    surface = hub()
    surface._COLAB_CLI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _colab_launch_lock_path()
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"a Colab launcher already owns session '{surface.SESSION}'; "
            f"refusing a concurrent lane (lock: {lock_path})"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({
        "pid": os.getpid(),
        "pid_start_ticks": _process_start_ticks(os.getpid()),
        "session": surface.SESSION,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "owner_token": uuid.uuid4().hex,
    }) + "\n")
    handle.flush()
    return handle


def release_colab_launch_lock(handle) -> None:
    """Release the process-scoped Colab session ownership lock."""
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
