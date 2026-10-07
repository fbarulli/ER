"""Colab self-watch: the detached release + delivery guarantee.

Split from cli/colab.py (capability module, phase 1). Collaborators still
owned by cli.colab are resolved at call time through the running colab module
(registered by colab.py itself in ``sys.modules["__colab_runtime_self__"]``),
so behavior is identical: ``monkeypatch`` on ``cli.colab.<name>`` keeps
working and the ``python -m cli.colab`` runtime identity never sees a stale
second copy.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

def _colab():
    """The RUNNING cli.colab module (never a second import copy)."""
    return sys.modules["__colab_runtime_self__"]

# ── default self-watch (owner order 2026-10-07): release + delivery is
# baked in, no operator arg.  The colab twin of the kaggle lane's
# `_spawn_autowatch` / `autowatch_kernel`: every executed remote launch
# carries its own detached watcher that proves delivery happened and the
# VM was released, even when the launcher process itself dies. ──

_SELF_WATCH_POLL_SECONDS = max(float(_colab()._LOG_POLL_SECONDS), 30.0)
_SELF_WATCH_BUDGET_SECONDS = 24 * 3600
_SELF_WATCH_TRANSCRIPT_MAX_BYTES = 30 * 1024 * 1024

def _self_watch_root(run_id: str) -> Path:
    """Receipt + captured-log folder for one watcher (lane receipts roof)."""
    return _colab().TRAINING_RESULTS / f"self_watch_{run_id}"

def _session_listed() -> bool:
    """Read the session state through the exact surface main()/stop() use.

    An unreachable CLI means unknown state, which must default to live: a
    false 'absent' would forfeit the release guarantee the watcher exists for.
    """
    try:
        result = _colab().colab("sessions", check=False, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return True
    return _colab().SESSION in (result.stdout or "")

def spawn_self_watch(*, what: str, run_id: str) -> dict[str, object]:
    """Detach the release/delivery self-watch for one executed remote run.

    One canonical spawn surface (tests reference it; the kaggle lane's
    `_spawn_autowatch` is the model).  Runs as its own session (setsid) so a
    wrapper timeout or a shell death cannot orphan the guarantee: the
    watcher's whole job is to survive the launcher.  Dry-run, --preflight-only,
    and explicit --keep-alive retention never reach here — main() guards the
    spawn point (after the first healthy provisioning stream).
    """
    # The watcher appends to the SAME per-run lane transcript (owner order:
    # one file).  It is a detached child, so it never truncates: the launcher
    # opened lane.log fresh (start_live_log) and this child only appends.
    from cli import log_capture
    log_path = log_capture.lane_log("colab", _colab().LANE_LOG_NAME)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The runbook's "never python -m cli.colab for a launch" rule stands: this
    # child is a watcher, not a launch — it provisions nothing.
    command = [sys.executable, "-m", "cli.colab", "--what", "self-watch",
               "--self-watch-what", what, "--self-watch-run", run_id]
    environment = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(_colab().TRAIN_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    with log_path.open("ab") as handle:
        handle.write(f"[spawn_self_watch {datetime.now(ZoneInfo('Europe/Paris'))} "
                     f"launching watcher for what={what} run_id={run_id}]\n"
                     .encode())
        handle.flush()
        subprocess.Popen(
            command, cwd=_colab().TRAIN_ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
    return {"self_watch": "spawned", "what": what, "run_id": run_id,
            "log": str(log_path)}

def _self_watch_delivery_state(run_id: str) -> dict[str, object]:
    """Verify the run's delivery/retention artifacts landed (or keep the log).

    On a confirmed delivery the receipt file list IS the retention evidence;
    when nothing landed, copies of the lane transcripts travel into the
    receipt folder so the failure stays inspectable after the VM is gone.
    """
    found: list[str] = []
    for candidate in (_colab().TRAINING_RESULTS / run_id,
                      _colab().TRAINING_RESULTS / f"colab_bundle_{run_id}"):
        if candidate.is_dir():
            found.extend(
                path.relative_to(_colab().TRAINING_RESULTS).as_posix()
                for path in sorted(candidate.rglob("*")) if path.is_file()
            )
    state: dict[str, object] = {
        "delivered": bool(found),
        "artifact_count": len(found),
        "artifacts": found[:64],
    }
    if not state["delivered"]:
        captured: list[str] = []
        destination = _self_watch_root(run_id)
        for name in (f"colab_system_{_colab().SESSION}.log", f"training_{_colab().SESSION}.log",
                     "lane.log", "system.log", "training.log"):
            try:
                from cli import log_capture
                source = log_capture.lane_log("colab", name)
                if (source.is_file()
                        and source.stat().st_size <= _SELF_WATCH_TRANSCRIPT_MAX_BYTES):
                    destination.mkdir(parents=True, exist_ok=True)
                    kept = destination / f"captured_{name}"
                    shutil.copy2(source, kept)
                    captured.append(kept.name)
            except OSError:
                continue
        state["transcripts_captured"] = captured
    return state

def _write_self_watch_receipt(plan: dict[str, object]) -> str:
    """One small receipt with the release proof; loud, never fatal, on error."""
    path = _self_watch_root(str(plan["run_id"])) / "self_watch_receipt.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(
            _colab()._stamp(),
            f"[self-watch] receipt write failed ({exc}); continuing",
            file=sys.stderr,
        )
        return ""
    return str(path)

def self_watch(
    *, what: str, run_id: str, execute: bool = True,
    poll_seconds: float | None = None, budget_seconds: float | None = None,
) -> dict[str, object]:
    """Terminal watcher: poll -> verify delivery -> prove release, every time.

    Polls the lane's session listing until the session is absent (the
    launcher's own `finally` teardown won), the local launcher lock shows the
    launcher died with the VM still listed, or the watch budget expires.  It
    never retains a VM: a still-listed session at a break with the lock free
    runs the lane's own `stop()` release guarantee — the always-release rule.
    Every executed outcome writes one receipt with the release proof and the
    delivery/retention artifact list.  ``execute=False`` is the plan-only
    mode: nothing is polled, nothing is stopped, no receipt is written.
    """
    poll = max(float(poll_seconds if poll_seconds is not None
                     else _SELF_WATCH_POLL_SECONDS), 1.0)
    budget = float(budget_seconds if budget_seconds is not None
                   else _SELF_WATCH_BUDGET_SECONDS)
    plan: dict[str, object] = {
        "what": what, "run_id": run_id, "session": _colab().SESSION,
        "mode": "executed" if execute else "dry-run",
        "poll_seconds": poll,
    }
    if not execute:
        return plan
    lock_path = _colab()._colab_launch_lock_path()
    deadline = time.monotonic() + budget
    polls = 0
    listed = True
    outcome = "released"
    while True:
        polls += 1
        listed = _session_listed()
        if not listed:
            outcome = "released"
            break
        if not _colab()._colab_launch_lock_is_held(lock_path):
            outcome = "launcher_exit_detected"
            break
        if time.monotonic() >= deadline:
            outcome = "watch_budget_expired"
            break
        time.sleep(poll)
    plan.update({"polls": polls, "outcome": outcome})
    if listed:
        confirmed = _colab().stop()
        plan["release"] = {"guarantee_ran": True, "confirmed": bool(confirmed)}
    else:
        plan["release"] = {"guarantee_ran": False, "observed_absent": True}
    plan["delivery"] = _self_watch_delivery_state(run_id)
    plan["receipt"] = _write_self_watch_receipt(plan)
    print(
        _colab()._stamp(),
        f"[self-watch] run_id={run_id} outcome={outcome} release={plan['release']}",
        flush=True,
    )
    return plan
