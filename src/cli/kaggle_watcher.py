"""The ONE detached terminal watcher every kaggle-family lane shares.

An ER push spawned a hardcoded ER watcher (``cli.kaggle_lane --what
autowatch``) that polled to a terminal state, downloaded the run, released the
session, and wrote a receipt. The laya lane had no such owner: its
``main --execute`` path was push-only, so a finished laya GPU smoke was never
downloaded. Rather than fork a parallel watcher, the lane-specific pieces live
on a :class:`KernelWatcherSpec` (watcher identity, fetch/stop owners, receipt
path, progress-log roof and detached entry argv) and one :class:`KernelWatcher`
runs the terminal-handler loop for every lane.

The watcher IS the terminal handler: poll to any terminal state, then download
(results on complete / partial artifacts on error) and release the session. It
is spawned detached (setsid) so wrapper timeouts and shell deaths never orphan a
live kernel. Polling is transition-only at the configured cadence — never a
tight loop.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from cli.kaggle_lifecycle import KernelLifecycle

#: Kernel states that end the watcher's poll loop.
TERMINAL_STATES = frozenset({"complete", "error", "cancelAcknowledged"})

FetchCallable = Callable[..., dict[str, Any]]
StopCallable = Callable[..., dict[str, Any]]


class KernelWatcherSpec(BaseModel):
    """Everything one lane's detached watcher needs, resolved once.

    The lane-owned pieces — the watcher ``which``/``kind`` identity, the
    fetch/stop owners, the receipt path, the progress ``log_path`` (the lane log
    roof), the append flag and the detached ``entry_argv`` — ride this spec, so
    a lane never spells an ER path and ER keeps its behavior.
    """

    model_config = ConfigDict(frozen=True)

    which: str
    kind: str
    configured_slug: str | None
    fetch_output: FetchCallable
    fetch_failure: FetchCallable
    stop: StopCallable
    stream_logs: FetchCallable
    kernel_status: FetchCallable
    #: Best-effort launch-aid capture run before/while streaming; ``None`` for a
    #: lane whose push already captures (ER). It persists a usable handle.
    capture_session: Callable[[str], dict[str, Any]] | None = None
    log_lane: Callable[[str], None]
    write_json: Callable[..., None]
    receipt_path: Path
    log_path: Path
    poll_seconds: float
    stream_join_seconds: float
    append: bool
    entry_argv: tuple[str, ...]
    cwd: Path
    source_dir: Path


class KernelWatcher:
    """Polls a pushed kernel to terminal, then harvests, releases, and receipts."""

    def __init__(self, spec: KernelWatcherSpec):
        self._spec = spec

    @property
    def spec(self) -> KernelWatcherSpec:
        return self._spec

    def spawn(self) -> dict[str, Any]:
        """Launch this watcher detached (setsid) for a hands-off terminal pass.

        The launch marker is appended to the lane's progress roof, then the
        detached process re-derives its own spec from the lane entry
        (``entry_argv``) — no callable crosses the process boundary.
        """
        spec = self._spec
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        # A receipt from a PREVIOUS watch is stale: clear it before the new
        # watcher starts so no reader (or re-spawned watcher) mistakes the old
        # run's terminal plan for this one's. Best-effort; absence is fine.
        if spec.receipt_path.exists():
            try:
                spec.receipt_path.unlink()
            except OSError as error:
                spec.log_lane(f"[{spec.which}] stale receipt not cleared: {error}")
        with spec.log_path.open("ab") as handle:
            handle.write(self._launch_marker().encode())
        subprocess.Popen(
            list(spec.entry_argv), cwd=spec.cwd,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            env={**os.environ,
                 "ER_KAGGLE_LANE_APPEND": "1" if spec.append else "0",
                 "PYTHONPATH": os.pathsep.join(filter(None, [
                     str(spec.source_dir), os.environ.get("PYTHONPATH")]))},
            start_new_session=True)
        return {"autowatch": "spawned", "kernel": spec.which,
                "log": str(spec.log_path)}

    def autowatch(self, *, execute: bool, slug: str | None = None,
                  poll_seconds: float | None = None) -> dict[str, Any]:
        """Poll the kernel to terminal, then fetch + release + receipt.

        Dry-run returns the plan without touching the network. Intended to run
        detached (setsid) so wrapper timeouts cannot kill it mid-poll.
        """
        spec = self._spec
        resolved_poll = (poll_seconds if poll_seconds is not None
                         else spec.poll_seconds)
        slug = slug or spec.configured_slug
        if not slug:
            raise RuntimeError(
                f"config names no {spec.which} kernel slug; name the kernel "
                "(owner/slug) before autowatch")
        plan: dict[str, Any] = {"mode": "executed" if execute else "dry-run",
                                "kernel": slug, "poll_seconds": resolved_poll}
        if not execute:
            return plan
        # Clear a stale receipt from a prior watch (the detached path clears it
        # in spawn(); an explicit autowatch does not go through spawn()).
        if spec.receipt_path.exists():
            try:
                spec.receipt_path.unlink()
            except OSError as error:
                spec.log_lane(f"[{slug}] stale receipt not cleared: {error}")
        stream: dict[str, Any] = {}
        follower = threading.Thread(
            target=self._stream, args=(slug,),
            kwargs={"into": stream}, daemon=True,
            name=f"stream-{spec.which}")
        follower.start()
        # While the follower streams, (re)capture the session id: the push-time
        # capture can miss the proxy window, and the log-derived id never lands
        # when the stream 429s. The capture persists the kernel name as its
        # fallback, so the stop handle is never empty.
        self._capture_session(slug)
        status, polls = self._poll_terminal(slug, resolved_poll, stream)
        plan["status"] = status["status"]
        plan["polls"] = polls
        plan["log_terminal"] = bool(stream.get("terminal"))
        # The release runs even against an already-terminal kernel, so no
        # session survives a finished run.
        plan.update(KernelLifecycle.harvest_and_stop(
            kind=spec.kind, slug=slug, which=spec.which,
            status=status["status"], fetch_output=spec.fetch_output,
            fetch_failure=spec.fetch_failure, stop=spec.stop,
            configured_slug=spec.configured_slug))
        # The harvest outcome is a first-class Kaggle-run result: record it on
        # the ONE transcript, not only in the receipt.
        fetched = plan.get("fetch") or {}
        stopped = plan.get("stop") or {}
        spec.log_lane(
            f"[{slug}] harvest: status={status['status']} "
            f"fetch_verified={bool(fetched.get('verified'))} "
            f"stop={stopped.get('verdict') or stopped.get('stopped')}")
        follower.join(timeout=spec.stream_join_seconds)
        try:
            spec.write_json(plan, spec.receipt_path)
        except OSError as error:
            print(f"[kaggle-watcher] receipt write failed ({error}); "
                  "continuing", flush=True)
        plan["receipt"] = str(spec.receipt_path)
        return plan

    def _stream(self, slug: str, *, into: dict[str, Any]) -> None:
        """Tail the ONE kaggle-logs path into the shared transcript.

        Runs in a background thread; a failure is recorded with its full
        traceback so a dead follower is never mistaken for a slow session.
        """
        spec = self._spec
        try:
            into.update(spec.stream_logs(slug, follow=True,
                                         log_path=spec.log_path))
        except Exception as error:  # noqa: BLE001 - follower must not die silent
            spec.log_lane(traceback.format_exc())
            into["error"] = f"{type(error).__name__}: {error}"

    def _poll_terminal(self, slug: str, poll: float,
                       stream: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Poll at the configured cadence; log only state transitions.

        A status-query failure is retried (the platform hiccups under load)
        rather than ending the watch; a real transport error is recorded with
        its message. Polling holds neither a session nor a quota. The ONE log
        path's END_OF_LOG is used as a terminal cue: when the follow stream
        closes, the session has finished, so the exact status is confirmed once
        (END_OF_LOG cannot say complete vs error) instead of waiting a full poll.
        """
        spec = self._spec
        polls = 0
        last: str | None = None
        while True:
            polls += 1
            try:
                status = spec.kernel_status(slug=slug)
            except (RuntimeError, OSError) as error:
                spec.log_lane(f"[{slug}] status query failed; retrying: {error}")
                time.sleep(poll)
                continue
            if status["status"] != last:
                spec.log_lane(f"[{slug}] status={status['status']}")
                last = status["status"]
            if status["status"] in TERMINAL_STATES:
                return status, polls
            if stream.get("terminal"):
                confirmed = spec.kernel_status(slug=slug)
                if confirmed["status"] in TERMINAL_STATES:
                    spec.log_lane(
                        f"[{slug}] log END_OF_LOG -> status={confirmed['status']}")
                    return confirmed, polls
            time.sleep(poll)

    def _capture_session(self, slug: str) -> None:
        """Best-effort session capture; never blocks or fails the watch.

        ``capture_session`` is the lane's capture owner (it persists the id or
        the kernel-name fallback). A capture failure is logged with its message
        and the follower's own SSE capture remains the backup writer.
        """
        spec = self._spec
        if spec.capture_session is None:
            return
        try:
            spec.capture_session(slug)
        except Exception as error:  # noqa: BLE001 - best-effort launch aid
            spec.log_lane(f"[{slug}] session capture skipped: {error}")

    def _launch_marker(self) -> str:
        return (f"[_spawn_autowatch {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"launching watcher for {self._spec.which}]\n")
