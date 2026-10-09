"""Detached supervision, live progress, and terminal harvesting."""
from __future__ import annotations

import re
import sys
import time
import traceback
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from cli.log_capture import progress_frames_to_lines

if TYPE_CHECKING:
    from cli.kaggle_watcher import KernelWatcherSpec


#: A kernel self-reports its session id on stdout (the numeric suffix of
#: KAGGLE_CONTAINER_NAME, which cancel_kernel_session accepts); the log-stream
#: URL carries no id, so this line is the only source the follower can persist
#: for the verified in-place stop.
_SESSION_MARKER_RE = re.compile(r"\[kaggle-session\][^\n]*?session_id=(\d+)")


def _reported_session_id(text: str) -> int | None:
    match = _SESSION_MARKER_RE.search(text or "")
    return int(match.group(1)) if match else None


class ReconnectBackoff:
    """429-aware reconnect pacing for the live log stream.

    Kaggle throttles the SSE log endpoint; a dropped stream that reconnects
    immediately can self-inflict HTTP 429 and then worsen it (every reconnect
    is a fresh request). Pace reconnects with exponential backoff + jitter,
    honouring ``Retry-After`` when present, with a longer ceiling for 429s so
    the follower cannot hammer the endpoint.
    """

    def __init__(self, *, base_seconds: float = 1.0, cap_seconds: float = 60.0,
                 rate_limit_cap_seconds: float = 300.0, jitter: float = 0.25):
        self._base = max(0.1, float(base_seconds))
        self._cap = float(cap_seconds)
        self._rate_limit_cap = float(rate_limit_cap_seconds)
        self._jitter = float(jitter)

    @staticmethod
    def is_rate_limited(error: BaseException) -> bool:
        status = getattr(error, "status_code", None)
        if status is None:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
        if status == 429:
            return True
        text = str(error)
        return "429" in text or "Too Many Requests" in text

    @staticmethod
    def _retry_after(error: BaseException) -> float | None:
        header = None
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                header = headers.get("Retry-After")
            except AttributeError:
                header = None
        if not header:
            match = re.search(r"[Rr]etry-[Aa]fter[:=]\s*(\d+(?:\.\d+)?)",
                              str(error))
            header = match.group(1) if match else None
        try:
            return float(header) if header is not None else None
        except (TypeError, ValueError):
            return None

    def delay(self, attempt: int, error: BaseException | None = None) -> float:
        import random

        rate_limited = error is not None and self.is_rate_limited(error)
        ceiling = self._rate_limit_cap if rate_limited else self._cap
        delay = min(self._base * (2 ** max(0, int(attempt) - 1)), ceiling)
        if error is not None:
            retry_after = self._retry_after(error)
            if retry_after is not None:
                delay = max(delay, min(retry_after, self._rate_limit_cap))
        return delay * (1.0 + random.uniform(0.0, self._jitter))


class KaggleMonitor:
    """Detached supervision, live progress, and terminal harvesting."""

    @staticmethod
    def _watcher_spec(which: str, *, slug: str | None = None) -> "KernelWatcherSpec":
        """Resolve the ER lane's watcher parameters into the shared spec.

        ONE registry (``kernel_identity``) resolves the watcher identity, its
        kind and the configured slug, so the push paths and the explicit
        ``--what autowatch`` op share this resolution — no per-surface
        kind->which table here. ``entry_argv`` re-invokes this lane detached.
        """
        from cli import kaggle_lane as lane
        from core.manifest import atomic_write_json

        from cli.kaggle_watcher import KernelWatcherSpec

        identity = lane.kernel_identity(which)
        # The canonical watcher value (e.g. kind "bundle" -> which "cpu") is
        # what the detached `--kernel` arg and the log roof take.
        which = identity.which
        configured_slug = identity.slug(lane._spec())
        resolved_slug = slug or configured_slug
        if not resolved_slug:
            raise RuntimeError(
                f"config kaggle.{identity.slug_attr} is unset; name the {which} "
                "kernel in config before autowatch")
        spec = lane._spec()
        return KernelWatcherSpec(
            which=which, kind=identity.kind, configured_slug=configured_slug,
            fetch_output=lane.fetch_kernel_output,
            fetch_failure=lane.fetch_failed_kernel_log,
            stop=lane.stop_kernel,
            stream_logs=KaggleMonitor.kernel_logs,
            kernel_status=lane.kernel_status,
            log_lane=lane._log_lane,
            write_json=atomic_write_json,
            receipt_path=lane.staging_dir()
            / spec.files.autowatch_receipt.format(kind=identity.kind),
            log_path=lane.lane_log_path(),
            poll_seconds=spec.logs_poll_seconds,
            stream_join_seconds=spec.limits.stream_join_seconds,
            append=True,
            entry_argv=(sys.executable, "-m", "cli.kaggle_lane",
                        "--what", "autowatch", "--kernel", which, "--execute",
                        *(["--slug", resolved_slug] if resolved_slug else [])),
            cwd=lane.TRAIN_ROOT,
            source_dir=lane.TRAIN_ROOT / spec.files.source_dir,
        )

    @staticmethod
    def _spawn_autowatch(which: str, *, slug: str | None = None) -> dict[str, Any]:
        """Detached self-watch spawned by every live kernel push (no args needed).

        Runs as its own session (setsid) so wrapper timeouts and shell deaths
        never orphan a live kernel. One spawn per push, ever: the chain reuses
        the push paths as-is and must never spawn a second watcher on top. The
        lane-specific bits live on the spec; this is the ER caller.
        """
        from cli.kaggle_watcher import KernelWatcher

        return KernelWatcher(KaggleMonitor._watcher_spec(which, slug=slug)).spawn()

    @staticmethod
    def autowatch_kernel(which: str, *, execute: bool, slug: str | None = None,
                         poll_seconds: float | None = None) -> dict[str, Any]:
        """Hands-off terminal watcher: download then release, every single time.

        Replaces interactive supervision as the launch standard. One process:
        poll the kernel status until ANY terminal state, then (a) `complete`
        fetches + hash-verifies the result archive, (b) `error` fetches partial
        artifacts and the session log; (c) always pushes the stub replace that
        releases the session — the failed-kernel-solves case included. The
        release runs even when the watcher starts against an already-terminal
        kernel, so no session survives a finished run. Intended to run detached
        (setsid) so wrapper timeouts cannot kill it mid-poll.
        """
        from cli.kaggle_watcher import KernelWatcher

        return KernelWatcher(
            KaggleMonitor._watcher_spec(which, slug=slug)).autowatch(
            execute=execute, slug=slug, poll_seconds=poll_seconds)

    @staticmethod
    def supervise_kernels(*, kinds: Sequence[str], execute: bool,
                          poll_seconds: float | None = None,
                          max_polls: int | None = None) -> dict[str, Any]:
        """Dependable single-op harvest: poll terminal states, then fetch.

        One process per launch: each requested kernel's status is polled at the
        configured cadence until it reaches a terminal state, then
        fetch_kernel_output downloads + hash-verifies + installs it (idempotent
        and fail-closed; re-running after an interruption simply completes the
        job). Polling holds neither a session nor a quota. On a non-complete
        terminal state partial artifacts and logs are downloaded, the failure
        is recorded, and the session itself is released through a stub
        version replace (stop_kernel); the aggregate returns only when every
        kind has been handled. Dry-run prints the plan.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        resolved_poll = poll_seconds if poll_seconds is not None else spec.logs_poll_seconds
        # One registry (config SSOT) resolves every kind's slug and watcher
        # identity; the finalize job shares the CPU slug but keeps its own
        # watcher kind so the two CPU steps' receipts stay distinct.
        identities = lane.kernel_identities(spec)
        slugs = {kind: identity.slug(spec) for kind, identity in identities.items()}
        unknown = [kind for kind in kinds if kind not in slugs]
        if unknown:
            raise RuntimeError(f"unknown supervise kind(s): {unknown}")
        missing = [kind for kind in kinds if not slugs[kind]]
        if missing:
            raise RuntimeError(
                f"config kaggle kernel slug unset for {missing}; name them in config")
        terminal = {"complete", "error", "cancelAcknowledged"}
        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "poll_seconds": resolved_poll,
            "kernels": {kind: slugs[kind] for kind in kinds},
        }
        if not execute:
            return plan
        import threading
        # Max visibility by default: one live log follower per kernel, through
        # the ONE kaggle-logs path (follow=True tails the session).
        threads = {kind: threading.Thread(
            target=lane.kernel_logs, args=(slugs[kind],),
            kwargs={"follow": True}, daemon=True,
            name=f'stream-{kind}') for kind in kinds}
        for thread in threads.values():
            thread.start()
        history: dict[str, list[dict[str, str]]] = {}
        outstanding = list(kinds)
        max_polls = spec.limits.max_polls if max_polls is None else max_polls
        polls = 0
        while outstanding:
            polls += 1
            if polls > max_polls:
                raise RuntimeError(
                    f"supervise gave up after {polls} polls; still outstanding: {outstanding}")
            for kind in list(outstanding):
                try:
                    status = lane.kernel_status(slug=slugs[kind])
                except (RuntimeError, OSError) as error:
                    lane._log_lane(f"[{slugs[kind]}] status query failed; retrying: {error}")
                    continue
                stamp = f"{lane.datetime.now(ZoneInfo(lane._spec().limits.timezone)):%Y-%m-%dT%H:%M:%S %Z}"
                lane._log_lane(f"[{slugs[kind]}] status={status['status']}")
                history.setdefault(kind, []).append({"at": stamp, "status": status["status"]})
                if status["status"] in terminal:
                    outstanding.remove(kind)
                    handled = lane.KernelLifecycle.harvest_and_stop(
                        kind=kind, slug=slugs[kind],
                        which=identities[kind].which,
                        status=status["status"], fetch_output=lane.fetch_kernel_output,
                        fetch_failure=lane.fetch_failed_kernel_log, stop=lane.stop_kernel,
                        configured_slug=slugs[kind])
                    if "fetch" in handled:
                        plan.setdefault("fetch", {})[kind] = handled["fetch"]
                    plan.setdefault("stop", {})[kind] = handled["stop"]
                    fetched = handled.get("fetch") or {}
                    stopped = handled.get("stop") or {}
                    lane._log_lane(
                        f"[{slugs[kind]}] harvest: status={status['status']} "
                        f"fetch_verified={bool(fetched.get('verified'))} "
                        f"stop={stopped.get('verdict') or stopped.get('stopped')}")
                    if status["status"] != "complete" or "fetch_error" in handled:
                        plan.setdefault("failures", {})[kind] = {
                            "status": status["status"], "raw": status.get("raw", ""),
                            **handled.get("failures", {}).get(kind, {}),
                            **({"fetch_error": handled["fetch_error"]}
                               if "fetch_error" in handled else {}),
                            "stop": handled["stop"],
                        }
            if outstanding:
                time.sleep(resolved_poll)
        plan["status_history"] = history
        for thread in threads.values():
            thread.join(timeout=spec.limits.stream_join_seconds)  # stream followers close at session teardown
        receipt_path = lane.staging_dir() / lane._spec().files.supervise_receipt
        try:
            lane.atomic_write_json(plan, receipt_path)
        except OSError as error:
            print(lane._stamp(), f"[kaggle-lane] supervise receipt write failed ({error}); continuing",
                  flush=True)
        plan["receipt"] = str(receipt_path)
        return plan

    @staticmethod
    def _logs_api():
        """The installed ``kaggle`` API the ``kaggle kernels logs`` CLI wraps.

        ONE construction site: every log read goes through the same installed
        ``KaggleApi.kernels_logs`` / ``kernels_logs_stream`` the CLI uses — never
        a bespoke HTTP/SSE client.
        """
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api

    @staticmethod
    def _append_log(destination: Path, kernel: str, text: str) -> None:
        """Append one decoded log chunk to the ONE transcript, tqdm-safe.

        The transcript is opened by the pusher's first ``_log_lane``; the log
        writer appends so it never wipes the watcher's status lines. CR-separated
        tqdm frames are expanded at write time (shared formatter) so the tail
        always shows the latest training bar.
        """
        if not text:
            return
        body = progress_frames_to_lines(text)
        if not body.endswith("\n"):
            body += "\n"
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(body)
        for line in body.splitlines():
            print(f"[stream {kernel}] {line}", flush=True)

    @staticmethod
    def kernel_logs(slug: str, *, follow: bool = False,
                    log_path: Path | None = None) -> dict[str, Any]:
        """Read a kernel's REAL execution log through the installed kaggle API.

        The ONE log path for every lane surface: ``KaggleApi.kernels_logs`` /
        ``kernels_logs_stream`` — the exact installed calls the ``kaggle kernels
        logs`` CLI (``-f`` to follow) wraps — never a bespoke SSE/HTTP client.
        ``follow=False`` returns the latest session's persisted stdout/stderr;
        ``follow=True`` tails the live session to END_OF_LOG with a bounded,
        429-aware reconnect (``ReconnectBackoff``) that fails LOUD with the full
        traceback rather than tight-looping. Decoded output is appended to
        ``log_path`` (default: the ONE transcript ``kaggle.files.lane_log``) and
        the self-reported ``[kaggle-session] session_id=`` marker arms the
        verified in-place stop. The terminated stream is the terminal signal;
        the exact final status is still read once by the watcher's status poll.
        """
        from cli import kaggle_lane as lane

        owner, slash, kernel = slug.rpartition("/")
        if not slash or not owner or not kernel:
            raise RuntimeError(f"kernel slug must be owner/slug, got {slug!r}")
        destination = log_path or lane.lane_log_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        api = KaggleMonitor._logs_api()
        plan: dict[str, Any] = {"kernel": slug, "stream_log": str(destination),
                                "mode": "follow" if follow else "latest"}
        if not follow:
            text = api.kernels_logs(slug) or ""
            KaggleMonitor._append_log(destination, kernel, text)
            reported = _reported_session_id(text)
            if reported is not None:
                KaggleMonitor.record_kernel_handle(slug, reported)
                plan["session_id"] = reported
            plan["logged_chars"] = len(text)
            plan["terminal"] = True
            return plan
        # The installed stream adapts: live SSE while the session runs, the
        # persisted blob once it is done; either way it ends at END_OF_LOG
        # (terminal). A reconnect replays from index 0, so an event counter
        # skips the already-written prefix instead of duplicating it.
        backoff = ReconnectBackoff(
            base_seconds=float(lane._spec().limits.retry_seconds))
        attempts = 0
        seen = 0
        while True:
            try:
                for index, event in enumerate(api.kernels_logs_stream(slug)):
                    if index < seen:
                        continue
                    seen = index + 1
                    data = event.get("data")
                    if data is None:
                        continue
                    KaggleMonitor._append_log(destination, kernel, data)
                    reported = _reported_session_id(data)
                    if reported is not None:
                        KaggleMonitor.record_kernel_handle(slug, reported)
                        plan["session_id"] = reported
                plan["terminal"] = True
                return plan
            except Exception as error:  # noqa: BLE001
                attempts += 1
                delay = backoff.delay(attempts, error)
                lane._log_lane(
                    f"[{kernel}] log stream attempt {attempts} failed: "
                    f"{type(error).__name__}: "
                    f"{str(error)[:lane._spec().limits.error_tail_chars]}")
                if backoff.is_rate_limited(error):
                    lane._log_lane(
                        f"[{kernel}] log stream rate-limited (429); backing off "
                        f"{delay:.0f}s (attempt {attempts})")
                if attempts >= lane._spec().limits.stream_retries:
                    lane._log_lane(traceback.format_exc())
                    raise RuntimeError(
                        f"log read for {slug} failed after {attempts} attempts: "
                        f"{type(error).__name__}: {error}") from error
                time.sleep(delay)

    @staticmethod
    def clear_kernel_session_id(slug: str) -> None:
        """Drop a stale recorded session id before a fresh push of ``slug``.

        The id file is process-local state under ``logs/kaggle``; leaving a
        previous run's id in place would let ``stop`` cancel a session that is
        already gone (or, worse, a different run's session).
        """
        from cli import kaggle_lane as lane

        _, _, kernel = slug.rpartition("/")
        if not kernel:
            return
        path = lane.lane_logs_dir() / lane._spec().files.session_id_file.format(kernel=kernel)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def recorded_kernel_handle(slug: str) -> str | None:
        """The persisted handle for a pushed kernel (session id OR kernel name).

        ONE reader for ``logs/kaggle/<kernel>.session_id``: the launch-aid id
        when captured, else the kernel-name fallback. ``None`` when nothing was
        recorded.
        """
        from cli import kaggle_lane as lane

        _, _, kernel = slug.rpartition("/")
        if not kernel:
            return None
        path = (lane.lane_logs_dir()
                / lane._spec().files.session_id_file.format(kernel=kernel))
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    @staticmethod
    def record_kernel_handle(slug: str, session_id: int | None) -> Path:
        """Persist the ONE usable handle for a pushed kernel.

        The launch-recorded ``kernel_session_id`` when captured (the verified
        in-place stop's target); otherwise the kernel NAME, so status/stop/
        output always have a non-empty handle and no ``session_id=None``
        dead-end remains. ``stop_kernel`` reads a non-numeric handle and takes
        its version-replace fallback against that same kernel.
        """
        from cli import kaggle_lane as lane

        _, _, kernel = slug.rpartition("/")
        if not kernel:
            raise ValueError(f"kernel slug must be owner/slug, got {slug!r}")
        path = (lane.lane_logs_dir()
                / lane._spec().files.session_id_file.format(kernel=kernel))
        path.parent.mkdir(parents=True, exist_ok=True)
        lane.atomic_write_text(
            path, f"{session_id if session_id is not None else kernel}\n")
        return path

    @staticmethod
    def capture_kernel_session_id(slug: str, *, attempts: int = 3,
                                  retry_seconds: float = 3.0,
                                  timeout_seconds: float = 15.0) -> dict[str, Any]:
        """Record a running kernel's session id so ``stop`` can cancel it in place.

        Connects the midtier log-stream proxy, reads the ``kernel_session_id``
        embedded in the stream URL — the same id the verified ``stop`` feeds to
        the SDK's in-place ``cancel_kernel_session`` — writes
        ``logs/kaggle/<kernel>.session_id`` atomically, and closes WITHOUT
        following the stream. Retries because the proxy only serves a URL once
        the session is up. When no id can be read (proxy down / stream 429), the
        kernel NAME is persisted instead (``record_kernel_handle``) so the stop
        handle is never empty. The SDK call has no client timeout, so each
        attempt runs in a daemon thread bounded by ``timeout_seconds``: a hung
        connect can never stall the synchronous launch path that awaits this
        capture.
        """
        import threading
        owner, slash, kernel = slug.rpartition("/")
        if not slash or not owner or not kernel:
            raise RuntimeError(f"kernel slug must be owner/slug, got {slug!r}")
        plan: dict[str, Any] = {"kernel": slug, "session_id": None}

        def probe(outcome: dict[str, Any]) -> None:
            """One bounded attempt: read the URL, always close the response."""
            try:
                from kagglesdk.kaggle_client import KaggleClient
                from kagglesdk.kaggle_env import KaggleEnv
                from kagglesdk.kernels.types.kernels_api_service import (
                    ApiGetKernelSessionLogsStreamRequest)
                request = ApiGetKernelSessionLogsStreamRequest()
                request.user_name = owner
                request.kernel_slug = kernel
                response = (KaggleClient(env=KaggleEnv.PROD).kernels
                            .kernels_api_client.get_kernel_session_logs_stream(request))
                try:
                    outcome["url"] = str(getattr(response, "url", "") or "")
                finally:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
            except Exception as error:  # proxy not up yet / transport hiccup
                outcome["error"] = f"{type(error).__name__}: {str(error)[:160]}"

        total = max(1, attempts)
        for attempt in range(total):
            outcome: dict[str, Any] = {}
            worker = threading.Thread(target=probe, args=(outcome,), daemon=True,
                                      name=f"capture-{kernel}")
            worker.start()
            worker.join(timeout=timeout_seconds)
            if worker.is_alive():
                plan["session_id"] = None
                plan["error"] = (f"TimeoutError: no stream URL within "
                                 f"{timeout_seconds:g}s")
            elif "error" in outcome:
                plan["session_id"] = None
                plan["error"] = outcome["error"]
            else:
                match = re.search(r'(\d{3,})(?:\?.*)?$', outcome.get("url", ""))
                if match:
                    session_id = int(match.group(1))
                    session_file = KaggleMonitor.record_kernel_handle(
                        slug, session_id)
                    plan["session_id"] = session_id
                    plan["handle"] = str(session_id)
                    plan["session_id_file"] = str(session_file)
                    plan.pop("error", None)
                    return plan
                plan["session_id"] = None
            if attempt + 1 < total:
                time.sleep(retry_seconds)
        # No id landed (proxy down / 429): persist the kernel NAME as the
        # fallback handle so stop/status/output stay targetable.
        session_file = KaggleMonitor.record_kernel_handle(slug, None)
        plan["handle"] = kernel
        plan["session_id_file"] = str(session_file)
        return plan

