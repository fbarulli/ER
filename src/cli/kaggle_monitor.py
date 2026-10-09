"""Detached supervision, live progress, and terminal harvesting."""
from __future__ import annotations

import json
import os
import re
import sys
import time
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
            stream_logs=lane.stream_kernel_logs,
            kernel_status=lane.kernel_status,
            log_lane=lane._log_lane,
            write_json=atomic_write_json,
            receipt_path=lane.staging_dir()
            / spec.files.autowatch_receipt.format(kind=identity.kind),
            log_path=lane.lane_logs_dir()
            / spec.files.autowatch_log.format(which=which),
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
        # Max visibility by default: one live log-stream follower per kernel.
        threads = {kind: threading.Thread(
            target=lane.stream_kernel_logs, args=(slugs[kind],), daemon=True,
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
    def stream_kernel_logs(slug: str, log_path: Path | None = None) -> dict[str, Any]:
        """Follow a session's live log stream (max visibility, owner default).

        kaggle's CLI only shows status until teardown; the midtier's SSE log
        proxy (kagglesdk GET->KERNELS GetKernelSessionLogsStream) exposes the
        run's live stdout/stderr — the tqdm bars included. The proxy URL embeds
        the kernel_session_id, which also feeds the manual kill switch.
        Appends every decoded data payload post-processed (CR frames -> lines +
        tagged last bar, cli.log_capture) to log_path (default: the run's
        logs/kaggle/lane.log) and echoes decoded lines to the console. The run
        transcript is opened fresh once per run by the pusher's first _log_lane;
        the follower appends so it never wipes the watcher's status lines. A
        dropped SSE connection replays from the session's FIRST line, so the
        follower tracks how many lines it already persisted and skips the
        replayed prefix instead of truncating the shared transcript."""
        from cli import kaggle_lane as lane

        from kagglesdk.kaggle_client import KaggleClient
        from kagglesdk.kaggle_env import KaggleEnv
        from kagglesdk.common.types.file_download import FileDownload
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiGetKernelSessionLogsStreamRequest)
        from urllib3.exceptions import ProtocolError
        import requests

        owner, slash, kernel = slug.rpartition("/")
        if not slash or not owner or not kernel:
            raise RuntimeError(f"kernel slug must be owner/slug, got {slug!r}")
        destination = log_path or (lane.lane_logs_dir() /
                                   lane._spec().files.stream_log.format(kernel=kernel))
        destination.parent.mkdir(parents=True, exist_ok=True)
        plan: dict[str, Any] = {"kernel": slug, "stream_log": str(destination)}
        session_id: int | None = None
        # One follower per kernel: two writers race the transcript, each with
        # its own replay state. Refuse to start if the recorded pid is alive; a
        # stale pid left by a dead follower is overwritten.
        lock_path = lane.lane_logs_dir() / f"{kernel}.follower.pid"
        try:
            holder = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            holder = None
        if holder is not None:
            try:
                os.kill(holder, 0)
            except ProcessLookupError:
                holder = None  # the previous follower is gone
            except PermissionError:
                pass  # alive (owned by another uid on this box)
            if holder is not None:
                raise RuntimeError(
                    f"a stream follower for {slug} is already running "
                    f"(pid {holder}); refusing a second writer on the transcript")
        lane.atomic_write_text(lock_path, str(os.getpid()) + "\n")
        # On a dropped connection the midtier replays the WHOLE session from
        # line 0. We do NOT try to dedup that replay: byte/line/time counters all
        # drift because the replay is not a prefix of what we wrote, and the
        # drift swallows the live tail (the log freezes, then dumps late).
        # Instead the transcript is truncated and rewritten from the replay on
        # every reconnect, so it always mirrors the full current session.

        def emit(text: str) -> None:
            if not text.endswith("\n"):
                text += "\n"
            log_handle.write(text)
            log_handle.flush()

        def append_progress(payload_text: str | None, raw: str) -> None:
            """Append one captured chunk as grep-able, post-processed lines.

            SSE frames that carry tqdm's \r-separated progress bars are expanded
            at write time (shared helper, cli.log_capture) so the log tail always
            shows the last training bar; non-JSON events are kept verbatim.
            """
            if payload_text is None:
                emit(raw)
                return
            emit(progress_frames_to_lines(payload_text or ""))
        with destination.open("a", encoding="utf-8") as log_handle:
            # This follower owns only the section after the run's status lines
            # (the pusher's _log_lane writes those first). On a reconnect the
            # whole replay is rewritten into that section.
            log_handle.seek(0, os.SEEK_END)
            section_start = log_handle.tell()
            client = KaggleClient(env=KaggleEnv.PROD)
            attempts = 0
            backoff = ReconnectBackoff(
                base_seconds=float(lane._spec().limits.retry_seconds))
            while True:
                try:
                    request = ApiGetKernelSessionLogsStreamRequest()
                    request.user_name = owner
                    request.kernel_slug = kernel
                    response = client.kernels.kernels_api_client \
                        .get_kernel_session_logs_stream(request)
                    # FileDownload.prepare_from returns the live streamed requests.Response
                    # (text/event-stream, "data: {stream_name,time,data}" SSE frames).
                    plan["stream_url"] = str(getattr(response, "url", "") or "")
                    # The SSE proxy URL embeds the kernel_session_id (the same
                    # id the manual kill switch consumes): capture it once the
                    # stream URL is known so the verified stop can cancel the
                    # exact session instead of blind version replace.
                    url_match = re.search(r'(\d{3,})(?:\?.*)?$', str(plan['stream_url']) or '')
                    if url_match:
                        session_id = int(url_match.group(1))
                        lane.atomic_write_text(
                            lane.lane_logs_dir()
                            / lane._spec().files.session_id_file.format(kernel=kernel),
                            str(session_id) + '\n')
                    # Decode UTF-8 explicitly: iter_lines(decode_unicode=True) would use
                    # requests' latin-1 default and mangle the box-drawing progress bars.
                    for raw in response.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="replace") \
                            if isinstance(raw, bytes) else raw
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            emit(line)
                            continue
                        try:
                            payload = json.loads(line[5:].strip())
                        except (json.JSONDecodeError, ValueError):
                            payload = None
                        if isinstance(payload, dict):
                            data_text = str(payload.get("data", ""))
                            reported = _reported_session_id(data_text)
                            if reported is not None and session_id is None:
                                session_id = reported
                                plan["session_id"] = reported
                                lane.atomic_write_text(
                                    lane.lane_logs_dir()
                                    / lane._spec().files.session_id_file.format(
                                        kernel=kernel),
                                    str(reported) + "\n")
                            append_progress(data_text, line)
                            for chunk in (data_text.splitlines() or [""]):
                                print(f"[stream {kernel}] {chunk}", flush=True)
                        else:
                            append_progress(None, line)
                            print(f"[stream {kernel}] {line}", flush=True)
                    # A clean connection completed: the counter holds only the
                    # current burst, never whole-run history.
                    attempts = 0
                    break
                except Exception as error:  # noqa: BLE001
                    # A detached follower must outlive EVERY transport hiccup:
                    # the midtier drops live connections repeatedly, and a
                    # reconnect cap (stream_retries) is exactly what froze the
                    # transcript mid-run — lane.log stopped and the logs only
                    # appeared when the session ended. Keep reconnecting until
                    # the SESSION ends (a clean END_OF_LOG, handled above).
                    attempts += 1
                    lane._log_lane(f"[stream {kernel}] reconnect attempt {attempts}: "
                              f"{type(error).__name__}: {str(error)[:lane._spec().limits.error_tail_chars]}")
                    # The next attempt replays from line 0: drop this follower's
                    # section (not the run's status lines) and rewrite it from
                    # the replay (no dedup, no drift).
                    log_handle.flush()
                    log_handle.seek(section_start)
                    log_handle.truncate()
                    log_handle.seek(0, os.SEEK_END)
                    delay = backoff.delay(attempts, error)
                    if backoff.is_rate_limited(error):
                        lane._log_lane(
                            f"[stream {kernel}] rate-limited (429); backing off "
                            f"{delay:.0f}s before reconnect {attempts + 1}")
                    time.sleep(delay)
        plan["session_id"] = session_id
        return plan

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

    @staticmethod
    def kernel_logs(*, slug: str, poll_seconds: float | None = None, follow: bool,
                    execute: bool) -> dict[str, Any]:
        """Poll kernel status; on terminal states pull output logs locally.

        Colab streams VM stdout into local transcripts; Kaggle exposes no live
        stream, so this is the honest equivalent: status polling with the
        configured executable (cadence from config kaggle.logs_poll_seconds)
        and, on terminal states, `kernels output` fetch of the kernel's own log
        file into the lane logs dir (logs/kaggle/, TRAIN_ROOT-relative SSOT).
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        resolved_poll = poll_seconds if poll_seconds is not None else spec.logs_poll_seconds
        log_dir = lane.lane_logs_dir()
        plan: dict[str, Any] = {
            "kernel": slug,
            "poll_seconds": resolved_poll,
            "follow": follow,
            "log_dir": str(log_dir),
            "mode": "executed" if execute else "dry-run",
        }
        if not execute:
            return plan
        executable = lane._require_kaggle_executable(spec.kaggle_executable)
        history: list[dict[str, Any]] = []
        while True:
            status = lane.kernel_status(slug)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            lane._log_lane(f"[{slug}] status={status['status']}")
            history.append({"at": stamp, "status": status["status"]})
            if status["status"] in {"complete", "error", "cancelAcknowledged"} or not follow:
                break
            time.sleep(resolved_poll)
        log_dir.mkdir(parents=True, exist_ok=True)
        # Fail LOUD: `kaggle kernels output` exits 0 with zero files, so rc
        # alone is not success. The fetcher verifies files landed (and paces
        # 429s) before this method reports `log_fetched`.
        from cli.kaggle_download import DownloadError, KernelOutputFetcher

        try:
            download = KernelOutputFetcher(
                argv_prefix=(executable,), cwd=lane.TRAIN_ROOT).fetch(
                slug, log_dir / slug.replace("/", "__"), require_globs=())
        except DownloadError as error:
            plan["log_fetched"] = False
            plan["log_error"] = f"{error}\n{error.traceback_text}"
            plan["history"] = history
            raise DownloadError(
                f"kernel output for {slug} was empty or failed: {error}",
                traceback_text=error.traceback_text,
                stdout=error.stdout) from error
        plan["log_fetched"] = True
        plan["files"] = [str(path) for path in download.files]
        plan["download_attempts"] = download.attempts
        plan["history"] = history
        return plan

