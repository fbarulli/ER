"""Detached supervision, live progress, and terminal harvesting."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Sequence
from cli.log_capture import progress_frames_to_lines



class KaggleMonitor:
    """Detached supervision, live progress, and terminal harvesting."""

    @staticmethod
    def _spawn_autowatch(which: str, *, slug: str | None = None) -> dict[str, Any]:
        """Detached self-watch spawned by every live kernel push (no args needed).

        The watcher IS the terminal handler: poll to any terminal state, then
        download (results on complete / session log on error) and release the
        session via stub replace. Runs as its own session (setsid) so wrapper
        timeouts and shell deaths never orphan a live kernel: the failure the
        old supervise approach had. Dry-run never spawns; the explicit op
        (`--what autowatch`) shares this path for manual use. One spawn per
        push, ever: the chain reuses the push paths as-is and must never spawn
        a second watcher on top.
        """
        from cli import kaggle_lane as lane

        import subprocess
        watcher = lane.AUTOWATCH_WHICH.get(which)
        if watcher is None:
            raise RuntimeError(
                f"unknown autowatch kernel {which!r}; expected one of "
                f"{sorted(lane.AUTOWATCH_WHICH)}")
        which = watcher
        log_path = lane.lane_logs_dir() / lane._spec().files.autowatch_log.format(which=which)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # The pusher already opened the run's transcript fresh (its first
        # _log_lane truncated lane.log); append the launch marker and let the
        # detached watcher write the transcript itself through _log_lane +
        # stream_kernel_logs. Its stdout is discarded so no second file handle
        # competes for the same file (cosmetic echo would otherwise duplicate).
        with log_path.open("ab") as handle:
            handle.write(f"[_spawn_autowatch {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                         f"launching watcher for {which}]\n".encode())
        subprocess.Popen(
            [sys.executable, "-m", "cli.kaggle_lane", "--what", "autowatch",
             "--kernel", which, "--execute",
             *(["--slug", slug] if slug else [])],
            cwd=lane.TRAIN_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            env={**os.environ, "ER_KAGGLE_LANE_APPEND": "1",
                 "PYTHONPATH": os.pathsep.join(filter(None, [
                     str(lane.TRAIN_ROOT / lane._spec().files.source_dir), os.environ.get("PYTHONPATH")]))},
            start_new_session=True)
        return {"autowatch": "spawned", "kernel": which, "log": str(log_path)}

    @staticmethod
    def autowatch_kernel(which: str, *, execute: bool, slug: str | None = None,
                         poll_seconds: float | None = None) -> dict[str, Any]:
        """Hands-off terminal watcher: download then release, every single time.

        Replaces interactive supervision as the launch standard. One process:
        poll the kernel status until ANY terminal state, then (a) `complete`
        fetches + hash-verifies the result archive, (b) `error` fetches partial
        artifacts and the session log; (c) always pushes the stub replace that releases the
        session — the failed-kernel-solves case included. The release runs even
        when the watcher starts against an already-terminal kernel, so no
        session survives a finished run. Intended to run detached (setsid) so
        wrapper timeouts cannot kill it mid-poll.
        """
        from cli import kaggle_lane as lane

        spec = lane._spec()
        resolved_poll = poll_seconds if poll_seconds is not None else spec.logs_poll_seconds
        slugs = {"cpu": spec.cpu_kernel_slug, "gpu": spec.gpu_kernel_slug,
                 "embed": spec.embedding_kernel_slug}
        slug = slug or slugs.get(which)
        if not slug:
            raise RuntimeError(
                f"config kaggle.{which}_kernel_slug is unset; name the {which} "
                "kernel in config before autowatch")
        kind = {"cpu": "bundle", "gpu": "train", "embed": "embed"}[which]
        plan: dict[str, Any] = {
            "mode": "executed" if execute else "dry-run",
            "kernel": slug,
            "poll_seconds": resolved_poll,
        }
        if not execute:
            return plan
        import threading
        follower = threading.Thread(target=lane.stream_kernel_logs, args=(slug,),
                                    daemon=True, name=f"stream-{which}")
        follower.start()
        terminal = {"complete", "error", "cancelAcknowledged"}
        polls = 0
        while True:
            polls += 1
            try:
                status = lane.kernel_status(slug=slug)
            except (RuntimeError, OSError) as error:
                lane._log_lane(f"[{slug}] status query failed; retrying: {error}")
                time.sleep(resolved_poll)
                continue
            if status["status"] in terminal:
                break
            time.sleep(resolved_poll)
        plan["status"] = status["status"]
        plan["polls"] = polls
        plan.update(lane.KernelLifecycle.harvest_and_stop(
            kind=kind, slug=slug, which=which, status=status["status"],
            fetch_output=lane.fetch_kernel_output, fetch_failure=lane.fetch_failed_kernel_log,
            stop=lane.stop_kernel, configured_slug=slugs[which]))
        follower.join(timeout=spec.limits.stream_join_seconds)
        receipt = lane.staging_dir() / lane._spec().files.autowatch_receipt.format(kind=kind)
        try:
            lane.atomic_write_json(plan, receipt)
        except OSError as error:
            print(f"[kaggle-lane] autowatch receipt write failed ({error}); "
                  "continuing", flush=True)
        plan["receipt"] = str(receipt)
        return plan

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
        slugs = {
            "bundle": spec.cpu_kernel_slug,
            "train": spec.gpu_kernel_slug,
            "embed": spec.embedding_kernel_slug,
        }
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
                        which={"bundle": "cpu", "train": "gpu", "embed": "embed"}[kind],
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
        persisted_lines = 0
        attempt_lines = 0

        def emit(text: str) -> None:
            """Append one chunk as logical lines, skipping replayed prefix.

            The midtier SSE proxy replays from the session's first line on a
            reconnect; persisted_lines counts lines already written so each new
            attempt drops the prefix instead of duplicating it.
            """
            nonlocal persisted_lines, attempt_lines
            if not text.endswith("\n"):
                text += "\n"
            for piece in text.splitlines(keepends=True):
                if attempt_lines < persisted_lines:
                    attempt_lines += 1
                    continue
                log_handle.write(piece)
                attempt_lines += 1
                persisted_lines += 1
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
            client = KaggleClient(env=KaggleEnv.PROD)
            attempts = 0
            while True:
                attempt_lines = 0
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
                            append_progress(str(payload.get("data", "")), line)
                            for chunk in (str(payload.get("data", "")).splitlines()
                                          or [""]):
                                print(f"[stream {kernel}] {chunk}", flush=True)
                        else:
                            append_progress(None, line)
                            print(f"[stream {kernel}] {line}", flush=True)
                    # A clean connection completed: the counter holds only the
                    # current burst, never whole-run history.
                    attempts = 0
                    break
                except (ProtocolError, requests.exceptions.RequestException) as error:
                    # The midtier SSE proxy drops live connections mid-run; a
                    # replayed stream re-attaches at the session's FIRST line.
                    # persisted_lines already holds the prefix, so the next
                    # attempt appends only what emit has not seen.
                    attempts += 1
                    if attempts > lane._spec().limits.stream_retries:
                        # Server-side drops exhaust the cap; visibility only —
                        # never kill the watcher's status-poll contract on it.
                        lane._log_lane(
                            f"[stream {kernel}] follower exhausted after "
                            f"{attempts} reconnects; status-poll only for the rest of the session")
                        break
                    lane._log_lane(f"[stream {kernel}] reconnect attempt {attempts}: "
                              f"{type(error).__name__}: {str(error)[:lane._spec().limits.error_tail_chars]}")
                    time.sleep(lane._spec().limits.retry_seconds * attempts)
        plan["session_id"] = session_id
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
        fetch = [executable, "kernels", "output", slug, "-p", str(log_dir / slug.replace("/", "__"))]
        try:
            _, _ = lane._run_kaggle(fetch)
            plan["log_fetched"] = True
        except RuntimeError as error:
            plan["log_fetched"] = False
            plan["log_error"] = str(error)[-lane._spec().limits.error_tail_chars:]
        plan["history"] = history
        return plan

