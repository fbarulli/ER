"""Colab CLI transport: control channel, exec stream/capture, remote staging.

The single process boundary between the launcher and the Colab CLI: the safe
wrapper command, the serialized control lock, the CLI call itself, uploads with
transient retries, the streaming/capturing exec channel, the detached remote
stage launcher, and the visible result download.  Split from cli/colab.py (the
kaggle_lane.py owner-module pattern) exactly like colab_runtime /
colab_result_sync / colab_launch / the prewarm modules.

Collaborators still owned by cli.colab (the CLI state paths, config constants,
the live/training log state and helpers, the bootstrap preamble, receipts) are
re-read through ``colab_hub.hub()`` at call time, so the legacy ``from cli import colab``
monkeypatch surface keeps driving every call and the running colab identity
never sees a stale second copy.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from cli.colab_hub import hub, timed_colab
from cli.colab_reconnect import (
    ControlChannelLoss,
    ControlChannelLost,
    ControlChannelRecovery,
)


def _colab_command(*args: str) -> list[str]:
    """Build every Colab CLI command through the shared safe entrypoint."""
    surface = hub()
    surface._COLAB_CLI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    colab_executable = shutil.which("colab")
    if not colab_executable:
        return ["colab", *args]
    first_line = Path(colab_executable).read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("#!"):
        return ["colab", *args]
    colab_python = first_line[2:].strip()
    return [
        colab_python,
        str(surface._COLAB_CLI_ENTRYPOINT),
        "--config",
        str(surface._COLAB_CLI_CONFIG),
        *args,
    ]


_colab_control_lock = threading.RLock()


def _serialize_colab_control(function):
    """Keep one notebook kernel control request in flight per launcher."""
    def wrapped(*args, **kwargs):
        with _colab_control_lock:
            return function(*args, **kwargs)
    return wrapped


@_serialize_colab_control
@timed_colab("event")
def colab(*args: str, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a Colab CLI subcommand through the shared safe entrypoint."""
    surface = hub()
    display_cmd = ["colab", *args]
    cmd = surface._colab_command(*args)
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        print(surface._stamp(), f"\n[error] colab command failed: {' '.join(display_cmd)}", file=sys.stderr)
        if e.stdout:
            print(f"stdout:\n{e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"stderr:\n{e.stderr}", file=sys.stderr)
        traceback.print_exc()
        raise


@timed_colab("event")
def _ensure_remote_parent(remote: str) -> None:
    """Create an upload target's remote directory before the transfer.

    The Colab contents API answers 500 (never 404) when the parent of an upload
    target is missing, and the VM checkout only carries the directories Git
    tracks: a lane upload must therefore establish its own directory. Without
    this, every upload into a not-yet-checked-out directory fails as an opaque
    server error.
    """
    surface = hub()
    parent = str(PurePosixPath(remote).parent)
    surface.run_colab_exec_stream(
        surface.SESSION,
        "import pathlib\n"
        f"pathlib.Path({parent!r}).mkdir(parents=True, exist_ok=True)\n",
        timeout=surface._PROBE_TIMEOUT_SECONDS,
        log_name="upload_dir",
    )


@timed_colab("event")
def _upload_with_retries(source: Path, remote: str, *, timeout: int) -> None:
    """Retry transient Colab upload/control-channel failures."""
    surface = hub()
    surface._ensure_remote_parent(remote)
    for attempt in range(1, surface._REMOTE_UPLOAD_RETRIES + 1):
        try:
            surface.colab("upload", "-s", surface.SESSION, str(source), remote, timeout=timeout)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == surface._REMOTE_UPLOAD_RETRIES:
                raise
            delay = min(surface._COLAB.remote_upload_max_backoff_seconds,
                        surface._COLAB.remote_upload_backoff_seconds * attempt)
            print(
                f"[upload] retry {attempt}/{surface._REMOTE_UPLOAD_RETRIES - 1} for {source.name} "
                f"after transient failure; waiting {delay}s",
                flush=True,
            )
            time.sleep(delay)


@_serialize_colab_control
@timed_colab("event")
def run_colab_exec_stream(
    session: str,
    script: str,
    timeout: int | None = None,
    log_name: str | None = None,
    *,
    retry_safe: bool = False,
    exclude_from_live_log: bool = False,
    training_output: bool = False,
) -> None:
    """Execute a python script on the colab session via stdin, streaming stdout/stderr.

    log_name labels a stage in the one per-run launcher transcript
    (logs/colab/lane.log under the canonical logs root). The file is opened
    once per run, line-flushed, and survives VM teardown so every Colab stage
    is inspectable in one chronological log.
    """
    surface = hub()
    if surface._live_log and log_name:
        print(surface._stamp(), f"\n===== {log_name} =====", flush=True)

    def stream_output(pipe, prefix, captured, remote_output=None):
        pending = ""

        def emit(text: str) -> None:
            nonlocal pending
            if training_output:
                surface._write_training_log(text)
            pending += text
            # tqdm uses carriage returns instead of newlines. Emit each
            # progress update immediately so a long encode/training stage
            # cannot appear hung in the terminal.
            while "\n" in pending or "\r" in pending:
                newline_positions = [p for p in (pending.find("\n"), pending.find("\r")) if p >= 0]
                end = min(newline_positions)
                unit = pending[:end].rstrip()
                captured.append(pending[: end + 1])
                if remote_output is not None:
                    remote_output.append(pending[: end + 1])
                pending = pending[end + 1:]
                if unit:
                    context = surface._LiveLogSuppressed() if exclude_from_live_log else nullcontext()
                    with context:
                        print(f"{prefix} {unit}", flush=True)

        while True:
            chunk = pipe.read(1)
            if chunk == "":
                break
            emit(chunk)
        if pending:
            if training_output:
                surface._write_training_log(pending)
            captured.append(pending)
            if remote_output is not None:
                remote_output.append(pending)
            context = surface._LiveLogSuppressed() if exclude_from_live_log else nullcontext()
            with context:
                print(f"{prefix} {pending.rstrip()}", flush=True)
        pipe.close()

    attempts = surface._PROBE_RETRIES if retry_safe else 1
    for attempt in range(1, attempts + 1):
        process = subprocess.Popen(
            # colab exec has its own 30-second kernel-client timeout.  It must
            # match the caller's legitimate lane timeout; otherwise a live VM
            # computation is reported as failed after 30 seconds.
            surface._colab_command("exec", "-s", session, "--timeout", str(timeout or 30)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        captured: list[str] = []
        remote_output: list[str] = []
        heartbeat_stop = threading.Event()

        def emit_heartbeat() -> None:
            started = time.monotonic()
            while not heartbeat_stop.wait(30):
                print(
                    surface._stamp(),
                    f"[stream] {log_name or 'remote stage'} still active "
                    f"({time.monotonic() - started:.0f}s elapsed; awaiting remote output)",
                    flush=True,
                )

        heartbeat = threading.Thread(target=emit_heartbeat, daemon=True)
        heartbeat.start()
        out_thread = threading.Thread(target=stream_output, args=(process.stdout, "[out]", captured, remote_output))
        err_thread = threading.Thread(target=stream_output, args=(process.stderr, "[err]", captured))
        out_thread.start()
        err_thread.start()
        assert process.stdin is not None
        process.stdin.write(script)
        process.stdin.close()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            out_thread.join()
            err_thread.join()
            heartbeat_stop.set()
            heartbeat.join(timeout=1)
            output = "".join(captured)
            raise RuntimeError(
                f"Remote execution timed out after {timeout}s.\n"
                "--- complete remote output / traceback ---\n"
                f"{output}"
            ) from exc
        out_thread.join()
        err_thread.join()
        heartbeat_stop.set()
        heartbeat.join(timeout=1)
        output = "".join(captured)
        # Some CLI versions report notebook execution errors with exit code 0.
        # Preserve the fail-fast contract before provisioning the next stage.
        # CLI destructor diagnostics are local stderr, not notebook failures.
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', ''.join(remote_output))
        lines = clean_output.splitlines()
        traceback_heads = [index for index, line in enumerate(lines)
                           if 'Traceback (most recent call last)' in line]
        remote_traceback = bool(traceback_heads)
        recovered_traceback = remote_traceback and all(
            any(lines[prior].lstrip().startswith('[traceback] ')
                for prior in range(max(0, index - 3), index))
            for index in traceback_heads
        )
        if process.returncode == 0 and (not remote_traceback or recovered_traceback):
            return
        transient = ControlChannelRecovery.classify(
            surface.SESSION, output) is ControlChannelLoss.TRANSIENT
        if retry_safe and transient and attempt < attempts:
            delay = surface._PROBE_RETRY_BACKOFF_SECONDS * attempt
            print(
                surface._stamp(),
                f"[stream] transient kernel connection loss ({attempt}/{attempts}); "
                f"retrying safe stage in {delay}s",
                flush=True,
            )
            time.sleep(delay)
            continue
        raise RuntimeError(
            f"Remote execution failed with return code {process.returncode}.\n"
            "--- complete remote output / traceback ---\n"
            f"{output}"
        )


@_serialize_colab_control
@timed_colab("event")
def run_colab_exec_capture(
    session: str, script: str, timeout: int, *, training_output: bool = False,
) -> str:
    """Execute a bounded remote probe while retaining stdout for parsing.

    Probes serve the live training log.  They must reveal control-channel
    failures promptly instead of becoming an opaque multi-minute wait.
    """
    surface = hub()

    def report_probe_progress(message: str) -> None:
        """Make a blocked training-log probe observable in its durable log."""
        print(f"{surface._stamp()} {message}", flush=True)
        if training_output:
            surface._write_training_log(message + "\n")

    last_error = ""
    for attempt in range(1, surface._PROBE_RETRIES + 1):
        try:
            process = subprocess.Popen(
                surface._colab_command("exec", "-s", session, "--timeout", str(timeout)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            captured: list[str] = []
            heartbeat_stop = threading.Event()

            def stream_probe_output() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    captured.append(line)
                    if line.rstrip() == f"[colab] Session '{session}' not found.":
                        continue
                    # The normal worker probe emits one JSON payload, which
                    # is decoded below.  Surface non-JSON diagnostics now so
                    # an upstream client/kernel failure is visible at once.
                    if not line.lstrip().startswith(("{", "[")):
                        if "jupyter_kernel_client" in line or "KernelClient" in line:
                            continue
                        stripped = line.strip()
                        if stripped in ("}", "]", "},", "],", "{", "["):
                            continue
                        report_probe_progress(f"[probe-out] {line.rstrip()}")

            def emit_heartbeat() -> None:
                started = time.monotonic()
                while not heartbeat_stop.wait(15):
                    report_probe_progress(
                        f"[probe] awaiting remote log/status "
                        f"({time.monotonic() - started:.0f}s; timeout={timeout}s)"
                    )

            reader = threading.Thread(target=stream_probe_output, daemon=True)
            heartbeat = threading.Thread(target=emit_heartbeat, daemon=True)
            reader.start()
            heartbeat.start()
            assert process.stdin is not None
            process.stdin.write(script)
            process.stdin.close()
            process.wait(timeout=timeout + 30)
            reader.join()
            heartbeat_stop.set()
            heartbeat.join(timeout=1)
            output = "".join(captured)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            reader.join()
            heartbeat_stop.set()
            heartbeat.join(timeout=1)
            last_error = f"probe timeout: {exc}"
            report_probe_progress(
                f"[probe] timeout after {timeout}s on attempt {attempt}/{surface._PROBE_RETRIES}"
            )
            traceback.print_exc()
        else:
            if process.returncode == 0:
                return output
            last_error = f"rc={process.returncode}: {output[-2000:]}"
            report_probe_progress(
                f"[probe] remote command rc={process.returncode}; stderr tail:"
            )
            report_probe_progress(output[-2000:])
        if attempt < surface._PROBE_RETRIES:
            delay = surface._PROBE_RETRY_BACKOFF_SECONDS * attempt
            time.sleep(delay)
    raise RuntimeError(f"remote log probe failed after {surface._PROBE_RETRIES} attempts: {last_error}")


def _parse_remote_json(output: str) -> dict:
    """Read the last JSON object from a Colab probe without trusting banners."""
    clean_output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    for line in reversed(clean_output.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"remote log probe returned no JSON: {clean_output[-1000:]}")


@timed_colab("event")
def run_detached_stage(stage: str, command_expr: list[str], timeout: int) -> None:
    """Run a VM stage outside the notebook kernel and stream its durable log."""
    surface = hub()
    # Two launches can occur within the same UTC second (especially after a
    # failed preflight). Microseconds keep the remote result root unique and
    # prevent FileExistsError from aborting before workers launch.
    stamp = datetime.now(timezone.utc).strftime("%m%dT%H%M%S%fZ")
    remote_log = f"{surface.REMOTE_ROOT}/results/logs/colab_stages/{stage}_{stamp}.log"
    remote_status = f"{remote_log}.status"
    remote_pid = f"{remote_log}.pid"
    launch = surface._BOOTSTRAP + f"""
import json, os, pathlib, shlex, subprocess, sys
log_path = pathlib.Path({remote_log!r})
status_path = pathlib.Path({remote_status!r})
pid_path = pathlib.Path({remote_pid!r})
log_path.parent.mkdir(parents=True, exist_ok=True)
running_pid = None
if status_path.is_file():
    print(json.dumps({{"pid": None, "log": str(log_path), "status": str(status_path)}}), flush=True)
else:
    try:
        candidate = int(pid_path.read_text(encoding="utf-8"))
        os.kill(candidate, 0)
        running_pid = candidate
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        pass
    if running_pid is None:
        status_path.unlink(missing_ok=True)
        pid_path.unlink(missing_ok=True)
        command = ["timeout", "--signal=TERM", "--kill-after=60", str({timeout})] + {command_expr}
        command_text = " ".join(shlex.quote(part) for part in command)
        wrapped = (
            "echo '[stage] started'; "
            + command_text
            + "; rc=$?; echo '[stage] exited rc='$rc; "
            + "printf '%s\\n' \\\"$rc\\\" > "
            + shlex.quote(str(status_path))
        )
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            child = subprocess.Popen(
                ["/bin/bash", "-lc", wrapped],
                cwd={surface.REMOTE_ROOT!r},
                env={{**os.environ, "PYTHONUNBUFFERED": "1"}},
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(str(child.pid), encoding="utf-8")
        running_pid = child.pid
print(json.dumps({{"pid": running_pid, "log": str(log_path), "status": str(status_path)}}), flush=True)
"""
    print(surface._stamp(), f"[{stage}] starting detached remote stage; durable log={remote_log}", flush=True)
    recovery = ControlChannelRecovery.for_running_launcher()
    try:
        launched = surface._parse_remote_json(
            recovery.run(
                lambda: surface.run_colab_exec_capture(surface.SESSION, launch, timeout=120),
                context=f"{stage} stage launch",
            )
        )
        print(surface._stamp(), f"[{stage}] remote pid={launched.get('pid')}", flush=True)
        offset = 0
        delay = surface._LOG_POLL_INITIAL_SECONDS
        probes = 0
        poll_started = time.perf_counter()
        poll_seconds = 0.0
        while True:
            probes += 1
            poll_seconds = time.perf_counter() - poll_started
            probe = surface._BOOTSTRAP + f"""
import json, pathlib
log_path = pathlib.Path({remote_log!r})
status_path = pathlib.Path({remote_status!r})
offset = {offset}
data = b""
if log_path.is_file():
    with log_path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
payload = {{
    "offset": offset + len(data),
    "chunk": data.decode("utf-8", errors="replace"),
    "done": status_path.is_file(),
    "returncode": status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else None,
}}
print(json.dumps(payload), flush=True)
"""
            try:
                # A detached trainer outlives a transient log-probe failure: the
                # recovery re-attaches the control channel and re-reads the
                # durable log, instead of turning the drop into a run failure.
                payload = recovery.run(
                    lambda: surface._parse_remote_json(
                        surface.run_colab_exec_capture(
                            surface.SESSION, probe,
                            timeout=surface._PROBE_TIMEOUT_SECONDS)),
                    context=f"{stage} log/status probe",
                )
            except ControlChannelLost:
                raise
            except RuntimeError as exc:
                # A probe failure that is not a channel loss must never hide a
                # dead runtime: the all-tracks poll tolerates it inside its
                # stage budget, every other stage keeps its fail-fast rule.
                if stage != 'all_tracks' or time.perf_counter() - poll_started >= timeout:
                    raise
                message = f"[{stage}] log/status unavailable; detached training continues: {exc}"
                surface._write_training_log(message + "\n")
                print(f"{surface._stamp()} {message}", flush=True)
                time.sleep(surface._LOG_POLL_SECONDS)
                continue
            offset = int(payload["offset"])
            if payload["chunk"]:
                for line in str(payload["chunk"]).splitlines():
                    if stage == 'all_tracks' and line.startswith('[track/'):
                        surface._write_training_log(line + '\n')
                    print(f"[{stage}] {line}", flush=True)
            if payload["done"]:
                returncode = int(payload["returncode"])
                if returncode:
                    raise RuntimeError(
                        f"remote stage {stage} failed (rc={returncode}); "
                        f"system log contains the streamed output; remote log={remote_log}"
                    )
                print(
                    f"[{stage}] completed successfully after {poll_seconds:.2f}s "
                    f"of polling ({probes} probe(s))",
                    flush=True,
                )
                return
            # A stage whose work is seconds long must not pay a full poll
            # interval, nor a round trip per interval, merely to be noticed.
            # Start tight and back off to the configured steady interval, so
            # short stages are seen almost immediately and long ones cost the
            # same few probes as before.
            delay = min(surface._LOG_POLL_SECONDS, max(surface._LOG_POLL_INITIAL_SECONDS, delay * 2))
            time.sleep(delay)
    except BaseException as exc:
        raise RuntimeError(
            f"remote stage {stage} lost its control connection; "
            f"system log contains the streamed output; remote log={remote_log}; cause={exc}"
        ) from exc


def _format_bytes(value: int) -> str:
    """Format transfer progress without hiding the raw byte count."""
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f}{unit} ({value} bytes)"
        amount /= 1024.0
    raise AssertionError("unreachable byte-format branch")


def _local_file_size(path: Path) -> int:
    """Return partial-download size while tolerating a missing destination."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _download_file_with_visibility(
    *,
    remote: str,
    local: Path,
    worker: int | None,
    index: int,
    total: int,
    run_id: str,
) -> int:
    """Download one result while exposing progress and connection failures."""
    surface = hub()
    relative = local.relative_to(surface.TRAINING_RESULTS / run_id)
    worker_label = str(worker) if worker is not None else "all"
    print(
        surface._stamp(),
        f"[download] worker={worker_label} file={index}/{total} starting "
        f"remote={remote} destination={local}",
        flush=True,
    )
    surface._result_event(
        run_id,
        "download_file",
        "started",
        worker=worker,
        file=str(relative),
        index=index,
        total=total,
    )
    stop_heartbeat = threading.Event()

    def report_progress() -> None:
        while not stop_heartbeat.wait(surface._RESULT_DOWNLOAD_HEARTBEAT_SECONDS):
            received = _local_file_size(local)
            print(
                surface._stamp(),
                f"[download] worker={worker_label} file={index}/{total} active "
                f"received={_format_bytes(received)}; waiting for transfer",
                flush=True,
            )

    heartbeat = threading.Thread(
        target=report_progress,
        name=f"result-download-heartbeat-{worker}-{index}",
        daemon=True,
    )
    heartbeat.start()
    try:
        surface.colab(
            "download",
            "-s",
            surface.SESSION,
            remote,
            str(local),
            timeout=surface._RESULT_DOWNLOAD_TIMEOUT_SECONDS,
        )
    except BaseException as exc:
        received = _local_file_size(local)
        surface._result_event(
            run_id,
            "download_file",
            "failed",
            worker=worker,
            file=str(relative),
            index=index,
            total=total,
            received_bytes=received,
            error=f"{type(exc).__name__}: {exc}",
        )
        print(
            surface._stamp(),
            f"[download] worker={worker_label} file={index}/{total} FAILED "
            f"received={_format_bytes(received)} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    finally:
        stop_heartbeat.set()
        heartbeat.join()
    received = _local_file_size(local)
    surface._result_event(
        run_id,
        "download_file",
        "completed",
        worker=worker,
        file=str(relative),
        index=index,
        total=total,
        received_bytes=received,
    )
    print(
        surface._stamp(),
        f"[download] worker={worker_label} file={index}/{total} completed "
        f"received={_format_bytes(received)}",
        flush=True,
    )
    return received
