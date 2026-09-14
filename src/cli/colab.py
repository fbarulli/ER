"""colab_backend.py — run EuromonitoR TRAIN work on a Colab GPU VM.

The Colab CLI (google-colab-cli) provisions a Colab runtime (T4 default —
free-tier GPU, enough for sentence-transformer fine-tuning), pushes code +
data, executes a lane, and pulls results back.

Lanes (post second-series rename — the old second03/second04 scripts are
now the src/training/ module chain):
  train  — full-chain GPU training: data_prep -> train.py (contrastive,
           OnlineContrastiveLoss, holdout 50/25/25). The production run:
           the CPU lane proved the chain but 15s/step * 740 steps is 3h;
           the T4 does ~1.5-2s/step.
  hpo    — masking-enabled Optuna TPE search. Each trial trains on 50%,
           selects on the dev 25%, and does not read the test 25%.
  sims   — the configured zero-shot embedding lane. The current config uses
           minilm_l6 and scores against the same canonical fingerprint
           contract as training.
  smoke  — the 1k chain check on GPU (fast verification the remote
           environment reproduces the local results contract).

Every lane reuses the shared bootstrap: the VM clones the configured public
training branch, regenerates all derived CSVs (byte-deterministic: canonicals
and gates reproduce identically), runs the lane, and pulls results back.

Usage:
  python colab_backend.py --what train
  python colab_backend.py --what train --train-frac 0.25 --epochs 2
  python colab_backend.py --what hpo
  python colab_backend.py --what hpo --resume-hpo
  er-colab --what hpo --gpu A100
  python colab_backend.py --what sims
  python colab_backend.py --what smoke
  python colab_backend.py --what stop
  python ... --keep-alive   # keep VM alive for debugging on failure
  python ... --what train --resume-run <run-id>
"""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
# AUDIT FIX (round 2 F15, round 3): RESULTS/DATA come from the config SSOT
# via lib.common (config/paths.yaml paths.results_dir/data_dir) — were
# re-derived inline (HERE / "artifacts" / "results"), a second declaration
# that happened to match today.
from core.common import (
    F,
    RESULTS,
    TRAINING_RESULTS,
    TRAIN_ROOT,
    embedding_model_keys,
    sweep_cfg,
    hpo_cfg,
    load_config,
    resolve_model,
    training_cfg,
)
from core.manifest import sha256_file
from core.schemas import ResultBundleManifest, StageManifest


# smoke sample size + train defaults: the config SSOT (config/training.yaml
# sweep: block via lib.common.sweep_cfg / training_cfg) — were inline
# literals (1000 / 0.25 / 2) that could silently diverge from the configs.
# AUDIT FIX (round 2 F07, round 3): the train-frac default reads
# sweep.train_fracs[0] — the 0.25 literal was the last one still inline.
_SMOKE_SAMPLE = int(sweep_cfg()["smoke_sample"])
_TRAIN_FRAC_DEFAULT = float(sweep_cfg()["train_fracs"][0])
_EPOCHS_DEFAULT = int(training_cfg().training.epochs)
_RERANK_MODEL = str(sweep_cfg()["rerank_model"])

_COLAB = training_cfg().colab
_SIMS_MODEL = str(_COLAB.sims_model)
REPOSITORY = _COLAB.repository
BRANCH = _COLAB.branch
GIT_REMOTE_NAME = _COLAB.git_remote_name
SESSION = _COLAB.session
GPU = _COLAB.gpu
REMOTE_ROOT = _COLAB.remote_root
_HPO_MODE = _COLAB.hpo_mode
_HPO_WORKERS = _COLAB.hpo_workers
_HPO_TRIAL_JOBS_DEFAULT = int(hpo_cfg()["n_jobs"])
_HPO_PERSISTENCE = str(hpo_cfg()["persistence"])
_TRAIN_WORKERS = _COLAB.train_workers
_SMOKE_WORKERS = _COLAB.smoke_workers
_MIXED_TRAIN_WORKERS = _COLAB.mixed_train_workers
_MIXED_SIMS_WORKERS = _COLAB.mixed_sims_workers
_MIXED_MINING_PROFILE = _COLAB.mixed_mining_profile
_MASKING_ENABLED = training_cfg().masking.enabled
_MASKING_PROFILE = str(training_cfg().masking.profile)
_COLLAPSE_GUARDRAIL_PROFILE = str(training_cfg().collapse_guardrail.profile)
_DVC_WORKERS = _COLAB.dvc_workers
_LOG_POLL_SECONDS = _COLAB.log_poll_seconds
_PROBE_TIMEOUT_SECONDS = _COLAB.probe_timeout_seconds
_PROBE_RETRIES = _COLAB.probe_retries
_PROBE_RETRY_BACKOFF_SECONDS = _COLAB.probe_retry_backoff_seconds
_MASK_EFFECT_AFTER_TRAIN = _COLAB.mask_effect_after_train
_SMOKE_EPOCHS = _COLAB.smoke_epochs
_WORKER_TIMEOUT_SECONDS = _COLAB.worker_timeout_seconds
_RESULT_DOWNLOAD_TIMEOUT_SECONDS = _COLAB.result_download_timeout_seconds
_RESULT_DOWNLOAD_HEARTBEAT_SECONDS = _COLAB.result_download_heartbeat_seconds
_RESULT_ARCHIVE_NAME = _COLAB.result_archive_name
_RESULT_MANIFEST_NAME = _COLAB.result_manifest_name
_RESULT_DOWNLOAD_EXCLUDED_DIRS = frozenset(_COLAB.result_download_excluded_dirs)
_WORKER_MONITOR_SECONDS = _COLAB.worker_monitor_seconds
_HPO_RESUME_DIR = TRAINING_RESULTS / "hpo_resume"
# The installed Colab CLI writes its diagnostic log under $HOME even when a
# config path is supplied. This workspace's home is read-only, so isolate the
# CLI state/history in a visible, root-local folder for every launcher run.
_COLAB_CLI_STATE_DIR = TRAIN_ROOT / "colab_cli_state"
_COLAB_CLI_CONFIG = _COLAB_CLI_STATE_DIR / "sessions.json"
_COLAB_CLI_ENTRYPOINT = Path(__file__).with_name("colab_cli_entry.py")
LIVE_LOG_PATH: Path | None = None
TRAINING_LOG_PATH: Path | None = None
_live_log = None
_training_log = None
_training_log_lock = threading.Lock()
_result_event_lock = threading.Lock()
_original_stdout = None
_original_stderr = None
_SUPPRESS_LIVE_LOG = False


class _Tee:
    """Mirror launcher output to the terminal and the root live log."""

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        if not _SUPPRESS_LIVE_LOG:
            self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        if not _SUPPRESS_LIVE_LOG:
            self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()


class _LiveLogSuppressed:
    """Temporarily keep streamed worker training out of the system log."""

    def __enter__(self):
        global _SUPPRESS_LIVE_LOG
        self._previous = _SUPPRESS_LIVE_LOG
        _SUPPRESS_LIVE_LOG = True

    def __exit__(self, exc_type, exc_value, traceback_value):
        global _SUPPRESS_LIVE_LOG
        _SUPPRESS_LIVE_LOG = self._previous
        return False


def _write_training_log(text: str) -> None:
    """Write trainer output to the dedicated local training log immediately."""
    if _training_log is None or not text:
        return
    with _training_log_lock:
        _training_log.write(text)
        _training_log.flush()


def _result_event(
    run_id: str,
    stage: str,
    state: str,
    *,
    worker: int | None = None,
    **details: object,
) -> None:
    """Persist ordered result-transfer state without measuring duration."""
    root = TRAINING_RESULTS / run_id
    root.mkdir(parents=True, exist_ok=True)
    event = {"stage": stage, "state": state, **details}
    if worker is not None:
        event["worker"] = int(worker)
    event_path = root / training_cfg().colab.result_events_file
    with _result_event_lock:
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    suffix = f" worker={worker}" if worker is not None else ""
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[result-state] {stage} {state}{suffix}" + (f" | {detail_text}" if detail_text else ""), flush=True)

# The clone contains the committed raw export and number-token reference;
# data_prep regenerates deduped data and all downstream CSVs on the VM.


def check_colab_cli() -> None:
    """Ensure the colab CLI is installed and authenticated."""
    try:
        colab("--help")
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            "colab CLI not found or not authenticated.\n"
            "Run: uv tool install google-colab-cli\n"
            "Then: colab sessions  (to complete OAuth sign-in)"
        )


def _colab_command(*args: str) -> list[str]:
    """Build every Colab CLI command through the shared safe entrypoint."""
    _COLAB_CLI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    colab_executable = shutil.which("colab")
    if not colab_executable:
        return ["colab", *args]
    first_line = Path(colab_executable).read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("#!"):
        return ["colab", *args]
    colab_python = first_line[2:].strip()
    return [
        colab_python,
        str(_COLAB_CLI_ENTRYPOINT),
        "--config",
        str(_COLAB_CLI_CONFIG),
        *args,
    ]


def colab(*args: str, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a Colab CLI subcommand through the shared safe entrypoint."""
    display_cmd = ["colab", *args]
    cmd = _colab_command(*args)
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        print(f"\n[error] colab command failed: {' '.join(display_cmd)}", file=sys.stderr)
        if e.stdout:
            print(f"stdout:\n{e.stdout[-1000:]}", file=sys.stderr)
        if e.stderr:
            print(f"stderr:\n{e.stderr[-1000:]}", file=sys.stderr)
        raise


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

    log_name labels a stage in the root colab_system.log transcript. The file is
    opened once per invocation, line-flushed, and survives VM teardown so
    every Colab stage is inspectable in one chronological log.
    """
    if _live_log and log_name:
        print(f"\n===== {log_name} =====", flush=True)

    def stream_output(pipe, prefix, captured):
        pending = ""

        def emit(text: str) -> None:
            nonlocal pending
            if training_output:
                _write_training_log(text)
            pending += text
            # tqdm uses carriage returns instead of newlines. Emit each
            # progress update immediately so a long encode/training stage
            # cannot appear hung in the terminal.
            while "\n" in pending or "\r" in pending:
                newline_positions = [p for p in (pending.find("\n"), pending.find("\r")) if p >= 0]
                end = min(newline_positions)
                unit = pending[:end].rstrip()
                captured.append(pending[: end + 1])
                pending = pending[end + 1:]
                if unit:
                    context = _LiveLogSuppressed() if exclude_from_live_log else nullcontext()
                    with context:
                        print(f"{prefix} {unit}", flush=True)

        while True:
            chunk = pipe.read(1)
            if chunk == "":
                break
            emit(chunk)
        if pending:
            if training_output:
                _write_training_log(pending)
            captured.append(pending)
            context = _LiveLogSuppressed() if exclude_from_live_log else nullcontext()
            with context:
                print(f"{prefix} {pending.rstrip()}", flush=True)
        pipe.close()

    attempts = _PROBE_RETRIES if retry_safe else 1
    for attempt in range(1, attempts + 1):
        process = subprocess.Popen(
            # colab exec has its own 30-second kernel-client timeout.  It must
            # match the caller's legitimate lane timeout; otherwise a live VM
            # computation is reported as failed after 30 seconds.
            _colab_command("exec", "-s", session, "--timeout", str(timeout or 30)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        captured: list[str] = []
        out_thread = threading.Thread(target=stream_output, args=(process.stdout, "[out]", captured))
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
            output = "".join(captured)
            raise RuntimeError(
                f"Remote execution timed out after {timeout}s.\n"
                "--- complete remote output / traceback ---\n"
                f"{output}"
            ) from exc
        out_thread.join()
        err_thread.join()
        if process.returncode == 0:
            return
        output = "".join(captured)
        transient = "connection was lost" in output.lower()
        if retry_safe and transient and attempt < attempts:
            delay = _PROBE_RETRY_BACKOFF_SECONDS * attempt
            print(
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


def run_colab_exec_capture(session: str, script: str, timeout: int) -> str:
    """Execute a remote probe while retaining stdout for structured parsing."""
    last_error = ""
    for attempt in range(1, _PROBE_RETRIES + 1):
        try:
            process = subprocess.Popen(
                _colab_command("exec", "-s", session, "--timeout", str(timeout)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            captured: list[str] = []

            def stream_probe_output() -> None:
                assert process.stdout is not None
                for line in process.stdout:
                    captured.append(line)

            reader = threading.Thread(target=stream_probe_output, daemon=True)
            reader.start()
            assert process.stdin is not None
            process.stdin.write(script)
            process.stdin.close()
            process.wait(timeout=timeout + 30)
            reader.join()
            output = "".join(captured)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            reader.join()
            last_error = f"probe timeout: {exc}"
            print(
                f"[probe] timeout after {timeout}s on attempt {attempt}/{_PROBE_RETRIES}",
                flush=True,
            )
            traceback.print_exc()
        else:
            if process.returncode == 0:
                return output
            last_error = f"rc={process.returncode}: {output[-2000:]}"
            print(
                f"[probe] remote command rc={process.returncode}; stderr tail:",
                flush=True,
            )
            print(output[-2000:], flush=True)
        if attempt < _PROBE_RETRIES:
            delay = _PROBE_RETRY_BACKOFF_SECONDS * attempt
            print(f"[probe] transient remote failure ({attempt}/{_PROBE_RETRIES}); retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"remote log probe failed after {_PROBE_RETRIES} attempts: {last_error}")


def _parse_remote_json(output: str) -> dict:
    """Read the last JSON object from a Colab probe without trusting banners."""
    for line in reversed(output.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"remote log probe returned no JSON: {output[-1000:]}")


def run_detached_stage(stage: str, command_expr: str, timeout: int) -> None:
    """Run a VM stage outside the notebook kernel and stream its durable log."""
    # Two launches can occur within the same UTC second (especially after a
    # failed preflight). Microseconds keep the remote result root unique and
    # prevent FileExistsError from aborting before workers launch.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    remote_log = f"{REMOTE_ROOT}/results/logs/colab_stages/{stage}_{stamp}.log"
    remote_status = f"{remote_log}.status"
    remote_pid = f"{remote_log}.pid"
    launch = _BOOTSTRAP + f"""
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
                cwd={REMOTE_ROOT!r},
                env={{**os.environ, "PYTHONUNBUFFERED": "1"}},
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(str(child.pid), encoding="utf-8")
        running_pid = child.pid
print(json.dumps({{"pid": running_pid, "log": str(log_path), "status": str(status_path)}}), flush=True)
"""
    print(f"[{stage}] starting detached remote stage; durable log={remote_log}", flush=True)
    try:
        launched = _parse_remote_json(
            run_colab_exec_capture(SESSION, launch, timeout=120)
        )
        print(f"[{stage}] remote pid={launched.get('pid')}", flush=True)
        offset = 0
        while True:
            probe = _BOOTSTRAP + f"""
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
            payload = _parse_remote_json(
                run_colab_exec_capture(SESSION, probe, timeout=_PROBE_TIMEOUT_SECONDS)
            )
            offset = int(payload["offset"])
            if payload["chunk"]:
                for line in str(payload["chunk"]).splitlines():
                    print(f"[{stage}] {line}", flush=True)
            if payload["done"]:
                returncode = int(payload["returncode"])
                if returncode:
                    raise RuntimeError(
                        f"remote stage {stage} failed (rc={returncode}); "
                        f"system log contains the streamed output; remote log={remote_log}"
                    )
                print(f"[{stage}] completed successfully", flush=True)
                return
            time.sleep(_LOG_POLL_SECONDS)
    except BaseException as exc:
        raise RuntimeError(
            f"remote stage {stage} lost its control connection; "
            f"system log contains the streamed output; remote log={remote_log}; cause={exc}"
        ) from exc


def _resume_pointer_payload(run_id: str, workers: int) -> dict[str, dict[str, str]]:
    """Load locally mirrored DVC pointers without exposing cache internals."""
    payload: dict[str, dict[str, str]] = {}
    for number in range(1, workers + 1):
        pointer_dir = TRAINING_RESULTS / run_id / f"worker_{number}" / ".resume"
        pointers = {
            path.name: base64.b64encode(path.read_bytes()).decode("ascii")
            for path in pointer_dir.glob("*.dvc")
        } if pointer_dir.is_dir() else {}
        if not pointers:
            raise FileNotFoundError(
                f"no locally mirrored DVC resume pointer for {run_id} worker {number}"
            )
        payload[str(number)] = pointers
    return payload


def _mirror_resume_pointers(run_id: str, pointers: dict[str, dict[str, str]]) -> None:
    """Persist DVC pointers locally as the launcher tails remote workers."""
    for worker, entries in pointers.items():
        pointer_dir = TRAINING_RESULTS / run_id / f"worker_{worker}" / ".resume"
        pointer_dir.mkdir(parents=True, exist_ok=True)
        for name, encoded in entries.items():
            (pointer_dir / name).write_bytes(base64.b64decode(encoded))


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
    relative = local.relative_to(TRAINING_RESULTS / run_id)
    worker_label = str(worker) if worker is not None else "all"
    print(
        f"[download] worker={worker_label} file={index}/{total} starting "
        f"remote={remote} destination={local}",
        flush=True,
    )
    _result_event(
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
        while not stop_heartbeat.wait(_RESULT_DOWNLOAD_HEARTBEAT_SECONDS):
            received = _local_file_size(local)
            print(
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
        colab(
            "download",
            "-s",
            SESSION,
            remote,
            str(local),
            timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS,
        )
    except BaseException as exc:
        received = _local_file_size(local)
        _result_event(
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
            f"[download] worker={worker_label} file={index}/{total} FAILED "
            f"received={_format_bytes(received)} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    finally:
        stop_heartbeat.set()
        heartbeat.join()
    received = _local_file_size(local)
    _result_event(
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
        f"[download] worker={worker_label} file={index}/{total} completed "
        f"received={_format_bytes(received)}",
        flush=True,
    )
    return received


def _hpo_resume_pointer_payload() -> dict[str, str]:
    """Load locally mirrored HPO pointers for a fresh Colab VM."""
    if not _HPO_RESUME_DIR.is_dir():
        return {}
    return {
        path.relative_to(_HPO_RESUME_DIR).as_posix(): base64.b64encode(
            path.read_bytes()
        ).decode("ascii")
        for path in sorted(_HPO_RESUME_DIR.rglob("*.dvc"))
        if path.is_file()
    }


def _mirror_hpo_resume_pointers() -> None:
    """Mirror HPO DVC pointers before the VM is torn down, even on failure."""
    remote_dir = f"{REMOTE_ROOT}/results"
    names = [name for name in _list_remote(remote_dir) if "/.resume/" in name]
    for name in names:
        rel = Path(name).relative_to(remote_dir)
        local = _HPO_RESUME_DIR / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        colab("download", "-s", SESSION, name, str(local), timeout=600)
    if names:
        print(f"[resume] mirrored {len(names)} HPO DVC pointer(s) -> {_HPO_RESUME_DIR}", flush=True)


def run_detached_train_and_tail(args: list[str]) -> None:
    """Start training outside the Jupyter cell and mirror its remote log live.

    The process has its own session and status file, so a transient notebook
    client disconnect cannot kill training or swallow its traceback.
    """
    # Keep single-worker roots collision-proof for rapid retries as well.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    remote_log = f"{REMOTE_ROOT}/results/logs/colab_train_{stamp}.log"
    remote_status = f"{remote_log}.status"
    launch = _BOOTSTRAP + _remote_auth_env_script() + f"""
import json, os, pathlib, shlex, subprocess, sys
log_path = pathlib.Path({remote_log!r})
status_path = pathlib.Path({remote_status!r})
log_path.parent.mkdir(parents=True, exist_ok=True)
status_path.unlink(missing_ok=True)
train_args = [sys.executable, *{args!r}]
command = " ".join(shlex.quote(part) for part in train_args)
process_log = pathlib.Path(str(log_path) + ".processes")
ps_command = f"ps -eo pid,ppid,pgid,etime,stat,%cpu,%mem,rss,args >> {{shlex.quote(str(process_log))}} 2>&1"
wrapped = f"{{ps_command}}; timeout --signal=TERM --kill-after=60 {_WORKER_TIMEOUT_SECONDS} {{command}}; rc=$?; {{ps_command}}; printf '%s\\n' \\"$rc\\" > {{shlex.quote(str(status_path))}}; exit $rc"
with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
    child = subprocess.Popen(
        ["/bin/bash", "-lc", wrapped],
        cwd={REMOTE_ROOT!r},
        env={{**os.environ, "PYTHONUNBUFFERED": "1"}},
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
print(json.dumps({{"pid": child.pid, "log": str(log_path), "status": str(status_path)}}), flush=True)
"""
    print("[run] starting detached train.py on the VM; streaming its remote log ...", flush=True)
    # Resume bootstrap restores each worker's checkpoint through DVC before
    # it emits the launch JSON.  A full checkpoint pull can legitimately take
    # longer than the short probe budget, so use the configured worker
    # timeout for this one-time preflight.
    launched = _parse_remote_json(
        run_colab_exec_capture(SESSION, launch, timeout=_WORKER_TIMEOUT_SECONDS)
    )
    print(f"[train] remote pid={launched['pid']} log={launched['log']}", flush=True)

    offset = 0
    while True:
        probe = _BOOTSTRAP + f"""
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
        payload = _parse_remote_json(run_colab_exec_capture(SESSION, probe, timeout=_PROBE_TIMEOUT_SECONDS))
        offset = int(payload["offset"])
        if payload["chunk"]:
            for line in str(payload["chunk"]).splitlines():
                print(f"[out] {line}", flush=True)
        if payload["done"]:
            returncode = int(payload["returncode"])
            if returncode:
                raise RuntimeError(
                    f"remote training failed (rc={returncode}); "
                    f"full remote log was streamed above"
                )
            print("[train] remote process completed successfully", flush=True)
            return
        time.sleep(_LOG_POLL_SECONDS)


def run_parallel_train_and_tail(
    args: list[str], workers: int, *, resume_run: str | None = None,
    run_labels: list[str] | None = None,
    masking_profiles: list[str] | None = None,
    collapse_guardrail_profiles: list[str] | None = None,
    prepared_bundles: list[Path] | None = None,
) -> tuple[str, int]:
    """Run isolated full-data trainers concurrently and mirror worker logs."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    remote_base = (
        f"{REMOTE_ROOT}/results/concurrent_train_{resume_run}"
        if resume_run
        else f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    )
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    remote_bundles = (
        _upload_prepared_bundles(run_id=run_id, bundles=prepared_bundles)
        if prepared_bundles is not None
        else None
    )
    remote_input_loop = (
        "for name in ():"
        if prepared_bundles is not None
        else 'for name in (F["canonical_records"], F["gate_results"], F["labeled_pairs"]):'
    )
    resume_pointers = _resume_pointer_payload(run_id, workers) if resume_run else {}
    launch = _BOOTSTRAP + _remote_auth_env_script() + f"""
import base64, json, os, pathlib, shutil, shlex, subprocess, sys, time, traceback
from core.common import F
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
run_id = base.name.removeprefix("concurrent_train_")
base.mkdir(parents=True, exist_ok={bool(resume_run)!r})
resume_pointers = {resume_pointers!r}
run_labels = {run_labels!r}
masking_profiles = {masking_profiles!r}
collapse_guardrail_profiles = {collapse_guardrail_profiles!r}
started = []
for number in range(1, {workers} + 1):
    worker_profile = (
        run_labels[number - 1]
        if run_labels and number <= len(run_labels)
        else ""
    )
    training_name = f"{{run_id}}-worker_{{number}}" + (f"-{{worker_profile}}" if worker_profile else "")
    worker_profile = (
        worker_profile if worker_profile in ("mining_enabled", "masking_only") else ""
    )
    out = base / f"worker_{{number}}"
    print(f"[resume-preflight] worker {{number}}: preparing {{out}}", flush=True)
    if {bool(resume_run)!r}:
        out.mkdir(exist_ok=True)
        pointer_dir = out / ".resume"
        pointer_dir.mkdir(exist_ok=True)
        for name, encoded in resume_pointers[str(number)].items():
            (pointer_dir / name).write_bytes(base64.b64decode(encoded))
        print(f"[resume-preflight] worker {{number}}: pointer files written", flush=True)
        {remote_input_loop}
            relative = name.relative_to(root / "results")
            source = root / "results" / relative
            destination = out / relative
            if not destination.is_file():
                if not source.is_file():
                    raise FileNotFoundError(f"resume worker input missing: {{source}}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        try:
            from training.dvc_store import restore_pointer
            pointers = sorted(pointer_dir.glob("*.dvc"))
            if not pointers:
                raise RuntimeError(
                    f"[resume-preflight] worker {{number}} has no DVC resume pointer; "
                    "the previous checkpoint cannot be restored"
                )
            restored_checkpoint = False
            for pointer in pointers:
                print(f"[resume-preflight] worker {{number}}: restoring {{pointer.name}}", flush=True)
                outputs = restore_pointer(out, pointer)
                print(f"[resume-preflight] worker {{number}}: DVC pull complete", flush=True)
                checkpoint_outputs = [
                    path for path in outputs
                    if "_checkpoints" in path.relative_to(out).parts
                ]
                if checkpoint_outputs:
                    restored_checkpoint = True
                    for checkpoint_root in checkpoint_outputs:
                        candidates = sorted(
                            checkpoint_root.glob("checkpoint-*"),
                            key=lambda path: int(path.name.removeprefix("checkpoint-")),
                        )
                        if not candidates:
                            raise RuntimeError(
                                f"[resume-preflight] restored checkpoint root is empty: "
                                f"{{checkpoint_root}}"
                            )
                        latest = candidates[-1]
                        required = (
                            "optimizer.pt",
                            "scheduler.pt",
                            "rng_state.pth",
                            "checkpoint_manifest.json",
                            "trainer_state.json",
                        )
                        missing = [
                            name for name in required if not (latest / name).is_file()
                        ]
                        if missing:
                            raise RuntimeError(
                                f"[resume-preflight] {{latest}} is not resumable; "
                                f"missing {{', '.join(missing)}}"
                            )
            if not restored_checkpoint:
                raise RuntimeError(
                    f"[resume-preflight] worker {{number}} restored no checkpoint "
                    "pointer; refusing to start training"
                )
            print(f"[resume-preflight] worker {{number}}: restored {{len(pointers)}} pointer(s)", flush=True)
        except BaseException:
            print(f"[resume-preflight] worker {{number}} traceback:", flush=True)
            traceback.print_exc()
            raise
    else:
        out.mkdir()
        for name in (
            F["canonical_records"],
            F["gate_results"],
            F["labeled_pairs"],
        ):
            relative = name.relative_to(root / "results")
            source = root / "results" / relative
            if not source.is_file():
                raise FileNotFoundError(f"worker input missing: {{source}}")
            destination = out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    worker_args = [sys.executable, *{args!r}]
    if {remote_bundles is not None!r}:
        worker_args[worker_args.index("training.train")] = "training.train_prepared"
        worker_args.extend(["--bundle", {remote_bundles!r}[number - 1]])
    if masking_profiles is not None and {remote_bundles is None!r}:
        worker_args.extend(["--masking-profile", masking_profiles[number - 1]])
    if collapse_guardrail_profiles is not None and {remote_bundles is None!r}:
        worker_args.extend([
            "--collapse-guardrail-profile",
            collapse_guardrail_profiles[number - 1],
        ])
    command = " ".join(shlex.quote(part) for part in worker_args)
    log_path, status_path = out / "training.log", out / "training.status"
    live_status_path = out / "live_status.json"
    wandb_dir = out / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root / "src"), "EUROMONITOR_RESULTS_DIR": str(out),
           "EUROMONITOR_MLRUNS_DIR": str(out / "mlruns"), "WANDB_DIR": str(wandb_dir),
           "WANDB_RUN_NAME": training_name,
           "EUROMONITOR_RUN_ID": training_name,
           "EUROMONITOR_MINING_PROFILE": worker_profile,
           "EUROMONITOR_REMOTE_TRAINING": "1"}}
    live_status_path.write_text(json.dumps({{
        "updated_at": time.time(), "event": "launched", "step": 0,
        "wandb_run_name": env["WANDB_RUN_NAME"],
    }}) + "\\n", encoding="utf-8")
    process_log = out / "processes.log"
    # Every worker launch owns a fresh diagnostics log.  Resume restores
    # checkpoints, not log history; stale snapshots from an earlier attempt
    # must not be mistaken for this run's lifecycle.
    process_log.write_text("", encoding="utf-8")
    ps_command = f"ps -eo pid,ppid,pgid,etime,stat,%cpu,%mem,rss,args >> {{shlex.quote(str(process_log))}} 2>&1"
    diagnostics = "free -h || true; nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true"
    wrapped = (
        f"echo '[worker-process] starting pid=$$'; {{ps_command}}; "
        f"echo '[worker-process] resource snapshot before training'; {{diagnostics}}; "
        f"timeout --signal=TERM --kill-after=60 {_WORKER_TIMEOUT_SECONDS} {{command}}; rc=$?; "
        f"echo '[worker-process] exited rc='$rc; {{ps_command}}; "
        f"echo '[worker-process] resource snapshot after training'; {{diagnostics}}; "
        f"printf '%s\\n' \\"$rc\\" > {{shlex.quote(str(status_path))}}; exit $rc"
    )
    # Resume restores model state only.  Worker training logs always start
    # fresh so a new attempt cannot append to a prior run's output.
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        child = subprocess.Popen(["/bin/bash", "-lc", wrapped], cwd=root, env=env,
            stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True)
    print(f"[train-launch] worker {{number}} started pid={{child.pid}}", flush=True)
    started.append({{"worker": number, "pid": child.pid}})
print(json.dumps({{"base": str(base), "workers": started}}), flush=True)
"""
    print(f"[run] starting {workers} isolated full-data trainers; streaming all worker logs ...", flush=True)
    # Resume bootstrap restores each worker's checkpoint through DVC before
    # it emits the launch JSON. A full checkpoint pull can legitimately take
    # longer than the short probe budget, so use the configured worker
    # timeout for this one-time preflight.
    launched = _parse_remote_json(
        run_colab_exec_capture(SESSION, launch, timeout=_WORKER_TIMEOUT_SECONDS)
    )
    print(f"[train] remote workers={launched['workers']} base={launched['base']}", flush=True)
    offsets = {str(item["worker"]): 0 for item in launched["workers"]}
    live_signatures: dict[str, str] = {}
    while True:
        probe = _BOOTSTRAP + f"""
import json, pathlib
base = pathlib.Path({remote_base!r})
offsets = {offsets!r}
payload = {{"offsets": {{}}, "chunks": {{}}, "status": {{}}, "resume": {{}}, "live": {{}}}}
for number in range(1, {workers} + 1):
    key = str(number)
    out = base / f"worker_{{number}}"
    log_path, status_path = out / "training.log", out / "training.status"
    live_status_path = out / "live_status.json"
    offset = int(offsets.get(key, 0))
    data = b""
    if log_path.is_file():
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    payload["offsets"][key] = offset + len(data)
    payload["chunks"][key] = data.decode("utf-8", errors="replace")
    payload["status"][key] = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else None
    if live_status_path.is_file():
        try:
            payload["live"][key] = json.loads(live_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    pointer_dir = out / ".resume"
    payload["resume"][key] = {{
        path.name: base64.b64encode(path.read_bytes()).decode("ascii")
        for path in pointer_dir.glob("*.dvc")
    }} if pointer_dir.is_dir() else {{}}
payload["done"] = all(value is not None for value in payload["status"].values())
print(json.dumps(payload), flush=True)
"""
        payload = _parse_remote_json(run_colab_exec_capture(SESSION, probe, timeout=_PROBE_TIMEOUT_SECONDS))
        offsets = {str(key): int(value) for key, value in payload["offsets"].items()}
        _mirror_resume_pointers(run_id, payload["resume"])
        for worker, live in payload["live"].items():
            signature = json.dumps(live, sort_keys=True)
            if live_signatures.get(worker) == signature:
                continue
            live_signatures[worker] = signature
            metrics = []
            for key, label in (("train_loss", "train_loss"), ("dev_average_precision", "dev_ap"),
                               ("dev_accuracy", "dev_acc")):
                if live.get(key) is not None:
                    metrics.append(f"{label}={float(live[key]):.4f}")
            if live.get("rss_mb") is not None:
                metrics.append(f"rss={float(live['rss_mb']):.0f}MB")
            if live.get("gpu_free_gb") is not None:
                metrics.append(
                    f"gpu={float(live.get('gpu_allocated_gb', 0)):.2f}G alloc/"
                    f"{float(live['gpu_free_gb']):.2f}G free"
                )
            position = f"step {live.get('step', 0)}/{live.get('max_steps', '?')}"
            print(f"[worker {worker}] {live.get('event', 'running')} | {position}" +
                  (" | " + " | ".join(metrics) if metrics else "") +
                  (f" | W&B {live['wandb_url']}" if live.get("wandb_url") else ""), flush=True)
        for worker, chunk in payload["chunks"].items():
            # Forward the complete worker log. Detached workers write to the
            # remote file; keep it out of the root system log to avoid a
            # second copy of the worker's training log.
            with _LiveLogSuppressed():
                for line in str(chunk).splitlines():
                    _write_training_log(f"[worker {worker}] {line}\n")
                    print(f"[worker {worker}] {line}", flush=True)
        if payload["done"]:
            failed = {worker: rc for worker, rc in payload["status"].items() if int(rc) != 0}
            if failed:
                raise RuntimeError(f"parallel trainers failed: {failed}")
            download_verified_training_results(remote_base, workers)
            print(f"[train] all {workers} remote workers completed successfully", flush=True)
            return remote_base, workers
        time.sleep(_LOG_POLL_SECONDS)


def _sha256_file(path: Path) -> str:
    """Hash one extracted result for manifest verification."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_remote_result_archive(remote_base: str, workers: int) -> str:
    """Build one manifest-backed archive on the VM before transfer."""
    archive_path = f"{remote_base}/{_RESULT_ARCHIVE_NAME}"
    script = f"""
import hashlib, json, pathlib, tarfile

base = pathlib.Path({remote_base!r})
archive_path = base / {_RESULT_ARCHIVE_NAME!r}
manifest_path = base / {_RESULT_MANIFEST_NAME!r}
excluded_dirs = set({_RESULT_DOWNLOAD_EXCLUDED_DIRS!r})

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

included = []
excluded = []
for worker in range(1, {workers + 1}):
    worker_root = base / f"worker_{{worker}}"
    if not worker_root.is_dir():
        raise FileNotFoundError(f"missing remote worker directory: {{worker_root}}")
    for path in sorted(worker_root.rglob("*")):
        if path.is_symlink():
            excluded.append({{
                "worker": worker,
                "path": path.relative_to(worker_root).as_posix(),
                "reason": "symlink_not_transferred",
            }})
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(worker_root)
        blocked = next((part for part in relative.parts if part in excluded_dirs), None)
        if blocked is not None:
            excluded.append({{
                "worker": worker,
                "path": relative.as_posix(),
                "reason": "configured_directory:" + blocked,
            }})
            continue
        included.append({{
            "worker": worker,
            "path": relative.as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }})

manifest = {{
    "schema_version": "1",
    "run_id": base.name.removeprefix("concurrent_train_"),
    "workers": {workers},
    "included": included,
    "excluded": excluded,
}}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
archive_path.unlink(missing_ok=True)
with tarfile.open(archive_path, "w:gz") as archive:
    for entry in included:
        worker_root = base / f"worker_{{entry['worker']}}"
        source = worker_root / entry["path"]
        archive.add(source, arcname=f"worker_{{entry['worker']}}/{{entry['path']}}")
    archive.add(manifest_path, arcname={_RESULT_MANIFEST_NAME!r})
print("[result-archive] included={{}} excluded={{}} archive_bytes={{}}".format(
    len(included), len(excluded), archive_path.stat().st_size
), flush=True)
"""
    print(
        f"[download] preparing one remote result archive for {workers} worker(s): "
        f"{archive_path}",
        flush=True,
    )
    run_colab_exec_stream(
        SESSION,
        script,
        timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS,
        log_name="result_archive",
    )
    return archive_path


def _verify_result_bundle(root: Path, run_id: str, workers: int) -> ResultBundleManifest:
    """Validate manifest coverage, paths, sizes, and hashes after extraction."""
    manifest_path = root / _RESULT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"result archive is missing its manifest: {manifest_path}")
    manifest = ResultBundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.run_id != run_id or manifest.workers != workers:
        raise RuntimeError(
            f"result manifest identity mismatch: run_id={manifest.run_id!r}, "
            f"workers={manifest.workers}; expected {run_id!r}, {workers}"
        )
    expected: set[str] = set()
    for entry in manifest.included:
        relative = Path(entry.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe result manifest path: {entry.path!r}")
        key = Path(f"worker_{entry.worker}") / relative
        key_text = key.as_posix()
        if key_text in expected:
            raise RuntimeError(f"duplicate result manifest path: {key_text}")
        expected.add(key_text)
        actual = root / key
        if not actual.is_file():
            raise RuntimeError(f"result archive missing manifest file: {key_text}")
        size = actual.stat().st_size
        if size != entry.size:
            raise RuntimeError(
                f"result size mismatch for {key_text}: {size} != {entry.size}"
            )
        digest = _sha256_file(actual)
        if digest != entry.sha256:
            raise RuntimeError(f"result SHA-256 mismatch for {key_text}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for worker_root in sorted(root.glob("worker_*"))
        if worker_root.is_dir()
        for path in worker_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected:
        missing = sorted(expected - actual_paths)
        unexpected = sorted(actual_paths - expected)
        raise RuntimeError(
            f"result archive coverage mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    return manifest


def _extract_result_archive(
    archive_path: Path, local_base: Path, run_id: str, workers: int
) -> ResultBundleManifest:
    """Safely extract and atomically replace the worker result directories."""
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-result-", dir=local_base.parent))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise RuntimeError(f"unsafe result archive member: {member.name!r}")
            archive.extractall(temporary)
        manifest = _verify_result_bundle(temporary, run_id, workers)
        for worker in range(1, workers + 1):
            source = temporary / f"worker_{worker}"
            target = local_base / source.name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            shutil.move(str(source), str(target))
        local_manifest = local_base / _RESULT_MANIFEST_NAME
        local_manifest.unlink(missing_ok=True)
        shutil.move(str(temporary / _RESULT_MANIFEST_NAME), str(local_manifest))
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def download_verified_training_results(remote_base: str, workers: int) -> None:
    """Transfer one manifest-backed result archive before VM teardown."""
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    local_base = TRAINING_RESULTS / run_id
    local_base.mkdir(parents=True, exist_ok=True)
    _result_event(run_id, "download", "started", workers=workers)
    remote_archive = _prepare_remote_result_archive(remote_base, workers)
    local_archive = local_base / _RESULT_ARCHIVE_NAME
    _download_file_with_visibility(
        remote=remote_archive,
        local=local_archive,
        worker=None,
        index=1,
        total=1,
        run_id=run_id,
    )
    manifest = _extract_result_archive(local_archive, local_base, run_id, workers)
    _result_event(
        run_id,
        "download",
        "completed",
        workers=workers,
        files=len(manifest.included),
        excluded_files=len(manifest.excluded),
        archive=str(local_archive),
        destination=str(local_base),
    )
    print(
        f"[download] verified result archive -> {local_base} "
        f"({len(manifest.included)} included, {len(manifest.excluded)} excluded)",
        flush=True,
    )

def _publish_local_hpo_model_snapshot(generation: Path, model_dir: Path) -> None:
    """Publish one HPO model snapshot through the DVC publisher lane."""
    from training.hpo_persistence import best_effort_dvc_publish, build_snapshot

    snapshot = build_snapshot(
        generation=generation,
        sequence=time.time_ns(),
        optuna_db=None,
        include=[model_dir],
        scope=model_dir.name,
    )
    if not best_effort_dvc_publish(snapshot):
        raise RuntimeError(f"local HPO DVC publication failed: {snapshot}")
    print(f"[hpo-durability-local] published {model_dir.name}", flush=True)


def publish_local_hpo_results(run_id: str, persistence: str) -> None:
    """Generate HPO reports and persist snapshots before VM teardown."""
    from training.generate_training_report import generate_report

    generation = TRAINING_RESULTS / "hpo_runs" / run_id
    if not generation.is_dir():
        raise FileNotFoundError(f"local HPO archive missing: {generation}")
    for model_dir in sorted((generation / "models").iterdir()):
        if not model_dir.is_dir():
            continue
        metrics = sorted(model_dir.glob("*_holdout_*_fold_metrics.csv"))
        pairs = sorted(model_dir.glob("*_fold*_pairs.csv"))
        if metrics and pairs:
            pointer = model_dir / F["results_pointer"].name
            pointer_data = json.loads(pointer.read_text(encoding="utf-8")) if pointer.is_file() else {}
            report_tag = str(pointer_data.get("run_tag") or model_dir.name)
            generate_report(
                metrics[-1], pairs, model_dir / f"report_{report_tag}",
                sorted(model_dir.glob("*_fold*_train_scores.csv")),
                sorted(model_dir.glob("*_fold*_random_easy_scores.csv")),
            )
            print(f"[report-local] HPO {model_dir.name}: report generated", flush=True)
    if persistence != "dvc":
        print(f"[hpo-dvc] skipped: persistence={persistence}", flush=True)
        return

    env = _local_auth_env()
    previous = {key: os.environ.get(key) for key in ("DVC_API_KEY", "DAGSHUB_USER_TOKEN")}
    try:
        for key in ("DVC_API_KEY", "DAGSHUB_USER_TOKEN"):
            if key in env:
                os.environ[key] = env[key]
        model_dirs = [
            model_dir
            for model_dir in sorted((generation / "models").iterdir())
            if model_dir.is_dir()
        ]
        print(
            f"[hpo-dvc] publishing with {_DVC_WORKERS} DVC worker(s); "
            "HPO model/trial workers remain separate",
            flush=True,
        )
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=_DVC_WORKERS, thread_name_prefix="hpo-dvc"
        ) as pool:
            futures = [
                pool.submit(_publish_local_hpo_model_snapshot, generation, model_dir)
                for model_dir in model_dirs
            ]
            for future in futures:
                future.result()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def start_live_log() -> None:
    """Start the root-level live Colab log, replacing the prior run's log."""
    global LIVE_LOG_PATH, TRAINING_LOG_PATH, _live_log, _training_log
    global _original_stdout, _original_stderr
    if _live_log is not None:
        _live_log.close()
    LIVE_LOG_PATH = F["colab_live_log"]
    TRAINING_LOG_PATH = F["colab_training_log"]
    _live_log = LIVE_LOG_PATH.open("w", encoding="utf-8")
    _training_log = TRAINING_LOG_PATH.open("w", encoding="utf-8")
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = _Tee(_original_stdout, _live_log)
    sys.stderr = _Tee(_original_stderr, _live_log)
    print(f"[log] capturing Colab output -> {LIVE_LOG_PATH}", flush=True)


def close_live_log() -> None:
    global _live_log, _training_log, _original_stdout, _original_stderr
    if _live_log is not None:
        sys.stdout = _original_stdout or sys.stdout
        sys.stderr = _original_stderr or sys.stderr
        _live_log.close()
        _live_log = None
    if _training_log is not None:
        _training_log.flush()
        _training_log.close()
        _training_log = None
        _original_stdout = None
        _original_stderr = None


def _verify_session_handshake() -> None:
    """Fail before checkout if the CLI cannot execute on the VM."""
    try:
        heartbeat = run_colab_exec_capture(
            SESSION,
            "import os, socket, sys; print({'pid': os.getpid(), 'python': sys.version.split()[0], 'host': socket.gethostname()})",
            timeout=60,
        )
    except BaseException as exc:
        raise RuntimeError(
            f"Colab session '{SESSION}' failed the control-channel handshake "
            f"before training: {exc}"
        ) from exc
    print(f"[session] control-channel handshake passed: {heartbeat.strip()}")


def _forget_cached_session() -> None:
    """Drop only this launcher's stale local session record before reprovisioning."""
    if not _COLAB_CLI_CONFIG.is_file():
        return
    try:
        state = json.loads(_COLAB_CLI_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(state, dict) or SESSION not in state:
        return
    state.pop(SESSION, None)
    temporary = _COLAB_CLI_CONFIG.with_name(_COLAB_CLI_CONFIG.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, _COLAB_CLI_CONFIG)
    print(f"[session] removed stale cached record for '{SESSION}'", flush=True)


def ensure_session() -> None:
    """Provision and verify the session before any training stage starts."""
    r = colab("sessions", check=False)
    if r.returncode == 0 and SESSION in (r.stdout or ""):
        print(f"[session] '{SESSION}' already active; verifying control channel ...")
        try:
            _verify_session_handshake()
            return
        except BaseException as exc:
            print(f"[session] cached session is stale; reprovisioning ({exc})", flush=True)
            _forget_cached_session()
    else:
        # The CLI may retain a named session locally after the VM has been
        # torn down.  Never let that record prevent a fresh allocation.
        _forget_cached_session()
    accelerator = [] if GPU.upper() == "CPU" else ["--gpu", GPU]
    print(f"[session] provisioning {SESSION} ({'cpu' if not accelerator else f'gpu={GPU}'}) ...")
    colab("new", "-s", SESSION, *accelerator, timeout=300)
    print("[session] provisioned; running control-channel handshake ...")
    _verify_session_handshake()


def prepare_remote_layout(*, minimal_runtime: bool = False) -> None:
    """Restore the configured branch using the established Colab checkout flow."""
    script = f"""
import pathlib, shutil, subprocess

root = pathlib.Path({REMOTE_ROOT!r})
remote_name = {GIT_REMOTE_NAME!r}
if root.exists() and not (root / ".git").is_dir():
    shutil.rmtree(root)
if (root / ".git").is_dir():
    remotes = subprocess.run(
        ["git", "remote"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.split()
    if remote_name not in remotes:
        if remote_name != "origin" and "origin" in remotes:
            subprocess.run(
                ["git", "remote", "rename", "origin", remote_name],
                cwd=root,
                check=True,
            )
        else:
            raise RuntimeError(
                f"configured git remote {{remote_name!r}} is absent in {{root}}; "
                f"available remotes={{remotes}}"
            )
    # A prior sparse/detached runtime must be returned to the stable branch
    # checkout used by the original Colab launcher before control cells import
    # project modules from REMOTE_ROOT/src.
    subprocess.run(["git", "sparse-checkout", "disable"], cwd=root, check=False)
    subprocess.run(["git", "fetch", remote_name, {BRANCH!r}], cwd=root, check=True)
    subprocess.run(
        ["git", "checkout", "-B", {BRANCH!r}, remote_name + "/" + {BRANCH!r}],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "pull", "--ff-only", remote_name, {BRANCH!r}], cwd=root, check=True)
else:
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--origin", remote_name, "--depth", "1",
        "--branch", {BRANCH!r},
        {REPOSITORY!r}, str(root),
    ], check=True)
for path in [root / "artifacts" / "data", root / "artifacts" / "results"]:
    path.mkdir(parents=True, exist_ok=True)
print("[repo] ready", {REPOSITORY!r}, "branch", {BRANCH!r},
      "prepared_runtime=" + str({minimal_runtime!r}), "at", root)
"""
    run_colab_exec_stream(SESSION, script, timeout=600, log_name="checkout", retry_safe=True)


def install_deps(*, minimal_runtime: bool = False) -> None:
    print(
        "[deps] installing "
        + ("prepared training runtime" if minimal_runtime else "full lane dependencies")
        + " on the VM ..."
    )
    # Run pip outside the notebook kernel. A kernel disconnect can interrupt
    # the control channel, but the detached process keeps writing a durable
    # log/status pair that the launcher can retrieve before teardown.
    # A prepared bundle removes *data preparation*, not runtime dependencies.
    # In particular, train.py shells out to ``dvc`` when remote checkpoint
    # durability is enabled.  Keep the smoke install equal to the full lane so
    # its dependency contract cannot drift and fail after training has begun.
    packages = (
        "'sentence-transformers', 'datasets', 'accelerate', 'evaluate', "
        "'scikit-learn', 'pandas', 'numpy', 'mlflow', 'optuna', "
        "'psycopg[binary]', 'wandb', 'dvc', 'dagshub'"
    )
    run_detached_stage(
        "00_deps",
        "[sys.executable, '-m', 'pip', 'install', " + packages + "]",
        timeout=900,
    )
    verify_remote_runtime_dependencies()


def verify_remote_runtime_dependencies() -> None:
    """Fail before a worker starts if its declared runtime is incomplete."""
    print("[deps] verifying Python imports and required command-line tools ...")
    modules = (
        "accelerate", "datasets", "dagshub", "dvc", "evaluate", "matplotlib",
        "mlflow", "numpy", "optuna", "pandas", "psycopg", "pydantic",
        "scipy", "sentence_transformers", "sentencepiece", "sklearn", "torch",
        "transformers", "wandb", "yaml",
    )
    script = _BOOTSTRAP + f"""
import importlib, subprocess
modules = {modules!r}
failed = {{}}
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed[name] = f"{{type(exc).__name__}}: {{exc}}"
if failed:
    raise RuntimeError(f"runtime import preflight failed: {{failed}}")
tool = subprocess.run(["dvc", "version"], capture_output=True, text=True)
if tool.returncode:
    raise RuntimeError(f"DVC executable preflight failed: {{tool.stderr.strip()}}")
print(f"[deps] imports={{len(modules)}} dvc={{tool.stdout.splitlines()[0]}} status=validated", flush=True)
"""
    run_colab_exec_stream(
        SESSION, script, timeout=300, log_name="dependency_preflight", retry_safe=True,
    )


def log_gpu_profile() -> None:
    """Record the runtime hardware before training, including CPU smoke runs."""
    script = """import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print({'hardware': 'gpu', 'name': p.name, 'total_gb': round(total / 1e9, 2), 'free_gb': round(free / 1e9, 2), 'torch': torch.__version__}, flush=True)
else:
    print({'hardware': 'cpu', 'threads': torch.get_num_threads(), 'torch': torch.__version__}, flush=True)
"""
    run_colab_exec_stream(SESSION, script, timeout=120, log_name="runtime_profile", retry_safe=True)


_BOOTSTRAP = f"""
import sys, runpy, pathlib, os
sys.path.insert(0, "{REMOTE_ROOT}/src")
os.environ["PYTHONPATH"] = "{REMOTE_ROOT}/src" + os.pathsep + os.environ.get("PYTHONPATH", "")
(pathlib.Path("{REMOTE_ROOT}/results")).mkdir(parents=True, exist_ok=True)
(pathlib.Path("{REMOTE_ROOT}/artifacts/data")).mkdir(parents=True, exist_ok=True)
"""


def _env_value(name: str) -> str | None:
    """Read a simple KEY=VALUE entry without printing or cloning secrets."""
    env_path = TRAIN_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None


def _wandb_env_script() -> str:
    """Inject only the API key into the remote process, never remote disk."""
    key = _env_value("WANDB_API_KEY")
    if not key:
        print("[wandb] WANDB_API_KEY absent from .env; run will remain local-only")
        return ""
    print("[wandb] API key loaded from local .env and injected into VM process")
    return f"os.environ['WANDB_API_KEY'] = {key!r}\n"


def _optuna_env_script() -> str:
    """Inject the shared PostgreSQL control-plane URL into the VM only."""
    url = _env_value("OPTUNA_STORAGE_URL")
    if not url:
        print("[hpo-control] OPTUNA_STORAGE_URL absent; concurrent HPO is disabled")
        return ""
    if not url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("OPTUNA_STORAGE_URL must use a PostgreSQL URL")
    print("[hpo-control] PostgreSQL Optuna URL loaded from local .env and injected into VM process")
    return f"os.environ['OPTUNA_STORAGE_URL'] = {url!r}\n"


def _remote_auth_env_script() -> str:
    """Credential exports used by remote subprocess launch cells only."""
    key = _env_value("DVC_API_KEY")
    if key:
        print("[dvc] DVC_API_KEY loaded from local .env and injected into VM process")
        dvc = f"os.environ['DVC_API_KEY'] = {key!r}\nos.environ['DAGSHUB_USER_TOKEN'] = {key!r}\n"
    else:
        print("[dvc] DVC_API_KEY absent from .env; durable DVC upload will fail")
        dvc = ""
    return _wandb_env_script() + _optuna_env_script() + dvc


def run_data_prep() -> None:
    """Regenerate the derived CSVs on the VM (byte-deterministic replay).

    AUDIT FIX 2026-09-08: 'data_prep only' assumed derived inputs that are
    NOT derived on the VM — the chain is dedupe -> build_reference --verify
    -> data_prep. Running all three keeps the VM replay identical to the
    local worktree replay (byte-comparable outputs).
    """
    print("[run] dedupe + reference-verify + data_prep on the VM ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
for step in ("src/training/dedupe.py", "src/training/build_second04_pairs.py", "src/training/build_reference.py --verify", "src/training/data_prep.py", "src/training/labeled_pairs.py"):
    print("== " + step, flush=True)
    rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/" + step.split()[0]] + step.split()[1:]).returncode
    if rc != 0:
        raise RuntimeError(f"data-prep stage failed: {{step}} (rc={{rc}})")
"""
    # dedupe 1-2 min + reference verify ~3 min + data_prep ~2 min
    run_colab_exec_stream(SESSION, script, timeout=1800, log_name="data_prep")


def verify_remote_models(model_keys: list[str]) -> None:
    """Validate the Git-shipped model bundles before starting any worker."""
    keys = sorted(set(model_keys))
    if not keys:
        return
    config = load_config()
    registry = config["models"]
    unknown = sorted(set(keys) - set(registry))
    if unknown:
        raise KeyError(f"unknown local model registry key(s): {unknown}")
    model_root_relative = str(config["paths"]["models_dir"])
    print(
        f"[models] source=git-shipped requested={keys} status=validation-start",
        flush=True,
    )
    script = _BOOTSTRAP + f"""
from pathlib import Path
from core.common import resolve_model
root = {REMOTE_ROOT!r}
requested = {keys!r}
model_root = (Path(root) / {model_root_relative!r}).resolve()

def bundle_bytes(path):
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

for key in requested:
    path = Path(resolve_model(key))
    if model_root not in path.parents:
        raise RuntimeError(
            f"resolved Git-shipped model {{key}} escapes {{model_root}}: {{path}}"
        )
    print(
        f"[models] key={{key}} path={{path.relative_to(Path(root))}} "
        f"status=validated bytes={{bundle_bytes(path)}}",
        flush=True,
    )
print(
    f"[models] source=git-shipped requested={{requested}} status=validated",
    flush=True,
)
"""
    run_colab_exec_stream(
        SESSION,
        script,
        timeout=600,
        log_name="model_validation",
        retry_safe=False,
    )


def verify_training_inputs() -> None:
    """Use frozen CSV inputs and materialize derived calibration input."""
    print("[data] validating frozen training CSVs from the cloned branch ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
from core.common import F
required = [
    F["dataset_deduped"],
    F["number_reference"],
    F["canonical_records"],
    F["gate_results"],
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("frozen training CSVs missing: " + ", ".join(missing))
for path in required:
    print(f"[data] {{path}}: {{path.stat().st_size:,}} bytes", flush=True)
calibration_path = F["labeled_pairs"]
if not calibration_path.is_file():
    print(
        f"[data] derived calibration input missing; generating {{calibration_path}}",
        flush=True,
    )
    rc = subprocess.run(
        [sys.executable, "src/training/labeled_pairs.py"],
        cwd={REMOTE_ROOT!r},
    ).returncode
    if rc != 0:
        raise RuntimeError(f"labeled-pairs generation failed (rc={{rc}})")
if not calibration_path.is_file():
    raise FileNotFoundError(f"derived calibration input missing after generation: {{calibration_path}}")
print(f"[data] {{calibration_path}}: {{calibration_path.stat().st_size:,}} bytes", flush=True)
"""
    run_colab_exec_stream(SESSION, script, timeout=120, log_name="01_data_check", retry_safe=True)


def verify_remote_prepared_inputs() -> None:
    """Check only the small calibration input used by prepared GPU workers."""
    print("[data] validating local-prepared GPU runtime inputs ...")
    script = _BOOTSTRAP + f"""
from core.common import resolve_model
from pathlib import Path
model = Path(resolve_model({str(training_cfg().training.base_model)!r}))
if not model.is_dir():
    raise FileNotFoundError(f"prepared runtime model bundle missing: {{model}}")
print(f"[data] model={{model}}")
print("[data] remote preparation disabled; worker consumes uploaded bundle")
"""
    run_colab_exec_stream(
        SESSION,
        script,
        timeout=120,
        log_name="01_prepared_input_check",
        retry_safe=True,
    )


def run_train(
    frac: float, epochs: int, sample: int | None, workers: int = 1,
    *, resume_run: str | None = None, model: str | None = None,
    run_label: str | None = None, masking_profile: str | None = None,
    collapse_guardrail_profile: str | None = None,
) -> tuple[str, int]:
    """Full-chain GPU training on the VM."""
    print("[run] train.py on the configured VM runtime ...")
    # AUDIT 2026-09-09: --mask-frac 0.15 REMOVED — it hardcoded a value that
    # silently contradicted the SSOT (masking.frac: 1.00 in
    # config/training.yaml). train.py's own default resolves from the config
    # now; the CLI flag remains for explicit overrides.
    args = ["-u", "-m", "training.train",
        "--split", "holdout",
        "--loss", "contrastive",
        "--train-frac", str(frac),
        "--epochs", str(epochs),
        # Reports and plots are intentionally not generated by the Colab lane.
        "--no-plot"]
    if model is not None:
        registry = load_config()["models"]
        if model not in registry:
            raise KeyError(
                f"Colab training model must be a local registry key; "
                f"got {model!r}, expected one of {sorted(registry)}"
            )
        # Resolve inside the remote checkout. A local absolute path would
        # not exist on the VM and would bypass the Git-shipped model contract.
        args.extend(["--model", model])
    if sample is not None:
        args.extend(["--sample", str(sample)])
    if workers == 1:
        args.extend(["--masking-profile", masking_profile or _MASKING_PROFILE])
        args.extend([
            "--collapse-guardrail-profile",
            collapse_guardrail_profile or _COLLAPSE_GUARDRAIL_PROFILE,
        ])
    if not _MASK_EFFECT_AFTER_TRAIN:
        args.append("--no-mask-effect")
    if resume_run:
        args.append("--resume")
    profiles = _expand_worker_profiles(
        masking_profile or _MASKING_PROFILE,
        workers,
        "masking",
    )
    prepared_bundles = _prepare_local_training_bundles(
        profiles=profiles,
        model=model,
        sample=sample,
    )
    if workers == 1 and resume_run is None:
        return run_single_train_and_stream(
            args,
            run_label=run_label,
            prepared_bundle=prepared_bundles[0],
        )
    return run_parallel_train_and_tail(
        args, workers, resume_run=resume_run,
        run_labels=(
            _expand_worker_profiles(run_label, workers, "run label")
            if run_label else None
        ),
        masking_profiles=profiles,
        collapse_guardrail_profiles=_expand_worker_profiles(
            collapse_guardrail_profile or _COLLAPSE_GUARDRAIL_PROFILE,
            workers,
            "collapse guardrail",
        ),
        prepared_bundles=prepared_bundles,
    )


def _expand_worker_profiles(raw: str, workers: int, label: str) -> list[str]:
    """Resolve one profile or one explicit profile per concurrent worker."""
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) == 1:
        return values * workers
    if len(values) != workers:
        raise ValueError(
            f"{label} profile count must be 1 or exactly {workers}; got {len(values)}"
        )
    return values


def _prepare_local_training_bundles(
    *,
    profiles: list[str],
    model: str | None,
    sample: int | None,
    payload: str = "full",
) -> list[Path]:
    """Build and validate one complete input bundle per worker locally."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = RESULTS / "prepared_training" / stamp
    root.mkdir(parents=True, exist_ok=False)
    model_key = model or str(training_cfg().training.base_model)
    bundles: list[Path] = []
    for number, profile in enumerate(profiles, start=1):
        bundle = root / f"worker_{number}_{profile}.pkl.gz"
        command = [
            sys.executable,
            "-u",
            "-m",
            "training.train",
            "--model",
            model_key,
            "--payload",
            payload,
            "--masking-profile",
            profile,
            "--collapse-guardrail-profile",
            _COLLAPSE_GUARDRAIL_PROFILE,
            "--prepare-bundle",
            str(bundle),
            "--no-mask-effect",
            "--no-plot",
        ]
        if sample is not None:
            command.extend(["--sample", str(sample)])
        env = {
            **os.environ,
            "PYTHONPATH": str(TRAIN_ROOT / "src"),
            "WANDB_MODE": "offline",
            "EUROMONITOR_RUN_ID": f"local-prepare-{stamp}-worker_{number}",
        }
        print(
            f"[local-prepare] worker={number} profile={profile} "
            f"bundle={bundle}",
            flush=True,
        )
        subprocess.run(command, cwd=TRAIN_ROOT, env=env, check=True)
        from training.prepared_bundle import load_prepared_bundle

        manifest, _ = load_prepared_bundle(bundle)
        print(
            f"[local-prepare] validated worker={number} "
            f"rows={manifest.n_df:,} payload={manifest.n_payload:,} "
            f"pos={manifest.n_pos:,} neg={manifest.n_neg:,} "
            f"sha256={manifest.sha256}",
            flush=True,
        )
        bundles.append(bundle)
    return bundles


def _upload_prepared_bundles(
    *,
    run_id: str,
    bundles: list[Path],
) -> list[str]:
    """Upload only the locally prepared bundles and their manifests."""
    remote_dir = f"{REMOTE_ROOT}/prepared_training/{run_id}"
    mkdir_script = _BOOTSTRAP + f"""
import pathlib
pathlib.Path({remote_dir!r}).mkdir(parents=True, exist_ok=True)
"""
    run_colab_exec_stream(
        SESSION,
        mkdir_script,
        timeout=120,
        log_name="prepared_bundle_mkdir",
        retry_safe=False,
    )
    remote_paths: list[str] = []
    for number, bundle in enumerate(bundles, start=1):
        for source in (bundle, bundle.with_suffix(bundle.suffix + ".json")):
            remote = f"{remote_dir}/worker_{number}/{source.name}"
            run_colab_exec_stream(
                SESSION,
                _BOOTSTRAP
                + f"""
import pathlib
pathlib.Path({str(Path(remote).parent)!r}).mkdir(parents=True, exist_ok=True)
""",
                timeout=120,
                log_name=f"prepared_bundle_dir_{number}",
                retry_safe=False,
            )
            print(f"[upload] prepared bundle file={source} -> {remote}", flush=True)
            colab(
                "upload",
                "-s",
                SESSION,
                str(source),
                remote,
                timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS,
            )
        remote_paths.append(f"{remote_dir}/worker_{number}/{bundle.name}")
    return remote_paths


def run_single_train_and_stream(
    args: list[str], *, run_label: str | None = None,
    prepared_bundle: Path | None = None,
) -> tuple[str, int]:
    """Run one worker in the Colab exec stream so W&B is visible immediately."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    remote_base = f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    if prepared_bundle is not None:
        remote_bundle = _upload_prepared_bundles(
            run_id=Path(remote_base).name.removeprefix("concurrent_train_"),
            bundles=[prepared_bundle],
        )[0]
        args = list(args)
        args[args.index("training.train")] = "training.train_prepared"
        args.extend(["--bundle", remote_bundle])
    remote_input_loop = (
        "for name in ():"
        if prepared_bundle is not None
        else 'for name in (F["canonical_records"], F["gate_results"], F["labeled_pairs"]):'
    )
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import os, pathlib, shutil, subprocess, sys
from core.common import F
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
out = base / "worker_1"
base.mkdir(parents=True, exist_ok=False)
out.mkdir()
{remote_input_loop}
    relative = name.relative_to(root / "results")
    source = root / "results" / relative
    if not source.is_file():
        raise FileNotFoundError(f"worker input missing: {{source}}")
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
wandb_dir = out / "wandb"
wandb_dir.mkdir(parents=True, exist_ok=True)
env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root / "src"),
       "EUROMONITOR_RESULTS_DIR": str(out), "EUROMONITOR_MLRUNS_DIR": str(out / "mlruns"),
       "WANDB_DIR": str(wandb_dir),
       "WANDB_RUN_NAME": {f'{Path(remote_base).name.removeprefix("concurrent_train_")}-{run_label}' if run_label else Path(remote_base).name.removeprefix("concurrent_train_")!r},
       "EUROMONITOR_RUN_ID": {f'{Path(remote_base).name.removeprefix("concurrent_train_")}-{run_label}' if run_label else Path(remote_base).name.removeprefix("concurrent_train_")!r},
       "EUROMONITOR_MINING_PROFILE": {run_label if run_label in ("mining_enabled", "masking_only") else ""!r},
       "EUROMONITOR_REMOTE_TRAINING": "1"}}
command = [sys.executable, *{args!r}]
log_path = out / "training.log"
print(f"[train-launch] worker 1 streaming directly: {{' '.join(command)}}", flush=True)
with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
    rc = process.wait()
if rc:
    raise RuntimeError(f"worker 1 failed (rc={{rc}}); log={{log_path}}")
print(f"[train] worker 1 completed; log={{log_path}}", flush=True)
"""
    print("[run] starting one trainer with direct live stdout streaming ...", flush=True)
    run_colab_exec_stream(
        SESSION,
        script,
        timeout=_WORKER_TIMEOUT_SECONDS,
        log_name="train",
        exclude_from_live_log=True,
        training_output=True,
    )
    download_verified_training_results(remote_base, 1)
    print("[train] single worker completed; results downloaded", flush=True)
    return remote_base, 1


def run_hpo(
    mode: str | None = None,
    *,
    resume: bool = False,
    trial_jobs: int = _HPO_TRIAL_JOBS_DEFAULT,
    persistence: str | None = None,
) -> None:
    """Sweep every configured backbone, then evaluate and rerank each winner."""
    mode = mode or _HPO_MODE
    persistence = persistence or _HPO_PERSISTENCE
    print(
        f"[run] round-robin HPO (mode={mode}, model_workers={_HPO_WORKERS}, "
        f"trial_jobs={trial_jobs}, resume={resume}, persistence={persistence}) ..."
    )
    run_id = (
        datetime.now(timezone.utc).strftime("hpo_%Y%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    mask_effect_flag = "--mask-effect" if _MASK_EFFECT_AFTER_TRAIN else "--no-mask-effect"
    resume_pointers = _hpo_resume_pointer_payload() if resume else {}
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import concurrent.futures, json, os, pathlib, shutil, subprocess, sys, time
from datetime import datetime, timezone
from core.common import F, hpo_cfg, resolve_model
root = pathlib.Path("{REMOTE_ROOT}")
hpo_root = root / "results" / "hpo_runs" / "{run_id}"
hpo_root.mkdir(parents=True, exist_ok=False)
(hpo_root / "generation.json").write_text(json.dumps({{
    "run_id": "{run_id}", "created_at": datetime.now(timezone.utc).isoformat(),
    "persistence": "{persistence}", "mode": "{mode}", "trial_jobs": {trial_jobs},
}}, indent=2, sort_keys=True), encoding="utf-8")
base = [sys.executable, "-u", "-m", "training.train", "--split", "holdout", "--loss", "contrastive", "--payload", "full", "--no-plot", "{mask_effect_flag}"]
hpo_base = list(base)
hpo_base.extend(["--n-jobs", str({trial_jobs})])
if {resume!r}:
    hpo_base.append("--resume")
resume_pointers = {resume_pointers!r}
model_keys = hpo_cfg()["models"]
required = {{"epochs", "lr", "warmup_ratio", "weight_decay"}}
mode = "{mode}"
workers = {_HPO_WORKERS}

for relative, encoded in resume_pointers.items():
    target = root / "results" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(__import__("base64").b64decode(encoded))

if {resume!r}:
    from training.dvc_store import restore_pointer
    pointers = sorted(
        path for path in (root / "results").rglob("*.dvc")
        if path.parent.name == ".resume"
    )
    if not pointers:
        raise RuntimeError(
            "[resume-preflight] no HPO DVC resume pointers are available; "
            "the previous Optuna database/checkpoints cannot be restored"
        )
    restored_db = False
    for pointer in pointers:
        source = pointer.parent.parent
        restore_pointer(source, pointer)
        restored_db = restored_db or pointer.name.endswith(".optuna.db.dvc")
    if not restored_db:
        raise RuntimeError(
            "[resume-preflight] restored HPO pointers but no Optuna database "
            "pointer was found; refusing to start the sweep"
        )
    print(f"[resume-preflight] restored {{len(pointers)}} HPO pointer(s)", flush=True)

def run_logged(args, label, extra_env=None):
    log_path = hpo_root / "logs" / (
        f"colab_{{label}}_{{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {{**os.environ, "PYTHONUNBUFFERED": "1"}}
    if extra_env:
        env.update(extra_env)
    print(f"[subprocess] {{' '.join(args)}} -> {{log_path}}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=root,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
    if rc:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"{{label}} failed (rc={{rc}}); log={{log_path}}\\n" + "\\n".join(tail)
        )
    return log_path

def worker_setup(model_key):
    out = hpo_root / "models" / model_key
    out.mkdir(parents=True, exist_ok=True)
    for name in (F["canonical_records"], F["gate_results"]):
        source, target = root / "results" / name.name, out / name.name
        if not source.is_file():
            raise FileNotFoundError(f"worker input missing: {{source}}")
        shutil.copy2(source, target)
    return out, {{
        "EUROMONITOR_RESULTS_DIR": str(out),
        "EUROMONITOR_MLRUNS_DIR": str(out / "mlruns"),
        "WANDB_RUN_NAME": f"{run_id}_hpo_{{model_key}}",
        "EUROMONITOR_REMOTE_TRAINING": "1",
        "EUROMONITOR_DISABLE_DVC_CHECKPOINTS": "1",
        "EUROMONITOR_HPO_RETENTION_MODE": "1",
        "EUROMONITOR_HPO_GENERATION_ID": "{run_id}",
        "EUROMONITOR_HPO_MODEL_KEY": model_key,
    }}

def run_model(model_key):
    model = resolve_model(model_key)
    out, env = worker_setup(model_key)
    model_tag = str(model).rstrip("/").rsplit("/", 1)[-1]
    best_path = out / f"train_{{model_tag}}-dlr_hpo_best.json"
    print(f"== HPO {{model_key}}: {{model}} (dev-selected; test withheld)", flush=True)
    run_logged(hpo_base + ["--model", str(model), "--hpo"], f"hpo_{{model_key}}", env)
    if not best_path.is_file():
        raise RuntimeError(f"missing HPO winner for {{model_key}}: {{best_path}}")
    best = json.loads(best_path.read_text())
    params = best["config"]
    missing = required - set(params)
    if missing:
        raise RuntimeError(f"HPO best config for {{model_key}} lacks {{sorted(missing)}}")
    final = base + [
        "--model", str(model),
        "--epochs", str(params["epochs"]),
        "--lr", str(params["lr"]),
        "--warmup-ratio", str(params["warmup_ratio"]),
        "--weight-decay", str(params["weight_decay"]),
        "--rerank", "{_RERANK_MODEL}",
    ]
    print(f"== FINAL {{model_key}}: selected dev config -> held-out test + rerank", flush=True)
    final_env = dict(env)
    if final_env:
        final_env["WANDB_RUN_NAME"] = f"{run_id}_final_{{model_key}}"
    run_logged(final, f"final_{{model_key}}", final_env)
    return {{"model_key": model_key, "model": str(model), "best": params,
            "results_dir": str(out.relative_to(root / "results"))}}

if mode == "parallel_same_vm":
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(model_keys))) as pool:
        summary = [future.result() for future in [pool.submit(run_model, key) for key in model_keys]]
else:
    summary = [run_model(key) for key in model_keys]
(hpo_root / "hpo_round_robin_summary.json").write_text(
    json.dumps({{"run_id": "{run_id}", "models": summary, "rerank_model": "{_RERANK_MODEL}"}}, indent=2),
    encoding="utf-8",
)
archive = pathlib.Path(shutil.make_archive(
    str(hpo_root), "gztar", root_dir=hpo_root.parent, base_dir=hpo_root.name
))
print(f"[hpo-archive] {{archive}}", flush=True)
print(json.dumps({{"hpo_run_id": "{run_id}", "hpo_round_robin": summary, "rerank_model": "{_RERANK_MODEL}"}}, sort_keys=True), flush=True)
"""
    try:
        run_colab_exec_stream(SESSION, script, timeout=8 * 3600 * 3, log_name="training_hpo")
        remote_archive = f"{REMOTE_ROOT}/results/hpo_runs/{run_id}.tar.gz"
        local_archive = TRAINING_RESULTS / "hpo_runs" / f"{run_id}.tar.gz"
        local_archive.parent.mkdir(parents=True, exist_ok=True)
        colab("download", "-s", SESSION, remote_archive, str(local_archive), timeout=3600)
        local_root = TRAINING_RESULTS / "hpo_runs"
        shutil.unpack_archive(local_archive, local_root, format="gztar")
        print(f"[hpo-archive] preserved -> {local_root / run_id}", flush=True)
    finally:
        # The VM is normally stopped by main() immediately after this
        # returns/raises. Keep the pointer files locally so a later
        # --resume-hpo can restore the DVC objects on a fresh VM.
        try:
            _mirror_hpo_resume_pointers()
        except Exception as exc:
            print(f"[warn] could not mirror HPO resume pointers: {exc}", file=sys.stderr, flush=True)
    return run_id


def run_sims() -> None:
    """Run the configured zero-shot embedding model lane on the VM."""
    print(f"[run] zero_shot_sims --models {_SIMS_MODEL} on the VM ...")
    run_id = datetime.now(timezone.utc).strftime("zero_shot_%Y%m%dT%H%M%S%fZ")
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import os, subprocess, sys
os.environ["EUROMONITOR_RUN_ID"] = {run_id!r}
os.environ["WANDB_RUN_NAME"] = {run_id!r}
rc = subprocess.run(
    [sys.executable, "-m", "training.zero_shot_sims", "--models", {_SIMS_MODEL!r}],
    cwd={REMOTE_ROOT!r},
).returncode
if rc != 0:
    raise RuntimeError(f"zero-shot similarity subprocess failed (rc={{rc}})")
"""
    run_colab_exec_stream(SESSION, script, timeout=2 * 3600, log_name="sims")
    print("[sims] remote zero-shot completed; downloading verified results ...", flush=True)
    download_results(skip_checkpoints=True, require_manifests=True)


def run_mixed(
    frac: float,
    epochs: int,
    *,
    model: str | None = None,
) -> tuple[str, int]:
    """Run one masked trainer and one zero-shot worker on the same VM."""
    model_key = model or str(training_cfg().training.base_model)
    if model_key not in set(embedding_model_keys()):
        raise ValueError(
            "mixed lane requires an embedding model registry key: "
            f"{model_key!r}"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    remote_base = f"{REMOTE_ROOT}/results/concurrent_train_mixed_{stamp}"
    train_args = [
        "-u",
        "-m",
        "training.train",
        "--split",
        "holdout",
        "--loss",
        "contrastive",
        "--train-frac",
        str(frac),
        "--epochs",
        str(epochs),
        "--model",
        model_key,
        "--no-plot",
    ]
    sims_args = [
        "-u",
        "-m",
        "training.zero_shot_sims",
        "--models",
        model_key,
    ]
    mixed_workers = _MIXED_TRAIN_WORKERS + _MIXED_SIMS_WORKERS
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import concurrent.futures, json, os, pathlib, shutil, subprocess, sys, threading, time
from core.common import F

root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
base.mkdir(parents=True, exist_ok=False)
worker_specs = [
    ("train", {train_args!r}, {_MIXED_MINING_PROFILE!r}, {_MASKING_ENABLED!r}),
    ("zero_shot", {sims_args!r}, {_MIXED_MINING_PROFILE!r}, False),
]

def emit_snapshot(label, out, proc):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commands = (
        ["ps", "-eo", "pid,ppid,pgid,etime,stat,%cpu,%mem,rss,args", "--forest"],
        ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
    )
    sections = []
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            body = (result.stdout or result.stderr or "").rstrip()
            sections.append("$ " + " ".join(command) + "\\n" + body)
        except Exception as exc:
            sections.append("$ " + " ".join(command) + "\\nERROR " + repr(exc))
    header = (
        "[mixed-monitor] timestamp=" + stamp
        + " worker=" + label
        + " child_pid=" + str(proc.pid)
        + " child_returncode=" + str(proc.poll())
        + "\\n"
    )
    text = header + "\\n".join(sections) + "\\n"
    print(text, end="", flush=True)
    with (out / "processes.log").open("a", encoding="utf-8") as handle:
        handle.write(text)

def monitor_worker(label, out, proc, stop):
    emit_snapshot(label, out, proc)
    while not stop.wait({int(_WORKER_MONITOR_SECONDS)}):
        emit_snapshot(label, out, proc)

def run_worker(number, label, command_args, profile, masking_applied):
    out = base / f"worker_{{number}}"
    out.mkdir()
    (out / "wandb").mkdir()
    for name in (F["canonical_records"], F["gate_results"]):
        source = root / "results" / name.name
        if not source.is_file():
            raise FileNotFoundError(f"worker input missing: {{source}}")
        shutil.copy2(source, out / name.name)
    log_path = out / ("training.log" if label == "train" else "zero_shot.log")
    (out / "worker_spec.json").write_text(json.dumps({{
        "label": label,
        "model_key": {model_key!r},
        "mining_profile": profile,
        "masking_requested": {_MASKING_ENABLED!r},
        "masking_applied": masking_applied,
        "masking_note": (
            "training augmentation is applied by training.train"
            if masking_applied else
            "zero-shot scoring has no training augmentation stage"
        ),
    }}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    env = {{
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(root / "src"),
        "EUROMONITOR_RESULTS_DIR": str(out),
        "EUROMONITOR_MLRUNS_DIR": str(out / "mlruns"),
        "WANDB_DIR": str(out / "wandb"),
        "WANDB_RUN_NAME": f"{{base.name}}-{{label}}",
        "EUROMONITOR_RUN_ID": f"{{base.name}}-{{label}}",
        "EUROMONITOR_MINING_PROFILE": profile,
        "EUROMONITOR_REMOTE_TRAINING": "1",
    }}
    command = [sys.executable, *command_args]
    print(f"[mixed] starting {{label}}: {{' '.join(command)}}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            process_group = str(os.getpgid(proc.pid))
        except ProcessLookupError:
            process_group = "exited"
        print(f"[mixed] worker={{label}} pid={{proc.pid}} pgid={{process_group}} monitor_interval={int(_WORKER_MONITOR_SECONDS)}s", flush=True)
        monitor_stop = threading.Event()
        monitor = threading.Thread(
            target=monitor_worker,
            args=(label, out, proc, monitor_stop),
            name="mixed-monitor-" + label,
            daemon=True,
        )
        monitor.start()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(f"[{{label}}] {{line}}", end="", flush=True)
                log.write(line)
        finally:
            monitor_stop.set()
            monitor.join()
            emit_snapshot(label, out, proc)
    rc = proc.wait()
    (out / "worker.status").write_text(f"{{rc}}\\n", encoding="utf-8")
    if rc:
        raise RuntimeError(f"mixed worker {{label}} failed (rc={{rc}}); log={{log_path}}")
    return label

with concurrent.futures.ThreadPoolExecutor(max_workers={mixed_workers}) as pool:
    futures = [
        pool.submit(run_worker, number, label, command, profile, masking_applied)
        for number, (label, command, profile, masking_applied) in enumerate(worker_specs, start=1)
    ]
    completed = [future.result() for future in futures]
print(json.dumps({{"base": str(base), "completed": completed}}), flush=True)
"""
    print(
        f"[run] mixed lane: {_MIXED_TRAIN_WORKERS} masked trainer + "
        f"{_MIXED_SIMS_WORKERS} zero-shot worker on {model_key} ...",
        flush=True,
    )
    run_colab_exec_stream(
        SESSION,
        script,
        timeout=_WORKER_TIMEOUT_SECONDS,
        log_name="mixed",
        training_output=True,
    )
    download_verified_training_results(remote_base, mixed_workers)
    print("[mixed] both workers completed; results downloaded", flush=True)
    return remote_base, 2


def _list_remote(pattern_dir: str) -> list[str]:
    """List remote files via a stdin-exec (same channel the lanes use)."""
    import json as _json

    script = (
        "import pathlib, json\n"
        f"files = sorted(str(p) for p in pathlib.Path('{pattern_dir}').rglob('*') if p.is_file())\n"
        "print('@@FILES@@' + json.dumps(files))\n"
    )
    proc = subprocess.Popen(
        _colab_command("exec", "-s", SESSION, "--timeout", "120"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(script, timeout=120)
    if proc.returncode != 0:
        raise SystemExit(f"remote listing failed: {err[-500:]}")
    for line in out.splitlines():
        if line.startswith("@@FILES@@"):
            return _json.loads(line[len("@@FILES@@"):])
    raise SystemExit(f"remote listing returned no marker; out={out[-500:]}")


def _download_remote_manifests(*, required: bool = True) -> list[StageManifest]:
    """Pull and validate the completion records produced by remote stages.

    The manifests live outside the normal results-download tree, so they
    must be fetched explicitly before any artifact can be
    trusted.  A lane that produced no completion records is incomplete by
    definition: do not tear down its only copy while claiming success.
    """
    remote_dir = f"{REMOTE_ROOT}/results/manifests"
    names = _list_remote(remote_dir)
    if not names and not required:
        print("[download] no stage manifests (frozen CSV lane)")
        return []
    if not names:
        raise RuntimeError(
            f"remote manifest directory is empty: {remote_dir}; refusing "
            "to download unverifiable lane results"
        )

    local_dir = RESULTS / "manifests"
    local_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[StageManifest] = []
    for name in names:
        remote = Path(name)
        try:
            rel = remote.relative_to(remote_dir)
        except ValueError as exc:
            raise RuntimeError(f"remote manifest escaped manifest dir: {name}") from exc
        if rel.parent != Path(".") or remote.suffix != ".json":
            raise RuntimeError(f"unexpected remote manifest path: {name}")
        local = local_dir / rel
        print(f"[download] manifest {rel}")
        colab("download", "-s", SESSION, name, str(local), timeout=600)
        try:
            manifest = StageManifest.model_validate_json(
                local.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RuntimeError(f"invalid remote manifest {name}: {exc}") from exc
        if manifest.status != "complete":
            raise RuntimeError(
                f"remote manifest {name} has status {manifest.status!r}; "
                "stage did not complete"
            )
        manifests.append(manifest)
    return manifests


def _local_path_for_remote(remote_path: str) -> Path:
    """Map an absolute path in the mirrored remote repo back to this repo."""
    try:
        rel = Path(remote_path).relative_to(REMOTE_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"manifest output is outside remote project root: {remote_path}"
        ) from exc
    return TRAIN_ROOT / rel


def _verify_manifest_downloads(manifests: list[StageManifest]) -> None:
    """Fail if a manifest-listed expected output is absent or byte-different."""
    problems: list[str] = []
    for manifest in manifests:
        output_names = {Path(entry.path).name for entry in manifest.outputs}
        for expected in manifest.expected_outputs:
            if expected not in output_names:
                problems.append(
                    f"{manifest.stage}: expected output absent from manifest: {expected}"
                )
        for entry in manifest.outputs:
            local = _local_path_for_remote(entry.path)
            if not local.is_file():
                problems.append(f"{manifest.stage}: missing local output: {local}")
                continue
            actual = sha256_file(local)
            if actual != entry.sha256:
                problems.append(
                    f"{manifest.stage}: sha256 mismatch for {local} "
                    f"(remote {entry.sha256[:12]}, local {actual[:12]})"
                )
    if problems:
        raise RuntimeError(
            "Colab download integrity verification failed:\n  - "
            + "\n  - ".join(problems)
        )


def download_results(
    skip_checkpoints: bool = True, *, require_manifests: bool = False
) -> list[StageManifest]:
    """Pull the result artifacts back to the repo results dir.

    AUDIT FIX 2026-09-08: the generic rglob included _checkpoints (~1.9 GB
    of model weights) for EVERY lane — checkpoints are pulled explicitly by
    download_checkpoints() only when --what train asks for them.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifests = _download_remote_manifests(required=require_manifests)
    files = _list_remote(f"{REMOTE_ROOT}/results")
    for name in files:
        rel = Path(name).relative_to(f"{REMOTE_ROOT}/results")
        if skip_checkpoints and rel.parts[0] == "_checkpoints":
            continue
        local = RESULTS / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {rel}")
        # RULING 2026-09-10 (silent-degradation audit): LOUD-RAISE.
        # This runs after the lane and before stop() destroys the VM: a
        # swallowed failure here is a silent data drop — main() would
        # tear down the only remaining copy and print "[done] artifacts
        # saved" over a partial results dir. Re-raise instead (colab()
        # already printed the command + stderr tail; stop() still runs
        # via main()'s finally unless --keep-alive, so the remote copy
        # survives for a re-pull).
        try:
            colab("download", "-s", SESSION, name, str(local), timeout=600)
        except subprocess.CalledProcessError:
            print(
                f"[error] results download failed for {rel} — local copy "
                f"at {local} is absent/partial; refusing to continue "
                f"because teardown would delete the only remote copy "
                f"(re-run the lane, or re-pull from a --keep-alive VM)",
                file=sys.stderr,
            )
            raise
    _verify_manifest_downloads(manifests)
    return manifests


def download_checkpoints(manifests: list[StageManifest] | None = None) -> None:
    """Pull the trained checkpoints (model weights) back.

    Called after --what train: the trained model IS the deliverable of the
    production run; results CSVs alone don't carry it.
    """
    # Re-fetch and re-verify after this separately downloaded tree too.  A
    # future stage may list a checkpoint as an output; then it receives the
    # same hash gate as ordinary results instead of becoming a blind spot.
    if manifests is None:
        manifests = _download_remote_manifests(required=False)
    print("[download] checkpoints ...")
    files = _list_remote(f"{REMOTE_ROOT}/results/_checkpoints")
    for name in files:
        rel = Path(name).relative_to(f"{REMOTE_ROOT}/results")
        local = RESULTS / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {rel}")
        # RULING 2026-09-10 (silent-degradation audit): LOUD-RAISE.
        # The trained model weights ARE the deliverable of --what train
        # (results CSVs alone don't carry it, see docstring); a swallowed
        # download here leaves the lane's only artifact on a VM that
        # stop() is about to destroy. Re-raise so the operator can
        # re-pull before teardown (--keep-alive keeps the VM up).
        try:
            colab("download", "-s", SESSION, name, str(local), timeout=1200)
        except subprocess.CalledProcessError:
            print(
                f"[error] checkpoint download failed for {rel} — the "
                f"trained weights were NOT pulled local; continuing would "
                f"let stop() destroy the only copy. Re-run the lane or "
                f"re-pull manually before the VM is gone (--keep-alive "
                f"keeps it up)",
                file=sys.stderr,
            )
            raise
    _verify_manifest_downloads(manifests)


def stop() -> None:
    print(f"[stop] tearing down '{SESSION}'")
    # RULING 2026-09-10 (silent-degradation audit): JUSTIFIED-KEEP.
    # stop() runs in main()'s finally — if the lane itself raised, the
    # lane's exception is the root cause and must stay the error the
    # operator sees; raising here would MASK it with a teardown failure
    # and abort nothing (no local data or pipeline state depends on the
    # VM being gone). But it must not be silent: an unreleased VM burns
    # Colab GPU quota until manually reaped, so warn loudly with the
    # consequence + the exact recovery command.
    try:
        result = colab("stop", "-s", SESSION, check=False)
        if result.returncode:
            print(
                f"[warn] VM release command returned rc={result.returncode}; "
                f"stdout={result.stdout[-2000:]!r} stderr={result.stderr[-2000:]!r}",
                file=sys.stderr,
            )
        status = colab("sessions", check=False)
        if SESSION in (status.stdout or ""):
            print(
                f"[warn] teardown verification still lists '{SESSION}'; "
                "the VM may still be live and consuming quota.",
                file=sys.stderr,
            )
        elif status.returncode == 0:
            print("[stop] teardown verified: session is no longer listed")
        else:
            print(
                f"[warn] could not verify teardown; sessions command returned "
                f"rc={status.returncode}: {status.stderr[-1000:]!r}",
                file=sys.stderr,
            )
    except subprocess.SubprocessError as exc:
        print(
            f"[warn] VM release request failed — the VM '{SESSION}' may "
            f"STILL BE LIVE and burning Colab GPU quota until it times "
            f"out or is reaped. After handling the failure above, reclaim "
            f"it with: colab stop -s {SESSION}   (or 'colab sessions' "
            f"to check). Original error: {exc}",
            file=sys.stderr,
        )
    print("[stop] VM release requested")


def main() -> None:
    global GPU
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", required=True,
                    choices=["train", "hpo", "sims", "mixed", "smoke", "stop"],
                    help="what to run on the VM")
    ap.add_argument("--train-frac", type=float, default=_TRAIN_FRAC_DEFAULT,
                    help=f"train fraction for --what train (default "
                    f"{_TRAIN_FRAC_DEFAULT:g})")
    ap.add_argument("--epochs", type=int, default=_EPOCHS_DEFAULT,
                    help=f"epochs for --what train (default {_EPOCHS_DEFAULT} = "
                    "config/training.yaml training.epochs)")
    ap.add_argument(
        "--workers", type=int, default=_TRAIN_WORKERS,
        help=f"concurrent full-data trainers for --what train (default {_TRAIN_WORKERS} = "
        "config/training.yaml colab.train_workers; use 1 for a single run)",
    )
    ap.add_argument(
        "--sample", type=int, default=None,
        help="optional smoke cap for --what train; full data when omitted",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="model registry key for --what train (for example minilm_l6)",
    )
    ap.add_argument(
        "--run-label",
        default=None,
        help="experiment label for W&B (for example mining_enabled or masking_only)",
    )
    ap.add_argument(
        "--masking-profile",
        default=_MASKING_PROFILE,
        help="config/training.yaml masking_profiles entry",
    )
    ap.add_argument(
        "--collapse-guardrail-profile",
        default=_COLLAPSE_GUARDRAIL_PROFILE,
        help="config/training.yaml collapse_guardrail_profiles entry",
    )
    ap.add_argument(
        "--resume-run",
        default=None,
        help="resume this existing concurrent_train_<id> run on the VM",
    )
    ap.add_argument(
        "--resume-hpo",
        action="store_true",
        help="restore the previous HPO Optuna database/checkpoints before the sweep",
    )
    ap.add_argument(
        "--hpo-persistence",
        choices=["dvc", "local", "none"],
        default=_HPO_PERSISTENCE,
        help="HPO durability backend (default from config/training.yaml)",
    )
    ap.add_argument(
        "--gpu",
        default=GPU,
        help=f"Colab accelerator request (default {GPU}; e.g. A100 when available)",
    )
    ap.add_argument(
        "--allow-gpu",
        action="store_true",
        help="required acknowledgement before a non-CPU runtime can be provisioned",
    )
    ap.add_argument(
        "--hpo-mode",
        choices=["sequential", "parallel_same_vm"],
        default=_HPO_MODE,
        help="HPO scheduling mode (default from config/training.yaml)",
    )
    ap.add_argument(
        "--hpo-jobs",
        type=int,
        default=_HPO_TRIAL_JOBS_DEFAULT,
        help="concurrent Optuna trials per backbone (3 models x 3 jobs = 9 A100 workers)",
    )
    ap.add_argument(
        "--refresh-data",
        action="store_true",
        help="explicitly regenerate frozen CSV inputs before training",
    )
    ap.add_argument("--keep-alive", action="store_true",
                    help="do not tear down the VM on completion/failure")
    args = ap.parse_args()

    GPU = args.gpu
    if GPU.upper() != "CPU" and not args.allow_gpu:
        raise ValueError("GPU launch requires --allow-gpu")

    dvc_jobs = int(training_cfg().colab.dvc_jobs)
    if args.what == "train":
        print(
            f"[workers] lane=train trainers={args.workers} "
            f"dvc_publishers={_DVC_WORKERS} dvc_transfer_jobs={dvc_jobs}",
            flush=True,
        )
    elif args.what == "smoke":
        print(
            f"[workers] lane=smoke trainers={_SMOKE_WORKERS} "
            f"dvc_publishers={_DVC_WORKERS} dvc_transfer_jobs={dvc_jobs}",
            flush=True,
        )
    elif args.what == "hpo":
        print(
            f"[workers] lane=hpo model_workers={_HPO_WORKERS} "
            f"trial_jobs={args.hpo_jobs} dvc_publishers={_DVC_WORKERS} "
            f"dvc_transfer_jobs={dvc_jobs}",
            flush=True,
        )
    elif args.what == "mixed":
        print(
            f"[workers] lane=mixed train_workers={_MIXED_TRAIN_WORKERS} "
            f"zero_shot_workers={_MIXED_SIMS_WORKERS} "
            f"dvc_publishers={_DVC_WORKERS} dvc_transfer_jobs={dvc_jobs}",
            flush=True,
        )

    if args.what == "stop":
        stop()
        return

    start_live_log()
    check_colab_cli()
    local_training_run: tuple[str, int] | None = None
    local_mixed_run: tuple[str, int] | None = None
    local_hpo_run: str | None = None

    try:
        ensure_session()
        prepared_train_runtime = args.what in {"train", "smoke"}
        prepare_remote_layout(minimal_runtime=prepared_train_runtime)
        install_deps(minimal_runtime=prepared_train_runtime)
        if args.what in {"train", "smoke", "mixed"}:
            required_models = [
                args.model or str(training_cfg().training.base_model)
            ]
        elif args.what == "hpo":
            required_models = list(hpo_cfg()["models"])
            required_models.append(str(sweep_cfg()["rerank_model"]))
        elif args.what == "sims":
            required_models = [_SIMS_MODEL]
        else:
            required_models = []
        if required_models:
            verify_remote_models(required_models)
        log_gpu_profile()
        if args.refresh_data and prepared_train_runtime:
            raise ValueError("--refresh-data is incompatible with local-prepared GPU training")
        if args.refresh_data:
            run_data_prep()
        elif prepared_train_runtime:
            verify_remote_prepared_inputs()
        else:
            verify_training_inputs()
        # AUDIT FIX 2026-09-08: --what sims used to run FULL TRAINING first
        # (run_train was unconditional) — hours of unintended GPU quota
        # for a lane that only needs the configured zero-shot scoring.
        if args.what == "sims":
            run_sims()
        elif args.what == "mixed":
            local_mixed_run = run_mixed(args.train_frac, args.epochs, model=args.model)
        elif args.what == "smoke":
            local_training_run = run_train(
                args.train_frac, _SMOKE_EPOCHS, sample=_SMOKE_SAMPLE,
                workers=_SMOKE_WORKERS, run_label=args.run_label,
                masking_profile=args.masking_profile,
                collapse_guardrail_profile=args.collapse_guardrail_profile,
            )
        elif args.what == "hpo":
            if args.hpo_jobs < 1:
                raise ValueError("--hpo-jobs must be >= 1")
            local_hpo_run = run_hpo(
                args.hpo_mode,
                resume=args.resume_hpo,
                trial_jobs=args.hpo_jobs,
                persistence=args.hpo_persistence,
            )
        else:
            local_training_run = run_train(
                args.train_frac, args.epochs, sample=args.sample, workers=args.workers,
                resume_run=args.resume_run, model=args.model,
                run_label=args.run_label,
                masking_profile=args.masking_profile,
                collapse_guardrail_profile=args.collapse_guardrail_profile,
            )
        if local_hpo_run is not None:
            print("[hpo] publishing snapshots on local CPU ...", flush=True)
            publish_local_hpo_results(local_hpo_run, args.hpo_persistence)
    except BaseException:
        print(
            "[launcher] traceback before teardown; result transfer is incomplete:",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        # Release the runtime after every outcome.  Retaining a failed VM can
        # consume quota indefinitely; callers that intentionally need live
        # recovery must make that choice explicitly with --keep-alive.
        if not args.keep_alive:
            stop()
        else:
            print("\n[info] --keep-alive specified. VM is still running.")
        close_live_log()

    print("\n[done] artifacts persisted to the configured DVC remote")


if __name__ == "__main__":
    main()
