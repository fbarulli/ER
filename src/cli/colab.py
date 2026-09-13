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
  sims   — the deberta zero-shot lane (GPU-only: 3.9s/text on CPU — the
           CPU lane leaves its column absent by design, see
           src/training/zero_shot_sims.py). Scores with --models deberta_v3_base
           against the same canonical fingerprint contract.
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
import json
import os
import shutil
import subprocess
import sys
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
    sweep_cfg,
    hpo_cfg,
    load_config,
    resolve_model,
    training_cfg,
)
from core.manifest import sha256_file
from core.schemas import StageManifest

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
_DVC_WORKERS = _COLAB.dvc_workers
_LOG_POLL_SECONDS = _COLAB.log_poll_seconds
_PROBE_TIMEOUT_SECONDS = _COLAB.probe_timeout_seconds
_PROBE_RETRIES = _COLAB.probe_retries
_PROBE_RETRY_BACKOFF_SECONDS = _COLAB.probe_retry_backoff_seconds
_MASK_EFFECT_AFTER_TRAIN = _COLAB.mask_effect_after_train
_SMOKE_EPOCHS = _COLAB.smoke_epochs
_WORKER_TIMEOUT_SECONDS = _COLAB.worker_timeout_seconds
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
_post_training_event_lock = threading.Lock()
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


def _post_training_event(
    run_id: str,
    stage: str,
    state: str,
    *,
    worker: int | None = None,
    **details: object,
) -> None:
    """Persist ordered local post-training state without measuring duration."""
    root = TRAINING_RESULTS / run_id
    root.mkdir(parents=True, exist_ok=True)
    event = {"stage": stage, "state": state, **details}
    if worker is not None:
        event["worker"] = int(worker)
    event_path = root / training_cfg().colab.post_training_events_file
    with _post_training_event_lock:
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    suffix = f" worker={worker}" if worker is not None else ""
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[post-training-state] {stage} {state}{suffix}" + (f" | {detail_text}" if detail_text else ""), flush=True)

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
) -> tuple[str, int]:
    """Run isolated full-data trainers concurrently and mirror worker logs."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    remote_base = (
        f"{REMOTE_ROOT}/results/concurrent_train_{resume_run}"
        if resume_run
        else f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    )
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    resume_pointers = _resume_pointer_payload(run_id, workers) if resume_run else {}
    launch = _BOOTSTRAP + _remote_auth_env_script() + f"""
import base64, json, os, pathlib, shutil, shlex, subprocess, sys, time, traceback
from core.common import F
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
run_id = base.name.removeprefix("concurrent_train_")
base.mkdir(parents=True, exist_ok={bool(resume_run)!r})
command = " ".join(shlex.quote(part) for part in [sys.executable, *{args!r}])
resume_pointers = {resume_pointers!r}
run_labels = {run_labels!r}
started = []
for number in range(1, {workers} + 1):
    worker_profile = (
        run_labels[number - 1]
        if run_labels and number <= len(run_labels)
        else ""
    )
    training_name = f"{{run_id}}-{{worker_profile}}" if worker_profile else "{{run_id}}"
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
        for name in (F["canonical_records"], F["gate_results"]):
            source = root / "results" / name.name
            if not (out / name.name).is_file():
                if not source.is_file():
                    raise FileNotFoundError(f"resume worker input missing: {{source}}")
                shutil.copy2(source, out / name.name)
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
        for name in (F["canonical_records"], F["gate_results"]):
            source = root / "results" / name.name
            if not source.is_file():
                raise FileNotFoundError(f"worker input missing: {{source}}")
            shutil.copy2(source, out / name.name)
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


def download_verified_training_results(remote_base: str, workers: int) -> None:
    """Materialize raw worker outputs before the VM is released.

    DVC publication is deliberately local now, after reports are generated;
    this transfer therefore cannot claim DVC verification yet.
    """
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    local_base = TRAINING_RESULTS / run_id
    _post_training_event(run_id, "download", "started", workers=workers)
    downloaded = 0
    for number in range(1, workers + 1):
        remote_dir = f"{remote_base}/worker_{number}"
        local_dir = local_base / f"worker_{number}"
        for name in _list_remote(remote_dir):
            remote = Path(name)
            rel = remote.relative_to(remote_dir)
            # Checkpoints are part of the deliverable: the local uniformity
            # audit and final prediction notebook must run against the actual
            # fine-tuned weights. DVC verification still happens at each save,
            # but the selected local post-processing lane needs the materialized
            # checkpoint tree before teardown.
            if remote.suffix not in {
                ".csv", ".json", ".log", ".png", ".safetensors", ".bin",
                ".pt", ".pth", ".npz", ".pkl", ".pickle", ".dvc",
                ".yaml", ".yml", ".txt", ".jsonl", ".html", ".db", ".sqlite3",
            }:
                continue
            local = local_dir / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            colab("download", "-s", SESSION, name, str(local), timeout=600)
            downloaded += 1
    _post_training_event(
        run_id,
        "download",
        "completed",
        workers=workers,
        files=downloaded,
        destination=str(local_base),
    )
    print(f"[download] raw training outputs -> {local_base} ({downloaded} files)", flush=True)


def generate_local_training_reports(remote_base: str, workers: int) -> None:
    """Generate CPU-side reports after remote training results are downloaded."""
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    local_base = TRAINING_RESULTS / run_id
    from training.generate_training_report import generate_report
    from core.common import load_dataset_deduped
    from pipeline import build_training_data
    from training.uniformity import run_uniformity_audit

    # These inputs are identical for every downloaded worker. Build them once
    # so report generation does not rewrite the same payload audit repeatedly.
    uniformity_cfg = training_cfg().evaluation.uniformity
    _post_training_event(run_id, "report_inputs", "started")
    data = load_dataset_deduped()
    payload = build_training_data(data, payload_variant="full")["payload"]
    _post_training_event(
        run_id,
        "report_inputs",
        "completed",
        rows=len(data),
        payload_entries=len(payload),
    )

    for number in range(1, workers + 1):
        worker = local_base / f"worker_{number}"
        _post_training_event(run_id, "reports", "started", worker=number)
        metrics = sorted(worker.glob("*_holdout_*_fold_metrics.csv"))
        pair_paths = sorted(worker.glob("*_fold*_pairs.csv"))
        if not metrics or not pair_paths:
            raise FileNotFoundError(
                f"cannot generate local report for worker {number}: "
                f"metrics={len(metrics)} pairs={len(pair_paths)} under {worker}"
            )
        pointer_path = worker / F["results_pointer"].name
        run_tag = run_id
        if pointer_path.is_file():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            run_tag = str(pointer.get("run_tag") or run_tag)
        report_dir = worker / f"report_{run_tag}"
        checkpoints = sorted(
            worker.glob("_checkpoints/**/checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1])
            if path.name.split("-")[-1].isdigit()
            else -1,
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"uniformity audit requires a downloaded fine-tuned checkpoint under {worker / '_checkpoints'}"
            )
        uniformity = {}
        if uniformity_cfg.enabled:
            scope = uniformity_cfg.checkpoint_scope
            selected_checkpoints = checkpoints if scope == "all" else [checkpoints[-1]]
            base_model_ref = str(training_cfg().training.base_model)
            try:
                base_model = Path(resolve_model(base_model_ref))
            except FileNotFoundError as exc:
                base_model = None
                print(
                    f"[report-local] worker {number}: uniformity skipped; "
                    f"configured base model is not materialized locally: "
                    f"{base_model_ref} ({exc})",
                    flush=True,
                )
            if base_model is None:
                for checkpoint in selected_checkpoints:
                    uniformity[checkpoint.name] = {
                        "status": "skipped_base_model_unavailable",
                        "base_model": base_model_ref,
                    }
            for checkpoint in (selected_checkpoints if base_model is not None else []):
                try:
                    uniformity[checkpoint.name] = run_uniformity_audit(
                        checkpoint,
                        report_dir / "uniformity" / checkpoint.name,
                        base_model=base_model,
                        df=data,
                        payload=payload,
                        n_pairs=uniformity_cfg.sample_pairs,
                        seed=uniformity_cfg.seed,
                        threshold=uniformity_cfg.threshold,
                    )
                except Exception as exc:
                    # Uniformity is an optional diagnostic. Record a named
                    # failure without dumping the same traceback once per
                    # checkpoint; the report event remains inspectable.
                    print(
                        f"[report-local] worker {number}: uniformity audit "
                        f"failed for {checkpoint}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    uniformity[checkpoint.name] = {
                        "status": "error",
                        "error_type": "uniformity_audit_failed",
                        "error": str(exc),
                    }
        generate_report(
            metrics[-1],
            pair_paths,
            report_dir,
            sorted(worker.glob("*_fold*_train_scores.csv")),
            sorted(worker.glob("*_fold*_random_easy_scores.csv")),
            uniformity_summary={"checkpoints": uniformity},
        )
        _post_training_event(
            run_id,
            "reports",
            "completed",
            worker=number,
            report_dir=str(report_dir),
            uniformity_checkpoints=len(uniformity),
        )
        print(f"[report-local] worker {number}: {report_dir}", flush=True)


def generate_local_mask_effect(remote_base: str, workers: int) -> None:
    """Run masked-vs-unmasked scoring on local CPU after teardown.

    mask_effect_after_train only controls whether the VM spends remote GPU
    time on it (--mask-effect vs --no-mask-effect in the worker launch;
    blocked on the VM anyway by the remote-training gate). The downloaded
    results are always scored here on CPU; workers without a visibility log
    or checkpoint are skipped per worker with a message, never silently.
    """
    import numpy as np
    import pandas as pd

    from core.common import training_cfg as _training_cfg

    mask_cfg = _training_cfg().masking
    embed_batch = int(_training_cfg().training.batch_size_embed)
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    for number in range(1, workers + 1):
        worker = TRAINING_RESULTS / run_id / f"worker_{number}"
        pointer_path = worker / F["results_pointer"].name
        if not pointer_path.is_file():
            _post_training_event(run_id, "mask_effect", "skipped", worker=number, reason="missing_results_pointer")
            continue
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_tag = str(pointer.get("run_tag") or run_id)
        visibility = worker / "logs" / run_tag / "mask_visibility.csv"
        if not visibility.is_file():
            visibility = worker / "logs" / "mask_visibility.csv"
        if not visibility.is_file():
            print(f"[mask-effect-local] worker {number}: visibility log absent; skipped", flush=True)
            _post_training_event(run_id, "mask_effect", "skipped", worker=number, reason="missing_visibility_log")
            continue
        audit = pd.read_csv(visibility)
        if audit.empty:
            _post_training_event(run_id, "mask_effect", "skipped", worker=number, reason="empty_visibility_log")
            continue
        model_tag = str(pointer.get("model", "")).rstrip("/").rsplit("/", 1)[-1]
        roots = sorted((worker / "_checkpoints" / model_tag).glob(f"r{run_tag}_f*"))
        if not roots:
            print(f"[mask-effect-local] worker {number}: checkpoint root absent; skipped", flush=True)
            _post_training_event(run_id, "mask_effect", "skipped", worker=number, reason="missing_checkpoint_root")
            continue
        candidates = sorted(
            roots[0].glob("checkpoint-*"),
            key=lambda path: int(path.name.removeprefix("checkpoint-")),
        )
        source = candidates[-1] if candidates else roots[0]
        for candidate in candidates:
            state_path = candidate / "trainer_state.json"
            if not state_path.is_file():
                continue
            best = json.loads(state_path.read_text(encoding="utf-8")).get("best_model_checkpoint")
            if best:
                best_path = roots[0] / Path(best).name
                if best_path.exists():
                    source = best_path
                    break

        if "target_text" not in audit.columns:
            canonical = pd.read_csv(worker / F["canonical_records"].name, dtype=str)
            by_gtin = dict(zip(canonical["gtin"].astype(str), canonical["canonical"].astype(str), strict=True))
            audit["target_text"] = audit["barcode"].astype(str).map(by_gtin)
        audit = audit.dropna(subset=["target_text"])
        if audit.empty:
            _post_training_event(run_id, "mask_effect", "skipped", worker=number, reason="no_resolved_targets")
            continue
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(str(source), device="cpu")
            masked = audit["masked_text"].astype(str).tolist()
            anchors = audit["anchor_text"].astype(str).tolist()
            targets = audit["target_text"].astype(str).tolist()
            embeddings = model.encode(
                masked + anchors + targets,
                batch_size=embed_batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            n = len(audit)
            masked_scores = np.einsum("ij,ij->i", embeddings[:n], embeddings[2 * n:])
            unmasked_scores = np.einsum("ij,ij->i", embeddings[n:2 * n], embeddings[2 * n:])
            result = pd.DataFrame({
                "realized_extent": audit["realized_extent"],
                "sim_to_target": masked_scores.round(4),
                "masked_score": masked_scores.round(4),
                "unmasked_score": unmasked_scores.round(4),
                "masked_minus_unmasked": (masked_scores - unmasked_scores).round(4),
                "barcode": audit["barcode"],
                "anchor_text": anchors,
                "masked_text": masked,
            })
            midpoint = (float(mask_cfg.mask_lo) + float(mask_cfg.mask_hi)) / 2.0
            result["bucket"] = np.where(
                result["realized_extent"] >= midpoint,
                f"high(>={midpoint:.2f})",
                f"low(<{midpoint:.2f})",
            )
            log_dir = worker / "logs" / run_tag
            log_dir.mkdir(parents=True, exist_ok=True)
            result.to_csv(log_dir / "mask_effect.csv", index=False)
            result.to_csv(worker / "logs" / "mask_effect.csv", index=False)
            metrics = {
                "mask_n": float(len(result)),
                "mask_masked_mean_cosine": float(result["masked_score"].mean()),
                "mask_masked_median_cosine": float(result["masked_score"].median()),
                "mask_unmasked_mean_cosine": float(result["unmasked_score"].mean()),
                "mask_unmasked_median_cosine": float(result["unmasked_score"].median()),
                "mask_mean_cosine_delta": float(result["masked_minus_unmasked"].mean()),
                "mask_median_cosine_delta": float(result["masked_minus_unmasked"].median()),
            }
            (worker / "local_mask_effect_metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(7, 4.5))
            axis.hist(result["unmasked_score"], bins=20, alpha=0.55, label="non-masked")
            axis.hist(result["masked_score"], bins=20, alpha=0.55, label="masked")
            axis.set(xlabel="cosine similarity to target", ylabel="count",
                     title="Masked vs non-masked positive performance")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(worker / f"mask_effect_{run_tag}.png", dpi=150)
            plt.close(figure)
            metric_paths = sorted(worker.glob("*_holdout_*_fold_metrics.csv"))
            if metric_paths:
                metrics_frame = pd.read_csv(metric_paths[-1])
                for key, value in metrics.items():
                    metrics_frame.loc[metrics_frame["status"].eq("ok"), key] = value
                metrics_frame.to_csv(metric_paths[-1], index=False)
                shared = worker / F["fold_metrics"].name
                metrics_frame.to_csv(shared, index=False)
            _post_training_event(
                run_id,
                "mask_effect",
                "completed",
                worker=number,
                rows=len(result),
                checkpoint=str(source),
            )
            print(f"[mask-effect-local] worker {number}: CPU scoring complete ({len(result)} rows)", flush=True)
        except Exception as exc:
            print(
                f"[mask-effect-local] worker {number} failed; full traceback:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            _post_training_event(
                run_id,
                "mask_effect",
                "failed",
                worker=number,
                error=str(exc),
            )


def _local_auth_env() -> dict[str, str]:
    """Return local-only credentials for post-training publishers."""
    env = dict(os.environ)
    dvc_key = _env_value("DVC_API_KEY")
    if dvc_key:
        env["DVC_API_KEY"] = dvc_key
        env["DAGSHUB_USER_TOKEN"] = dvc_key
    wandb_key = _env_value("WANDB_API_KEY")
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key
    return env


def _publish_local_training_worker(run_id: str, number: int, workers: int) -> None:
    """Publish one isolated worker bundle; safe to submit concurrently."""
    source = TRAINING_RESULTS / run_id / f"worker_{number}"
    if not source.is_dir():
        raise FileNotFoundError(f"local worker output missing: {source}")
    _post_training_event(
        run_id,
        "dvc_publish",
        "started",
        worker=number,
        workers=workers,
        source=str(source),
    )
    print(f"[dvc-local] publishing worker {number}/{workers}: {source}", flush=True)
    try:
        subprocess.run(
            [
                sys.executable, "-u", "-m", "training.dvc_store",
                "--source", str(source), "--run-id", run_id,
                "--worker", str(number),
            ],
            cwd=TRAIN_ROOT,
            env={**_local_auth_env(), "PYTHONPATH": str(TRAIN_ROOT / "src")},
            check=True,
        )
    except BaseException as exc:
        _post_training_event(
            run_id,
            "dvc_publish",
            "failed",
            worker=number,
            error=str(exc),
        )
        raise
    _post_training_event(run_id, "dvc_publish", "completed", worker=number)


def publish_local_training_results(remote_base: str, workers: int) -> None:
    """Publish isolated worker bundles concurrently through DVC."""
    from concurrent.futures import ThreadPoolExecutor

    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    publisher_workers = min(workers, _DVC_WORKERS)
    _post_training_event(
        run_id,
        "dvc_publish",
        "dispatching",
        workers=workers,
        publisher_workers=publisher_workers,
        dvc_workers=_DVC_WORKERS,
        dvc_jobs=int(training_cfg().colab.dvc_jobs),
    )
    with ThreadPoolExecutor(max_workers=publisher_workers, thread_name_prefix="dvc-worker") as pool:
        futures = [
            pool.submit(_publish_local_training_worker, run_id, number, workers)
            for number in range(1, workers + 1)
        ]
        for future in futures:
            future.result()
    _post_training_event(run_id, "dvc_publish", "completed", workers=workers)
    print("[dvc-local] local result publication completed", flush=True)


def publish_local_wandb_artifacts(remote_base: str, workers: int) -> None:
    """Attach the final local report bundle to the existing W&B run."""
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    _post_training_event(run_id, "wandb_publish", "started", workers=workers)
    api_key = _env_value("WANDB_API_KEY")
    if not api_key:
        print(
            "[wandb-local] WANDB_API_KEY is absent; skipping W&B artifact "
            "publication. DVC publication and teardown may continue.",
            flush=True,
        )
        _post_training_event(run_id, "wandb_publish", "skipped", reason="missing_api_key")
        return
    import os

    try:
        import wandb
    except Exception:
        print(
            "[wandb-local] W&B import failed; DVC publication and teardown "
            "may continue. Full traceback:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        _post_training_event(run_id, "wandb_publish", "skipped", reason="wandb_import_failed")
        return

    project = str(training_cfg().tracking.wandb.project)
    for number in range(1, workers + 1):
        worker = TRAINING_RESULTS / run_id / f"worker_{number}"
        status_path = worker / "live_status.json"
        try:
            status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.is_file()
                else {}
            )
        except Exception:
            print(
                f"[wandb-local] worker {number}: could not read {status_path}; "
                "skipping this worker. Full traceback:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            continue
        wandb_run_id = status.get("wandb_run_id")
        if not wandb_run_id:
            print(
                f"[wandb-local] worker {number}: no W&B run id in {status_path}; "
                "skipping this worker",
                flush=True,
            )
            _post_training_event(
                run_id,
                "wandb_publish",
                "skipped",
                worker=number,
                reason="missing_wandb_run_id",
            )
            continue
        try:
            os.environ["WANDB_API_KEY"] = api_key
            os.environ["WANDB_DIR"] = str(worker / "wandb")
            run = wandb.init(
                project=project,
                id=str(wandb_run_id),
                resume="allow",
                settings=wandb.Settings(x_disable_stats=True, x_disable_machine_info=True),
            )
            artifact = wandb.Artifact(f"run-{run_id}-downloadable", type="training-result")
            selected = []
            for path in sorted(worker.iterdir()):
                if path.name in {F["canonical_records"].name, F["gate_results"].name, "wandb", "mlruns"}:
                    continue
                if path.is_file() and path.suffix in {
                    ".csv", ".json", ".jsonl", ".log", ".png", ".dvc", ".yaml", ".yml", ".txt",
                }:
                    selected.append(path)
                elif path.is_dir() and (
                    path.name.startswith("report_")
                    or path.name in {"logs", "_checkpoints", ".resume"}
                ):
                    selected.append(path)
            if not selected:
                print(
                    f"[wandb-local] worker {number}: no downloadable files; skipping",
                    flush=True,
                )
                run.finish(exit_code=1)
                _post_training_event(
                    run_id,
                    "wandb_publish",
                    "skipped",
                    worker=number,
                    reason="no_downloadable_files",
                )
                continue
            for path in selected:
                if path.is_dir():
                    artifact.add_dir(str(path), name=path.name)
                else:
                    artifact.add_file(str(path), name=path.name)
            run.log_artifact(artifact)
            run.finish(exit_code=0)
            _post_training_event(
                run_id,
                "wandb_publish",
                "completed",
                worker=number,
                files=len(selected),
                artifact=artifact.name,
            )
            print(f"[wandb-local] worker {number}: final artifact uploaded", flush=True)
        except Exception:
            print(
                f"[wandb-local] worker {number} publication failed; "
                "DVC publication and teardown may continue. Full traceback:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            _post_training_event(
                run_id,
                "wandb_publish",
                "failed",
                worker=number,
                error="worker publication exception",
            )


def finalize_local_training_run(remote_base: str, workers: int) -> None:
    """Complete every local post-run publication before remote teardown.

    A training run is not complete when the GPU workers exit.  It is complete
    only after the raw outputs are downloaded, CPU reports are generated, the
    result bundle is verified/published through DVC, and the same bundle is
    attached to each W&B run.  Keep this as one explicit gate so teardown
    cannot destroy a remote-only result after a partial post-run step.
    """
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    _post_training_event(run_id, "post_training", "started", workers=workers)
    print("[post-training] finalizing downloaded results on local CPU ...", flush=True)
    _post_training_event(run_id, "mask_effect", "started", workers=workers)
    generate_local_mask_effect(remote_base, workers)
    _post_training_event(run_id, "mask_effect", "completed", workers=workers)
    generate_local_training_reports(remote_base, workers)
    publish_local_training_results(remote_base, workers)
    publish_local_wandb_artifacts(remote_base, workers)
    _post_training_event(run_id, "post_training", "completed", workers=workers)
    print("[post-training] DVC and W&B publication verified", flush=True)


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


def ensure_session() -> None:
    """Provision and verify the session before any training stage starts."""
    r = colab("sessions", check=False)
    if SESSION in (r.stdout or ""):
        print(f"[session] '{SESSION}' already active; verifying control channel ...")
        _verify_session_handshake()
        return
    accelerator = [] if GPU.upper() == "CPU" else ["--gpu", GPU]
    print(f"[session] provisioning {SESSION} ({'cpu' if not accelerator else f'gpu={GPU}'}) ...")
    colab("new", "-s", SESSION, *accelerator, timeout=300)
    print("[session] provisioned; running control-channel handshake ...")
    _verify_session_handshake()


def prepare_remote_layout() -> None:
    """Clone/update the configured training branch and create runtime dirs."""
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
    subprocess.run(["git", "pull", "--ff-only", remote_name, {BRANCH!r}], cwd=root, check=True)
else:
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--origin", remote_name, "--depth", "1", "--branch", {BRANCH!r},
        {REPOSITORY!r}, str(root),
    ], check=True)
for path in [root / "artifacts" / "data", root / "artifacts" / "results"]:
    path.mkdir(parents=True, exist_ok=True)
print("[repo] ready", {REPOSITORY!r}, "branch", {BRANCH!r}, "at", root)
"""
    run_colab_exec_stream(SESSION, script, timeout=600, log_name="checkout", retry_safe=True)


def install_deps() -> None:
    print("[deps] installing dependencies on the VM ...")
    # Run pip outside the notebook kernel. A kernel disconnect can interrupt
    # the control channel, but the detached process keeps writing a durable
    # log/status pair that the launcher can retrieve before teardown.
    run_detached_stage(
        "00_deps",
        "[sys.executable, '-m', 'pip', 'install', "
        "'sentence-transformers', 'datasets', 'accelerate', 'evaluate', "
        "'scikit-learn', 'pandas', 'numpy', 'mlflow', 'optuna', "
        "'psycopg[binary]', 'wandb', 'dvc', 'dagshub']",
        timeout=900,
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


def materialize_remote_models(model_keys: list[str]) -> None:
    """Pull and validate only the model bundles required by this lane."""
    keys = sorted(set(model_keys))
    if not keys:
        return
    config = load_config()
    registry = config["models"]
    unknown = sorted(set(keys) - set(registry))
    if unknown:
        raise KeyError(f"unknown local model registry key(s): {unknown}")
    model_roots = [
        str(config["paths"]["models_dir"]),
        str(config["paths"]["models_dir_sibling"]),
    ]
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import subprocess
from pathlib import Path
import yaml
from core.common import load_config
from core.common import resolve_model
root = {REMOTE_ROOT!r}
cfg = load_config()
registry = cfg["models"]
model_roots = [Path(root) / relative for relative in {model_roots!r}]
requested = {keys!r}

def exact_dvc_target(target):
    pointer = Path(str(target) + ".dvc")
    if pointer.is_file():
        return target
    for dvc_yaml in Path(root).rglob("dvc.yaml"):
        document = yaml.safe_load(dvc_yaml.read_text(encoding="utf-8")) or {{}}
        for stage in document.get("stages", {{}}).values():
            for output in stage.get("outs", []):
                raw_path = output.get("path") if isinstance(output, dict) else output
                if raw_path is None:
                    continue
                candidate = (dvc_yaml.parent / str(raw_path)).resolve()
                if candidate == target.resolve():
                    return target
    return None

targets = []
unavailable = []
for key in requested:
    try:
        resolve_model(key)
        print(f"[models] {{key}} already materialized locally", flush=True)
        continue
    except FileNotFoundError:
        pass
    candidates = [root / Path(registry[key]) for root in model_roots]
    addressable = [target for target in candidates if exact_dvc_target(target)]
    if len(addressable) != 1:
        unavailable.append((key, [str(target) for target in candidates]))
        continue
    targets.append(str(addressable[0].relative_to(Path(root))))
if unavailable:
    raise RuntimeError(
        "requested model bundles are not individually DVC-addressable; "
        "refusing to pull the monolithic models output: "
        + repr(unavailable)
    )
if targets:
    print(f"[models] pulling DVC targets: {{targets}}", flush=True)
    pull = subprocess.run(
        ["dvc", "pull", "--jobs", str({int(_COLAB.dvc_jobs)}), *targets],
        cwd=root,
        text=True,
    )
    if pull.returncode:
        raise RuntimeError(f"DVC model materialization failed (rc={{pull.returncode}})")
for key in requested:
    print(f"[models] validating {{key}}", flush=True)
    print(f"[models] {{key}} -> {{resolve_model(key)}}", flush=True)
"""
    run_colab_exec_stream(
        SESSION,
        script,
        timeout=3600,
        log_name="model_materialization",
        retry_safe=True,
    )


def verify_training_inputs() -> None:
    """Use the frozen CSV inputs committed on the training branch."""
    print("[data] validating frozen training CSVs from the cloned branch ...")
    script = _BOOTSTRAP + f"""
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
"""
    run_colab_exec_stream(SESSION, script, timeout=120, log_name="01_data_check", retry_safe=True)


def run_train(
    frac: float, epochs: int, sample: int | None, workers: int = 1,
    *, resume_run: str | None = None, model: str | None = None,
    run_label: str | None = None,
) -> tuple[str, int]:
    """Full-chain GPU training on the VM."""
    print("[run] train.py on the VM (GPU) ...")
    # AUDIT 2026-09-09: --mask-frac 0.15 REMOVED — it hardcoded a value that
    # silently contradicted the SSOT (masking.frac: 1.00 in
    # config/training.yaml). train.py's own default resolves from the config
    # now; the CLI flag remains for explicit overrides.
    args = ["-u", "-m", "training.train",
        "--split", "holdout",
        "--loss", "contrastive",
        "--train-frac", str(frac),
        "--epochs", str(epochs),
        # Reports and plots are CPU-side post-processing. Generate them after
        # the DVC-verified download instead of spending GPU time on them.
        "--no-plot"]
    if model is not None:
        registry = load_config()["models"]
        if model not in registry:
            raise KeyError(
                f"Colab training model must be a local registry key; "
                f"got {model!r}, expected one of {sorted(registry)}"
            )
        # Resolve inside the remote checkout. A local absolute path would
        # not exist on the VM and would bypass the DVC-owned model contract.
        args.extend(["--model", model])
    if sample is not None:
        args.extend(["--sample", str(sample)])
    if not _MASK_EFFECT_AFTER_TRAIN:
        args.append("--no-mask-effect")
    if resume_run:
        args.append("--resume")
    if workers == 1 and resume_run is None:
        return run_single_train_and_stream(args, run_label=run_label)
    return run_parallel_train_and_tail(
        args, workers, resume_run=resume_run,
        run_labels=(
            [item.strip() for item in run_label.split(",") if item.strip()]
            if run_label else None
        ),
    )


def run_single_train_and_stream(
    args: list[str], *, run_label: str | None = None
) -> tuple[str, int]:
    """Run one worker in the Colab exec stream so W&B is visible immediately."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    remote_base = f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import os, pathlib, shutil, subprocess, sys
from core.common import F
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
out = base / "worker_1"
base.mkdir(parents=True, exist_ok=False)
out.mkdir()
for name in (F["canonical_records"], F["gate_results"]):
    source = root / "results" / name.name
    if not source.is_file():
        raise FileNotFoundError(f"worker input missing: {{source}}")
    shutil.copy2(source, out / name.name)
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
    print("[train] single worker completed; local post-processing deferred", flush=True)
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


def run_sims_deberta() -> None:
    """The configured zero-shot model lane on the VM."""
    print(f"[run] zero_shot_sims --models {_SIMS_MODEL} on the VM ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/src/training/zero_shot_sims.py",
                     "--models", {_SIMS_MODEL!r}]).returncode
if rc != 0:
    raise RuntimeError(f"zero-shot similarity subprocess failed (rc={{rc}})")
"""
    run_colab_exec_stream(SESSION, script, timeout=2 * 3600, log_name="sims_deberta")


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
                    choices=["train", "hpo", "sims", "smoke", "stop"],
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

    if args.what == "stop":
        stop()
        return

    start_live_log()
    check_colab_cli()
    local_training_run: tuple[str, int] | None = None
    local_hpo_run: str | None = None
    publication_complete = False

    try:
        ensure_session()
        prepare_remote_layout()
        install_deps()
        if args.what in {"train", "smoke"}:
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
            materialize_remote_models(required_models)
        log_gpu_profile()
        if args.refresh_data:
            run_data_prep()
        else:
            verify_training_inputs()
        # AUDIT FIX 2026-09-08: --what sims used to run FULL TRAINING first
        # (run_train was unconditional) — hours of unintended GPU quota
        # for a lane that only needs the deberta scoring.
        if args.what == "sims":
            run_sims_deberta()
        elif args.what == "smoke":
            local_training_run = run_train(
                args.train_frac, _SMOKE_EPOCHS, sample=_SMOKE_SAMPLE,
                workers=_SMOKE_WORKERS, run_label=args.run_label,
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
            )
        if local_training_run is not None:
            remote_base, workers = local_training_run
            finalize_local_training_run(remote_base, workers)
        if local_hpo_run is not None:
            print("[post-training] publishing HPO snapshots on local CPU ...", flush=True)
            publish_local_hpo_results(local_hpo_run, args.hpo_persistence)
        publication_complete = True
    except BaseException:
        print(
            "[launcher] traceback before teardown; publication is incomplete and "
            "the VM will be kept alive for recovery:",
            flush=True,
        )
        traceback.print_exc()
        raise
    finally:
        # A successful run is released only after every local publication step
        # has completed.  On a failed/cancelled run, preserve the VM so a
        # partial transfer or publication can be recovered instead of deleting
        # the only remaining copy of the results.
        if not args.keep_alive and publication_complete:
            stop()
        elif not args.keep_alive:
            print(
                "[stop] skipped: training/publication did not complete; "
                "remote results remain available for recovery",
                file=sys.stderr,
                flush=True,
            )
        else:
            print("\n[info] --keep-alive specified. VM is still running.")
        close_live_log()

    print("\n[done] artifacts persisted to the configured DVC remote")


if __name__ == "__main__":
    main()
