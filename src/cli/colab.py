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
  smoke  — the chain check (fast verification the remote environment
           reproduces the local results contract). Runs on CPU by default and
           reads the full deduped CSV already in the cloned Colab checkout;
           no CSV or prepared bundle is uploaded. Its config-owned 128-row
           cap is applied only in memory by train.py, never by deleting or
           rewriting source rows.

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
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
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


# Smoke and normal training defaults come from the Colab runtime config.
# Sweep fractions remain exclusive to the sweep lane.
_SMOKE_SAMPLE = int(sweep_cfg()["smoke_sample"])
_TRAIN_FRAC_DEFAULT = float(training_cfg().colab.train_fraction)
_EPOCHS_DEFAULT = int(training_cfg().training.epochs)
_TRAIN_LOSS = str(training_cfg().training.loss)
_RERANK_MODEL = str(sweep_cfg()["rerank_model"])

_COLAB = training_cfg().colab
_SIMS_MODEL = str(_COLAB.sims_model)
REPOSITORY = _COLAB.repository
BRANCH = _COLAB.branch
GIT_REMOTE_NAME = _COLAB.git_remote_name
# Keep the config session as the default, while allowing concurrent launches
# to select an isolated named VM without editing the shared configuration.
SESSION = os.environ.get("EUROMONITOR_COLAB_SESSION", _COLAB.session)
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
# The VM's asserted distributions and installer preference are config-owned
# (config/training.yaml colab.runtime_packages / colab.prefer_uv_install):
# trimming or re-pinning the remote stack must not require editing this file.
_RUNTIME_PACKAGES = _COLAB.runtime_packages
_PREFER_UV_INSTALL = bool(_COLAB.prefer_uv_install)
# Reusing a prepared bundle whose inputs are byte-identical saves the whole
# local build (531 s measured cold).  Config-owned so it can be turned off
# when a lane needs to prove a bundle was built rather than reused.
_CACHE_PREPARED_BUNDLES = bool(_COLAB.cache_prepared_bundles)
_MASKING_ENABLED = training_cfg().masking.enabled
_MASKING_PROFILE = str(training_cfg().masking.profile)
_COLLAPSE_GUARDRAIL_PROFILE = str(training_cfg().collapse_guardrail.profile)
_DVC_ENABLED = bool(_COLAB.dvc_enabled)
_DVC_WORKERS = _COLAB.dvc_workers if _DVC_ENABLED else 0
_DVC_DISABLED_FLAG = "0" if _DVC_ENABLED else "1"
_LOG_POLL_SECONDS = _COLAB.log_poll_seconds
_LOG_POLL_INITIAL_SECONDS = float(_COLAB.log_poll_initial_seconds)
_PROBE_TIMEOUT_SECONDS = _COLAB.probe_timeout_seconds
_PROBE_RETRIES = _COLAB.probe_retries
_PROBE_RETRY_BACKOFF_SECONDS = _COLAB.probe_retry_backoff_seconds
_MASK_EFFECT_AFTER_TRAIN = _COLAB.mask_effect_after_train
_SMOKE_EPOCHS = _COLAB.smoke_epochs
_WORKER_TIMEOUT_SECONDS = _COLAB.worker_timeout_seconds
_RESULT_DOWNLOAD_TIMEOUT_SECONDS = _COLAB.result_download_timeout_seconds
_RESULT_DOWNLOAD_HEARTBEAT_SECONDS = _COLAB.result_download_heartbeat_seconds
_REMOTE_UPLOAD_RETRIES = 3
_RESULT_ARCHIVE_NAME = _COLAB.result_archive_name
_RESULT_MANIFEST_NAME = _COLAB.result_manifest_name
_RESULT_DOWNLOAD_EXCLUDED_DIRS = frozenset(_COLAB.result_download_excluded_dirs)
# Poll interval for the incremental result sync that runs during training.
# Small enough that a finished checkpoint is on the laptop well before the run
# ends, large enough that the remote listing does not compete with the trainer
# for the control channel.
_INCREMENTAL_SYNC_SECONDS = 30
# The trainer writes this file last inside a checkpoint directory, so its
# presence is what distinguishes a finished checkpoint from one mid-write.
_CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
# Checkpoints live at
# ``worker_N/_checkpoints/<model>/<run>_f0/checkpoint-<step>/<file>``, so a
# checkpoint file is four levels below the ``_checkpoints`` root; that is the
# only depth either the step lookup or the file listing needs, because a
# checkpoint can only be identified by listing something inside it.  The walk
# is bounded to stay inside the exec budget however large the run's log and
# wandb trees grow -- an unbounded walk is what timed out on T4.
_CHECKPOINT_LISTING_DEPTH = 4
# The directory holding every checkpoint of a run, directly under a worker.
_CHECKPOINT_ROOT_NAME = "_checkpoints"
# Bookkeeping written beside a locally retained checkpoint, recording which
# remote checkpoint it is and the score that won it the slot.
_LATEST_BEST_MARKER = "latest_best.json"
_WORKER_MONITOR_SECONDS = _COLAB.worker_monitor_seconds
_FINAL_INFERENCE = _COLAB.final_inference
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


# The lane's fixed split, asserted rather than derived: training consumes the
# deduped catalog with the held-out IDs removed, and inference scores precisely
# that held-out population.
_EXPECTED_TRAINING_ROWS = 58_529
_EXPECTED_INFERENCE_ROWS = 3_000
_EXPECTED_SOURCE_ROWS = _EXPECTED_TRAINING_ROWS + _EXPECTED_INFERENCE_ROWS


def training_lifecycle_preflight(
    *, workers: int, model: str | None, masking_profile: str,
    train_only: bool = False,
) -> dict[str, object]:
    """Validate the local train/holdout contract without contacting Colab."""
    import pandas as pd

    training_path = _validation_input_path(_COLAB.training_dataset_csv)
    sample_path = _validation_input_path(_FINAL_INFERENCE.input_csv)
    source_path = _validation_input_path(_FINAL_INFERENCE.source_csv)
    frames = {
        "source": pd.read_csv(source_path, usecols=["product_id"], dtype=str),
        "training": pd.read_csv(training_path, usecols=["product_id"], dtype=str),
        "inference": pd.read_csv(sample_path, usecols=["product_id"], dtype=str),
    }
    ids = {key: set(frame["product_id"]) for key, frame in frames.items()}
    for name, frame in frames.items():
        duplicate_count = int(frame["product_id"].duplicated().sum())
        if duplicate_count:
            raise RuntimeError(
                f"{name} input contains {duplicate_count:,} duplicate product ID row(s)"
            )
    overlap = ids["training"] & ids["inference"]
    if overlap:
        raise RuntimeError(
            f"training/inference overlap contains {len(overlap):,} product IDs"
        )
    reconstructed = ids["training"] | ids["inference"]
    if reconstructed != ids["source"]:
        missing = len(ids["source"] - reconstructed)
        extra = len(reconstructed - ids["source"])
        raise RuntimeError(
            "training remainder plus inference sample does not reconstruct source: "
            f"missing={missing:,} extra={extra:,}"
        )
    if (
        len(frames["training"]) != _EXPECTED_TRAINING_ROWS
        or len(frames["inference"]) != _EXPECTED_INFERENCE_ROWS
        or len(frames["source"]) != _EXPECTED_SOURCE_ROWS
    ):
        raise RuntimeError(
            "unexpected split sizes: "
            f"training={len(frames['training']):,} "
            f"(expected {_EXPECTED_TRAINING_ROWS:,}), "
            f"inference={len(frames['inference']):,} "
            f"(expected {_EXPECTED_INFERENCE_ROWS:,}), "
            f"source={len(frames['source']):,} "
            f"(expected {_EXPECTED_SOURCE_ROWS:,})"
        )
    profiles = _expand_worker_profiles(masking_profile, workers, "masking")
    model_key = model or str(training_cfg().training.base_model)
    return {
        "contacts_colab": False,
        "workers": workers,
        "model": model_key,
        "masking_profiles": profiles,
        "training_dataset": str(training_path),
        "training_rows": len(frames["training"]),
        "inference_dataset": str(sample_path),
        "inference_rows": len(frames["inference"]),
        "source_dataset": str(source_path),
        "source_rows": len(frames["source"]),
        "product_id_overlap": 0,
        "reconstructs_source": True,
        "train_only": train_only,
        "final_inference_enabled": not train_only,
        "prepared_train_argv": [
            sys.executable, "-u", "-m", "training.train", "--model", model_key,
            "--dataset", str(training_path), "--prepare-bundle", "<worker-bundle>",
        ],
        "remote_completion_argv": (
            None if train_only else [
                "<remote-python>", "-m", "training.complete_colab_worker",
                "--validation-input", "<uploaded-dataset_deduped_sample_3000.csv>",
                "--training-input", "<uploaded-dataset_deduped_train_minus_3000.csv>",
            ]
        ),
        "successful_worker_order": (
            ["train", "write_success_status", "download", "teardown"]
            if train_only else
            ["train", "resolve_best_checkpoint", "heldout_sku_inference",
             "dvc_publish", "write_success_status", "download", "teardown"]
        ),
    }


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


def _record_remote_run(remote_base: str, *, workers: int, lane: str) -> None:
    """Persist the remote location before uploads or training begin."""
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    root = TRAINING_RESULTS / run_id
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lane": lane,
        "remote_base": remote_base,
        "run_id": run_id,
        "session": SESSION,
        "workers": workers,
    }
    (root / "remote_run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[run] remote metadata recorded -> {root / 'remote_run.json'}", flush=True)


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


def _colab_launch_lock_path() -> Path:
    lock_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", SESSION)
    return _COLAB_CLI_STATE_DIR / f"launcher-{lock_name}.lock"


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


def acquire_colab_launch_lock():
    """Prevent independent launchers from sharing and tearing down one VM.

    Every lane intentionally uses the configured session name.  Without an
    inter-process lock, a previously interrupted local launcher can keep
    running and execute its ``finally: stop()`` while a later launch is using
    that same session.  The resulting kernel 404 is indistinguishable from a
    Colab-side failure, so refuse the second launch before it touches Colab.
    """
    _COLAB_CLI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _colab_launch_lock_path()
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"a Colab launcher already owns session '{SESSION}'; "
            f"refusing a concurrent lane (lock: {lock_path})"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({
        "pid": os.getpid(),
        "pid_start_ticks": _process_start_ticks(os.getpid()),
        "session": SESSION,
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


_colab_control_lock = threading.RLock()


def _serialize_colab_control(function):
    """Keep one notebook kernel control request in flight per launcher."""
    def wrapped(*args, **kwargs):
        with _colab_control_lock:
            return function(*args, **kwargs)
    return wrapped


@_serialize_colab_control
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


def _upload_with_retries(source: Path, remote: str, *, timeout: int) -> None:
    """Retry transient Colab upload/control-channel failures."""
    for attempt in range(1, _REMOTE_UPLOAD_RETRIES + 1):
        try:
            colab("upload", "-s", SESSION, str(source), remote, timeout=timeout)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == _REMOTE_UPLOAD_RETRIES:
                raise
            delay = min(30, 5 * attempt)
            print(
                f"[upload] retry {attempt}/{_REMOTE_UPLOAD_RETRIES - 1} for {source.name} "
                f"after transient failure; waiting {delay}s",
                flush=True,
            )
            time.sleep(delay)
@_serialize_colab_control
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
        heartbeat_stop = threading.Event()

        def emit_heartbeat() -> None:
            started = time.monotonic()
            while not heartbeat_stop.wait(30):
                print(
                    f"[stream] {log_name or 'remote stage'} still active "
                    f"({time.monotonic() - started:.0f}s elapsed; awaiting remote output)",
                    flush=True,
                )

        heartbeat = threading.Thread(target=emit_heartbeat, daemon=True)
        heartbeat.start()
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


@_serialize_colab_control
def run_colab_exec_capture(
    session: str, script: str, timeout: int, *, training_output: bool = False,
) -> str:
    """Execute a bounded remote probe while retaining stdout for parsing.

    Probes serve the live training log.  They must reveal control-channel
    failures promptly instead of becoming an opaque multi-minute wait.
    """
    def report_probe_progress(message: str) -> None:
        """Make a blocked training-log probe observable in its durable log."""
        print(message, flush=True)
        if training_output:
            _write_training_log(message + "\n")

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
                f"[probe] timeout after {timeout}s on attempt {attempt}/{_PROBE_RETRIES}"
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
        if attempt < _PROBE_RETRIES:
            delay = _PROBE_RETRY_BACKOFF_SECONDS * attempt
            time.sleep(delay)
    raise RuntimeError(f"remote log probe failed after {_PROBE_RETRIES} attempts: {last_error}")


def _parse_remote_json(output: str) -> dict:
    """Read the last JSON object from a Colab probe without trusting banners."""
    clean_output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    for line in reversed(clean_output.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"remote log probe returned no JSON: {clean_output[-1000:]}")


def run_detached_stage(stage: str, command_expr: list[str], timeout: int) -> None:
    """Run a VM stage outside the notebook kernel and stream its durable log."""
    # Two launches can occur within the same UTC second (especially after a
    # failed preflight). Microseconds keep the remote result root unique and
    # prevent FileExistsError from aborting before workers launch.
    stamp = datetime.now(timezone.utc).strftime("%m%dT%H%M%S%fZ")
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
        delay = _LOG_POLL_INITIAL_SECONDS
        probes = 0
        poll_started = time.perf_counter()
        poll_seconds = 0.0
        while True:
            probes += 1
            poll_seconds = time.perf_counter() - poll_started
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
            delay = min(_LOG_POLL_SECONDS, max(_LOG_POLL_INITIAL_SECONDS, delay * 2))
            time.sleep(delay)
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
    stamp = datetime.now(timezone.utc).strftime("%m%dT%H%M%S%fZ")
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
    worker_losses: list[str] | None = None,
    masking_profiles: list[str] | None = None,
    collapse_guardrail_profiles: list[str] | None = None,
    prepared_bundles: list[Path] | None = None,
    final_inference: bool = True,
    inference_sample: int | None = None,
    inference_device: str | None = None,
    remote_checkout_inputs: bool = False,
    remote_checkout_bundles: list[str] | None = None,
    remote_validation_csv: str | None = None,
    incremental_sync: bool = True,
) -> tuple[str, int]:
    """Run isolated full-data trainers concurrently and mirror worker logs."""
    if worker_losses is not None and len(worker_losses) != workers:
        raise ValueError(
            f"worker loss list must contain exactly {workers} values; "
            f"got {len(worker_losses)}"
        )
    stamp = _lane_run_stamp()
    remote_base = (
        f"{REMOTE_ROOT}/results/concurrent_train_{resume_run}"
        if resume_run
        else f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    )
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    _record_remote_run(remote_base, workers=workers, lane="train")
    if prepared_bundles is not None and remote_checkout_bundles is not None:
        raise ValueError("prepared bundles must be uploaded or checkout-native, not both")
    if remote_checkout_bundles is not None:
        if len(remote_checkout_bundles) != workers:
            raise ValueError("checkout bundle list must contain one bundle per worker")
        remote_bundles = []
        for raw in remote_checkout_bundles:
            relative = Path(raw)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("checkout bundle path must stay inside the checkout")
            remote_bundles.append(str(Path(REMOTE_ROOT) / relative))
    elif prepared_bundles is not None:
        remote_bundles = _upload_prepared_bundles(run_id=run_id, bundles=prepared_bundles)
    else:
        remote_bundles = None
    if not final_inference:
        remote_validation_inputs = {"sample": "", "source": "", "training": ""}
    elif remote_validation_csv is not None:
        remote_validation_inputs = {
            "sample": remote_validation_csv,
            "source": remote_validation_csv,
            "training": remote_validation_csv,
        }
    else:
        remote_validation_inputs = _upload_validation_inputs(run_id)
    remote_input_loop = (
        "for name in ():"
        if prepared_bundles is not None or remote_checkout_inputs
        else 'for name in (F["canonical_records"], F["gate_results"], F["labeled_pairs"]):'
    )
    resume_pointers = _resume_pointer_payload(run_id, workers) if resume_run else {}
    launch = _BOOTSTRAP + _remote_auth_env_script(
        # train_prepared is deliberately remote-only and refuses to run
        # without W&B.  A Git-shipped bundle changes transport, not tracking.
        include_wandb=remote_bundles is not None or not remote_checkout_inputs
    ) + f"""
import base64, json, os, pathlib, shutil, shlex, subprocess, sys, time, traceback
from core.common import F
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
run_id = base.name.removeprefix("concurrent_train_")
base.mkdir(parents=True, exist_ok={bool(resume_run)!r})
resume_pointers = {resume_pointers!r}
run_labels = {run_labels!r}
worker_losses = {worker_losses!r}
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
        if {remote_bundles is None and not remote_checkout_inputs!r}:
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
    if worker_losses is not None:
        loss_index = worker_args.index("--loss") + 1
        worker_args[loss_index] = worker_losses[number - 1]
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
    completion_args = [
        sys.executable, "-m", "training.complete_colab_worker",
        "--source", str(out), "--run-id", run_id, "--worker", str(number),
        "--validation-input", {remote_validation_inputs['sample']!r},
        "--validation-source", {remote_validation_inputs['source']!r},
        "--training-input", {remote_validation_inputs['training']!r},
    ]
    if {inference_sample!r} is not None:
        completion_args.extend(["--sample", str({inference_sample!r})])
    if {inference_device is not None!r}:
        completion_args.extend(["--device", {inference_device!r}])
    if not { _DVC_ENABLED!r}:
        completion_args.append("--skip-dvc")
    completion_command = " ".join(shlex.quote(part) for part in completion_args)
    completion_clause = (
        f'if [ "$rc" -eq 0 ]; then echo "[worker-process] training complete; running validation inference and final DVC publication"; {{completion_command}}; rc=$?; fi; '
        if {final_inference!r} else ""
    )
    log_path, status_path = out / "training.log", out / "training.status"
    live_status_path = out / "live_status.json"
    wandb_dir = out / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root / "src"), "EUROMONITOR_RESULTS_DIR": str(out),
           "EUROMONITOR_MLRUNS_DIR": str(out / "mlruns"), "WANDB_DIR": str(wandb_dir),
           "WANDB_RUN_NAME": training_name,
           "EUROMONITOR_RUN_ID": training_name,
           "EUROMONITOR_MINING_PROFILE": worker_profile,
           "EUROMONITOR_REMOTE_TRAINING": "1", "EUROMONITOR_DISABLE_DVC_CHECKPOINTS": {_DVC_DISABLED_FLAG!r}}}
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
        f"{{completion_clause}}"
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
    # Fetch finished artifacts from every worker while they train, so the end
    # of the run is a short delta rather than the whole result set.  Stopped
    # before the authoritative download so the two cannot race on one file.
    syncer = (
        _IncrementalResultSync(remote_base, run_id, workers=workers)
        if incremental_sync else None
    )
    if syncer is not None:
        syncer.start()
    offsets = {str(item["worker"]): 0 for item in launched["workers"]}
    live_signatures: dict[str, str] = {}
    try:
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
            try:
                payload = _parse_remote_json(
                    run_colab_exec_capture(
                        SESSION, probe, timeout=_PROBE_TIMEOUT_SECONDS,
                        training_output=True,
                    )
                )
            except RuntimeError as exc:
                # The trainer is detached and continues writing remotely.  A
                # transient empty/control-channel reply must not turn a log
                # read into a training failure followed by VM teardown.
                # A lost kernel or missing session is not transient: there
                # can be no remote worker left to poll.  Propagate it so
                # main's finally tears down local state and releases the
                # session lock for the next launch.
                detail = str(exc).lower()
                if (
                    "connection was lost" in detail
                    or f"session '{SESSION}' not found".lower() in detail
                ):
                    raise
                message = f"[probe] log/status unavailable; continuing worker: {exc}"
                _write_training_log(message + "\n")
                print(message, flush=True)
                time.sleep(_LOG_POLL_SECONDS)
                continue
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
    finally:
        if syncer is not None:
            syncer.stop()


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

# ``_checkpoints`` is excluded because a run writes one large checkpoint per
# evaluation, but the checkpoint the trainer selected must still reach the
# laptop.  trainer_state.json records best_model_checkpoint and best_metric --
# the same choice training restores at the end -- so keep that one directory
# per worker and drop the rest.
def _best_checkpoint(worker_root):
    best = None
    pattern = "_checkpoints/**/checkpoint-*/trainer_state.json"
    for state_path in sorted(worker_root.glob(pattern)):
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            continue
        recorded = state.get("best_model_checkpoint")
        if not recorded:
            continue
        selected = state_path.parent.parent / pathlib.Path(str(recorded)).name
        if not selected.is_dir():
            continue
        metric = state.get("best_metric")
        rank = (
            float(metric) if metric is not None else float("-inf"),
            int(state.get("global_step") or 0),
        )
        if best is None or rank > best[0]:
            best = (rank, selected)
    return best[1] if best is not None else None

for worker in range(1, {workers + 1}):
    worker_root = base / f"worker_{{worker}}"
    if not worker_root.is_dir():
        raise FileNotFoundError(f"missing remote worker directory: {{worker_root}}")
    best_checkpoint = _best_checkpoint(worker_root)
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
        if best_checkpoint is not None:
            try:
                path.relative_to(best_checkpoint)
            except ValueError:
                pass
            else:
                included.append({{
                    "worker": worker,
                    "path": relative.as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }})
                continue
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


class _IncrementalResultSync:
    """Keep the newest *best* checkpoint on the laptop while training runs.

    The end-of-run download is one archive transferred after the whole
    lifecycle (train -> inference -> DVC publish) has collapsed to a single
    moment, so the multi-GB checkpoint set is dead time at the end.  Training
    writes a new ``checkpoint-<step>`` directory steadily, and each directory
    is immutable once written, so the best one can be held locally as it
    appears.

    Retention is *latest-if-better, overwrite*: each worker keeps exactly one
    local directory, ``latest_best``, and a remote checkpoint replaces it only
    when its score strictly beats the score already held.  "Best" is dev
    average precision when the run reports it, else lowest train loss, both
    read from the ``live_status.json`` the trainer already publishes.

    This is a pure latency optimisation and never a correctness dependency:

    * a checkpoint is copied only once its ``checkpoint_manifest.json`` exists,
      which the trainer writes last, so a directory still being written is
      never captured half-finished;
    * everything it fetches is re-fetched and hash-verified by the
      authoritative :func:`download_verified_training_results` pass, which
      remains the only source of truth for a complete run;
    * every failure is swallowed and logged; a flaky sync must not fail a run
      whose data still arrives by the normal path.
    """

    def __init__(self, remote_base: str, run_id: str, *, workers: int) -> None:
        self.remote_base = remote_base
        self.run_id = run_id
        self.workers = workers
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # worker -> (score, remote checkpoint dir) currently held locally.
        self._held: dict[int, tuple[float, str]] = {}
        self._synced_bytes = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name=f"result-sync-{self.run_id}", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Long enough for an in-flight checkpoint to land, short enough that
            # a stuck sync cannot delay teardown: whatever it misses is still
            # covered by the authoritative download.
            self._thread.join(timeout=120)

    def synced_bytes(self) -> int:
        return self._synced_bytes

    def _loop(self) -> None:
        while not self._stop.wait(_INCREMENTAL_SYNC_SECONDS):
            try:
                self._pass()
            except BaseException as exc:  # never fail the run from the syncer
                print(
                    f"[result-sync] pass failed ({type(exc).__name__}: {exc}); "
                    "final download still covers every file",
                    flush=True,
                )

    def _pass(self) -> None:
        for worker in range(1, self.workers + 1):
            try:
                self._sync_worker(worker)
            except BaseException as exc:
                # A listing failure, a timeout, or a download error all land
                # here: one pass must never escape into the run.
                print(
                    f"[result-sync] worker {worker} pass skipped "
                    f"({type(exc).__name__}: {exc}); final download covers it",
                    flush=True,
                )

    def _sync_worker(self, worker: int) -> None:
        """Copy this worker's best-so-far checkpoint, if it beats the local one.

        The trainer's ``live_status.json`` heartbeat names the checkpoint it
        just finished, so one small read replaces walking the run's tree.  That
        walk is what timed out on the T4 lane: listing the whole checkpoint,
        log, and wandb tree on a 30 s cadence cannot finish inside the exec
        budget while training is competing for the same control channel.
        """
        heartbeat = self._heartbeat(worker)
        if heartbeat is None:
            return
        step, score = heartbeat
        directory = self._checkpoint_directory(worker, step)
        if directory is None:
            return
        if worker in self._held:
            cached_score, cached_dir = self._held[worker]
        else:
            cached_score, cached_dir = self._best_local(worker)
            self._held[worker] = (cached_score, cached_dir)
        if directory == cached_dir or score <= cached_score:
            return
        files = _list_remote(directory, max_depth=_CHECKPOINT_LISTING_DEPTH)
        if not files:
            return
        if not any(
            Path(name).name == _CHECKPOINT_MANIFEST_NAME for name in files
        ):
            # The manifest is written last, so its absence means this
            # checkpoint is still being written and must not be copied yet.
            return
        destination = (
            TRAINING_RESULTS / self.run_id / f"worker_{worker}" / "latest_best"
        )
        staging = destination.with_name("latest_best.staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        transferred = 0
        checkpoint_name = Path(directory).name
        # Path of the checkpoint relative to the worker root, so the marker
        # records where it actually lives rather than just its leaf name.
        checkpoint_rel = (
            Path(directory)
            .relative_to(Path(f"{self.remote_base}/worker_{worker}"))
            .as_posix()
        )
        for name in sorted(files):
            target = staging / checkpoint_name / Path(name).relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            _download_one_remote_file(name, target)
            transferred += target.stat().st_size
        (staging / _LATEST_BEST_MARKER).write_text(
            json.dumps(
                {
                    "checkpoint": directory,
                    "relative": checkpoint_rel,
                    "score": score,
                    "step": step,
                    "worker": worker,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        # Swap whole directories so the laptop never holds a half checkpoint.
        superseded = destination.with_name(f"latest_best.superseded.{os.getpid()}")
        if destination.exists():
            os.replace(destination, superseded)
        os.replace(staging, destination)
        shutil.rmtree(superseded, ignore_errors=True)
        self._held[worker] = (score, directory)
        self._synced_bytes += transferred
        print(
            f"[result-sync] worker {worker} kept {checkpoint_name} "
            f"(step {step}, score={score:.4f}, "
            f"{_format_bytes(transferred)}) as latest_best",
            flush=True,
        )

    def _heartbeat(self, worker: int) -> tuple[int, float] | None:
        """The step and score the trainer last reported, if it reported one."""
        remote = f"{self.remote_base}/worker_{worker}/live_status.json"
        try:
            payload = json.loads(_read_remote_text(remote))
        except BaseException:
            return None
        step = payload.get("step")
        if step is None:
            return None
        average_precision = payload.get("dev_average_precision")
        if average_precision is not None:
            return (int(step), float(average_precision))
        loss = payload.get("train_loss")
        if loss is not None:
            return (int(step), -float(loss))
        return None

    def _checkpoint_directory(self, worker: int, step: int) -> str | None:
        """Locate ``checkpoint-<step>`` without walking the whole run tree.

        The trainer nests it under model and run folders whose names are not
        known here, so the two levels above it are matched by glob.  This is a
        bounded probe of one small subtree, not a walk of the whole run: only
        ``_checkpoints/<model>/<run>_f0`` is examined.
        """
        root = f"{self.remote_base}/worker_{worker}/{_CHECKPOINT_ROOT_NAME}"
        try:
            matches = _list_remote(root, max_depth=_CHECKPOINT_LISTING_DEPTH)
        except BaseException:
            return None
        wanted = f"checkpoint-{step}"
        # The listing reports files, so the checkpoint is the directory
        # *containing* them: matching on the file's own name would only ever
        # compare against names like checkpoint_manifest.json and find nothing.
        for name in matches:
            candidate = Path(name)
            if candidate.parent.name == wanted:
                return str(candidate.parent)
        return None

    def _best_local(self, worker: int) -> tuple[float, str | None]:
        """Score and checkpoint name already held locally, if any."""
        pointer = TRAINING_RESULTS / self.run_id / f"worker_{worker}" / "latest_best"
        marker = pointer / _LATEST_BEST_MARKER
        if not marker.is_file():
            return (float("-inf"), None)
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return (float(payload["score"]), str(payload["checkpoint"]))
        except (OSError, ValueError, KeyError, TypeError):
            return (float("-inf"), None)


def _download_one_remote_file(remote: str, local: Path) -> None:
    """Fetch one remote file through the module's Colab CLI wrapper.

    Indirection that keeps the incremental syncer patchable: inside this module
    the name ``colab`` is both the module and the CLI wrapper function, so a
    test cannot patch the wrapper without shadowing the module.
    """
    colab(
        "download", "-s", SESSION, remote, str(local),
        timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS,
    )


def _read_remote_text(remote: str) -> str:
    """Read one small remote file through the shared stdin-exec channel."""
    script = (
        "import base64, pathlib\n"
        f"p = pathlib.Path({remote!r})\n"
        "print('@@TEXT@@' + base64.b64encode("
        "p.read_bytes() if p.is_file() else b'').decode('ascii'))\n"
    )
    proc = subprocess.Popen(
        _colab_command("exec", "-s", SESSION, "--timeout", "60"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate(script, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"remote read failed: {err[-200:]}")
    for line in out.splitlines():
        if line.startswith("@@TEXT@@"):
            payload = line[len("@@TEXT@@"):]
            if not payload:
                raise RuntimeError(f"remote file is missing: {remote}")
            return base64.b64decode(payload).decode("utf-8")
    raise RuntimeError("remote read returned no marker")


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
    cached = state.get(SESSION)
    keep_alive_pid = cached.get("keep_alive_pid") if isinstance(cached, dict) else None
    if isinstance(keep_alive_pid, int) and keep_alive_pid != os.getpid():
        proc_cmdline = Path(f"/proc/{keep_alive_pid}/cmdline")
        try:
            command = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            command = ""
        if _is_keep_alive_daemon(command):
            try:
                os.kill(keep_alive_pid, 15)
                print(f"[session] stopped stale local keep-alive pid={keep_alive_pid}", flush=True)
            except ProcessLookupError:
                pass
    state.pop(SESSION, None)
    temporary = _COLAB_CLI_CONFIG.with_name(_COLAB_CLI_CONFIG.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, _COLAB_CLI_CONFIG)
    print(f"[session] removed stale cached record for '{SESSION}'", flush=True)


def _is_keep_alive_daemon(command: str) -> bool:
    """Whether a /proc command line is this wrapper's keep-alive daemon."""
    return (
        _COLAB_CLI_ENTRYPOINT.name in command
        and "keep-alive" in command
    )


def keep_alive_daemon_pids() -> list[int]:
    """PIDs of keep-alive daemons serving THIS session.

    The CLI records the pid of the daemon it spawned, but a backstop must not
    depend on the CLI's bookkeeping being present or correct, so the process
    table is the source of truth and the recorded pid is only a hint.
    """
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (
                (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            )
        except OSError:
            continue
        if _is_keep_alive_daemon(command) and SESSION in command:
            found.append(int(entry.name))
    return sorted(found)


def stop_keep_alive_daemon(*, reason: str) -> int:
    """Stop this session's keep-alive daemon and report how many were stopped.

    The daemon is what provisioning needs, so it is always allowed to start.
    On a lane that must never be retained it is stopped as soon as the launcher
    owns the run: the launcher's own teardown remains the primary release, and
    this is the backstop that keeps a crash from leaving the VM held open by
    its own daemon.  A daemon that cannot be found is reported loudly rather
    than passed over in silence.
    """
    pids = keep_alive_daemon_pids()
    if not pids:
        print(
            f"[session] no keep-alive daemon found for '{SESSION}' ({reason}); "
            "the VM is released by the launcher's own teardown",
            flush=True,
        )
        return 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError as exc:
            print(
                f"[session] could not stop keep-alive pid={pid} ({exc!r}); "
                "the VM is released by the launcher's own teardown",
                flush=True,
            )
            continue
        print(f"[session] stopped keep-alive daemon pid={pid} ({reason})", flush=True)
    return len(pids)


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


def _runtime_install_command(
    packages: list[str], *, prefer_uv: bool, wheel_paths: list[str],
) -> str:
    """Build the remote command that installs one lane's runtime packages.

    ``uv`` resolves and downloads the same wheels several times faster than
    pip, and the installed Colab CLI already prefers it for its own
    ``colab install`` subcommand.  ``--python sys.executable`` targets exactly
    the interpreter that will import these packages, so the fast path cannot
    land them in a different environment than the pip fallback does.

    A configured prebuilt wheel replaces its distribution and is used only when
    its ABI/platform tag matches the interpreter actually running — a compiled
    extension from another Python or architecture is worthless, and a silent
    mismatch would install nothing while looking successful.

    Which installer ran, which prebuilt wheel was used or rejected, and any
    downgrade to pip are all printed, so the durable stage log never hides the
    slow path (no silent fallbacks).
    """
    program = f"""\
import pathlib, shutil, subprocess, sys, sysconfig

packages = {packages!r}
root = pathlib.Path({REMOTE_ROOT!r})
tag = "cp{{}}{{}}".format(*sys.version_info[:2])
platform = sysconfig.get_platform().replace("-", "_")
requirements = []
provided = set()
for relative in {wheel_paths!r}:
    wheel = root / relative
    filename = wheel.name
    distribution = filename.split("-")[0].replace("_", "-").lower()
    if not wheel.is_file():
        print(f"[deps] prebuilt wheel missing: {{wheel}}", flush=True)
        continue
    if tag not in filename or platform not in filename:
        print(
            f"[deps] prebuilt wheel {{filename}} does not match {{tag}}/{{platform}}; "
            "building from the index instead",
            flush=True,
        )
        continue
    requirements.append(str(wheel))
    provided.add(distribution)
    print(f"[deps] prebuilt wheel={{wheel}} replaces {{distribution}}", flush=True)
packages = [
    package for package in packages
    if package.split("==")[0].split("[")[0].replace("_", "-").lower() not in provided
]
install = requirements + packages
uv = shutil.which("uv") if {prefer_uv!r} else None
if uv:
    command = [uv, "pip", "install", "--python", sys.executable, *install]
    print("[deps] installer=uv", " ".join(command), flush=True)
    if subprocess.call(command) == 0:
        raise SystemExit(0)
    print("[deps] uv install failed; falling back to pip", flush=True)
elif {prefer_uv!r}:
    print("[deps] uv is absent on the VM; falling back to pip", flush=True)
else:
    print("[deps] uv disabled by configuration; using pip", flush=True)
command = [sys.executable, "-m", "pip", "install", *install]
print("[deps] installer=pip", " ".join(command), flush=True)
raise SystemExit(subprocess.call(command))
"""
    return f"[sys.executable, '-c', {program!r}]"


def install_deps(*, minimal_runtime: bool = False) -> None:
    packages = list(
        _RUNTIME_PACKAGES.prepared if minimal_runtime else _RUNTIME_PACKAGES.full
    )
    print(
        "[deps] installing "
        + ("prepared training runtime" if minimal_runtime else "full lane dependencies")
        + f" on the VM ({len(packages)} distributions: {', '.join(packages)}) ...",
        flush=True,
    )
    # Run the installer outside the notebook kernel. A kernel disconnect can
    # interrupt the control channel, but the detached process keeps writing a
    # durable log/status pair that the launcher can retrieve before teardown.
    run_detached_stage(
        "00_deps",
        _runtime_install_command(
            packages,
            prefer_uv=_PREFER_UV_INSTALL,
            wheel_paths=list(_RUNTIME_PACKAGES.prebuilt_wheels),
        ),
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


def _remote_auth_env_script(
    *, include_optuna: bool = False, include_wandb: bool = True,
) -> str:
    """Credential exports used by remote subprocess launch cells only."""
    wandb = _wandb_env_script() if include_wandb else ""
    if not _DVC_ENABLED:
        return wandb + (_optuna_env_script() if include_optuna else "")
    key = _env_value("DVC_API_KEY")
    if key:
        print("[dvc] DVC_API_KEY loaded from local .env and injected into VM process")
        dvc = f"os.environ['DVC_API_KEY'] = {key!r}\nos.environ['DAGSHUB_USER_TOKEN'] = {key!r}\n"
    else:
        print("[dvc] DVC_API_KEY absent from .env; durable DVC upload will fail")
        dvc = ""
    optuna = _optuna_env_script() if include_optuna else ""
    return wandb + optuna + dvc


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


def run_train(
    frac: float, epochs: int, sample: int | None, workers: int = 1,
    *, resume_run: str | None = None, model: str | None = None,
    dataset_csv: str | None = None,
    inference_sample: int | None = None,
    inference_device: str | None = None,
    run_label: str | None = None, masking_profile: str | None = None,
    collapse_guardrail_profile: str | None = None,
    loss: str = _TRAIN_LOSS,
    worker_losses: list[str] | None = None,
    train_only: bool = False,
    remote_dataset_csv: str | None = None,
    remote_prepared_bundles: list[str] | None = None,
    remote_validation_csv: str | None = None,
    incremental_sync: bool = True,
) -> tuple[str, int]:
    """Full-chain GPU training on the VM."""
    print("[run] train.py on the configured VM runtime ...")
    # AUDIT 2026-09-09: --mask-frac 0.15 REMOVED — it hardcoded a value that
    # silently contradicted the SSOT (masking.frac: 1.00 in
    # config/training.yaml). train.py's own default resolves from the config
    # now; the CLI flag remains for explicit overrides.
    args = ["-u", "-m", "training.train",
        "--split", "holdout",
        "--loss", loss,
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
    # `training.train_prepared` receives the frozen dataframe inside its
    # bundle.  Keep the checkout dataset available for final inference, but
    # do not pass raw-trainer-only `--dataset` to that entrypoint.
    if remote_dataset_csv is not None and remote_prepared_bundles is None:
        remote_dataset = Path(remote_dataset_csv)
        if remote_dataset.is_absolute() or ".." in remote_dataset.parts:
            raise ValueError("remote dataset path must stay inside the checkout")
        args.extend(["--dataset", str(Path(REMOTE_ROOT) / remote_dataset)])
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
    profiles = _training_bundle_profiles(masking_profile, workers)
    prepared_bundles: list[Path] | None = None
    if remote_dataset_csv is None and remote_prepared_bundles is None:
        bundle_request = {
            "profiles": profiles,
            "model": model,
            "sample": sample,
        }
        if dataset_csv is not None:
            bundle_request["dataset_csv"] = dataset_csv
        prepared_bundles = _prepare_local_training_bundles(**bundle_request)
    if workers == 1 and resume_run is None:
        if worker_losses is not None:
            raise ValueError("worker_losses requires at least two concurrent workers")
        if remote_prepared_bundles is not None and len(remote_prepared_bundles) != 1:
            raise ValueError("a single-worker run requires exactly one checkout bundle")
        return run_single_train_and_stream(
            args,
            run_label=run_label,
            prepared_bundle=prepared_bundles[0] if prepared_bundles else None,
            remote_checkout_bundle=(
                remote_prepared_bundles[0] if remote_prepared_bundles else None
            ),
            final_inference=not train_only,
            inference_sample=inference_sample,
            inference_device=inference_device,
            # A remote dataset already shares the checkout's training_data
            # tree; the legacy worker-input copy assumes every input is under
            # RESULTS and is not applicable to this path.
            copy_remote_inputs=remote_dataset_csv is None,
            remote_validation_csv=remote_validation_csv,
            # Calibration pairs are now a versioned checkout input.
            prepare_remote_labeled_pairs=False,
            # Checkout-native prepared bundles still use the configured W&B
            # mirror; the remote dataset flag must not disable credentials.
            include_wandb=True,
            incremental_sync=incremental_sync,
        )
    return run_parallel_train_and_tail(
        args, workers, resume_run=resume_run,
        run_labels=(
            _expand_worker_profiles(run_label, workers, "run label")
            if run_label else None
        ),
        worker_losses=worker_losses,
        masking_profiles=profiles,
        collapse_guardrail_profiles=_expand_worker_profiles(
            collapse_guardrail_profile or _COLLAPSE_GUARDRAIL_PROFILE,
            workers,
            "collapse guardrail",
        ),
        prepared_bundles=prepared_bundles,
        final_inference=not train_only,
        inference_sample=inference_sample,
        inference_device=inference_device,
        remote_checkout_inputs=remote_dataset_csv is not None,
        remote_checkout_bundles=remote_prepared_bundles,
        remote_validation_csv=remote_validation_csv,
        incremental_sync=incremental_sync,
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


def _training_bundle_profiles(masking_profile: str | None, workers: int) -> list[str]:
    """Resolve the masking profiles one lane's prepared bundles are built for.

    Single definition shared by `run_train` (which builds the bundles) and
    `main` (which may start that build early), so a prewarm can never be built
    for a different request than the run asks for.
    """
    return _expand_worker_profiles(
        masking_profile or _MASKING_PROFILE, workers, "masking"
    )


def _bundle_request_key(request: dict) -> tuple:
    """Identity of one local bundle request, for prewarm reuse checks."""
    return (
        tuple(request["profiles"]),
        request["model"],
        request["sample"],
        request.get("dataset_csv"),
        request.get("payload", "full"),
    )


class _BundlePrewarm:
    """One local bundle build running while the VM provisions and installs.

    The build is pure local CPU work over immutable local inputs; nothing on
    the VM feeds it, so it can overlap the remote deps stage instead of
    following it.  Measured on the 2026-09-15 T4 launch: 33.3 s of local
    bundle construction sat strictly after a 62.7 s remote install.
    """

    def __init__(self, request: dict) -> None:
        self.request = request
        self.key = _bundle_request_key(request)
        self.bundles: list[Path] | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._build, daemon=True)

    def _build(self) -> None:
        try:
            self.bundles = _build_local_training_bundles(**self.request)
        except BaseException as exc:  # re-raised in the owning run_train call
            self.error = exc
            # Also print: a prewarm abandoned by a mismatched request would
            # otherwise fail invisibly (no silent drops).
            print(f"[local-prepare] concurrent build failed: {exc!r}", flush=True)

    def join(self) -> list[Path]:
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.bundles is None:
            raise RuntimeError("the concurrent bundle build returned no bundles")
        return self.bundles


_BUNDLE_PREWARM: _BundlePrewarm | None = None


def start_local_bundle_prewarm(**request) -> None:
    """Build this lane's prepared bundles while the VM installs its runtime."""
    global _BUNDLE_PREWARM
    prewarm = _BundlePrewarm(request)
    _BUNDLE_PREWARM = prewarm
    prewarm.thread.start()
    print(
        "[local-prepare] building "
        f"{len(request['profiles'])} bundle(s) concurrently with the VM "
        "dependency install",
        flush=True,
    )


def drain_local_bundle_prewarm() -> None:
    """Never leave a prewarm thread writing into a closing live log."""
    global _BUNDLE_PREWARM
    prewarm, _BUNDLE_PREWARM = _BUNDLE_PREWARM, None
    if prewarm is not None and prewarm.thread.is_alive():
        print("[local-prepare] waiting for the concurrent build to finish ...", flush=True)
        prewarm.thread.join()


def _take_prewarmed_bundles(**request) -> list[Path] | None:
    """Hand over the in-flight build when it matches this exact request."""
    global _BUNDLE_PREWARM
    prewarm, _BUNDLE_PREWARM = _BUNDLE_PREWARM, None
    if prewarm is None:
        return None
    if prewarm.key != _bundle_request_key(request):
        print(
            "[local-prepare] concurrent build was started for a different "
            f"request {prewarm.key}; rebuilding for {_bundle_request_key(request)}",
            flush=True,
        )
        return None
    print("[local-prepare] joining the build started before the VM setup", flush=True)
    return prewarm.join()


def _lane_bundle_request(args: argparse.Namespace) -> dict | None:
    """The exact local bundle request a lane's `run_train` call will make.

    Mirrors the per-lane worker/sample arguments in `main`; the profile
    derivation itself comes from `_training_bundle_profiles`, the same
    function `run_train` uses.  Returns None for lanes that prepare no
    bundles (sims, mixed, hpo, stop).
    """
    if args.what == "dual-train":
        workers, sample = 2, args.sample
        dataset_csv = None
    elif args.what == "train":
        if (
            args.workers == 1
            and args.model is None
            and args.sample is None
            and args.resume_run is None
        ):
            return None
        workers, sample = args.workers, args.sample
        dataset_csv = None
    else:
        return None
    request = {
        "profiles": _training_bundle_profiles(args.masking_profile, workers),
        "model": args.model,
        "sample": sample,
    }
    if dataset_csv is not None:
        request["dataset_csv"] = dataset_csv
    return request


_BUNDLE_CACHE_DIRNAME = "_cache"
# Every file whose content can change what a prepared bundle contains.  A miss
# on any of them must invalidate the cache: a stale bundle would train on data
# the operator did not ask for, which is the silent-staleness defect class this
# repository treats as a bug rather than an inconvenience.
_BUNDLE_SOURCE_DIRS = ("src/core", "src/training")
_BUNDLE_SOURCE_FILES = ("src/pipeline.py",)


def _tree_digest() -> str:
    """Digest every config file and bundle-producing source file."""
    digest = hashlib.sha256()
    paths = sorted(
        [path for name in _BUNDLE_SOURCE_DIRS for path in (TRAIN_ROOT / name).glob("*.py")]
        + [TRAIN_ROOT / name for name in _BUNDLE_SOURCE_FILES]
        + sorted((TRAIN_ROOT / "config").glob("*"))
    )
    for path in paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(TRAIN_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _bundle_cache_dir(
    *,
    profiles: list[str],
    model_key: str,
    sample: int | None,
    payload: str,
    training_dataset: Path,
) -> Path | None:
    """Content address for one bundle request, or None when caching is off."""
    if not _CACHE_PREPARED_BUNDLES:
        return None
    request = json.dumps(
        {
            "profiles": list(profiles),
            "model": model_key,
            "sample": sample,
            "payload": payload,
            "collapsed_guardrail": _COLLAPSE_GUARDRAIL_PROFILE,
            "dataset_sha256": sha256_file(training_dataset),
            "sources_sha256": _tree_digest(),
        },
        sort_keys=True,
    )
    key = hashlib.sha256(request.encode("utf-8")).hexdigest()[:32]
    return RESULTS / "prepared_training" / _BUNDLE_CACHE_DIRNAME / key


def _bundle_manifest(bundle: Path):
    from training.prepared_bundle import load_prepared_bundle

    return load_prepared_bundle(bundle)[0]


def _cached_bundles(cache_dir: Path, *, profiles: list[str]) -> list[Path] | None:
    """The cached bundles for this request, when every worker's pair is intact.

    `load_prepared_bundle` is the validation: it re-checks the manifest against
    the current encoder-text contract and refuses a mismatch, so a cached
    bundle cannot outlive the model-input spec even if the digest missed it.
    """
    expected = [
        cache_dir / f"worker_{number}_{profile}.pkl.gz"
        for number, profile in enumerate(profiles, start=1)
    ]
    for bundle in expected:
        if not bundle.is_file() or not bundle.with_suffix(bundle.suffix + ".json").is_file():
            return None
        try:
            _bundle_manifest(bundle)
        except Exception as exc:
            print(
                f"[local-prepare] cached bundle {bundle} is not reusable "
                f"({exc!r}); rebuilding",
                flush=True,
            )
            return None
    return expected


def _populate_bundle_cache(cache_dir: Path, bundles: list[Path]) -> None:
    """Publish freshly built bundles under their content address."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for bundle in bundles:
            for source in (bundle, bundle.with_suffix(bundle.suffix + ".json")):
                shutil.copy2(source, cache_dir / source.name)
    except OSError as exc:
        print(f"[local-prepare] could not populate {cache_dir} ({exc!r})", flush=True)
        return
    print(f"[local-prepare] cached {len(bundles)} bundle(s) at {cache_dir}", flush=True)


def _build_local_training_bundles(
    *,
    profiles: list[str],
    model: str | None,
    sample: int | None,
    dataset_csv: str | None = None,
    payload: str = "full",
) -> list[Path]:
    """Build and validate one complete input bundle per worker locally.

    The single bundle builder.  It deliberately does NOT consult the prewarm:
    it is what the prewarm thread itself runs, so looking the prewarm up here
    would make that thread join itself.

    The build is deterministic in its inputs and expensive (531 s measured on
    this host once the payload stage is cold), and every launch rebuilt it from
    scratch.  A content-keyed cache under `results/prepared_training/_cache`
    now serves a bundle whose inputs — dataset bytes, every bundle-producing
    source file, every config file, and the requested model/profile/payload —
    are byte-identical to one already built.
    """
    model_key = model or str(training_cfg().training.base_model)
    training_dataset = _validation_input_path(
        dataset_csv or _COLAB.training_dataset_csv
    )
    cache_dir = _bundle_cache_dir(
        profiles=profiles, model_key=model_key, sample=sample, payload=payload,
        training_dataset=training_dataset,
    )
    cached = _cached_bundles(cache_dir, profiles=profiles) if cache_dir else None
    if cached is not None:
        for number, bundle in enumerate(cached, start=1):
            manifest = _bundle_manifest(bundle)
            print(
                f"[local-prepare] cache hit worker={number} "
                f"rows={manifest.n_df:,} payload={manifest.n_payload:,} "
                f"pos={manifest.n_pos:,} neg={manifest.n_neg:,} "
                f"sha256={manifest.sha256} bundle={bundle}",
                flush=True,
            )
        return cached
    stamp = datetime.now(timezone.utc).strftime("%m%dT%H%M%S%fZ")
    root = RESULTS / "prepared_training" / stamp
    root.mkdir(parents=True, exist_ok=False)
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
            "--dataset",
            str(training_dataset),
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
    if cache_dir is not None:
        _populate_bundle_cache(cache_dir, bundles)
    return bundles


def _prepare_local_training_bundles(
    *,
    profiles: list[str],
    model: str | None,
    sample: int | None,
    dataset_csv: str | None = None,
    payload: str = "full",
) -> list[Path]:
    """Bundles for one training call: the in-flight build when it matches.

    The only entry point that consumes a prewarm, so the build it hands over
    runs once and the overlap in `start_local_bundle_prewarm` is real.
    """
    request = {
        "profiles": profiles,
        "model": model,
        "sample": sample,
        "payload": payload,
    }
    if dataset_csv is not None:
        request["dataset_csv"] = dataset_csv
    prewarmed = _take_prewarmed_bundles(**request)
    if prewarmed is not None:
        return prewarmed
    return _build_local_training_bundles(**request)


def _upload_prepared_bundles(
    *,
    run_id: str,
    bundles: list[Path],
) -> list[str]:
    """Upload only the locally prepared bundles and their manifests."""
    remote_dir = f"{REMOTE_ROOT}/prepared_training/{run_id}"
    worker_dirs = [
        f"{remote_dir}/worker_{number}" for number in range(1, len(bundles) + 1)
    ]
    # One exec for every worker directory. Each notebook exec carries ~1.5 s of
    # round trip (measured 1.43-1.63 s in the 2026-09-15 T4 history), so the
    # former per-file mkdir made setup latency grow with the worker count while
    # creating exactly the same directories.
    run_colab_exec_stream(
        SESSION,
        _BOOTSTRAP
        + f"""
import pathlib
for worker_dir in {worker_dirs!r}:
    pathlib.Path(worker_dir).mkdir(parents=True, exist_ok=True)
""",
        timeout=120,
        log_name="prepared_bundle_mkdir",
        retry_safe=False,
    )
    remote_paths: list[str] = []
    for number, bundle in enumerate(bundles, start=1):
        for source in (bundle, bundle.with_suffix(bundle.suffix + ".json")):
            remote = f"{remote_dir}/worker_{number}/{source.name}"
            print(f"[upload] prepared bundle file={source} -> {remote}", flush=True)
            _upload_with_retries(
                source, remote, timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS
            )
        remote_paths.append(f"{remote_dir}/worker_{number}/{bundle.name}")
    return remote_paths


def _validation_input_path(configured_value: str) -> Path:
    """Resolve one configured validation artifact inside the repository."""
    configured = Path(configured_value)
    source = configured if configured.is_absolute() else TRAIN_ROOT / configured
    source = source.resolve()
    if not source.is_relative_to(TRAIN_ROOT.resolve()):
        raise ValueError(
            "colab.final_inference.input_csv must stay inside the repository"
        )
    if _FINAL_INFERENCE.enabled and not source.is_file():
        raise FileNotFoundError(f"configured final-inference CSV is missing: {source}")
    return source


def _remote_checkout_copy(source: Path) -> str | None:
    """Return the VM's own checkout copy of `source` when its bytes match.

    `prepare_remote_layout` has already put the configured branch at
    REMOTE_ROOT, so any validation input that is committed to the branch is
    on the VM before the first upload.  Re-uploading it costs the launcher
    seconds per run (measured: 14.7 s for the 46 MB deduped source CSV on the
    2026-09-15 T4 launch) and transfers bytes the VM already has.

    The reuse is content-verified, not assumed: the remote file is hashed and
    must equal the local digest, so a locally modified or absent input falls
    back to a normal upload instead of silently training on the wrong rows.
    """
    relative = source.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()
    remote = f"{REMOTE_ROOT}/{relative}"
    probe = _BOOTSTRAP + f"""
import hashlib, pathlib
path = pathlib.Path({remote!r})
if path.is_file():
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    print(digest.hexdigest(), flush=True)
else:
    print("", flush=True)
"""
    try:
        reported = run_colab_exec_capture(SESSION, probe, timeout=120).strip()
    except Exception as exc:
        print(
            f"[upload] could not verify the VM checkout copy of {source.name} "
            f"({exc}); uploading instead",
            flush=True,
        )
        return None
    if reported and reported == sha256_file(source):
        return remote
    return None


_RUN_STAMP_FORMAT = "%m%dT%H%M%S%fZ"
# How long the prewarmed upload waits for the session gate before giving up and
# letting the lane upload serially.  Generous enough for a slow GPU allocation,
# finite so a lane that never releases the gate cannot hang teardown.
_PREWARM_GATE_TIMEOUT_SECONDS = 900
# How long starting a new prewarm waits for a previous one to retire.  Bounded
# so that starting a lane never depends on an earlier lane's transfer.
_PREWARM_RETIRE_SECONDS = 30


class _ValidationUploadPrewarm:
    """One lane's validation uploads, travelling during the VM dependency install.

    The validation CSVs are read from this repository and pushed to the VM;
    nothing the VM does produces them, so the transfer can overlap the install
    instead of following it.  Measured before the overlap: 36.79 s of uploads
    strictly after a 42.96 s dependency install.

    The thread waits for :func:`ensure_session` before its first transfer.
    Started earlier, it raced session provisioning, failed on a VM that did
    not exist yet, and left the whole payload to be re-uploaded serially at
    the join -- which is the cost the overlap exists to remove.
    """

    def __init__(self, stamp: str) -> None:
        self.stamp = stamp
        self.remote_paths: dict[str, str] | None = None
        self.error: BaseException | None = None
        self.session_ready = threading.Event()
        self.thread = threading.Thread(target=self._upload, daemon=True)

    def _upload(self) -> None:
        # A GPU allocation can take minutes; the upload is worthless until the
        # control channel answers, so wait rather than transfer into a void.
        # Bounded: if the gate is never released -- a lane that failed before
        # releasing it -- this thread must still end, because the launcher's
        # drain joins it and an unbounded wait would hang teardown forever.
        if not self.session_ready.wait(_PREWARM_GATE_TIMEOUT_SECONDS):
            print(
                f"[upload] session was not ready within "
                f"{_PREWARM_GATE_TIMEOUT_SECONDS}s; uploading serially instead",
                flush=True,
            )
            return
        try:
            self.remote_paths = _perform_validation_upload(self.stamp)
        except BaseException as exc:  # handed back to the owning run, or fallen back from
            self.error = exc
            print(
                f"[upload] concurrent validation upload failed: {exc!r}", flush=True
            )

    def join(self) -> dict[str, str]:
        self.thread.join()
        if self.error is not None:
            raise self.error
        if self.remote_paths is None:
            raise RuntimeError("the concurrent validation upload returned no paths")
        return self.remote_paths


_VALIDATION_UPLOAD_PREWARM: _ValidationUploadPrewarm | None = None


def start_validation_upload_prewarm() -> str:
    """Begin this lane's validation uploads; returns the run id they belong to."""
    global _VALIDATION_UPLOAD_PREWARM
    if _VALIDATION_UPLOAD_PREWARM is not None:
        # Running a second prewarm over an unfinished one would race two
        # threads onto the same remote paths, and the first thread would never
        # be joined.  Release and retire it before starting its replacement --
        _VALIDATION_UPLOAD_PREWARM.session_ready.set()
        # but never block indefinitely: starting a lane must stay instant, so
        # a previous upload that will not finish is abandoned to its own
        # bounded wait rather than stalling this one.
        _VALIDATION_UPLOAD_PREWARM.thread.join(timeout=_PREWARM_RETIRE_SECONDS)
    stamp = datetime.now(timezone.utc).strftime(_RUN_STAMP_FORMAT)
    prewarm = _ValidationUploadPrewarm(stamp)
    _VALIDATION_UPLOAD_PREWARM = prewarm
    prewarm.thread.start()
    print(
        f"[upload] sending validation inputs for run {stamp} concurrently with "
        "the VM dependency install",
        flush=True,
    )
    return stamp


def drain_validation_upload_prewarm() -> None:
    """Never leave an upload thread writing while the live log closes."""
    global _VALIDATION_UPLOAD_PREWARM
    prewarm, _VALIDATION_UPLOAD_PREWARM = _VALIDATION_UPLOAD_PREWARM, None
    if prewarm is not None and prewarm.thread.is_alive():
        print("[upload] waiting for the concurrent upload to finish ...", flush=True)
        # The thread may still be blocked on the session gate; release it so a
        # drain at exit cannot hang forever on a VM that never came up.
        prewarm.session_ready.set()
        # Bounded: an upload to a dead VM can stall, and teardown must not wait
        # on it indefinitely.  Whatever lands is verified by the download.
        prewarm.thread.join(timeout=_PREWARM_RETIRE_SECONDS)
        if prewarm.thread.is_alive():
            print(
                "[upload] concurrent upload did not finish within "
                f"{_PREWARM_RETIRE_SECONDS}s; continuing with teardown",
                flush=True,
            )


def release_validation_upload_prewarm() -> None:
    """Let the prewarmed upload start, now that the session can accept it."""
    prewarm = _VALIDATION_UPLOAD_PREWARM
    if prewarm is not None:
        prewarm.session_ready.set()


def _lane_run_stamp() -> str:
    """Adopt the prewarmed run identity so the uploads belong to this run."""
    if _VALIDATION_UPLOAD_PREWARM is not None:
        return _VALIDATION_UPLOAD_PREWARM.stamp
    return datetime.now(timezone.utc).strftime(_RUN_STAMP_FORMAT)


def _upload_validation_inputs(run_id: str) -> dict[str, str]:
    """Validation inputs for one run: the in-flight transfer when it matches.

    The only entry point that consumes an upload prewarm, so the transfer it
    hands over runs once and the overlap in `start_validation_upload_prewarm`
    is real.
    """
    global _VALIDATION_UPLOAD_PREWARM
    if _VALIDATION_UPLOAD_PREWARM is not None:
        prewarm, _VALIDATION_UPLOAD_PREWARM = _VALIDATION_UPLOAD_PREWARM, None
        if prewarm.stamp == run_id:
            print(
                "[upload] joining the validation upload started before the VM "
                "setup",
                flush=True,
            )
            # By the time a lane joins, the session is up: release the gate so
            # the overlap actually happens instead of the thread waiting on a
            # signal that this join is itself blocking.
            prewarm.session_ready.set()
            try:
                return prewarm.join()
            except Exception as exc:
                print(
                    f"[upload] concurrent validation upload failed ({exc!r}); "
                    "uploading serially instead",
                    flush=True,
                )
        else:
            print(
                f"[upload] concurrent upload belongs to run {prewarm.stamp}, not "
                f"{run_id}; uploading serially",
                flush=True,
            )
            # This prewarm no longer belongs to the caller, but it still owns
            # a thread waiting on the session gate.  Release and retire it
            # before proceeding so it cannot linger for the full gate timeout
            # (and so its one intended transfer is not silently abandoned).
            prewarm.session_ready.set()
            prewarm.thread.join(timeout=_PREWARM_RETIRE_SECONDS)
            if prewarm.thread.is_alive():
                print(
                    "[upload] mismatched concurrent upload is still running; "
                    "continuing with the serial upload",
                    flush=True,
                )
    return _perform_validation_upload(run_id)


def _perform_validation_upload(run_id: str) -> dict[str, str]:
    """Transfer the immutable source, training complement, and SKU holdout.

    The worker itself.  It deliberately does NOT consult the prewarm: it is
    what the prewarm thread runs, so looking the prewarm up here would make
    that thread join itself.
    """
    sources = {
        "source": _validation_input_path(_FINAL_INFERENCE.source_csv),
        "training": _validation_input_path(_COLAB.training_dataset_csv),
        "sample": _validation_input_path(_FINAL_INFERENCE.input_csv),
    }
    remote_dir = f"{REMOTE_ROOT}/prepared_training/{run_id}/validation"
    remotes: dict[str, str] = {}
    remote_by_source: dict[str, str] = {}
    if not _FINAL_INFERENCE.enabled:
        for key, source in sources.items():
            remotes[key] = remote_by_source.setdefault(
                str(source.resolve()), f"{remote_dir}/{key}_{source.name}"
            )
        return remotes
    run_colab_exec_stream(
        SESSION,
        _BOOTSTRAP
        + f"""
import pathlib
pathlib.Path({remote_dir!r}).mkdir(parents=True, exist_ok=True)
""",
        timeout=120,
        log_name="validation_input_dir",
        retry_safe=False,
    )
    for key, source in sources.items():
        source_key = str(source.resolve())
        if source_key in remote_by_source:
            remotes[key] = remote_by_source[source_key]
            print(
                f"[upload] validation {key}={source} reusing "
                f"{remote_by_source[source_key]}",
                flush=True,
            )
            continue
        checkout_copy = _remote_checkout_copy(source)
        if checkout_copy is not None:
            remote_by_source[source_key] = checkout_copy
            remotes[key] = checkout_copy
            print(
                f"[upload] validation {key}={source} reused the verified VM "
                f"checkout copy {checkout_copy} (sha256 matches; not uploaded)",
                flush=True,
            )
            continue
        remote = f"{remote_dir}/{key}_{source.name}"
        print(f"[upload] validation {key}={source} -> {remote}", flush=True)
        _upload_with_retries(
            source, remote, timeout=_RESULT_DOWNLOAD_TIMEOUT_SECONDS
        )
        remote_by_source[source_key] = remote
        remotes[key] = remote
    return remotes


def run_single_train_and_stream(
    args: list[str], *, run_label: str | None = None,
    prepared_bundle: Path | None = None, final_inference: bool = True,
    remote_checkout_bundle: str | None = None,
    inference_sample: int | None = None,
    inference_device: str | None = None,
    copy_remote_inputs: bool = True,
    remote_validation_csv: str | None = None,
    prepare_remote_labeled_pairs: bool = False,
    include_wandb: bool = True,
    incremental_sync: bool = True,
) -> tuple[str, int]:
    """Run one worker in the Colab exec stream so W&B is visible immediately."""
    stamp = _lane_run_stamp()
    remote_base = f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    _record_remote_run(remote_base, workers=1, lane="train")
    if not final_inference:
        remote_validation_inputs = {"sample": "", "source": "", "training": ""}
    elif remote_validation_csv is not None:
        remote_validation_inputs = {
            "sample": remote_validation_csv,
            "source": remote_validation_csv,
            "training": remote_validation_csv,
        }
    else:
        remote_validation_inputs = _upload_validation_inputs(run_id)
    if prepared_bundle is not None and remote_checkout_bundle is not None:
        raise ValueError("choose either an uploaded or checkout prepared bundle")
    if remote_checkout_bundle is not None:
        relative = Path(remote_checkout_bundle)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkout bundle path must stay inside the checkout")
        remote_bundle = str(Path(REMOTE_ROOT) / relative)
        args = list(args)
        args[args.index("training.train")] = "training.train_prepared"
        args.extend(["--bundle", remote_bundle])
    elif prepared_bundle is not None:
        remote_bundle = _upload_prepared_bundles(
            run_id=Path(remote_base).name.removeprefix("concurrent_train_"),
            bundles=[prepared_bundle],
        )[0]
        args = list(args)
        args[args.index("training.train")] = "training.train_prepared"
        args.extend(["--bundle", remote_bundle])
    remote_input_loop = (
        "for name in ():"
        if remote_checkout_bundle is not None or prepared_bundle is not None or not copy_remote_inputs
        else 'for name in (F["canonical_records"], F["gate_results"], F["labeled_pairs"]):'
    )
    script = _BOOTSTRAP + _remote_auth_env_script(include_wandb=include_wandb) + f"""
import os, pathlib, shutil, subprocess, sys
from core.common import F
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
out = base / "worker_1"
base.mkdir(parents=True, exist_ok=False)
out.mkdir()
print(f"[worker] setup complete: results={{out}}", flush=True)
print("[worker] resolving versioned checkout inputs", flush=True)
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
       "EUROMONITOR_REMOTE_TRAINING": "1", "EUROMONITOR_DISABLE_DVC_CHECKPOINTS": {_DVC_DISABLED_FLAG!r}}}
if {prepare_remote_labeled_pairs!r}:
    calibration = out / "training" / "labeled_pairs.csv"
    if not calibration.is_file():
        print(f"[data] generating worker calibration input: {{calibration}}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "training.labeled_pairs"],
            cwd=root, env=env, check=True,
        )
command = [sys.executable, *{args!r}]
log_path = out / "training.log"
print("[worker] inputs ready; launching training process", flush=True)
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
print("[worker] training process finished", flush=True)
run_completion = {final_inference!r}
completion = [
    sys.executable, "-m", "training.complete_colab_worker",
    "--source", str(out), "--run-id", {run_id!r}, "--worker", "1",
        "--validation-input", {remote_validation_inputs['sample']!r},
        "--validation-source", {remote_validation_inputs['source']!r},
    "--training-input", {remote_validation_inputs['training']!r},
]
if {inference_sample!r} is not None:
    completion.extend(["--sample", str({inference_sample!r})])
if {inference_device is not None!r}:
    completion.extend(["--device", {inference_device!r}])
if not { _DVC_ENABLED!r}:
    completion.append("--skip-dvc")
if run_completion:
    print("[worker] starting final validation inference", flush=True)
    print("[train] training complete; running validation inference and final DVC publication", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            completion, cwd=root, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        completion_rc = process.wait()
    if completion_rc:
        raise RuntimeError(
            f"worker 1 completion failed (rc={{completion_rc}}); log={{log_path}}"
        )
print(f"[train] worker 1 completed; log={{log_path}}", flush=True)
"""
    print(
        "[run] starting one trainer as a detached remote stage; polling its durable log ...",
        flush=True,
    )
    # Stream finished artifacts back while the trainer runs, so the end-of-run
    # download is a short delta instead of the whole result set.  The stream is
    # stopped before the authoritative download so the two never race on the
    # same local file.
    syncer = _IncrementalResultSync(remote_base, run_id, workers=1) if incremental_sync else None
    if syncer is not None:
        syncer.start()
    try:
        # A long-lived ``colab exec`` stream can stall before the kernel begins
        # evaluating the worker cell. Run the exact same script outside the
        # notebook kernel instead; its log, PID, and exit status are then
        # independently visible through the short polling probes.
        run_detached_stage(
            "train",
            ["/usr/bin/python3", "-c", script],
            timeout=_WORKER_TIMEOUT_SECONDS,
        )
    finally:
        if syncer is not None:
            syncer.stop()
    download_verified_training_results(remote_base, 1)
    print(
        f"[train] single worker completed; results downloaded "
        f"({_format_bytes(syncer.synced_bytes() if syncer is not None else 0)} arrived during the run)",
        flush=True,
    )
    return remote_base, 1


def run_hpo(
    mode: str | None = None,
    *,
    resume: bool = False,
    trial_jobs: int = _HPO_TRIAL_JOBS_DEFAULT,
    persistence: str | None = None,
    loss: str = _TRAIN_LOSS,
) -> None:
    """Sweep every configured backbone, then evaluate and rerank each winner."""
    mode = mode or _HPO_MODE
    persistence = persistence or _HPO_PERSISTENCE
    print(
        f"[run] round-robin HPO (mode={mode}, model_workers={_HPO_WORKERS}, "
        f"trial_jobs={trial_jobs}, resume={resume}, persistence={persistence}) ..."
    )
    run_id = (
        datetime.now(timezone.utc).strftime("hpo_%m%dT%H%M%SZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    mask_effect_flag = "--mask-effect" if _MASK_EFFECT_AFTER_TRAIN else "--no-mask-effect"
    resume_pointers = _hpo_resume_pointer_payload() if resume else {}
    script = _BOOTSTRAP + _remote_auth_env_script(include_optuna=True) + f"""
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
base = [sys.executable, "-u", "-m", "training.train", "--split", "holdout", "--loss", "{loss}", "--payload", "full", "--no-plot", "{mask_effect_flag}"]
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
        f"colab_{{label}}_{{datetime.now(timezone.utc).strftime('%m%dT%H%M%SZ')}}.log"
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
    run_id = datetime.now(timezone.utc).strftime("zero_shot_%m%dT%H%M%S%fZ")
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
    loss: str = _TRAIN_LOSS,
) -> tuple[str, int]:
    """Run one masked trainer and one zero-shot worker on the same VM."""
    model_key = model or str(training_cfg().training.base_model)
    if model_key not in set(embedding_model_keys()):
        raise ValueError(
            "mixed lane requires an embedding model registry key: "
            f"{model_key!r}"
        )
    stamp = datetime.now(timezone.utc).strftime("%m%dT%H%M%S%fZ")
    remote_base = f"{REMOTE_ROOT}/results/concurrent_train_mixed_{stamp}"
    _record_remote_run(remote_base, workers=2, lane="mixed")
    train_args = [
        "-u",
        "-m",
        "training.train",
        "--split",
        "holdout",
        "--loss",
        loss,
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
    stamp = time.strftime("%m%dT%H%M%SZ", time.gmtime())
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


def _list_remote(pattern_dir: str, *, max_depth: int | None = None) -> list[str]:
    """List remote files via a stdin-exec (same channel the lanes use).

    ``max_depth`` bounds the walk.  An unbounded ``rglob`` over a live run
    root is what timed out a listing during the 2026-09-15 T4 run: the tree
    holds every checkpoint, ``wandb`` file, and log the run has produced so
    far, and the walk exceeded the 120 s exec budget.  Callers that only need
    a shallow slice pass a depth and get a bounded walk.
    """
    import json as _json

    if max_depth is None:
        collect = "p for p in root.rglob('*')"
    else:
        collect = (
            f"p for p in root.rglob('*') "
            f"if len(p.relative_to(root).parts) <= {int(max_depth)}"
        )
    script = (
        "import pathlib, json\n"
        f"root = pathlib.Path({pattern_dir!r})\n"
        f"files = sorted(str(p) for p in {collect} if p.is_file())\n"
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


def stop_local_launch_owner(*, timeout_seconds: float = 15.0) -> None:
    """Ask a verified launcher owner to exit after its VM is stopped.

    PID reuse makes a bare PID unsafe.  New lock records include Linux's
    process-start ticks, so this only sends SIGTERM when the recorded process
    is demonstrably still the same process that acquired the lock.
    """
    lock_path = _colab_launch_lock_path()
    if not _colab_launch_lock_is_held(lock_path):
        print("[stop] local launcher lock is already released")
        return
    owner = _read_colab_launch_owner(lock_path)
    if owner is None:
        print(
            f"[warn] local launcher lock remains held but has no readable owner metadata: {lock_path}",
            file=sys.stderr,
        )
        return
    pid = owner.get("pid")
    expected_start = owner.get("pid_start_ticks")
    if not isinstance(pid, int) or not isinstance(expected_start, int):
        print(
            "[warn] local launcher lock remains held by a legacy or unverifiable owner; "
            f"metadata={owner}. It cannot be signalled safely.",
            file=sys.stderr,
        )
        return
    actual_start = _process_start_ticks(pid)
    if actual_start != expected_start:
        print(
            "[warn] local launcher lock remains held, but its recorded owner no longer "
            f"matches pid={pid}; refusing to signal a potentially reused PID.",
            file=sys.stderr,
        )
        return
    if pid == os.getpid():
        print("[stop] current launcher owns the local session lock")
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"[stop] recorded launcher pid={pid} has already exited")
    except PermissionError:
        print(
            f"[warn] local launcher pid={pid} owns the lock but cannot be signalled",
            file=sys.stderr,
        )
        return
    else:
        print(f"[stop] requested shutdown from local launcher pid={pid}")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _colab_launch_lock_is_held(lock_path):
            print("[stop] local launcher lock released")
            return
        time.sleep(0.2)
    print(
        f"[warn] local launcher lock is still held after {timeout_seconds:g}s: {lock_path}",
        file=sys.stderr,
    )


def stop(*, stop_local_owner: bool = False) -> None:
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
    if stop_local_owner:
        stop_local_launch_owner()
    print("[stop] VM release requested")


def main() -> None:
    global GPU
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", required=True,
                    choices=["train", "dual-train", "hpo", "sims", "mixed", "smoke", "stop"],
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
        "--loss",
        choices=["contrastive", "mnrl", "triplet"],
        default=_TRAIN_LOSS,
        help="training loss (default: config/training.yaml training.loss)",
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
    ap.add_argument(
        "--keep-alive",
        action="store_true",
        help="CPU only: do not tear down the VM on completion/failure (refused for GPU launches)",
    )
    ap.add_argument(
        "--preflight-only", action="store_true",
        help="validate and print the train/inference lifecycle without contacting Colab",
    )
    ap.add_argument(
        "--train-only", action="store_true",
        help="train and collect artifacts without post-training validation inference",
    )
    args = ap.parse_args()

    GPU = args.gpu
    if GPU.upper() != "CPU" and not args.allow_gpu:
        raise ValueError("GPU launch requires --allow-gpu")

    # Two different decisions used to be one, and conflating them stopped every
    # GPU lane from provisioning at all:
    #
    #   the DAEMON   is spawned by `colab new` itself and is what provisioning
    #                needs.  It is always allowed, on every lane.
    #   RETENTION    is whether the VM is still running when the work ends.
    #                That is CPU-only, and a GPU lane never keeps it.
    os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = "1"
    if args.keep_alive and GPU.upper() != "CPU":
        # Retention is the operator-facing flag, and it stays refused rather
        # than downgraded to a warning: a caller who asked to keep a GPU VM
        # must not be able to mistake a warning for a retained VM, and a
        # retained GPU VM bills accelerator quota for as long as it lives.
        raise ValueError(
            "--keep-alive is CPU-only (a retained GPU VM consumes accelerator "
            f"quota indefinitely); requested --gpu {GPU}. Use --gpu CPU or drop "
            "--keep-alive."
        )

    if args.preflight_only:
        if args.what not in {"train", "smoke"}:
            raise ValueError("--preflight-only supports train/smoke lanes")
        print(json.dumps(training_lifecycle_preflight(
            workers=args.workers if args.what == "train" else _SMOKE_WORKERS,
            model=args.model,
            masking_profile=args.masking_profile,
            train_only=args.train_only,
        ), indent=2, sort_keys=True))
        return

    dvc_jobs = int(training_cfg().colab.dvc_jobs)
    if args.what == "train":
        print(
            f"[workers] lane=train trainers={args.workers} "
            f"dvc_publishers={_DVC_WORKERS} dvc_transfer_jobs={dvc_jobs}",
            flush=True,
        )
    elif args.what == "dual-train":
        print(
            f"[workers] lane=dual-train matcher_loss={args.loss} ann_loss=mnrl "
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
        stop(stop_local_owner=True)
        return

    start_live_log()
    launch_lock = None
    try:
        check_colab_cli()
        launch_lock = acquire_colab_launch_lock()
    except BaseException:
        close_live_log()
        raise
    local_training_run: tuple[str, int] | None = None
    local_mixed_run: tuple[str, int] | None = None
    local_hpo_run: str | None = None

    try:
        prepared_train_runtime = args.what in {"train", "dual-train"}
        # The local bundle build is pure local CPU work over immutable local
        # inputs, so start it before the VM is even provisioned: it then runs
        # under the remote checkout, install, model check, and profile instead
        # of after them.  run_train joins this exact build (or rebuilds when
        # the request differs).
        bundle_request = _lane_bundle_request(args)
        if bundle_request is not None:
            start_local_bundle_prewarm(**bundle_request)
        # The validation CSVs are read here and pushed to the VM, so they do
        # not have to wait for the dependency install to finish.  A resumed
        # run keeps its existing identity and uploads serially.
        if bundle_request is not None and not args.resume_run:
            start_validation_upload_prewarm()
        ensure_session()
        # The session exists now, so the prewarmed upload can run for real.  It
        # travels alongside prepare_remote_layout/install_deps below instead of
        # after them, which is the whole point of starting it early.
        release_validation_upload_prewarm()
        if GPU.upper() != "CPU":
            # Backstop, not the primary release: the launcher's own teardown
            # still runs in `finally`.  A GPU VM must never be left held open
            # by its own daemon if this process dies, so the daemon is stopped
            # now that provisioning is done and the launcher owns the run --
            # the VM then idle-terminates instead of burning accelerator quota
            # indefinitely.
            stop_keep_alive_daemon(reason=f"GPU lane ({GPU}) must never be retained")
        prepare_remote_layout(minimal_runtime=prepared_train_runtime)
        install_deps(minimal_runtime=prepared_train_runtime)
        if args.what in {"train", "dual-train", "smoke", "mixed"}:
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
        elif args.what != "smoke" and not prepared_train_runtime:
            verify_training_inputs()
        # AUDIT FIX 2026-09-08: --what sims used to run FULL TRAINING first
        # (run_train was unconditional) — hours of unintended GPU quota
        # for a lane that only needs the configured zero-shot scoring.
        if args.what == "sims":
            run_sims()
        elif args.what == "mixed":
            local_mixed_run = run_mixed(
                args.train_frac, args.epochs, model=args.model, loss=args.loss,
            )
        elif args.what == "smoke":
            smoke_sample = args.sample if args.sample is not None else _SMOKE_SAMPLE
            local_training_run = run_train(
                args.train_frac, _SMOKE_EPOCHS, sample=smoke_sample,
                workers=_SMOKE_WORKERS,
                inference_sample=smoke_sample,
                inference_device="cuda" if GPU.upper() != "CPU" else "cpu",
                run_label=args.run_label,
                masking_profile=args.masking_profile,
                collapse_guardrail_profile=args.collapse_guardrail_profile,
                loss=args.loss,
                # The smoke lane's full CSV is already in the Git checkout.
                # It tests inference against the same file without uploading
                # a separate held-out split.
                train_only=False,
                remote_dataset_csv="data/dataset_deduped.csv",
                # These bundles are committed with the other immutable
                # runtime inputs.  Colab only consumes them on the GPU; no
                # CSV or prepared-data upload is part of this lane.
                remote_prepared_bundles=[
                    "data/prepared/smoke_1000/worker_1_baseline.pkl.gz",
                    "data/prepared/smoke_1000/worker_2_baseline.pkl.gz",
                ] if smoke_sample == 1000 else None,
                remote_validation_csv=f"{REMOTE_ROOT}/data/dataset_deduped.csv",
                # The status/log poll is the only Colab control-channel user
                # during smoke; checkpoint syncing waits for full runs.
                incremental_sync=False,
            )
        elif args.what == "dual-train":
            if args.resume_run:
                raise ValueError("--resume-run is not supported for dual-train")
            # Both workers start from the same shipped base model and prepared
            # data, but write fully isolated checkpoints/results. The ANN
            # encoder is always trained with MNRL; the matcher uses --loss.
            local_training_run = run_train(
                args.train_frac, args.epochs, sample=args.sample, workers=2,
                model=args.model, run_label="matcher,ann_embedding",
                masking_profile=args.masking_profile,
                collapse_guardrail_profile=args.collapse_guardrail_profile,
                loss=args.loss,
                worker_losses=[args.loss, "mnrl"],
                train_only=args.train_only,
            )
        elif args.what == "hpo":
            if args.hpo_jobs < 1:
                raise ValueError("--hpo-jobs must be >= 1")
            local_hpo_run = run_hpo(
                args.hpo_mode,
                resume=args.resume_hpo,
                trial_jobs=args.hpo_jobs,
                persistence=args.hpo_persistence,
                loss=args.loss,
            )
        else:
            checkout_full_bundles = (
                list(_COLAB.full_prepared_bundles)
                if (
                    args.what == "train"
                    and args.workers == 1
                    and args.model is None
                    and args.sample is None
                    and args.resume_run is None
                )
                else None
            )
            local_training_run = run_train(
                args.train_frac, args.epochs, sample=args.sample, workers=args.workers,
                resume_run=args.resume_run, model=args.model,
                run_label=args.run_label,
                masking_profile=args.masking_profile,
                collapse_guardrail_profile=args.collapse_guardrail_profile,
                loss=args.loss,
                train_only=args.train_only,
                remote_dataset_csv=(
                    _FINAL_INFERENCE.source_csv if checkout_full_bundles else None
                ),
                remote_prepared_bundles=checkout_full_bundles,
                inference_device="cuda" if GPU.upper() != "CPU" else "cpu",
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
        # A lane that failed before its run_train call would otherwise leave
        # the concurrent bundle build writing into a closing log file.
        drain_local_bundle_prewarm()
        drain_validation_upload_prewarm()
        close_live_log()
        release_colab_launch_lock(launch_lock)

    if _DVC_ENABLED:
        print("\n[done] artifacts persisted to the configured DVC remote")
    else:
        print("\n[done] artifacts retained locally; DVC disabled")


if __name__ == "__main__":
    main()
