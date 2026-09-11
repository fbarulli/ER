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
import json
import subprocess
import sys
import threading
import time
import traceback
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
REPOSITORY = _COLAB.repository
BRANCH = _COLAB.branch
SESSION = _COLAB.session
GPU = _COLAB.gpu
REMOTE_ROOT = _COLAB.remote_root
_HPO_MODE = _COLAB.hpo_mode
_HPO_WORKERS = _COLAB.hpo_workers
_TRAIN_WORKERS = _COLAB.train_workers
_LOG_POLL_SECONDS = _COLAB.log_poll_seconds
_PROBE_TIMEOUT_SECONDS = _COLAB.probe_timeout_seconds
_PROBE_RETRIES = _COLAB.probe_retries
_PROBE_RETRY_BACKOFF_SECONDS = _COLAB.probe_retry_backoff_seconds
_MASK_EFFECT_AFTER_TRAIN = _COLAB.mask_effect_after_train
_SMOKE_EPOCHS = _COLAB.smoke_epochs
_WORKER_TIMEOUT_SECONDS = _COLAB.worker_timeout_seconds
_HPO_RESUME_DIR = TRAINING_RESULTS / "hpo_resume"
LIVE_LOG_PATH: Path | None = None
_live_log = None
_original_stdout = None
_original_stderr = None


class _Tee:
    """Mirror launcher output to the terminal and the root live log."""

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

# The clone contains the committed raw export and number-token reference;
# data_prep regenerates deduped data and all downstream CSVs on the VM.


def check_colab_cli() -> None:
    """Ensure the colab CLI is installed and authenticated."""
    try:
        subprocess.run(["colab", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            "colab CLI not found or not authenticated.\n"
            "Run: uv tool install google-colab-cli\n"
            "Then: colab sessions  (to complete OAuth sign-in)"
        )


def colab(*args: str, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a colab CLI subcommand."""
    cmd = ["colab", *args]
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        print(f"\n[error] colab command failed: {' '.join(cmd)}", file=sys.stderr)
        if e.stdout:
            print(f"stdout:\n{e.stdout[-1000:]}", file=sys.stderr)
        if e.stderr:
            print(f"stderr:\n{e.stderr[-1000:]}", file=sys.stderr)
        raise


def run_colab_exec_stream(session: str, script: str, timeout: int | None = None, log_name: str | None = None) -> None:
    """Execute a python script on the colab session via stdin, streaming stdout/stderr.

    log_name labels a stage in the root training.log transcript. The file is
    opened once per invocation, line-flushed, and survives VM teardown so
    every Colab stage is inspectable in one chronological log.
    """
    if _live_log and log_name:
        print(f"\n===== {log_name} =====", flush=True)

    def stream_output(pipe, prefix):
        for line in iter(pipe.readline, ''):
            print(f"{prefix} {line.rstrip()}", flush=True)
        pipe.close()

    process = subprocess.Popen(
        # colab exec has its own 30-second kernel-client timeout.  It must
        # match the caller's legitimate lane timeout; otherwise a live VM
        # computation is reported as failed after 30 seconds.
        ["colab", "exec", "-s", session, "--timeout", str(timeout or 30)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    out_thread = threading.Thread(target=stream_output, args=(process.stdout, "[out]"))
    err_thread = threading.Thread(target=stream_output, args=(process.stderr, "[err]"))
    out_thread.start()
    err_thread.start()

    process.stdin.write(script)
    process.stdin.close()

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"\n[error] Execution timed out after {timeout}s", file=sys.stderr)
        raise

    out_thread.join()
    err_thread.join()
    if process.returncode != 0:
        raise RuntimeError(
            f"Remote execution failed with return code {process.returncode}; "
            "see the timestamped Colab log for the full traceback"
        )


def run_colab_exec_capture(session: str, script: str, timeout: int) -> str:
    """Execute a remote probe, streaming its output while retaining stdout."""
    last_error = ""
    for attempt in range(1, _PROBE_RETRIES + 1):
        try:
            process = subprocess.Popen(
                ["colab", "exec", "-s", session, "--timeout", str(timeout)],
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
                    print(f"[probe-out] {line.rstrip()}", flush=True)

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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
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
    args: list[str], workers: int, *, resume_run: str | None = None
) -> None:
    """Run isolated full-data trainers concurrently and mirror worker logs."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_base = (
        f"{REMOTE_ROOT}/results/concurrent_train_{resume_run}"
        if resume_run
        else f"{REMOTE_ROOT}/results/concurrent_train_{stamp}"
    )
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    resume_pointers = _resume_pointer_payload(run_id, workers) if resume_run else {}
    launch = _BOOTSTRAP + _remote_auth_env_script() + f"""
import base64, json, os, pathlib, shutil, shlex, subprocess, sys, traceback
from core.common import F
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
base.mkdir(parents=True, exist_ok={bool(resume_run)!r})
command = " ".join(shlex.quote(part) for part in [sys.executable, *{args!r}])
resume_pointers = {resume_pointers!r}
started = []
for number in range(1, {workers} + 1):
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
            source = root / "results" / name
            if not (out / name).is_file():
                if not source.is_file():
                    raise FileNotFoundError(f"resume worker input missing: {{source}}")
                shutil.copy2(source, out / name)
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
            source = root / "results" / name
            if not source.is_file():
                raise FileNotFoundError(f"worker input missing: {{source}}")
            shutil.copy2(source, out / name)
    log_path, status_path = out / "training.log", out / "training.status"
    env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root / "src"), "EUROMONITOR_RESULTS_DIR": str(out),
           "EUROMONITOR_MLRUNS_DIR": str(out / "mlruns"), "WANDB_RUN_NAME": f"train_worker_{{number}}"}}
    process_log = out / "processes.log"
    ps_command = f"ps -eo pid,ppid,pgid,etime,stat,%cpu,%mem,rss,args >> {{shlex.quote(str(process_log))}} 2>&1"
    wrapped = f"{{ps_command}}; timeout --signal=TERM --kill-after=60 {_WORKER_TIMEOUT_SECONDS} {{command}}; rc=$?; {{ps_command}}; printf '%s\\n' \\"$rc\\" > {{shlex.quote(str(status_path))}}; exit $rc"
    with log_path.open("a" if {bool(resume_run)!r} else "w", encoding="utf-8", buffering=1) as log_file:
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
    while True:
        probe = _BOOTSTRAP + f"""
import json, pathlib
base = pathlib.Path({remote_base!r})
offsets = {offsets!r}
payload = {{"offsets": {{}}, "chunks": {{}}, "status": {{}}, "resume": {{}}}}
for number in range(1, {workers} + 1):
    key = str(number)
    out = base / f"worker_{{number}}"
    log_path, status_path = out / "training.log", out / "training.status"
    offset = int(offsets.get(key, 0))
    data = b""
    if log_path.is_file():
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    payload["offsets"][key] = offset + len(data)
    payload["chunks"][key] = data.decode("utf-8", errors="replace")
    payload["status"][key] = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else None
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
        for worker, chunk in payload["chunks"].items():
            for line in str(chunk).splitlines():
                print(f"[worker {worker}] {line}", flush=True)
        if payload["done"]:
            failed = {worker: rc for worker, rc in payload["status"].items() if int(rc) != 0}
            if failed:
                raise RuntimeError(f"parallel trainers failed: {failed}")
            publish_parallel_results(remote_base, workers)
            download_verified_training_results(remote_base, workers)
            print(f"[train] all {workers} remote workers completed successfully", flush=True)
            return
        time.sleep(_LOG_POLL_SECONDS)


def publish_parallel_results(remote_base: str, workers: int) -> None:
    """Publish completed workers sequentially through one remote process."""
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import pathlib, subprocess, sys, os
root = pathlib.Path({REMOTE_ROOT!r})
base = pathlib.Path({remote_base!r})
for number in range(1, {workers} + 1):
    source = base / f"worker_{{number}}"
    print(f"[dvc] central publish worker {{number}}/{{workers}}", flush=True)
    subprocess.run([
        sys.executable, "-u", "-m", "training.dvc_store",
        "--source", str(source), "--run-id", {run_id!r}, "--worker", str(number),
    ], cwd=root, env={{**os.environ, "PYTHONPATH": str(root / "src")}}, check=True)
print("[dvc] central publisher completed all workers", flush=True)
"""
    run_colab_exec_stream(
        SESSION, script,
        timeout=_WORKER_TIMEOUT_SECONDS * max(1, workers),
        log_name="03_dvc_publish",
    )


def download_verified_training_results(remote_base: str, workers: int) -> None:
    """Materialize only DVC-verified smoke outputs under training_results/."""
    run_id = Path(remote_base).name.removeprefix("concurrent_train_")
    local_base = TRAINING_RESULTS / run_id
    for number in range(1, workers + 1):
        remote_dir = f"{remote_base}/worker_{number}"
        local_dir = local_base / f"worker_{number}"
        for name in _list_remote(remote_dir):
            remote = Path(name)
            if remote.suffix not in {".csv", ".json", ".log"}:
                continue
            rel = remote.relative_to(remote_dir)
            local = local_dir / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            colab("download", "-s", SESSION, name, str(local), timeout=600)
    print(f"[download] DVC-verified training outputs -> {local_base}", flush=True)


def start_live_log() -> None:
    """Start the root-level live Colab log, replacing the prior run's log."""
    global LIVE_LOG_PATH, _live_log, _original_stdout, _original_stderr
    if _live_log is not None:
        _live_log.close()
    LIVE_LOG_PATH = TRAIN_ROOT / F["colab_live_log"]
    _live_log = LIVE_LOG_PATH.open("w", encoding="utf-8")
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = _Tee(_original_stdout, _live_log)
    sys.stderr = _Tee(_original_stderr, _live_log)
    print(f"[log] capturing Colab output -> {LIVE_LOG_PATH}", flush=True)


def close_live_log() -> None:
    global _live_log, _original_stdout, _original_stderr
    if _live_log is not None:
        sys.stdout = _original_stdout or sys.stdout
        sys.stderr = _original_stderr or sys.stderr
        _live_log.close()
        _live_log = None
        _original_stdout = None
        _original_stderr = None


def ensure_session() -> None:
    """Provision the session if it does not already exist."""
    r = colab("sessions", check=False)
    if SESSION in (r.stdout or ""):
        print(f"[session] '{SESSION}' already active")
        return
    print(f"[session] provisioning {SESSION} (gpu={GPU}) ...")
    colab("new", "-s", SESSION, "--gpu", GPU, timeout=300)
    print("[session] up")


def prepare_remote_layout() -> None:
    """Clone/update the configured training branch and create runtime dirs."""
    script = f"""
import pathlib, shutil, subprocess

root = pathlib.Path({REMOTE_ROOT!r})
if root.exists() and not (root / ".git").is_dir():
    shutil.rmtree(root)
if (root / ".git").is_dir():
    subprocess.run(["git", "pull", "--ff-only", "origin", {BRANCH!r}], cwd=root, check=True)
else:
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", {BRANCH!r},
        {REPOSITORY!r}, str(root),
    ], check=True)
for path in [root / "artifacts" / "data", root / "artifacts" / "results"]:
    path.mkdir(parents=True, exist_ok=True)
print("[repo] ready", {REPOSITORY!r}, "branch", {BRANCH!r}, "at", root)
"""
    run_colab_exec_stream(SESSION, script, timeout=600, log_name="00_checkout")


def install_deps() -> None:
    print("[deps] installing dependencies on the VM ...")
    # pip via sys.executable is guaranteed on a Colab VM (uv is NOT installed
    # there by default); streaming shows install progress live.
    # datasets + accelerate + transformers for the HF Trainer-based training
    # lane; sentence-transformers pins its own transformers requirement.
    install_script = (
        "import sys, subprocess\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
        "                'sentence-transformers', 'datasets', 'accelerate',\n"
        "                'evaluate', 'scikit-learn', 'pandas', 'numpy',\n"
        "                'mlflow', 'optuna', 'wandb', 'dvc', 'dagshub'], check=True)\n"
        "print('deps installed')"
    )
    run_colab_exec_stream(SESSION, install_script, timeout=900, log_name="00_deps")


def log_gpu_profile() -> None:
    """Record the exact accelerator and memory budget before training."""
    script = """import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable: refusing a CPU HPO run')
p = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info(0)
print({'name': p.name, 'total_gb': round(total / 1e9, 2), 'free_gb': round(free / 1e9, 2), 'torch': torch.__version__}, flush=True)
"""
    run_colab_exec_stream(SESSION, script, timeout=120, log_name="gpu_profile")


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


def _hf_env_script() -> str:
    """Pass a local HF token into the VM process without persisting it."""
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        key = _env_value(name)
        if key:
            print(f"[huggingface] {name} loaded from local .env and injected into VM process")
            return f"os.environ['HF_TOKEN'] = {key!r}\n"
    print("[huggingface] HF_TOKEN absent from .env; Hub requests will be anonymous")
    return ""


def _remote_auth_env_script() -> str:
    """Credential exports used by remote subprocess launch cells only."""
    key = _env_value("DVC_API_KEY")
    if key:
        print("[dvc] DVC_API_KEY loaded from local .env and injected into VM process")
        dvc = f"os.environ['DVC_API_KEY'] = {key!r}\nos.environ['DAGSHUB_USER_TOKEN'] = {key!r}\n"
    else:
        print("[dvc] DVC_API_KEY absent from .env; durable DVC upload will fail")
        dvc = ""
    return _wandb_env_script() + _hf_env_script() + dvc


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
for step in ("src/training/dedupe.py", "src/training/build_reference.py --verify", "src/training/data_prep.py"):
    print("== " + step, flush=True)
    rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/" + step.split()[0]] + step.split()[1:]).returncode
    if rc != 0:
        raise RuntimeError(f"data-prep stage failed: {{step}} (rc={{rc}})")
"""
    # dedupe 1-2 min + reference verify ~3 min + data_prep ~2 min
    run_colab_exec_stream(SESSION, script, timeout=1800, log_name="01_data_prep")


def verify_training_inputs() -> None:
    """Use the frozen CSV inputs committed on the training branch."""
    print("[data] validating frozen training CSVs from the cloned branch ...")
    script = _BOOTSTRAP + f"""
from core.common import DATA_DIR, F, RESULTS
required = [
    DATA_DIR / F["dataset_deduped"],
    DATA_DIR / F["number_reference"],
    RESULTS / F["canonical_records"],
    RESULTS / F["gate_results"],
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError("frozen training CSVs missing: " + ", ".join(missing))
for path in required:
    print(f"[data] {{path}}: {{path.stat().st_size:,}} bytes", flush=True)
"""
    run_colab_exec_stream(SESSION, script, timeout=120, log_name="01_data_check")


def run_train(
    frac: float, epochs: int, sample: int | None, workers: int = 1,
    *, resume_run: str | None = None,
) -> None:
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
        "--no-plot"]
    if sample is not None:
        args.extend(["--sample", str(sample)])
    if not _MASK_EFFECT_AFTER_TRAIN:
        args.append("--no-mask-effect")
    if resume_run:
        args.append("--resume")
    run_parallel_train_and_tail(args, workers, resume_run=resume_run)


def run_hpo(mode: str | None = None, *, resume: bool = False) -> None:
    """Sweep every configured backbone, then evaluate and rerank each winner."""
    mode = mode or _HPO_MODE
    print(
        f"[run] round-robin HPO (mode={mode}, workers={_HPO_WORKERS}, "
        f"resume={resume}) ..."
    )
    resume_pointers = _hpo_resume_pointer_payload() if resume else {}
    script = _BOOTSTRAP + _remote_auth_env_script() + f"""
import concurrent.futures, json, os, pathlib, shutil, subprocess, sys
from datetime import datetime, timezone
from core.common import F, hpo_cfg, resolve_model
root = pathlib.Path("{REMOTE_ROOT}")
base = [sys.executable, "-u", "-m", "training.train", "--split", "holdout", "--loss", "contrastive", "--payload", "full", "--no-plot"]
hpo_base = list(base)
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
    log_path = root / "results" / "logs" / (
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
    if mode != "parallel_same_vm":
        return root / "results", {{}}
    out = root / "results" / "hpo_workers" / model_key
    out.mkdir(parents=True, exist_ok=True)
    for name in (F["canonical_records"], F["gate_results"]):
        source, target = root / "results" / name, out / name
        if not source.is_file():
            raise FileNotFoundError(f"worker input missing: {{source}}")
        shutil.copy2(source, target)
    return out, {{
        "EUROMONITOR_RESULTS_DIR": str(out),
        "EUROMONITOR_MLRUNS_DIR": str(out / "mlruns"),
        "WANDB_RUN_NAME": f"hpo_{{model_key}}",
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
        final_env["WANDB_RUN_NAME"] = f"final_{{model_key}}"
    run_logged(final, f"final_{{model_key}}", final_env)
    return {{"model_key": model_key, "model": str(model), "best": params,
            "results_dir": str(out.relative_to(root / "results"))
            if mode == "parallel_same_vm" else "results"}}

if mode == "parallel_same_vm":
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(model_keys))) as pool:
        summary = [future.result() for future in [pool.submit(run_model, key) for key in model_keys]]
else:
    summary = [run_model(key) for key in model_keys]
(root / "results" / "hpo_round_robin_summary.json").write_text(
    json.dumps({{"models": summary, "rerank_model": "{_RERANK_MODEL}"}}, indent=2),
    encoding="utf-8",
)
print(json.dumps({{"hpo_round_robin": summary, "rerank_model": "{_RERANK_MODEL}"}}, sort_keys=True), flush=True)
"""
    try:
        run_colab_exec_stream(SESSION, script, timeout=8 * 3600 * 3, log_name="training_hpo")
    finally:
        # The VM is normally stopped by main() immediately after this
        # returns/raises. Keep the pointer files locally so a later
        # --resume-hpo can restore the DVC objects on a fresh VM.
        try:
            _mirror_hpo_resume_pointers()
        except Exception as exc:
            print(f"[warn] could not mirror HPO resume pointers: {exc}", file=sys.stderr, flush=True)


def run_sims_deberta() -> None:
    """The deberta zero-shot lane (GPU-only) on the VM."""
    print("[run] zero_shot_sims --models deberta_v3_base on the VM (GPU) ...")
    script = _BOOTSTRAP + f"""
import subprocess, sys
rc = subprocess.run([sys.executable, "{REMOTE_ROOT}/src/training/zero_shot_sims.py",
                     "--models", "deberta_v3_base"]).returncode
if rc != 0:
    raise RuntimeError(f"zero-shot similarity subprocess failed (rc={{rc}})")
"""
    run_colab_exec_stream(SESSION, script, timeout=2 * 3600, log_name="03_sims_deberta")


def _list_remote(pattern_dir: str) -> list[str]:
    """List remote files via a stdin-exec (same channel the lanes use)."""
    import json as _json

    script = (
        "import pathlib, json\n"
        f"files = sorted(str(p) for p in pathlib.Path('{pattern_dir}').rglob('*') if p.is_file())\n"
        "print('@@FILES@@' + json.dumps(files))\n"
    )
    proc = subprocess.Popen(
        ["colab", "exec", "-s", SESSION],
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
        colab("stop", "-s", SESSION, check=False)
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
        "--refresh-data",
        action="store_true",
        help="explicitly regenerate frozen CSV inputs before training",
    )
    ap.add_argument("--keep-alive", action="store_true",
                    help="do not tear down the VM on completion/failure")
    args = ap.parse_args()

    GPU = args.gpu

    if args.what == "stop":
        stop()
        return

    start_live_log()
    check_colab_cli()

    try:
        ensure_session()
        prepare_remote_layout()
        install_deps()
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
            run_train(args.train_frac, _SMOKE_EPOCHS, sample=_SMOKE_SAMPLE, workers=_TRAIN_WORKERS)
        elif args.what == "hpo":
            run_hpo(args.hpo_mode, resume=args.resume_hpo)
        else:
            run_train(
                args.train_frac, args.epochs, sample=args.sample, workers=args.workers,
                resume_run=args.resume_run,
            )
        print("[dvc] remote artifacts are authoritative; local download disabled", flush=True)
    finally:
        # Default behavior is to aggressively teardown to prevent quota burning.
        if not args.keep_alive:
            stop()
        else:
            print("\n[info] --keep-alive specified. VM is still running.")
        close_live_log()

    print("\n[done] artifacts persisted to the configured DVC remote")


if __name__ == "__main__":
    main()
