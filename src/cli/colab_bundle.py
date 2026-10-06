"""Standalone CPU bundle lane: one raw export -> one delivery archive on the VM.

Copy-assembled from src/cli/colab.py's working machinery (colab.py itself is
untouched):

  provisioning order ... cli.colab main(), sims path (check_colab_cli ->
  ensure_session -> prepare_remote_layout(minimal_runtime=False) ->
  install_deps(minimal_runtime=False)); the VM is never stopped (no stop()
  call in this module — the session stays open by owner directive).

  upload ............... colab.py:3856 `_upload_with_retries(source,
  f"{REMOTE_ROOT}/dataset.csv", timeout=3600)` (REMOTE_ROOT-prefixed remote
  path, like every other working upload).

  remote exec .......... run_sims()'s shape (colab.py:3923): _BOOTSTRAP +
  a plain script string (module-level f-string, _BOOTSTRAP's own
  mechanism, body brace-free so no escaping is needed), then
  run_colab_exec_stream(SESSION, script, timeout=..., log_name=..., training_output=True).

  prepare_all on the VM . one detached "worker" using the train lanes'
  proven POPEN+file telemetry contract: bash-wrapped
  training.prepare_all with stdout/stderr in prepare_bundle.log
  (unbuffered, line-flushed), rc recorded in prepare_bundle.status via
  start_new_session=True so the launch exec returns; the lane POLLs the
  log locally (train-lane offset/chunk probe, training_output=True) and
  every tqdm bar streams to both transcripts. on rc!=0 the
  RuntimeError follows the prepare log tail (working-code inheritance,
  not captured-file buffering).

  delivery archive ..... the member list copied verbatim from the previous
  attempt in git history (git show 8ddc614:src/cli/colab.py, run_bundle
  remote section): remote tar at REMOTE_ROOT/bundle_delivery.tar.zst.

  resume ................. upload, extract, prep pattern (same
  _upload_with_retries call shape as the raw export, colab.py:3856): the
  frozen state uploads to REMOTE_ROOT/resume_state.tar.zst and is extracted
  with tarfile.extractall(REMOTE_ROOT) BEFORE the pin re-write; prepare_all
  then runs --run-dir REMOTE_ROOT/results/training_prep/<frozen id>
  --resume-from <choice> (runtime values injected via @placeholders@, never
  baked into a module constant).

  download ............. download_verified_training_results' download block
  (colab.py:2021-2033): local_base = TRAINING_RESULTS / run_id; local =
  local_base / archive name; _download_file_with_visibility(remote=...,
  local=..., worker=None, index=1, total=1, run_id=run_id) — the exact root
  contract that machinery enforces with
  local.relative_to(TRAINING_RESULTS / run_id).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.common import TRAIN_ROOT

import cli.colab as _colab
from cli.colab import (
    REMOTE_ROOT,
    SESSION,
    TRAINING_RESULTS,
    _BOOTSTRAP,
    _download_file_with_visibility,
    _lane_run_stamp,
    _parse_remote_json,
    _result_event,
    _timed_colab,
    _upload_with_retries,
    check_colab_cli,
    ensure_session,
    install_deps,
    prepare_remote_layout,
    run_colab_exec_capture,
    run_colab_exec_stream,
)

# The delivery archive keeps the VM-side fixed name from the 8ddc614 lane; a
# rerun on a live VM overwrites it (the download consumes it per invocation).
_ARCHIVE_NAME = "bundle_delivery.tar.zst"

# Resume uploads keep a fixed VM-side name too (extracted before the pin
# re-write, then the prepare_all CLI continues the frozen run in place).
_RESUME_STATE_ARCHIVE = "resume_state.tar.zst"

# prepare_all's own argparse choices (src/training/prepare_all.py main());
# copied locally so @RESUME_FROM@ substitution can never bend the emitted
# script while prepare_all fails loudly for anything it does not accept.
_RESUME_FROM_CHOICES = ("dedupe", "validation", "full_bundle", "suite_inputs")

# _BOOTSTRAP (cli.colab:2446) is a module-level f-string interpolating
# REMOTE_ROOT into a triple-quoted body; THAT mechanism is what this module
# copies. Every segment below is a module-level f-string with a brace-free
# body (no escaping, no brace collisions). The resume segment's runtime
# values are injected by .replace on the @RESUME_RUN_ID@/@RESUME_FROM@
# placeholders, exactly like the earlier directive required ("plain string,
# placeholder-replaced, brace-free, no regex/backslashes").
_BUNDLE_HEAD = _BOOTSTRAP + f"""
import glob, os, subprocess, sys, tarfile

root = "{REMOTE_ROOT}"
"""
# One prepare_all "worker" using the train lanes' POPEN+file telemetry
# contract verbatim (cli.colab run_train launcher, cli.colab:1408-1434):
# bash-wrapped command, `printf rc > status`, all output (tqdm bars on
# stderr included) into one log via Popen(start_new_session=True) so the
# kernel exec returns and the lane POLLs the log locally — the proven
# capture path. No multi-hour held-open exec, no stream loss.
_PREPARE_BUDGET_SECONDS = 4 * 3600
_PREPARE_LOG = "prepare_bundle.log"
_PREPARE_STATUS = "prepare_bundle.status"
_LAUNCH_PREPARE_SCRIPT = _BUNDLE_HEAD + f"""
import json, pathlib, shlex

base = pathlib.Path(root)
log_path, status_path = base / "{_PREPARE_LOG}", base / "{_PREPARE_STATUS}"
command_args = [sys.executable, "-u", "-m", "training.prepare_all"LAUNCH_ARGS_LIST]
command = " ".join(shlex.quote(part) for part in command_args)
wrapped = (
    "echo '[prepare-process] starting pid=$$'; "
    "echo '[prepare-process] resource snapshot before prepare'; free -h || true; "
    "timeout --signal=TERM --kill-after=60 {_PREPARE_BUDGET_SECONDS} " + command + "; rc=$?; "
    "echo '[prepare-process] exited rc='$rc; "
    "echo '[prepare-process] resource snapshot after prepare'; free -h || true; "
    "printf '%s\\\\n' \\"$rc\\" > " + shlex.quote(str(status_path)) + "; exit $rc"
)
env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(root / "src")}}
with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
    child = subprocess.Popen(["/bin/bash", "-lc", wrapped], cwd=root, env=env,
        stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True)
print("[prepare] pid=%d" % child.pid, flush=True)
"""
# prepare_all's lifecycle budget (the launch bash `timeout` value, mirroring
# the train lanes' _WORKER_TIMEOUT_SECONDS role; ~50 min locally, far longer
# on a VM CPU, so the full single-exec budget is kept).
_FRESH_LAUNCH_SCRIPT = _LAUNCH_PREPARE_SCRIPT.replace("LAUNCH_ARGS_LIST", "")
_RESUME_LAUNCH_SCRIPT = _LAUNCH_PREPARE_SCRIPT.replace(
    "LAUNCH_ARGS_LIST",
    ', "--run-dir", root + "/results/training_prep/@RESUME_RUN_ID@",'
    ' "--resume-from", "@RESUME_FROM@"')

# Poll cadence + failure shape lifted from the train lanes (cli.colab
# _LOG_POLL_SECONDS and the transient-probe tolerance at cli.colab:1482-1495):
# a transient control-channel hiccup must not kill a run that is alive on
# the VM; only a lost kernel/session is fatal.
_LOG_POLL_SECONDS = _colab._LOG_POLL_SECONDS
_PROBE_TIMEOUT_SECONDS = _colab._PROBE_TIMEOUT_SECONDS
_PROBE_RETRIES = _colab._PROBE_RETRIES


def _poll_prepare_log(deadline_seconds: int) -> None:
    """Stream the VM-side prepare log into both local transcripts.

    Chunk forwarding is the train lanes' contract (cli.colab:1501-1505):
    every new byte — each tqdm \r update included — is printed locally and
    written to the durable training log.  The loop exits when the detached
    process records its rc in the status file; any nonzero rc fails loudly.
    """
    import time as _time

    from cli.colab import _write_training_log
    started = _time.monotonic()
    offset = 0
    transit_budget = _PROBE_RETRIES
    while True:
        probe = _BOOTSTRAP + f"""
import json, os

root = "{REMOTE_ROOT}"
log_path = root + "/{_PREPARE_LOG}"
status_path = root + "/{_PREPARE_STATUS}"
offset = {offset}
payload = {{"offset": offset, "chunk": "", "status": None}}
try:
    with open(log_path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    payload["offset"] = offset + len(data)
    # Decoded before the JSON envelope, exactly like the train lanes' probe
    # (cli.colab:1460-1461): raw bytes are not JSON-serializable, and the
    # decoded text is what the forwarding loop prints.
    payload["chunk"] = data.decode("utf-8", errors="replace")
    if os.path.exists(status_path):
        payload["status"] = int(open(status_path).read().strip() or "-1")
except FileNotFoundError:
    pass
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
            detail = str(exc).lower()
            if ("connection was lost" in detail
                    or f"session '{SESSION}' not found".lower() in detail):
                raise
            transit_budget -= 1
            if transit_budget <= 0:
                raise
            message = f"[probe] prepare log unavailable; continuing: {exc}"
            _write_training_log(message + "\n")
            print(message, flush=True)
            _time.sleep(_LOG_POLL_SECONDS)
            continue
        transit_budget = _PROBE_RETRIES
        chunk = payload["chunk"]
        if chunk:
            # Both-transcript forwarding, exactly like worker telemetry
            # (cli.colab:1501-1505).
            for line in chunk.splitlines():
                _write_training_log(f"[prepare] {line}\n")
                print(f"[prepare] {line}", flush=True)
            offset = int(payload["offset"])
        status = payload["status"]
        if status is not None:
            if int(status) != 0:
                raise RuntimeError(
                    f"prepare_all failed on the VM (rc={status}); see the [prepare] log above")
            return
        if _time.monotonic() - started > deadline_seconds:
            raise RuntimeError(f"prepare poll deadline exceeded ({deadline_seconds}s)")
        _time.sleep(_LOG_POLL_SECONDS)


_RESUME_EXTRACT_SEGMENT = f"""
# resume: extract the owner-built frozen state
# (derived CSVs + second04 pairs + data_prep/labeled_pairs stage manifests,
# REMOTE_ROOT-relative inside the tarball).
from core.archive_reader import tar_archive
with tar_archive(root + "/resume_state.tar.zst") as tar:
    tar.extractall(root, filter="data")
print("[resume] frozen state extracted at " + root, flush=True)
"""
_DELIVERY_SEGMENT = f"""
# delivery: run dir + regenerated data artifacts (list from the 8ddc614 lane).
run_dir = sorted(glob.glob(root + "/results/training_prep/*"))[-1]
delivery = root + "/bundle_delivery.tar.zst"
from core.archive_reader import tar_archive
with tar_archive(delivery, "w") as tar:
    tar.add(run_dir, arcname="training_prep/" + os.path.basename(run_dir))
    for rel in ("data/canonical_records.csv", "data/gate_results.csv",
                "data/dataset_deduped.csv", "data/labeled_pairs.csv",
                "data/final_validation.csv", "data/number_tokens_reference.csv"):
        if os.path.exists(root + "/" + rel):
            tar.add(root + "/" + rel, arcname=rel)
    for name in ("track_setup",):
        member = root + "/data/" + name
        if os.path.isdir(member):
            tar.add(member, arcname="data/" + name)
    for name in ("full", "smoke_200"):
        member = root + "/data/prepared/" + name
        if os.path.isdir(member):
            tar.add(member, arcname="data/prepared/" + name)
print("[bundle] delivery archive ready", flush=True)
"""
_FINALIZE_SCRIPT = _BUNDLE_HEAD + _DELIVERY_SEGMENT


def _provision() -> None:
    """cli.colab main()'s provisioning order for a full-runtime CPU lane."""
    from core.common import resolve_model, training_cfg

    check_colab_cli()
    ensure_session()
    # The bundle lane regenerates every derived artifact from the uploaded
    # export (same git-inputs pattern the tracks lane uses for suite inputs):
    # the checkout carries only what the prep reads — src/, config/,
    # scripts/, requirements, the text checkpoint and the tracked smoke
    # inputs — never the stale derived CSVs or result archives on the branch.
    root = TRAIN_ROOT.resolve()
    checkout_paths = (
        Path(resolve_model(training_cfg().training.base_model)),
        TRAIN_ROOT / "data/prepared/smoke_200",
    )
    prepare_remote_layout(minimal_runtime=True, sparse_paths=tuple(
        path.resolve().relative_to(root).as_posix() for path in checkout_paths))
    install_deps(minimal_runtime=True)


@_timed_colab("step")
def run_bundle(
    dataset_csv: Path,
    *,
    resume_from: str | None = None,
    resume_run_id: str | None = None,
    resume_state: Path | None = None,
) -> None:
    """Upload the export, re-pin, prepare on the VM CPU, download the delivery.

    With --resume-from/--resume-run-id/--resume-state the frozen state tarball
    is uploaded as REMOTE_ROOT/resume_state.tar.zst and extracted at REMOTE_ROOT
    BEFORE the pin re-write, and prepare_all continues the frozen run via
    --run-dir/--resume-from; the delivery/download flow is unchanged.
    """
    source = Path(dataset_csv)
    if not source.is_file():
        raise FileNotFoundError(f"raw export not found: {source}")
    if resume_state is not None and not Path(resume_state).is_file():
        raise FileNotFoundError(f"resume state not found: {resume_state}")
    run_id = _lane_run_stamp()
    print(
        f"[bundle] lane={run_id} uploading raw export {source} -> "
        f"{REMOTE_ROOT}/dataset.csv ...",
        flush=True,
    )
    _result_event(run_id, "upload", "started", file=str(source))
    _upload_with_retries(source, f"{REMOTE_ROOT}/dataset.csv", timeout=3600)
    _result_event(run_id, "upload", "completed", remote=f"{REMOTE_ROOT}/dataset.csv")
    script = _FRESH_LAUNCH_SCRIPT
    if resume_state is not None:
        resume_state = Path(resume_state)
        print(
            f"[bundle] resume: uploading {resume_state} -> {REMOTE_ROOT}/{_RESUME_STATE_ARCHIVE} ...",
            flush=True,
        )
        _result_event(run_id, "upload", "started", file=str(resume_state))
        _upload_with_retries(
            resume_state, f"{REMOTE_ROOT}/{_RESUME_STATE_ARCHIVE}", timeout=3600
        )
        _result_event(
            run_id, "upload", "completed", remote=f"{REMOTE_ROOT}/{_RESUME_STATE_ARCHIVE}"
        )
        if resume_run_id is None or resume_from is None:
            raise ValueError(
                "--resume-state requires --resume-run-id and --resume-from "
                "(the frozen run id and the prepare_all --resume-from choice)"
            )
        script = _RESUME_LAUNCH_SCRIPT.replace(
            "@RESUME_RUN_ID@", resume_run_id
        ).replace("@RESUME_FROM@", resume_from)
        print(
            f"[bundle] resume: prepare_all --run-dir {REMOTE_ROOT}/results/training_prep/{resume_run_id} "
            f"--resume-from {resume_from}",
            flush=True,
        )
    # Launch the detached prepare (a quick exec), then POLL the VM-side log —
    # the train lanes' proven capture: every new byte, each tqdm \r update
    # included, streams to both transcripts until the status file records
    # the rc.  No multi-hour kernel exec is held open, so a silenced stream
    # can no longer hide the run's progress.
    run_colab_exec_stream(
        SESSION, script, timeout=300, log_name="bundle_launch", training_output=True
    )
    _poll_prepare_log(deadline_seconds=_PREPARE_BUDGET_SECONDS)
    run_colab_exec_stream(
        SESSION, _FINALIZE_SCRIPT, timeout=1800, log_name="bundle_delivery",
        training_output=True,
    )
    # Download block copied from download_verified_training_results
    # (cli.colab:2021-2033): the local root is TRAINING_RESULTS / run_id and
    # the SAME run_id goes to _download_file_with_visibility, which resolves
    # display/event paths with local.relative_to(TRAINING_RESULTS / run_id).
    local_base = TRAINING_RESULTS / run_id
    local_base.mkdir(parents=True, exist_ok=True)
    _result_event(run_id, "download", "started", workers=1)
    local = local_base / _ARCHIVE_NAME
    _download_file_with_visibility(
        remote=f"{REMOTE_ROOT}/bundle_delivery.tar.zst",
        local=local,
        worker=None,
        index=1,
        total=1,
        run_id=run_id,
    )
    _result_event(run_id, "download", "completed", archive=str(local), destination=str(local_base))
    print(f"[bundle] delivered -> {local}", flush=True)
    print(f"[bundle] {run_id} complete; the VM session stays open", flush=True)


def _resume_args(args: argparse.Namespace) -> dict:
    """Validate/echo the resume triplet; empty dict is an exact fresh run."""
    provided = [args.resume_from is not None, args.resume_state is not None,
                args.resume_run_id is not None]
    if any(provided) and not all(provided):
        raise ValueError(
            "--resume-from/--resume-state/--resume-run-id must be provided "
            "together (they describe one frozen run) or not at all"
        )
    if args.resume_from is not None:
        if args.resume_from not in _RESUME_FROM_CHOICES:
            raise ValueError(
                f"unknown --resume-from {args.resume_from!r}; prepare_all "
                f"accepts {sorted(_RESUME_FROM_CHOICES)}"
            )
        candidate = Path(args.resume_run_id)
        # Guard copied from cli.colab's runtime-checkout-path validation
        # (colab.py:2261-2266); one extension, because the value lands inside
        # the emitted remote script: braces are refused alongside the other
        # script-breaking characters.
        if (candidate.is_absolute() or ".." in candidate.parts or not candidate.parts
                or len(candidate.parts) != 1
                or any(char in args.resume_run_id for char in "\n\r\\*?[]!{}")):
            raise ValueError(
                "resume run id must be a single plain training_prep "
                f"run-directory name: {args.resume_run_id!r}"
            )
        return {"resume_from": args.resume_from, "resume_run_id": args.resume_run_id,
                "resume_state": args.resume_state}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-csv", type=Path, default=TRAIN_ROOT / "dataset.csv",
                    help="raw export to upload as the VM's dataset.csv (default: "
                         "repo-root dataset.csv, the export the committed pins "
                         "expect)")
    ap.add_argument("--resume-from", default=None,
                    help="prepare_all --resume-from choice for a frozen run "
                         "(dedupe|validation|full_bundle|suite_inputs; e.g. "
                         "validation re-runs the ~45 min token phase onward)")
    ap.add_argument("--resume-state", type=Path, default=None,
                    help="owner-built frozen state tarball (derived CSVs + "
                         "second04 pairs + stage manifests) uploaded and "
                         "extracted at REMOTE_ROOT before the pin re-write")
    ap.add_argument("--resume-run-id", default=None,
                    help="frozen training_prep run id to continue (e.g. "
                         "20261005T162137972376); prepare_all creates the run "
                         "dir itself on the VM")
    args = ap.parse_args()
    # CPU bundle lane: cli.colab main() sets the module-global GPU the same
    # way (GPU = args.gpu, colab.py:4623) and ensure_session reads it at call
    # time; CPU keeps provisioning off the accelerator. The daemon is allowed
    # by main()'s provisioning env (colab.py:4634) and this lane never calls
    # stop() — the owner keeps the session open.
    _colab.GPU = "CPU"
    os.environ["EUROMONITOR_KEEP_ALIVE_ALLOWED"] = "1"
    _provision()
    run_bundle(args.dataset_csv, **_resume_args(args))


if __name__ == "__main__":
    main()
