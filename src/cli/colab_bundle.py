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
  run_colab_exec_stream(SESSION, script, timeout=..., log_name=...).

  VM pin re-write ...... a PLAIN LINE FILTER (directive-specified; the deleted
  attempt's two-layer regex was the regression): open(config).readlines(),
  replace the two lines starting with '  source_export_expected_rows:' and
  '  source_export_expected_sha256:' by % formatting, write back, one
  confirmation line with the new rows+sha. Rows come from the same census the
  guard enforces (len of the parsed frame, core.common._validate_source_export);
  sha256 comes from the digest loop copied from colab.py:3224.

  prepare_all on the VM . subprocess.run([sys.executable, "-m",
  "training.prepare_all"], cwd=root) with NO stderr capture, NO log file
  and NO pump thread: stdout and stderr inherit to the streamed cell
  exactly like run_sims' subprocess (cli.colab:3929-3940) — live [out]
  lines carry every tqdm bar and tracebacks stream live; on rc!=0 the
  RuntimeError follows the streamed output (working-code inheritance, not
  captured-file buffering).

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
    _result_event,
    _timed_colab,
    _upload_with_retries,
    check_colab_cli,
    ensure_session,
    install_deps,
    prepare_remote_layout,
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
import glob, hashlib, os, subprocess, sys, tarfile
import pandas as pd

root = "{REMOTE_ROOT}"
csv_path = root + "/dataset.csv"
config_path = root + "/config/training.yaml"
"""
_RESUME_EXTRACT_SEGMENT = f"""
# resume: extract the owner-built frozen state BEFORE the pin re-write
# (derived CSVs + second04 pairs + data_prep/labeled_pairs stage manifests,
# REMOTE_ROOT-relative inside the tarball).
from core.archive_reader import tar_archive
with tar_archive(root + "/resume_state.tar.zst") as tar:
    tar.extractall(root, filter="data")
print("[resume] frozen state extracted at " + root, flush=True)
"""
_PIN_SEGMENT = f"""
# pins: point the audit pins at the uploaded export, plain line filter.
digest = hashlib.sha256()
with open(csv_path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
digest = digest.hexdigest()
rows = len(pd.read_csv(csv_path))
with open(config_path) as handle:
    source_lines = handle.readlines()
out_lines = []
for line in source_lines:
    if line.startswith("  source_export_expected_rows:"):
        out_lines.append("  source_export_expected_rows: %d" % rows + os.linesep)
    elif line.startswith("  source_export_expected_sha256:"):
        out_lines.append('  source_export_expected_sha256: "%s"' % digest + os.linesep)
    else:
        out_lines.append(line)
with open(config_path, "w") as handle:
    handle.writelines(out_lines)
print("[pins] source_export_expected_rows=%d source_export_expected_sha256=%s" % (rows, digest), flush=True)
"""
_PREPARE_FRESH_SEGMENT = f"""
# prepare: run the full CSV-to-inputs preparation; stdout and stderr inherit
# to the streamed cell exactly like run_sims' subprocess (live [out] lines
# with every tqdm bar; tracebacks stream live too).
rc = subprocess.run(
    [sys.executable, "-m", "training.prepare_all"],
    cwd=root,
).returncode
if rc != 0:
    raise RuntimeError("prepare_all failed on the VM (rc=%d); see the streamed output above" % rc)
"""
_PREPARE_RESUME_SEGMENT = f"""
# resume: continue the frozen preparation in place (prepare_all CLI); stdout
# and stderr inherit to the streamed cell (live [out] lines, tqdm bars,
# tracebacks).
# lifecycle: resume_from='validation' re-runs negative_supply+discriminator+
# validation+graph_inputs+full_bundle+suite_inputs+verify_handoff (~45 min
# token phase); the delivery/download flow below then works unchanged.
rc = subprocess.run(
    [sys.executable, "-m", "training.prepare_all",
     "--run-dir", root + "/results/training_prep/@RESUME_RUN_ID@",
     "--resume-from", "@RESUME_FROM@"],
    cwd=root,
).returncode
if rc != 0:
    raise RuntimeError("prepare_all failed on the VM (rc=%d); see the streamed output above" % rc)
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
_BUNDLE_REMOTE_SCRIPT = _BUNDLE_HEAD + _PIN_SEGMENT + _PREPARE_FRESH_SEGMENT + _DELIVERY_SEGMENT
_RESUME_REMOTE_SCRIPT = (
    _BUNDLE_HEAD + _RESUME_EXTRACT_SEGMENT + _PIN_SEGMENT
    + _PREPARE_RESUME_SEGMENT + _DELIVERY_SEGMENT
)


def _provision() -> None:
    """cli.colab main()'s provisioning order for a full-runtime CPU lane."""
    check_colab_cli()
    ensure_session()
    prepare_remote_layout(minimal_runtime=False)
    install_deps(minimal_runtime=False)


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
    script = _BUNDLE_REMOTE_SCRIPT
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
        script = _RESUME_REMOTE_SCRIPT.replace(
            "@RESUME_RUN_ID@", resume_run_id
        ).replace("@RESUME_FROM@", resume_from)
        print(
            f"[bundle] resume: prepare_all --run-dir {REMOTE_ROOT}/results/training_prep/{resume_run_id} "
            f"--resume-from {resume_from}",
            flush=True,
        )
    run_colab_exec_stream(SESSION, script, timeout=4 * 3600, log_name="bundle")
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
