"""Consolidated Colab lane: one base class, one CPU lane, one GPU lane.

Modeled on the kaggle lane (docs/kaggle-lane.md): a single cohesive owner
for transport dial-ins, staging, receipts, transcript logging, and the
CPU/GPU lane split.  The former three-file Colab layout —

  * cli.colab            — GPU launcher surface (train/tracks/hpo/sims/
                           mixed/smoke/stop) plus the legacy ``--what bundle``
                           upload lane (owner ruling 910ee17 keeps that flow
                           in place, byte-identical),
  * cli.colab_bundle     — standalone committed-export CPU bundle lane
                           (copy-assembled from cli.colab machinery),
  * cli.colab_data_bundle_prep — CPU data-bundle prep lane layered on
                           cli.colab.run_bundle,

was copy-assembly over an ever-growing import + monkeypatch surface.  This
module is the class SSOT those files delegate to.

Structure:

  ColabLaneBase          — transport dial-ins + receipts + the shared lane
                           contracts (transit-failure tolerance, delivery
                           root, delivery member list, checkout-relative
                           guard).  Every dial-in re-reads the ``cli.colab``
                           module attribute at call time, so the facades and
                           the offline test fakes keep driving every lane
                           through one patch surface.
  ColabCPULane           — the committed-export delivery lane (from
                           cli.colab_bundle) and the CPU data-bundle prep
                           parity lane (from cli.colab_data_bundle_prep).
  ColabGPULane           — the accelerator/retention boundary gates
                           cli.colab.main enforces before provisioning.

Envelope/telemetry/receipt/paths conventions follow the kaggle lane where
the concerns overlap (receipt events, sha256 digests, polling helpers,
plan-mode dry runs).  No default flips anywhere: the facades keep their
commands, flags, and byte-identical outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from core.common import TRAIN_ROOT, training_cfg
from cli import log_capture
from core.schemas import ColabBundlePlan


DELIVERY_ARCHIVE_NAME = "bundle_delivery.tar.zst"
RESUME_STATE_ARCHIVE = "resume_state.tar.zst"
RESUME_FROM_CHOICES = ("dedupe", "validation", "full_bundle", "suite_inputs")
DELIVERY_DATA_MEMBERS = (
    "data/canonical_records.csv",
    "data/gate_results.csv",
    "data/dataset_deduped.csv",
    "data/labeled_pairs.csv",
    "data/final_validation.csv",
    "data/number_tokens_reference.csv",
)
DELIVERY_TRACKED_DIRS = ("track_setup",)
DELIVERY_PREPARED_DIRS = ("full", "smoke_200")
PREPARE_BUDGET_SECONDS = 4 * 3600
PREPARE_LOG_NAME = "prepare_bundle.log"
PREPARE_STATUS_NAME = "prepare_bundle.status"
BUNDLE_LAUNCH_TIMEOUT_SECONDS = 300
BUNDLE_DELIVERY_TIMEOUT_SECONDS = 1800
RESUME_STATE_UPLOAD_TIMEOUT_SECONDS = 3600
MAX_PARALLEL_PREP_SESSIONS = 2




def _stamp() -> str:
    """Bracketed Europe/Paris (CET/CEST) wall-clock prefix for output."""
    return (f"[colab-lane {datetime.now(ZoneInfo('Europe/Paris')):%Y-%m-%dT%H:%M:%S %Z}]")




class ColabLaneBase:
    """Shared Colab lane core: transport dial-ins, receipts, lane contracts.

    The ``cli.colab`` module stays the single transport surface the offline
    fakes patch; every dial-in here re-reads the module attribute at call
    time instead of capturing a bound function, so nothing is frozen at
    construction.
    """

    kind = "base"

    def __init__(self, *, surface: Any = None) -> None:
        self._surface = surface

    @property
    def surface(self) -> Any:
        if self._surface is None:
            import cli.colab as surface

            self._surface = surface
        return self._surface

    @property
    def session(self) -> str:
        return self.surface.SESSION

    @property
    def remote_root(self) -> str:
        return self.surface.REMOTE_ROOT

    @property
    def training_results(self) -> Path:
        return self.surface.TRAINING_RESULTS

    def exec_stream(self, *args: Any, **kwargs: Any) -> None:
        return self.surface.run_colab_exec_stream(*args, **kwargs)

    def exec_capture(self, *args: Any, **kwargs: Any) -> Any:
        return self.surface.run_colab_exec_capture(*args, **kwargs)

    def upload_with_retries(self, *args: Any, **kwargs: Any) -> Any:
        return self.surface._upload_with_retries(*args, **kwargs)

    def download_with_visibility(self, **kwargs: Any) -> Any:
        return self.surface._download_file_with_visibility(**kwargs)

    def result_event(self, *args: Any, **kwargs: Any) -> None:
        return self.surface._result_event(*args, **kwargs)

    def parse_remote_json(self, output: str) -> dict:
        return self.surface._parse_remote_json(output)

    def transit_fatal(self, detail: str) -> bool:
        lowered = str(detail).lower()
        return (
            "connection was lost" in lowered
            or f"session '{self.session}' not found".lower() in lowered
        )

    @staticmethod
    def checkout_relative_guard(value: str, *, message: str) -> None:
        candidate = Path(value)
        outside = (candidate.is_absolute() or ".." in candidate.parts
                   or not candidate.parts or len(candidate.parts) != 1
                   or any(char in str(value) for char in "\n\r\\*?[]!{}"))
        if outside:
            raise ValueError(message)

    def delivery_root(self, run_id: str) -> Path:
        """SSOT of the TRAINING_RESULTS delivery root (commit 4d40d1e)."""
        return self.training_results / ("colab_bundle_" + run_id)

    def export_digest(self, dataset_csv: Path) -> str:
        digest = hashlib.sha256()
        with Path(dataset_csv).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def bundle_head(self) -> str:
        remote_root = self.remote_root
        return self.surface._BOOTSTRAP + f"""
import glob, os, subprocess, sys, tarfile

root = "{remote_root}"
"""


class ColabCPULane(ColabLaneBase):
    """CPU lanes: committed-export delivery + data-bundle prep parity."""

    kind = "cpu"

    def cohort_label(self, dataset_csv: Path) -> str:
        """Per-VM cohort tag: `full`, `50pct`, else the sanitized stem."""
        from cli.colab_data_bundle_prep import cohort_label as facade_label

        return facade_label(dataset_csv)

    def plan(self, dataset_csv: Path | None = None, *, resume_state: Path | None = None,
             resume_run_id: str | None = None, resume_from: str | None = None) -> ColabBundlePlan:
        source = Path(dataset_csv) if dataset_csv is not None else TRAIN_ROOT / "dataset.csv"
        if resume_state is not None and not Path(resume_state).is_file():
            raise FileNotFoundError(f"resume state not found: {resume_state}")
        return ColabBundlePlan(
            kind=self.kind,
            mode="dry-run",
            session=self.session,
            export=str(source),
            cohort=self.cohort_label(source),
            archive=f"{self.remote_root}/{DELIVERY_ARCHIVE_NAME}",
            resume_from=resume_from,
            resume_run_id=resume_run_id,
        )

    def provision(self, dataset_csv: Path | None = None) -> None:
        """cli.colab main()'s provisioning order for a full-runtime CPU lane."""
        surface = self.surface
        from core.common import resolve_model

        surface.check_colab_cli()
        surface.ensure_session()
        root = TRAIN_ROOT.resolve()
        chosen = dataset_csv if dataset_csv is not None else TRAIN_ROOT / "dataset.csv"
        export = chosen.name
        if export not in training_cfg().kaggle.export_csvs:
            raise ValueError(
                "bundle exports must be committed config kaggle.export_csvs "
                f"entries (no upload exists on this lane); got {export!r}")
        checkout_paths = (
            Path(resolve_model(training_cfg().training.base_model)),
            TRAIN_ROOT / "data/prepared/smoke_200",
            TRAIN_ROOT / export,
        )
        surface.prepare_remote_layout(minimal_runtime=True, sparse_paths=tuple(
            path.resolve().relative_to(root).as_posix() for path in checkout_paths))
        surface.install_deps(minimal_runtime=True)

    def launch_prepare_script(self) -> str:
        remote_root = self.remote_root
        return self.bundle_head() + f"""
import json, pathlib, shlex, shutil

# Cohort remap copies the committed export the sparse checkout carries onto
# dataset.csv — the kaggle lane's proven contract (cli.kaggle_lane). The
# clone is the only source of bytes this lane consumes.
chosen = pathlib.Path(root) / "@COHORT_EXPORT@"
if not chosen.is_file():
    raise SystemExit("committed cohort export absent from the checkout: @COHORT_EXPORT@")
if chosen.name != "dataset.csv":
    shutil.copy2(chosen, pathlib.Path(root) / "dataset.csv")
    print("[bundle-cpu] cohort remap: @COHORT_EXPORT@ -> dataset.csv", flush=True)

base = pathlib.Path(root)
log_path, status_path = base / "{PREPARE_LOG_NAME}", base / "{PREPARE_STATUS_NAME}"
command_args = [sys.executable, "-u", "-m", "training.prepare_all"LAUNCH_ARGS_LIST]
command = " ".join(shlex.quote(part) for part in command_args)
wrapped = (
    "echo '[prepare-process] starting pid=$$'; "
    "echo '[prepare-process] resource snapshot before prepare'; free -h || true; "
    "timeout --signal=TERM --kill-after=60 {PREPARE_BUDGET_SECONDS} " + command + "; rc=$?; "
    "echo '[prepare-process] exited rc='$rc; "
    "echo '[prepare-process] resource snapshot after prepare'; free -h || true; "
    "printf '%s\\\\n' \\"$rc\\" > " + shlex.quote(str(status_path)) + "; exit $rc"
)
env = {{**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(pathlib.Path(root) / "src")}}
with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
    child = subprocess.Popen(["/bin/bash", "-lc", wrapped], cwd=root, env=env,
        stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True)
print("[prepare] pid=%d" % child.pid, flush=True)
"""

    def poll_prepare_log(self, deadline_seconds: int) -> None:
        """Stream the VM-side prepare log into both local transcripts."""
        surface = self.surface
        remote_root = self.remote_root
        session = self.session
        started = time.monotonic()
        offset = 0
        transit_budget = surface._PROBE_RETRIES
        while True:
            probe = self.bundle_head() + f"""
import json, os

root = "{remote_root}"
log_path = root + "/{PREPARE_LOG_NAME}"
status_path = root + "/{PREPARE_STATUS_NAME}"
offset = {offset}
payload = {{"offset": offset, "chunk": "", "status": None}}
try:
    with open(log_path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    payload["offset"] = offset + len(data)
    payload["chunk"] = data.decode("utf-8", errors="replace")
    if os.path.exists(status_path):
        payload["status"] = int(open(status_path).read().strip() or "-1")
except FileNotFoundError:
    pass
print(json.dumps(payload), flush=True)
"""
            try:
                payload = self.parse_remote_json(
                    self.exec_capture(
                        session, probe, timeout=surface._PROBE_TIMEOUT_SECONDS,
                        training_output=True,
                    )
                )
            except RuntimeError as exc:
                if self.transit_fatal(exc):
                    raise
                transit_budget -= 1
                if transit_budget <= 0:
                    raise
                message = f"[probe] prepare log unavailable; continuing: {exc}"
                surface._write_training_log(message + "\n")
                print(_stamp(), message, flush=True)
                time.sleep(surface._LOG_POLL_SECONDS)
                continue
            transit_budget = surface._PROBE_RETRIES
            chunk = payload["chunk"]
            if chunk:
                for line in chunk.splitlines():
                    surface._write_training_log(f"[prepare] {line}\n")
                    print(f"[prepare] {line}", flush=True)
                offset = int(payload["offset"])
            status = payload["status"]
            if status is not None:
                if int(status) != 0:
                    raise RuntimeError(
                        f"prepare_all failed on the VM (rc={status}); see the [prepare] log above")
                return
            if time.monotonic() - started > deadline_seconds:
                raise RuntimeError(f"prepare poll deadline exceeded ({deadline_seconds}s)")
            time.sleep(surface._LOG_POLL_SECONDS)

    def delivery_segment(self) -> str:
        return f"""
# delivery: run dir + regenerated data artifacts (list from the 8ddc614 lane).
run_dir = sorted(glob.glob(root + "/results/training_prep/*"))[-1]
delivery = root + "/{DELIVERY_ARCHIVE_NAME}"
from core.archive_reader import tar_archive
with tar_archive(delivery, "w") as tar:
    tar.add(run_dir, arcname="training_prep/" + os.path.basename(run_dir))
    for rel in {DELIVERY_DATA_MEMBERS!r}:
        if os.path.exists(root + "/" + rel):
            tar.add(root + "/" + rel, arcname=rel)
    for name in {DELIVERY_TRACKED_DIRS!r}:
        member = root + "/data/" + name
        if os.path.isdir(member):
            tar.add(member, arcname="data/" + name)
    for name in {DELIVERY_PREPARED_DIRS!r}:
        member = root + "/data/prepared/" + name
        if os.path.isdir(member):
            tar.add(member, arcname="data/prepared/" + name)
print("[bundle] delivery archive ready", flush=True)
"""

    def delivery_script(self) -> str:
        return self.bundle_head() + self.delivery_segment()

    def run_delivery(self, dataset_csv: Path, *, resume_from: str | None = None,
                     resume_run_id: str | None = None,
                     resume_state: Path | None = None) -> None:
        """Prepare on the VM CPU from the cloned cohort export, download the delivery."""
        surface = self.surface
        source = Path(dataset_csv)
        if resume_state is not None and not Path(resume_state).is_file():
            raise FileNotFoundError(f"resume state not found: {resume_state}")
        run_id = surface._lane_run_stamp()
        print(
            _stamp(),
            f"[bundle] lane={run_id} cohort export {source.name} rides the sparse "
            f"checkout (no upload; the clone carries the bytes)",
            flush=True,
        )
        script = self.launch_prepare_script().replace(
            "LAUNCH_ARGS_LIST", "").replace("@COHORT_EXPORT@", source.name)
        if resume_state is not None:
            resume_state = Path(resume_state)
            print(
                _stamp(),
                f"[bundle] resume: uploading {resume_state} -> "
                f"{self.remote_root}/{RESUME_STATE_ARCHIVE} ...",
                flush=True,
            )
            self.result_event(run_id, "upload", "started", file=str(resume_state))
            self.upload_with_retries(
                resume_state,
                f"{self.remote_root}/{RESUME_STATE_ARCHIVE}",
                timeout=RESUME_STATE_UPLOAD_TIMEOUT_SECONDS,
            )
            self.result_event(
                run_id, "upload", "completed",
                remote=f"{self.remote_root}/{RESUME_STATE_ARCHIVE}"
            )
            if resume_run_id is None or resume_from is None:
                raise ValueError(
                    "--resume-state requires --resume-run-id and --resume-from "
                    "(the frozen run id and the prepare_all --resume-from choice)"
                )
            script = self.launch_prepare_script().replace(
                "LAUNCH_ARGS_LIST",
                ', "--run-dir", root + "/results/training_prep/@RESUME_RUN_ID@",'
                ' "--resume-from", "@RESUME_FROM@"'
            ).replace("@RESUME_RUN_ID@", resume_run_id).replace(
                "@RESUME_FROM@", resume_from).replace("@COHORT_EXPORT@", source.name)
            print(
                _stamp(),
                f"[bundle] resume: prepare_all --run-dir "
                f"{self.remote_root}/results/training_prep/{resume_run_id} "
                f"--resume-from {resume_from}",
                flush=True,
            )
        self.exec_stream(
            self.session, script, timeout=BUNDLE_LAUNCH_TIMEOUT_SECONDS,
            log_name="bundle_launch", training_output=True,
        )
        self.poll_prepare_log(deadline_seconds=PREPARE_BUDGET_SECONDS)
        self.exec_stream(
            self.session, self.delivery_script(),
            timeout=BUNDLE_DELIVERY_TIMEOUT_SECONDS, log_name="bundle_delivery",
            training_output=True,
        )
        local_base = self.delivery_root(run_id)
        local_base.mkdir(parents=True, exist_ok=True)
        self.result_event(run_id, "download", "started", workers=1)
        local = local_base / DELIVERY_ARCHIVE_NAME
        self.download_with_visibility(
            remote=f"{self.remote_root}/{DELIVERY_ARCHIVE_NAME}",
            local=local,
            worker=None,
            index=1,
            total=1,
            run_id=run_id,
        )
        self.result_event(run_id, "download", "completed", archive=str(local),
                          destination=str(local_base))
        print(_stamp(), f"[bundle] delivered -> {local}", flush=True)
        print(_stamp(), f"[bundle] {run_id} complete; the VM session stays open", flush=True)

    def resume_args(self, args: argparse.Namespace) -> dict:
        """Validate/echo the resume triplet; empty dict is an exact fresh run."""
        provided = [args.resume_from is not None, args.resume_state is not None,
                    args.resume_run_id is not None]
        if any(provided) and not all(provided):
            raise ValueError(
                "--resume-from/--resume-state/--resume-run-id must be provided "
                "together (they describe one frozen run) or not at all"
            )
        if args.resume_from is not None:
            if args.resume_from not in RESUME_FROM_CHOICES:
                raise ValueError(
                    f"unknown --resume-from {args.resume_from!r}; prepare_all "
                    f"accepts {sorted(RESUME_FROM_CHOICES)}"
                )
            self.checkout_relative_guard(
                args.resume_run_id,
                message="resume run id must be a single plain training_prep "
                        f"run-directory name: {args.resume_run_id!r}",
            )
            return {"resume_from": args.resume_from, "resume_run_id": args.resume_run_id,
                    "resume_state": args.resume_state}
        return {}

    @contextmanager
    def dual_transcript_streaming(self, transport: Any):
        """Force the shared transport's training_output for one prep run."""
        original = transport.run_colab_exec_stream

        def dual_transcript(*args, **kwargs):
            kwargs["training_output"] = True
            return original(*args, **kwargs)

        transport.run_colab_exec_stream = dual_transcript
        try:
            yield
        finally:
            transport.run_colab_exec_stream = original

    def qualify_session_transcripts(self, file_map: dict, session: str) -> None:
        """Per-session root/training transcript paths for the 2-parallel cap."""
        # One roof (owner order 2026-10-07): per-session transcripts live in
        # the canonical logs root, colab lane subdir.
        file_map["colab_live_log"] = log_capture.lane_log(
            "colab", f"colab_system_{session}.log")
        file_map["colab_training_log"] = log_capture.lane_log(
            "colab", f"training_{session}.log")

    def run_cpu_prep(self, dataset_csv: Path | None = None) -> None:
        """One CPU prep run with the parity capabilities layered on top of
        cli.colab.run_bundle (the untouched CSV-to-inputs lifecycle)."""
        from core.common import DATA_PATH

        surface = self.surface
        source = Path(dataset_csv) if dataset_csv is not None else Path(DATA_PATH)
        print(
            _stamp(),
            f"[cpu-prep] cohort={self.cohort_label(source)} dataset={source.name} "
            f"sha256={self.export_digest(source)[:12]} "
            f"max_parallel_sessions={MAX_PARALLEL_PREP_SESSIONS}",
            flush=True,
        )
        with self.dual_transcript_streaming(surface):
            surface.run_bundle(dataset_csv=source)


class ColabGPULane(ColabLaneBase):
    """GPU lane boundary gates, enforced by cli.colab.main before provisioning."""

    kind = "gpu"

    def enforce_boundary(
        self, gpu: str, *, allow_gpu: bool = False, keep_alive: bool = False,
    ) -> None:
        if gpu.upper() != "CPU" and not allow_gpu:
            raise ValueError("GPU launch requires --allow-gpu")
        if keep_alive and gpu.upper() != "CPU":
            raise ValueError(
                "--keep-alive is CPU-only (a retained GPU VM consumes accelerator "
                f"quota indefinitely); requested --gpu {gpu}. Use --gpu CPU or drop "
                "--keep-alive."
            )
