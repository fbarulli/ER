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

from core.common import TRAIN_ROOT
from cli import log_capture
from core.schemas import ColabBundlePlan
from cli.colab_lane_contracts import (  # noqa: F401
    BUNDLE_DELIVERY_TIMEOUT_SECONDS,
    BUNDLE_LAUNCH_TIMEOUT_SECONDS,
    DELIVERY_ARCHIVE_NAME,
    DELIVERY_DATA_MEMBERS,
    DELIVERY_PREPARED_DIRS,
    DELIVERY_TRACKED_DIRS,
    MAX_PARALLEL_PREP_SESSIONS,
    PREPARE_BUDGET_SECONDS,
    PREPARE_LOG_NAME,
    PREPARE_STATUS_NAME,
    RESUME_FROM_CHOICES,
    RESUME_STATE_ARCHIVE,
    RESUME_STATE_UPLOAD_TIMEOUT_SECONDS,
    ColabLaneBase,
    _stamp,
)
from cli.colab_lane_cpu_provision import ColabCPULaneProvision




class ColabCPULane(ColabCPULaneProvision, ColabLaneBase):
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
