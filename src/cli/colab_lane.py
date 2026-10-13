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

  colab_lane_contracts     — shared constants, the lane stamp, and
                             ColabLaneBase: transport dial-ins + receipts +
                             the shared lane contracts (transit-failure
                             tolerance, delivery root, delivery member list,
                             checkout-relative guard).  Every dial-in
                             re-reads the ``cli.colab`` module attribute at
                             call time, so the facades and the offline test
                             fakes keep driving every lane through one patch
                             surface.
  colab_lane_cpu_provision — the CPU lane's provisioning order and the
                             byte-exact prepare-launch script segments.
  colab_lane_cpu_poll      — the VM prepare-log poll (offset probes,
                             transit tolerance, the one lane transcript, exits).
  colab_lane_cpu_delivery  — the delivery archive, run_delivery's
                             launch/poll/collect/download phases, and the
                             frozen resume-state upload.
  ColabCPULane             — the composition: the committed-export delivery
                             lane (from cli.colab_bundle) and the CPU
                             data-bundle prep parity lane (from
                             cli.colab_data_bundle_prep).
  ColabGPULane             — the accelerator/retention boundary gates
                             cli.colab.main enforces before provisioning.

Envelope/telemetry/receipt/paths conventions follow the kaggle lane where
the concerns overlap (receipt events, size digests, polling helpers,
plan-mode dry runs).  No default flips anywhere: the facades keep their
commands, flags, and byte-identical outputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cli.colab_lane_contracts import (  # noqa: F401
    BUNDLE_DELIVERY_TIMEOUT_SECONDS,
    BUNDLE_LAUNCH_TIMEOUT_SECONDS,
    DELIVERY_ARCHIVE_NAME,
    MAX_PARALLEL_PREP_SESSIONS,
    PREPARE_BUDGET_SECONDS,
    PREPARE_LOG_NAME,
    PREPARE_STATUS_NAME,
    RESUME_FROM_CHOICES,
    RESUME_STATE_ARCHIVE,
    RESUME_STATE_UPLOAD_TIMEOUT_SECONDS,
    ColabDeliveryManifest,
    ColabLaneBase,
    _stamp,
)
from cli.colab_lane_cpu_delivery import ColabCPULaneDelivery
from cli.colab_lane_cpu_poll import ColabCPULanePoll
from cli.colab_lane_cpu_provision import ColabCPULaneProvision
from core.common import TRAIN_ROOT
from core.run_log import RunLogger
from core.schemas import ColabBundlePlan
from training.prepare_all_trace import timed

_LOG = RunLogger(__name__)


def cohort_label(dataset_csv: Path) -> str:
    """Per-VM cohort tag: `full` for the repo-root export, `50pct` for the
    50% cohort, else the sanitized stem.  The lane SSOT the prep facade
    delegates to (no facade round-trip)."""
    name = Path(dataset_csv).name
    if name == "dataset.csv":
        return "full"
    if "50pct" in name:
        return "50pct"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.rsplit(".", 1)[0])


class ColabCPULane(ColabCPULaneDelivery, ColabCPULanePoll, ColabCPULaneProvision, ColabLaneBase):
    """CPU lanes: committed-export delivery + data-bundle prep parity."""

    kind = "cpu"

    @timed
    def cohort_label(self, dataset_csv: Path) -> str:
        """Per-VM cohort tag: `full`, `50pct`, else the sanitized stem."""
        return cohort_label(dataset_csv)

    @timed
    def plan(self, dataset_csv: Path | None = None, *, resume_state: Path | None = None,
             resume_run_id: str | None = None, resume_from: str | None = None) -> ColabBundlePlan:
        source = Path(dataset_csv) if dataset_csv is not None else TRAIN_ROOT / "dataset.csv"
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

    @timed
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

    @timed
    def run_cpu_prep(self, dataset_csv: Path | None = None, *,
                     diagnostic: bool = False) -> None:
        """One CPU prep run with the parity capabilities layered on top of
        cli.colab.run_bundle (the untouched CSV-to-inputs lifecycle).

        ``diagnostic`` is forwarded to the remote prepare so a subsampled run
        emits its artifact without the pinned-evidence guard; production keeps
        the guard (the default).
        """
        from core.common import DATA_PATH

        surface = self.surface
        source = Path(dataset_csv) if dataset_csv is not None else Path(DATA_PATH)
        print(
            _stamp(),
            f"[cpu-prep] cohort={self.cohort_label(source)} dataset={source.name} "
            f"size={self.export_digest(source)} "
            f"max_parallel_sessions={MAX_PARALLEL_PREP_SESSIONS} "
            f"diagnostic={diagnostic}",
            flush=True,
        )
        with self.dual_transcript_streaming(surface), _LOG.section("colab_lane.cpu_prep.run_bundle"):
            surface.run_bundle(dataset_csv=source, diagnostic=diagnostic)

    @timed
    def run_committed_export(self, dataset_csv: Path, *, resume_from: str | None = None,
                             resume_run_id: str | None = None,
                             resume_state: Path | None = None,
                             keep_alive: bool = False) -> None:
        """The committed-export lane under its ONE transcript.

        Provision (class-owned sparse checkout) then deliver, all captured to
        the declared ``ColabSpec`` transcript — the single class entry a
        committed-export facade calls, so the transcript and the provisioning
        order are never re-wired at a call site.

        The VM session is RELEASED on every outcome (owner directive
        2026-10-12, release by default) through the canonical ``stop()``
        surface; ``keep_alive`` retains it explicitly, the same contract
        cli.colab.main enforces.
        """
        with self.transcript():
            try:
                self.provision(dataset_csv)
                self.run_delivery(dataset_csv, resume_from=resume_from,
                                  resume_run_id=resume_run_id, resume_state=resume_state)
            finally:
                if not keep_alive:
                    self.surface.stop()


class ColabGPULane(ColabLaneBase):
    """GPU lane boundary gates, enforced by cli.colab.main before provisioning."""

    kind = "gpu"

    @timed
    def prepared_bundles(self, args: argparse.Namespace) -> list[Path] | None:
        """The ONE prepared-bundle source a GPU launch consumes, or None.

        The CPU bundle lane (``--what bundle --gpu CPU``) runs
        ``training.prepare_all`` on the CPU VM; its ``full_bundle`` stage writes
        exactly the bundles ``config colab.full_prepared_bundles`` names, and
        the delivery installs them here.  A GPU launch REUSES them for the
        unmodified single-worker request and never prepares data itself
        (owner directive: no CPU prep on the GPU lane, none locally either).

        Any other request -- a different model, worker count, profile, loss or
        sampling -- has no CPU-lane bundle, and None is what makes the launch
        refuse it (see :meth:`enforce_data_preparation_boundary`) instead of
        quietly building one here.
        """
        if (
            args.what != "train"
            or args.workers != 1
            or args.model is not None
            or args.sample is not None
            or args.resume_run is not None
            or args.run_label is not None
            or args.dataset_csv is not None
            or args.train_frac != self.surface._TRAIN_FRAC_DEFAULT
            or args.epochs != self.surface._EPOCHS_DEFAULT
            or args.loss != self.surface._TRAIN_LOSS
            or args.masking_profile != self.surface._MASKING_PROFILE
            or args.collapse_guardrail_profile != self.surface._COLLAPSE_GUARDRAIL_PROFILE
        ):
            return None
        return [self.surface.TRAIN_ROOT / bundle
                for bundle in self.surface._COLAB.full_prepared_bundles]

    @timed
    def enforce_data_preparation_boundary(
        self, gpu: str, lane: str, *, prepared_inputs: bool,
    ) -> None:
        """Keep CPU data preparation out of accelerator launches."""
        if gpu.upper() == "CPU":
            return
        if lane in {"bundle", "hpo", "mixed"}:
            raise ValueError(
                f"--what {lane} performs raw CPU data preparation and cannot run "
                "in a GPU launch; prepare inputs on the CPU lane first"
            )
        if lane in {"train", "dual-train", "tracks"} and not prepared_inputs:
            raise ValueError(
                f"--what {lane} has no CPU-lane prepared bundle for this request; "
                "run the CPU bundle lane (er-colab --what bundle --gpu CPU) to "
                "build it, then launch the GPU with the default request"
            )

    @timed
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
