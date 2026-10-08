"""CPU lane committed-export delivery (phase owner).

Split phase of cli/colab_lane.py (the kaggle_lane.py owner-class pattern):
this owner owns the delivery half of the committed-export lane — the remote
delivery archive assembly (run dir + regenerated data artifacts), the
launch/poll/collect/download orchestration of run_delivery, the frozen
resume-state upload with its receipt events, and the TRAINING_RESULTS
delivery receipt.  Collaborators resolve at call time through the running
colab identity (``sys.modules["__colab_runtime_self__"]``) via the lane's
dial-ins.
"""
from __future__ import annotations

from pathlib import Path

from core.common import training_cfg
from core.run_log import RunLogger
from training.prepare_all_trace import timed
from cli.colab_lane_contracts import (
    BUNDLE_DELIVERY_TIMEOUT_SECONDS,
    BUNDLE_LAUNCH_TIMEOUT_SECONDS,
    DELIVERY_ARCHIVE_NAME,
    DELIVERY_DATA_MEMBERS,
    DELIVERY_PREPARED_DIRS,
    DELIVERY_TRACKED_DIRS,
    PREPARE_BUDGET_SECONDS,
    RESUME_STATE_ARCHIVE,
    RESUME_STATE_UPLOAD_TIMEOUT_SECONDS,
    _stamp,
)

#: The preparation run-directory base (config SSOT); the delivery and resume
#: segments spell no path literal.
_RUN_DIR_BASE = training_cfg().preparation.run_dir_base

_LOG = RunLogger(__name__)


class ColabCPULaneDelivery:
    """Delivery archive assembly + the run_delivery orchestration phases."""

    @timed
    def delivery_segment(self) -> str:
        """The remote delivery assembly, ending with its transport token.

        The archive is hashed once as it is written and the token lands beside
        it (``bundle_delivery.tar.zst.size``); the operator-side boundary is
        :func:`cli.colab_bundle_transport.verify_transport_digest`, the one
        integrity check of this VM crossing.
        """
        from cli.colab_bundle_transport import record_digest_script

        return f"""
# delivery: run dir + regenerated data artifacts (list from the 8ddc614 lane).
run_dir = sorted(glob.glob(root + "/results/" + {_RUN_DIR_BASE!r} + "/*"))[-1]
delivery = root + "/{DELIVERY_ARCHIVE_NAME}"
from core.archive_reader import tar_archive
with tar_archive(delivery, "w") as tar:
    tar.add(run_dir, arcname={_RUN_DIR_BASE!r} + "/" + os.path.basename(run_dir))
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
{record_digest_script('delivery', label='bundle')}print("[bundle] delivery archive ready", flush=True)
"""

    @timed
    def delivery_script(self) -> str:
        return self.bundle_head() + self.delivery_segment()

    @timed
    def run_delivery(self, dataset_csv: Path, *, resume_from: str | None = None,
                     resume_run_id: str | None = None,
                     resume_state: Path | None = None) -> None:
        """Prepare on the VM CPU from the cloned cohort export, download the delivery."""
        surface = self.surface
        source = Path(dataset_csv)
        self._validate_resume_state(resume_state)
        run_id = surface._lane_run_stamp()
        self._announce_lane_start(source, run_id)
        script = self._fresh_launch_script(source)
        if resume_state is not None:
            self._upload_resume_state(run_id, resume_state)
            if resume_run_id is None or resume_from is None:
                raise ValueError(
                    "--resume-state requires --resume-run-id and --resume-from "
                    "(the frozen run id and the prepare_all --resume-from choice)"
                )
            script = self._resume_launch_script(source, resume_from, resume_run_id)
        with _LOG.section("colab_lane.delivery.prepare"):
            self._launch_prepare(script)
            self.poll_prepare_log(deadline_seconds=PREPARE_BUDGET_SECONDS)
        with _LOG.section("colab_lane.delivery.collect"):
            self._collect_delivery_archive()
            self._download_delivery(run_id)

    @staticmethod
    def _validate_resume_state(resume_state: Path | None) -> None:
        """Fail loud before any lane work when a resume state is missing."""
        if resume_state is not None and not Path(resume_state).is_file():
            raise FileNotFoundError(f"resume state not found: {resume_state}")

    @timed
    def _announce_lane_start(self, source: Path, run_id: str) -> None:
        """The lane-start receipt: the cohort export rides the sparse checkout."""
        print(
            _stamp(),
            f"[bundle] lane={run_id} cohort export {source.name} rides the sparse "
            f"checkout (no upload; the clone carries the bytes)",
            flush=True,
        )

    @timed
    def _fresh_launch_script(self, source: Path) -> str:
        """The prepare launcher for a fresh run of the cloned cohort export."""
        return self.launch_prepare_script().replace(
            "LAUNCH_ARGS_LIST", "").replace("@COHORT_EXPORT@", source.name)

    @timed
    def _upload_resume_state(self, run_id: str, resume_state: Path) -> None:
        """Upload the frozen resume state with its started/completed events."""
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

    @timed
    def _resume_launch_script(self, source: Path, resume_from: str,
                              resume_run_id: str) -> str:
        """The prepare launcher continuing one frozen run, announced."""
        script = self.launch_prepare_script().replace(
            "LAUNCH_ARGS_LIST",
            f', "--run-dir", root + "/results/{_RUN_DIR_BASE}/@RESUME_RUN_ID@",'
            ' "--resume-from", "@RESUME_FROM@"'
        ).replace("@RESUME_RUN_ID@", resume_run_id).replace(
            "@RESUME_FROM@", resume_from).replace("@COHORT_EXPORT@", source.name)
        print(
            _stamp(),
            f"[bundle] resume: prepare_all --run-dir "
            f"{self.remote_root}/results/{_RUN_DIR_BASE}/{resume_run_id} "
            f"--resume-from {resume_from}",
            flush=True,
        )
        return script

    @timed
    def _launch_prepare(self, script: str) -> None:
        """Stream the prepare launcher onto the VM (bounded launch timeout)."""
        self.exec_stream(
            self.session, script, timeout=BUNDLE_LAUNCH_TIMEOUT_SECONDS,
            log_name="bundle_launch", training_output=True,
        )

    @timed
    def _collect_delivery_archive(self) -> None:
        """Assemble the delivery archive on the VM (bounded delivery timeout)."""
        self.exec_stream(
            self.session, self.delivery_script(),
            timeout=BUNDLE_DELIVERY_TIMEOUT_SECONDS, log_name="bundle_delivery",
            training_output=True,
        )

    @timed
    def _download_delivery(self, run_id: str) -> None:
        """Fetch the delivery archive into its TRAINING_RESULTS root with events."""
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
        self._verify_delivery_boundary(local, run_id)
        print(_stamp(), f"[bundle] delivered -> {local}", flush=True)
        print(_stamp(), f"[bundle] {run_id} complete; the VM session stays open", flush=True)

    @timed
    def _verify_delivery_boundary(self, local: Path, run_id: str) -> int:
        """The ONE integrity check of this VM crossing, then the token receipt.

        The VM recorded ``<archive>.size`` as it finished writing; this reads
        that token back over the existing control channel (a small text read,
        never a second archive transfer) and verifies the delivered archive's
        byte size once
        (:func:`cli.colab_bundle_transport.verify_transport_digest`). A
        mismatch fails loud with the partial kept; a VM whose lane script
        predates the token is reported as unverified, loudly, rather than
        silently trusted.
        """
        from cli.colab_bundle_transport import digest_sidecar, verify_transport_digest

        remote_token = f"{self.remote_root}/{digest_sidecar(local).name}"
        expected = ""
        try:
            expected = self.surface._read_remote_text(remote_token).strip()
        except Exception as error:  # noqa: BLE001 - absence is reported, not hidden
            print(_stamp(), f"[bundle] delivery digest token unreadable at "
                  f"{remote_token} ({error}); verifying locally only", flush=True)
        observed = verify_transport_digest(local, expected)
        if not expected:
            print(_stamp(), "[bundle] WARNING: the VM recorded no delivery digest "
                  "token; this delivery's bytes are unverified against the writer"
                  f" (local size={observed})", flush=True)
            self.result_event(run_id, "download", "digest_absent", archive=str(local),
                              archive_size=observed)
            return observed
        self.result_event(run_id, "download", "verified", archive=str(local),
                          archive_size=observed)
        print(_stamp(), f"[bundle] delivery verified size={observed}", flush=True)
        return observed
