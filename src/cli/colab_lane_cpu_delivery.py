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

import sys
import traceback
from pathlib import Path

from core.common import training_cfg
from core.run_log import RunLogger
from training.prepare_all_trace import timed
from cli.colab_lane_contracts import (
    BUNDLE_DELIVERY_TIMEOUT_SECONDS,
    BUNDLE_LAUNCH_TIMEOUT_SECONDS,
    BundleDelivery,
    ColabDeliveryManifest,
    DELIVERY_ARCHIVE_NAME,
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

    def receipt_root(self, run_id: str) -> str:
        """The lane's result-event root name: the delivery root's own name.

        ``_result_event`` keys receipts to ``TRAINING_RESULTS / <name>``; every
        receipt of this lane lands in the delivery root, so the archive and its
        events share ONE root (the run_bundle contract of commit 4d40d1e that
        the fb10b68 lane consolidation never wired).
        """
        return self.delivery_root(run_id).name

    @timed
    def delivery_segment(self, run_id: str) -> str:
        """The remote delivery assembly, ending with its transport token.

        The segment composes the ONE declared manifest
        (``ColabDeliveryManifest``): a missing required member names itself and
        the run directory it was expected under, before any byte is written.
        The archive's byte size is recorded once as it is written and the token
        lands beside it (``bundle_delivery.size``); the operator-side boundary
        reads that token back as a receipt only, never as a refusal (owner
        directive 2026-10-09: data is never checked).
        """
        from cli.colab_bundle_transport import record_digest_script

        return (
            ColabDeliveryManifest.delivery_assembly(run_id)
            + record_digest_script('delivery', label='bundle')
        )

    @timed
    def delivery_script(self, run_id: str) -> str:
        return self.bundle_head() + self.delivery_segment(run_id)

    @timed
    def run_delivery(self, dataset_csv: Path, *, resume_from: str | None = None,
                     resume_run_id: str | None = None,
                     resume_state: Path | None = None) -> None:
        """Prepare on the VM CPU, download the delivery, promote it run-ready.

        The delivery is the lane's committed-export contract; the promote step
        after a successful download makes BOTH training lanes consumable from
        this checkout (see ``promote_delivery``).
        """
        surface = self.surface
        source = Path(dataset_csv)
        run_id = surface._lane_run_stamp()
        self._announce_lane_start(source, run_id)
        script = self._fresh_launch_script(source, run_id)
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
            # The delivery composes exactly the run this lane pinned at launch
            # (fresh lane stamp or resumed frozen run), never a name-sorted glob.
            prep_run_id = resume_run_id or run_id
            self._collect_delivery_archive(prep_run_id)
            delivery = self._download_delivery(
                run_id, cohort=self.cohort_label(source),
                cohort_dataset=source.name, prep_run_id=prep_run_id)
        with _LOG.section("colab_lane.delivery.promote"):
            self.promote_delivery(delivery)
        print(_stamp(), f"[bundle] {run_id} complete; the VM session stays open", flush=True)

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
    def _fresh_launch_script(self, source: Path, run_id: str) -> str:
        """The prepare launcher for a fresh run of the cloned cohort export.

        ``--run-dir`` is pinned to this lane's run id, so the delivery reads
        back exactly the directory this launch created — a reused VM's newer
        or foreign run directory can never ship.
        """
        return self.launch_prepare_script().replace(
            "LAUNCH_ARGS_LIST",
            f', "--run-dir", root + "/results/{_RUN_DIR_BASE}/{run_id}"'
        ).replace("@COHORT_EXPORT@", source.name)

    @timed
    def _upload_resume_state(self, run_id: str, resume_state: Path) -> None:
        """Upload the frozen resume state with its started/completed events."""
        resume_state = Path(resume_state)
        receipt_root = self.receipt_root(run_id)
        print(
            _stamp(),
            f"[bundle] resume: uploading {resume_state} -> "
            f"{self.remote_root}/{RESUME_STATE_ARCHIVE} ...",
            flush=True,
        )
        self.result_event(receipt_root, "upload", "started", file=str(resume_state))
        self.upload_with_retries(
            resume_state,
            f"{self.remote_root}/{RESUME_STATE_ARCHIVE}",
            timeout=RESUME_STATE_UPLOAD_TIMEOUT_SECONDS,
        )
        self.result_event(
            receipt_root, "upload", "completed",
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
    def _collect_delivery_archive(self, run_id: str) -> None:
        """Assemble the delivery archive on the VM (bounded delivery timeout)."""
        self.exec_stream(
            self.session, self.delivery_script(run_id),
            timeout=BUNDLE_DELIVERY_TIMEOUT_SECONDS, log_name="bundle_delivery",
            training_output=True,
        )

    @timed
    def _download_delivery(self, run_id: str, *, cohort: str, cohort_dataset: str,
                           prep_run_id: str) -> BundleDelivery:
        """Fetch the delivery archive into its TRAINING_RESULTS root, typed.

        The return is the ``BundleDelivery`` seam: the landed archive and root
        the transport measured, plus the prep-run and cohort identity only the
        launching orchestration knows — so ``promote_delivery`` reads one model
        and no caller re-spells paths.
        """
        local_base = self.delivery_root(run_id)
        local_base.mkdir(parents=True, exist_ok=True)
        receipt_root = self.receipt_root(run_id)
        self.result_event(receipt_root, "download", "started", workers=1)
        local = local_base / DELIVERY_ARCHIVE_NAME
        self.download_with_visibility(
            remote=f"{self.remote_root}/{DELIVERY_ARCHIVE_NAME}",
            local=local,
            worker=None,
            index=1,
            total=1,
        )
        self.result_event(receipt_root, "download", "completed", archive=str(local),
                          destination=str(local_base))
        observed = self._record_delivery_size(local, receipt_root)
        print(_stamp(), f"[bundle] delivered -> {local}", flush=True)
        return BundleDelivery(run_id=run_id, prep_run_id=prep_run_id,
                              cohort=cohort, cohort_dataset=cohort_dataset,
                              archive=local, destination=local_base,
                              archive_size=observed)

    @timed
    def _record_delivery_size(self, local: Path, receipt_root: str) -> int:
        """Record the delivered archive's byte size and the writer's token.

        The VM recorded ``<archive>.size`` as it finished writing; this reads
        that token back over the existing control channel (a small text read,
        never a second archive transfer) and records it beside the delivered
        archive's measured size. Data is never checked, so a token that differs
        from the delivered bytes is reported, never fatal (owner directive
        2026-10-09).
        """
        from cli.colab_bundle_transport import digest_sidecar, transport_archive_size

        remote_token = f"{self.remote_root}/{digest_sidecar(local).name}"
        writer_token = ""
        try:
            writer_token = self.surface._read_remote_text(remote_token).strip()
        except Exception as error:  # noqa: BLE001 - absence is reported, not hidden
            print(_stamp(), f"[bundle] delivery size token unreadable at "
                  f"{remote_token} ({error})", flush=True)
        observed = transport_archive_size(local)
        self.result_event(receipt_root, "download", "measured", archive=str(local),
                          archive_size=observed, writer_token=writer_token or None)
        print(_stamp(), f"[bundle] delivered size={observed} "
              f"(writer token {writer_token or 'absent'})", flush=True)
        return observed

    @timed
    def install_delivery(self, archive: Path, *, checkout: Path | None = None,
                         results: Path | None = None) -> dict[str, int]:
        """Install a delivered bundle archive into the checkout (local, no VM).

        ``_download_delivery`` lands one immutable ``bundle_delivery.tar.zst``
        under its TRAINING_RESULTS root; this is the phase that makes it
        consumable by a launch. ``data/*`` members install under the checkout
        root (the prepared ``track_setup`` / ``prepared`` trees a GPU launch
        packages), and ``training_prep/*`` installs under ``results/`` (where
        ``prepared_package_candidates`` and ``run_history`` scan the prepared
        ``all_tracks_inputs`` package). A member under any other top-level name
        is not part of the delivery contract and is skipped, never guessed at.
        The archive is immutable and trusted, so no byte is re-checked (owner
        directive: data is never checked); extraction goes through the ONE
        ``core.archive_reader.tar_archive`` reader (no ``zstd`` binary needed)
        with tarfile's ``data`` filter for path safety.

        Returns the per-root installed FILE counts (directories excluded).
        """
        from core.archive_reader import tar_archive
        from core.common import DATA_DIR, RESULTS, TRAIN_ROOT

        checkout = Path(checkout) if checkout is not None else Path(TRAIN_ROOT)
        results = Path(results) if results is not None else Path(RESULTS)
        data_root = Path(DATA_DIR).name
        roots = {data_root: checkout, _RUN_DIR_BASE: results}
        installed = {key: 0 for key in roots}
        archive = Path(archive)
        with tar_archive(archive, "r") as tar:
            for member in tar.getmembers():
                top = member.name.split("/", 1)[0]
                root = roots.get(top)
                if root is None:
                    continue
                tar.extract(member, path=root, filter="data")
                if member.isfile():
                    installed[top] += 1
        print(
            _stamp(),
            f"[bundle] installed {archive.name}: {data_root}="
            f"{installed[data_root]} files -> {checkout}, {_RUN_DIR_BASE}="
            f"{installed[_RUN_DIR_BASE]} files -> {results}",
            flush=True,
        )
        return installed

    @timed
    def promote_delivery(self, delivery: BundleDelivery) -> None:
        """Promote a delivered bundle: BOTH training lanes end run-ready.

        (1) ``install_delivery`` lands the archive's data/ trees in this
        checkout and the fresh ``training_prep`` run under results/ (the GPU
        launch's inputs). (2) The kaggle lane's staging gets the SAME fresh
        run dir as the cohort's bundle install — the promoted install must BE
        the delivered archive, because ``default_prepared_input_package``
        resolves an existing kaggle install BEFORE the newest prep run.
        (3) The hosted ``bundle`` dataset is versioned from that install; a
        publish failure is recorded fail-soft with its reason and the
        installed data stays (the doctrine of the kaggle lane's
        ``_publish_after_verified_fetch``) — never silent success.
        (4) The laya corpus follow-up is printed as the named route, never
        run unattended.
        """
        from cli.kaggle_datasets import KaggleDatasets
        from core.common import RESULTS

        receipt_root = self.receipt_root(delivery.run_id)
        installed = self.install_delivery(delivery.archive)
        install = KaggleDatasets.stage_bundle_install(
            Path(RESULTS) / _RUN_DIR_BASE / delivery.prep_run_id,
            cohort=delivery.cohort, cohort_dataset=delivery.cohort_dataset,
            branch=training_cfg().colab.branch)
        self.result_event(receipt_root, "promote", "staged", install=str(install),
                          prep_run=delivery.prep_run_id, files=installed)
        self._publish_promoted_bundle(receipt_root)
        self._announce_laya_followup()

    @timed
    def _publish_promoted_bundle(self, receipt_root: str) -> None:
        """Version the hosted bundle dataset from the promoted install.

        Fail-soft with a recorded reason (installed data stays), full
        traceback on stderr — the publish itself resolves the newest install
        and refuses loud when none is present.
        """
        from cli.kaggle_datasets import KaggleDatasets

        try:
            plan = KaggleDatasets.publish_bundle_dataset("bundle", execute=True)
        except (RuntimeError, FileNotFoundError, OSError) as error:
            print(_stamp(), f"[bundle] promote publish failed and was recorded "
                  f"(installed data stays): {error}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            self.result_event(receipt_root, "promote", "publish_failed",
                              error=str(error))
            return
        self.result_event(receipt_root, "promote", "published",
                          dataset_version=plan.get("dataset_version"),
                          publish_stage=plan.get("stage"))

    @timed
    def _announce_laya_followup(self) -> None:
        """Print the laya corpus route (named follow-up; never executed).

        Laya finetune freshness flows through a corpus rebuild from the
        installed ``data/prepared/full`` trees, published via the laya lane to
        the hosted ``corpus`` dataset; the published-tip gate keeps it
        operator-driven, so this lane names the route instead of running it.
        """
        from core.hosted_dataset import hosted_registry

        corpus = hosted_registry().by_role("corpus").slug
        print(_stamp(), f"[bundle] laya follow-up (route, not run): rebuild the "
              f"corpus with scripts/laya_build_dataset.py on the installed "
              f"data/prepared/full trees, then cli.laya_lane "
              f"publish_laya_dataset to version {corpus}; the published-tip "
              f"gate keeps it operator-driven", flush=True)
