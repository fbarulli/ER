"""Colab lane contracts: the shared constants, stamp, and transport base.

Split phase of cli/colab_lane.py (the kaggle_lane.py owner-class pattern):
the module-level constants every lane phase reads plus ``ColabLaneBase`` —
transport dial-ins + receipts + the shared lane contracts (transit-failure
tolerance, delivery root, delivery member list, checkout-relative guard).
The ``cli.colab`` module stays the single transport surface the offline fakes
patch; every dial-in re-reads the module attribute at call time through the
shared ``colab_hub.hub()`` (the running ``__colab_runtime_self__`` identity), so the
facades and the test fakes keep driving every lane through one patch surface.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from cli.colab_hub import hub
from core.portable_archive import file_size
from training.prepare_all_trace import timed

if TYPE_CHECKING:
    from core.schemas import PreparationGraphSetupSpec

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


def _stamp(label: str = "colab-lane", *, now: datetime | None = None) -> str:
    """Bracketed Europe/Paris (CET/CEST) wall-clock prefix for output.

    The ONE stamp formatter every lane imports. ``label`` names the emitting
    lane; ``now`` is an optional pre-computed wall-clock so a caller that keeps
    its own ``datetime`` patched in tests passes it through.
    """
    now = now or datetime.now(ZoneInfo("Europe/Paris"))
    return f"[{label} {now:%Y-%m-%dT%H:%M:%S %Z}]"


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
            self._surface = hub()
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
        """Whether a transport failure is any control-channel loss.

        The markers live once, in cli.colab_reconnect: this contract only says
        that both a transient drop and a vanished session are fatal to the
        lane's poll, which is what its caller does with the answer.
        """
        from cli.colab_reconnect import ControlChannelLoss, ControlChannelRecovery

        return ControlChannelRecovery.classify(
            self.session, str(detail)) is not ControlChannelLoss.OTHER

    @staticmethod
    def checkout_relative_guard(value: str, *, message: str) -> None:
        """Refuse a value that is not one repository-relative checkout path.

        The shape rule (traversal, pattern characters, single component) is the
        shared checkout contract in cli.colab_runtime; this guard only owns the
        lane's message.
        """
        from cli.colab_runtime import is_checkout_relative_path

        if not is_checkout_relative_path(value, single_component=True):
            raise ValueError(message)

    def delivery_root(self, run_id: str) -> Path:
        """SSOT of the TRAINING_RESULTS delivery root (commit 4d40d1e)."""
        return self.training_results / ("colab_bundle_" + run_id)

    @contextmanager
    def transcript(self):
        """This lane's ONE per-run transcript.

        Owned here so every committed-export/relaunch caller captures its
        output to the declared ``ColabSpec`` transcript
        (``logs/colab/lane_<session>.log``) without re-wiring
        ``cli.colab.start_live_log``/``close_live_log`` at each facade — the
        path and the truncate-once/append contract stay in one place.
        """
        self.surface.start_live_log()
        try:
            yield
        finally:
            self.surface.close_live_log()

    @timed
    def export_digest(self, dataset_csv: Path) -> int:
        """The export's byte size (structural identity; content is never read)."""
        return file_size(dataset_csv)

    def bundle_head(self) -> str:
        remote_root = self.remote_root
        return self.surface._BOOTSTRAP + f"""
import glob, os, subprocess, sys, tarfile

root = "{remote_root}"
"""


class DeliveryMemberKind(str, Enum):
    """How a declared delivery member is present: one file or one directory."""

    FILE = "file"
    DIR = "dir"


class DeliveryMember(BaseModel):
    """One member of the declared delivery manifest, checkout-relative.

    ``ships`` separates the two roles the manifest plays: every member is a
    PRESENCE gate (membership only — bytes are never checked), and a DIR member
    is also what ``tar.add`` carries recursively. A member the directory add
    already ships inside it gates with ``ships=False`` so no tar member is
    duplicated.
    """

    model_config = ConfigDict(extra="forbid")

    kind: DeliveryMemberKind
    rel: str
    ships: bool = True
    #: Tar arcname when it differs from the checkout-relative path (the
    #: preparation run dir lives under ``results/`` but ships as its base).
    arc: str = ""

    def path(self, root: str) -> str:
        return root + "/" + self.rel

    def arcname(self) -> str:
        return self.arc or self.rel

    def present(self, root: str) -> bool:
        path = self.path(root)
        if self.kind is DeliveryMemberKind.DIR:
            return os.path.isdir(path)
        return os.path.isfile(path)

    def describe(self, root: str) -> str:
        return f"{self.kind.value} {self.rel!r} (expected at {self.path(root)})"


class BundleDelivery(BaseModel):
    """ONE typed delivery: the transport outcome handed to promotion.

    ``_download_delivery`` composes it from what the download landed (the
    archive, its delivery root, its measured bytes) plus the identity only
    ``run_delivery`` sees at launch: the ``training_prep`` run the archive
    carries (a resumed lane ships the frozen id, not the fresh stamp), the
    cohort tag and committed export the run prepared from. ``archive_size``
    is a recorded receipt, never a gate (data is never checked).
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    prep_run_id: str
    cohort: str
    cohort_dataset: str
    archive: Path
    destination: Path
    archive_size: int


class ColabDeliveryManifest:
    """The ONE delivery manifest: what a complete delivery carries, the presence
    gate that refuses a partial one, and the composed remote assembly script.

    The per-member ``if exists`` guards that silently omitted a missing input
    (and the second inline copy of them in ``cli.colab.run_bundle``) are gone;
    both lanes now emit exactly this module's assembly, which names every
    missing member and the run dir it was composed for before it writes a byte.
    The track_setup member names are DERIVED from the declared layout spec
    (``training.preparation.graph_setup``), never re-typed here;
    ``shared_embeddings`` is excluded because its producer is the GPU encode
    step, which never runs on the CPU lane.
    """

    @classmethod
    def members(cls, *, run_id: str) -> tuple[DeliveryMember, ...]:
        """Every declared member of this lane's delivery, checkout-relative."""
        from core.common import prepared_setup_layout, training_cfg

        layout = prepared_setup_layout()
        prep_run = f"{training_cfg().preparation.run_dir_base}/{run_id}"
        members: list[DeliveryMember] = [
            DeliveryMember(kind=DeliveryMemberKind.DIR, rel=f"results/{prep_run}",
                           arc=prep_run),
            *(DeliveryMember(kind=DeliveryMemberKind.FILE, rel=rel)
              for rel in DELIVERY_DATA_MEMBERS),
        ]
        for name in DELIVERY_TRACKED_DIRS:
            members.append(DeliveryMember(kind=DeliveryMemberKind.DIR,
                                          rel=f"data/{name}"))
            members.extend(
                DeliveryMember(kind=kind, rel=f"data/{name}/{rel}", ships=False)
                for kind, rel in cls._setup_tree_members(layout)
            )
        for name in DELIVERY_PREPARED_DIRS:
            members.append(DeliveryMember(kind=DeliveryMemberKind.DIR,
                                          rel=f"data/prepared/{name}"))
        members.extend(
            DeliveryMember(kind=DeliveryMemberKind.FILE, rel=rel, ships=False)
            for rel in training_cfg().colab.full_prepared_bundles
        )
        return tuple(members)

    @staticmethod
    def _setup_tree_members(
        layout: PreparationGraphSetupSpec,
    ) -> tuple[tuple[DeliveryMemberKind, str], ...]:
        """The setup-tree members as (kind, name), every name read off the layout
        spec; ``prepared_dir`` is the tree itself and its declared children ride
        inside it."""
        file, directory = DeliveryMemberKind.FILE, DeliveryMemberKind.DIR
        return (
            (file, layout.catalog), (file, layout.splits), (file, layout.pairs),
            (file, layout.pair_lineage), (file, layout.census),
            (file, layout.manifest), (file, layout.text_config),
            (file, layout.embedding_request), (file, layout.prepared_text),
            (file, layout.text_export_request), (file, layout.text_training_binding),
            (file, layout.shared_training_data),
            (file, layout.shared_training_projection),
            (directory, layout.prepared_dir),
            (file, f"{layout.prepared_dir}/{layout.input_manifest}"),
            (file, f"{layout.prepared_dir}/{layout.listings}"),
        )

    @classmethod
    def require_ready(cls, root: str, *, run_id: str) -> None:
        """Refuse to compose a delivery that is missing a declared member."""
        missing = [member.describe(root) for member in cls.members(run_id=run_id)
                   if not member.present(root)]
        if missing:
            raise RuntimeError(
                f"delivery refused for run {run_id!r}: this lane pinned prepare "
                f"--run-dir to exactly this run id at launch (a fresh lane stamp, "
                f"or the resumed frozen run), so no other run directory can ship; "
                f"declared members missing:\n" + "\n".join(missing)
            )

    @classmethod
    def delivery_assembly(cls, run_id: str) -> str:
        """The generated VM-side delivery: gate on the ONE manifest, then tar it.

        The emitted text is the single copy both lanes embed; it runs against
        the ``root`` global the lane head defines and raises with every missing
        member before a partial archive can install as a success.
        """
        return f"""
from cli.colab_lane_contracts import ColabDeliveryManifest
from core.archive_reader import tar_archive
ColabDeliveryManifest.require_ready(root, run_id={run_id!r})
delivery = root + "/{DELIVERY_ARCHIVE_NAME}"
with tar_archive(delivery, "w") as tar:
    for member in ColabDeliveryManifest.members(run_id={run_id!r}):
        if member.ships:
            tar.add(member.path(root), arcname=member.arcname())
print("[bundle] delivery archive ready", flush=True)
"""
