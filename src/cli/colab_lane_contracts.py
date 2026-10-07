"""Colab lane contracts: the shared constants, stamp, and transport base.

Split phase of cli/colab_lane.py (the kaggle_lane.py owner-class pattern):
the module-level constants every lane phase reads plus ``ColabLaneBase`` —
transport dial-ins + receipts + the shared lane contracts (transit-failure
tolerance, delivery root, delivery member list, checkout-relative guard).
The ``cli.colab`` module stays the single transport surface the offline fakes
patch; every dial-in re-reads the module attribute at call time through the
running colab identity (``sys.modules["__colab_runtime_self__"]``), so the
facades and the test fakes keep driving every lane through one patch surface.
"""
from __future__ import annotations

import sys
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from training.prepare_all_trace import timed

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


def _colab_hub() -> Any:
    """The RUNNING cli.colab module (never a second import copy)."""
    hub = sys.modules.get("__colab_runtime_self__")
    if hub is not None:
        return hub
    import cli.colab as surface

    return surface


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
            self._surface = _colab_hub()
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

    @timed
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
