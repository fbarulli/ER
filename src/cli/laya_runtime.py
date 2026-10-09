"""Laya lane runtime: the staging root, transcript roof, and CSV census.

The ``LayaRuntimeFactory`` binds ONE resolved ``LayaSpec`` + the lane's
``TRAIN_ROOT`` so no helper re-reads the global config. The transcript is the
ONE shared kaggle run log (``kaggle.logs_dir`` / ``kaggle.files.lane_log``),
truncated once per process.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from cli.log_capture import LaneTranscript
from core.laya_config import LayaSpec
from core.manifest import sha256_file

_PARIS = ZoneInfo("Europe/Paris")  # build once, not per log line

# The accuracy/F1 metric contract a harvest agent needs when a decision
# CSV carries ground-truth labels (owner order 2026-10-07): the EXPECTED
# row count + label distribution computed from the csv itself + the gold
# columns the harvest reads.
_METRIC_EXPECTATION_KEYS = ("expected_rows", "expected_label_distribution",
                            "metric_expectation")


class LayaRuntimeFactory:
    """The lane's runtime paths, transcript, and CSV census for one spec."""

    def __init__(self, spec: LayaSpec, train_root: Path):
        self._spec = spec
        self._train_root = Path(train_root)

    @property
    def spec(self) -> LayaSpec:
        return self._spec

    @property
    def train_root(self) -> Path:
        return self._train_root

    def staging_dir(self) -> Path:
        """The lane staging root (TRAIN_ROOT-relative; SSOT laya.staging_dir)."""
        return (self._train_root / self._spec.staging_dir).resolve()

    def lane_logs_dir(self) -> Path:
        """The ONE canonical transcript roof (config ``kaggle.logs_dir``).

        The laya lane shares the ER kaggle lane's transcript roof so both
        lanes' runs land in the same file; laya-only state (fetched session
        handles, follower locks) rides the same roof as separate files.
        """
        return LaneTranscript.roof_for(self._train_root)

    def lane_log_path(self) -> Path:
        """The declared single run transcript (``kaggle.files.lane_log``)."""
        return LaneTranscript.path_for(self._train_root)

    def _stamp(self) -> str:
        """Bracketed Europe/Paris (CET/CEST) wall-clock prefix.

        Mirrors kaggle_lane._stamp: the CET convention landed there (owner
        order 2026-10-07) and this lane follows it; the "[laya-lane UTC-stamp]"
        phrasing in the relaunch brief predates that convention.
        """
        return f"[laya-lane {self._bare_stamp()}]"

    @staticmethod
    def _bare_stamp() -> str:
        return f"{datetime.now(_PARIS):%Y-%m-%dT%H:%M:%S %Z}"

    def log_lane(self, line: str) -> None:
        """Timestamped lane logging into the ONE shared kaggle transcript.

        Same truncate-at-run-start/append-after semantics as the ER writer —
        one class owns the gate, so a laya process and an ER writer in the same
        run can never truncate each other's lines.
        """
        LaneTranscript.from_config(
            self._train_root, lane="laya-lane", stamp=self._bare_stamp,
        ).write(line)

    @staticmethod
    def measure_csv(path: Path, wanted_columns: tuple[str, ...]) -> dict[str, Any]:
        """Stdlib CSV census: header check + row count + sha256 + bytes.

        Deliberately NOT pandas: staging must run anywhere (including a box
        without the frame stack), and the contract is only the columns. Public
        because ``scripts/laya_metrics_pairs.py`` probes a built pair CSV
        through the lane's own measurement.
        """
        import csv as _csv

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"decision input not found: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = _csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"decision input has no header row: {path}")
            missing = [column for column in wanted_columns
                       if column not in header]
            if missing:
                raise ValueError(
                    f"decision input {path.name} is missing columns {missing} "
                    f"(header: {header})")
            rows = sum(1 for _ in reader)
        if rows == 0:
            raise ValueError(f"decision input has no data rows: {path}")
        return {"rows": rows, "columns": list(header),
                "sha256": sha256_file(path), "bytes": path.stat().st_size}

    def census_csv(self, path: Path, wanted_columns: tuple[str, ...],
                   label_column: str = "true_label") -> dict[str, Any]:
        """Stdlib CSV census: header check + row count + sha256 + label census.

        Deliberately NOT pandas: staging must run anywhere (including a box
        without the frame stack), and the contract is only the columns. The
        expectation block is only produced when the label column is present.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"decision input not found: {path}")
        digest = hashlib.sha256()
        labels: Counter = Counter()
        with path.open("rb") as handle:
            def lines():
                for raw in handle:
                    digest.update(raw)
                    yield raw.decode("utf-8")

            reader = csv.reader(lines())
            header = next(reader, None)
            if header is None:
                raise ValueError(f"decision input has no header row: {path}")
            missing = [column for column in wanted_columns if column not in header]
            if missing:
                raise ValueError(
                    f"decision input {path.name} is missing columns {missing} "
                    f"(header: {header})")
            label_at = header.index(label_column) if label_column in header else None
            rows = 0
            for record in reader:
                rows += 1
                if label_at is not None and record:
                    labels[record[label_at] if label_at < len(record) else ""] += 1
        if rows == 0:
            raise ValueError(f"decision input has no data rows: {path}")
        expectation: dict[str, Any] = {}
        if label_at is not None:
            unknown = sorted(set(labels) - {"0", "1"})
            if unknown:
                raise ValueError(
                    f"decision input {path.name} carries true_label values "
                    f"outside {{0, 1}}: {unknown}")
            expectation = {
                "expected_rows": sum(labels.values()),
                "expected_label_distribution": {
                    label: labels[label] for label in sorted(labels)},
                "metric_expectation": {
                    "accuracy_gold": "label",
                    "f1_gold": "identity_claim-vs-true_label",
                },
            }
        return {"rows": rows, "columns": list(header),
                "sha256": digest.hexdigest(), "bytes": path.stat().st_size,
                "expectation": expectation}
