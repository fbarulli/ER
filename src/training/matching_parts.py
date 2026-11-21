"""Bounded-memory candidate parts for the SKU-to-canonical matching lane.

The lane's candidate population is ``SKUs x (top_k + 1)`` report rows, and one
such row is ~16 KB (the gate audit carries the SKU and candidate text).  At the
graded 8,657 calibration SKUs that is a ~7 GB DataFrame, before the fold loop's
per-fold copies and its six audit-trace copies; measured RSS reached 21.3 GB and
the process was OOM-killed.  The full ~65k catalogue would not fit at all.

This module owns the streaming seam instead of one in-memory table:

* :class:`CandidatePartStore` — the scorer appends ONE parquet part per SKU
  chunk; folds, the holdout and the submission read back only the columns they
  decide on.  Peak memory is one chunk plus one fold, independent of catalogue
  size.
* :class:`DiagnosticsCsvWriter` / :class:`DiagnosticsParts` — an audit table is
  validated chunk by chunk and appended to its CSV as it is produced, so the
  diagnostics never exist as one frame either.

Everything here is disposable: the store lives under the configured scratch root
(``config/paths.yaml`` ``paths.worktrees_dir``) and is removed by ``close()``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Self

import pandas as pd

from core.run_log import RunLogger
from core.workspace import Workspace

_LOG = RunLogger(__name__)

_PART_PREFIX = "part-"
_PART_SUFFIX = ".parquet"
_PART_TEMPLATE = _PART_PREFIX + "{index:05d}" + _PART_SUFFIX


class CandidatePartStore:
    """The candidate table on disk: one parquet part per scored SKU chunk."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def build(cls, slot: str) -> CandidatePartStore:
        """The run's scratch slot under the configured worktree/scratch root."""
        workspace = Workspace.from_config()
        root = workspace.slot(f"{slot}_{os.getpid()}")
        # A crashed predecessor's parts are unreferenced scratch, never data; the
        # lane is single-process, so sweeping its own slot family is safe.
        for stale in root.parent.glob(f"{slot}_*"):
            _LOG.warning("removing stale matching scratch %s", stale)
            shutil.rmtree(stale)
        return cls(root)

    def path(self, name: str) -> Path:
        """One file inside the store (diagnostics parts, scratch CSVs)."""
        return self.root / name

    def write(self, index: int, frame: pd.DataFrame) -> None:
        """Persist one chunk. Parts are ordered and never re-read mid-run."""
        frame.to_parquet(self.root / _PART_TEMPLATE.format(index=index), index=False)

    def clear_parts(self) -> None:
        """Drop the scored candidate parts once a stage has consumed them."""
        for part in self.parts:
            part.unlink()

    @property
    def parts(self) -> tuple[Path, ...]:
        return tuple(sorted(self.root.glob(f"{_PART_PREFIX}*{_PART_SUFFIX}")))

    def iter_parts(
        self,
        columns: Sequence[str] | None = None,
        skus: set[str] | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Yield the parts in score order, projected/filtered to what a phase needs."""
        for part in self.parts:
            frame = pd.read_parquet(part, columns=None if columns is None else list(columns))
            if skus is not None:
                frame = frame[frame["SKU_ID"].isin(skus)]
            yield frame

    def read(self, columns: Sequence[str], skus: set[str] | None = None) -> pd.DataFrame:
        """The parts as one frame (only for a projection small enough to hold)."""
        frames = list(self.iter_parts(columns, skus))
        return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    def close(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        else:  # fail-soft: scratch only, but never silently
            _LOG.warning("matching scratch %s already absent at close", self.root)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class DiagnosticsCsvWriter:
    """Append validated audit chunks to one CSV in production order.

    The column contract is checked on EVERY chunk, at the producer, so a streamed
    CSV has exactly the schema the old single-frame write guaranteed.
    """

    def __init__(self, path: Path, expected_columns: frozenset[str]) -> None:
        self.path = path
        self.columns: tuple[str, ...] | None = None
        self._expected = expected_columns
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        actual = frozenset(frame.columns)
        if actual != self._expected:
            raise ValueError(
                "streamed diagnostics column contract violated: "
                f"missing={sorted(self._expected - actual)}, "
                f"unexpected={sorted(actual - self._expected)}"
            )
        frame.to_csv(
            self.path,
            mode="a",
            header=self.columns is None,
            index=False,
        )
        if self.columns is None:
            self.columns = tuple(frame.columns)
        self.rows += len(frame)

    def parts(self) -> DiagnosticsParts:
        if self.columns is None:
            raise RuntimeError(f"no diagnostics rows were written to {self.path}")
        return DiagnosticsParts((self.path,), self.columns, self.rows)


class DiagnosticsParts:
    """Ordered diagnostics CSV parts sharing one validated column set."""

    def __init__(self, paths: tuple[Path, ...], columns: tuple[str, ...], rows: int) -> None:
        self.paths = paths
        self.columns = columns
        self.rows = rows

    def write_csv(self, destination: Path) -> Path:
        """Move/concatenate the parts to ``destination`` without holding a frame."""
        if len(self.paths) == 1:
            # The one-part case (every lane stage) is a rename, never a second
            # copy of a multi-GB audit file.
            self.paths[0].replace(destination)
            self.paths = (destination,)
            return destination
        partial = destination.with_suffix(destination.suffix + ".partial")
        with partial.open("w", encoding="utf-8", newline="") as target:
            for position, part in enumerate(self.paths):
                with part.open("r", encoding="utf-8", newline="") as source:
                    if position:
                        source.readline()  # every part carries the same header
                    shutil.copyfileobj(source, target, length=1 << 20)
        partial.replace(destination)
        return destination
