"""Operational-cost instrumentation: latency, memory, refresh time.

MODEL_TRACKS_PLAN.md requires "Encoding/indexing/query latency, memory,
refresh time" as a first-class reported measurement. The pre-existing
:class:`core.training_profiler.TrainingProfiler` records an opt-in PyTorch
trace (``ER_TRAINING_PROFILE=1``) that only covers whichever three trainer
steps the schedule selects, and its outputs were never aggregated or
surfaced -- ``operator_summary.txt`` sat unread in the run directory and the
dashboard could not see it.

This module is the cheap, always-on complement: wall-clock per named section,
process peak RSS, and CUDA peak allocation when present. It does not replace
the torch trace, it summarises whatever the trace covers when the trace
exists.
"""

from __future__ import annotations

import json
import os
import re
import resource
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from core.common import performance_cfg
from pydantic import BaseModel, ConfigDict, Field, StrictInt, TypeAdapter, model_validator
from typing import Literal

PERFORMANCE_SCHEMA = "er-track-performance-v1"

#: Sections the plan names explicitly. Anything else measured is still
#: recorded, but these are the ones that must not go missing.
REQUIRED_SECTIONS = ("encode", "index_build", "query", "refresh")


def _peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS; this repo only
    runs on Linux, but the platform check keeps the value honest rather than
    silently 1000x wrong if that ever changes.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if os.uname().sysname == "Darwin" else int(peak) * 1024


def _cuda_peak_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except ImportError:  # CPU-only reporting can run without torch.
        return None


class SectionTiming(BaseModel):
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    calls: StrictInt = Field(ge=1)
    total_seconds: float = Field(ge=0)


class RefreshTiming(BaseModel):
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    refresh_seconds: float = Field(ge=0)


class ProfilerMetadata(BaseModel):
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    device: Literal['cpu', 'cuda']
    active_steps: int = Field(ge=0, le=3)
    includes_profiling_overhead: Literal[True]


class PerformanceSummary(BaseModel):
    model_config = ConfigDict(extra='allow', allow_inf_nan=False)
    schema_id: Literal['er-track-performance-v1'] = Field(alias='schema')
    track: Literal['text', 'gnn_only', 'hybrid']
    enabled: bool
    sections: dict[str, SectionTiming]
    missing_required_sections: list[str]
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    peak_rss_mb: float | None = Field(default=None, ge=0)
    peak_cuda_allocated_bytes: int | None = Field(default=None, ge=0)
    peak_cuda_allocated_mb: float | None = Field(default=None, ge=0)


    @model_validator(mode='after')
    def check_measurement_inventory(self):
        expected = [name for name in REQUIRED_SECTIONS if name not in self.sections]
        if self.missing_required_sections != expected:
            raise ValueError('missing_required_sections must match measured sections')
        if not self.enabled and self.sections:
            raise ValueError('disabled instrumentation cannot claim measured sections')
        return self


def load_refresh_timings(path: Path) -> list[dict]:
    """Validate an existing timing file; retain parsing/validation tracebacks."""
    return [row.model_dump() for row in TypeAdapter(list[RefreshTiming]).validate_json(path.read_text())]


class PerformanceRecorder:
    """Accumulate named wall-clock sections and memory peaks.

    Cheap enough to leave enabled: one ``monotonic()`` pair per section and
    one ``getrusage`` per memory sample.
    """

    def __init__(self, track: str, *, enabled: bool | None = None):
        self.track = track
        self.enabled = performance_cfg()["enabled"] if enabled is None else enabled
        self._sections: list[dict] = []
        self._counts: dict[str, int] = {}
        self._external: dict[str, dict] = {}
        self._cuda_peak: int | None = None

    def adopt(self, label: str, stats: dict) -> None:
        """Adopt a section that was measured in another process.

        The ANN refresh runs inside the trainer while the report is built by
        the postprocess lane, so its timings arrive as an aggregate rather than
        as individual samples. They still count towards the section being
        reported, so ``missing_required_sections`` stays truthful.
        """
        if self.enabled and stats:
            self._external[label] = SectionTiming.model_validate(stats).model_dump()

    @contextmanager
    def section(self, label: str):
        if not self.enabled:
            with nullcontext():
                yield
            return
        started = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            self._sections.append(
                {"section": label, "seconds": elapsed, "count": 1}
            )
            self._counts[label] = self._counts.get(label, 0) + 1
            self.sample_memory()

    def count(self, label: str, amount: int = 1) -> None:
        """Record a counted event (e.g. 250 queries) to derive per-item cost."""
        if self.enabled:
            self._counts[label] = self._counts.get(label, 0) + amount

    def record(self, label: str, seconds: float) -> None:
        """Add an already-measured duration.

        Used where the timing already exists (the postprocess lanes wrap each
        phase in their own ``time.monotonic()`` pairs for progress logging) and
        re-wrapping them would double the instrumentation.
        """
        if not self.enabled:
            return
        timing = SectionTiming(calls=1, total_seconds=seconds)
        self._sections.append(
            {"section": label, "seconds": timing.total_seconds, "count": timing.calls}
        )
        self._counts[label] = self._counts.get(label, 0) + 1
        self.sample_memory()

    def sample_memory(self) -> None:
        if not self.enabled:
            return
        peak = _cuda_peak_bytes()
        if peak is not None:
            self._cuda_peak = peak if self._cuda_peak is None else max(self._cuda_peak, peak)

    @staticmethod
    def _stats(values: list[float]) -> dict:
        ordered = sorted(values)
        if not ordered:
            return {"calls": 0, "total_seconds": 0.0}
        def at(q: float) -> float:
            idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
            return float(ordered[idx])
        return {
            "calls": len(ordered),
            "total_seconds": float(sum(ordered)),
            "median_seconds": at(0.5),
            "p95_seconds": at(0.95),
            "max_seconds": float(ordered[-1]),
        }

    def summary(self) -> dict:
        by_section: dict[str, list[float]] = {}
        for row in self._sections:
            by_section.setdefault(row["section"], []).append(float(row["seconds"]))
        sections = {
            label: {**self._stats(values), "events": self._counts.get(label)}
            for label, values in sorted(by_section.items())
        }
        # Externally measured sections (training-side refresh) are merged in
        # last so they are neither lost nor double counted.
        for label, stats in self._external.items():
            sections.setdefault(label, stats)
        # Per-item cost needs both a count and a time, so it is derived by the
        # caller passing ``count()`` for the matching event label.
        peak_rss = _peak_rss_bytes() if self.enabled else 0
        return {
            "schema": PERFORMANCE_SCHEMA,
            "track": self.track,
            "enabled": bool(self.enabled),
            "sections": sections,
            "missing_required_sections": [
                s for s in REQUIRED_SECTIONS if s not in sections
            ],
            "peak_rss_bytes": peak_rss or None,
            "peak_rss_mb": round(peak_rss / 1048576, 3) if peak_rss else None,
            "peak_cuda_allocated_bytes": self._cuda_peak,
            "peak_cuda_allocated_mb": (
                round(self._cuda_peak / 1048576, 3) if self._cuda_peak else None
            ),
        }

    def write(self, path: Path) -> Path:
        return self.write_payload(path, self.summary())

    def write_payload(self, path: Path, payload: dict) -> Path:
        """Write a summary that may carry extra blocks (e.g. the profiler)."""
        validated = PerformanceSummary.model_validate(payload)
        candidate = path.with_suffix(path.suffix + '.partial')
        candidate.write_text(validated.model_dump_json(indent=2, by_alias=True) + "\n")
        candidate.replace(path)
        return path


def summarize_profiler_directory(directory: Path) -> dict:
    """Fold an existing torch profile into the performance summary.

    ``TrainingProfiler`` is opt-in and only covers up to three steps, so this
    is reported as a separate ``torch_profile`` block rather than merged into
    the section timings -- merging a 3-step operator table into per-operation
    wall clock would double-count.
    """
    directory = Path(directory)
    manifest_path = directory / "profile_manifest.json"
    summary_path = directory / "operator_summary.txt"
    if not (manifest_path.is_file() or summary_path.is_file()):
        return {}
    block: dict = {"directory": str(directory)}
    if manifest_path.is_file():
        try:
            block["manifest"] = ProfilerMetadata.model_validate_json(manifest_path.read_text()).model_dump()
        except ValueError as exc:
            exc.add_note(f"Profiler manifest: {manifest_path}")
            raise
    if summary_path.is_file():
        text = summary_path.read_text()
        top = self_time_seconds(text)
        if top:
            block["top_self_time_seconds"] = top
        block["includes_profiling_overhead"] = True
        block["covers_at_most_steps"] = 3
    return {"torch_profile": block}


class OperatorTiming(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    name: str = Field(min_length=1)
    seconds: float = Field(ge=0)


def self_time_seconds(operator_summary: str) -> dict[str, float]:
    """Read the named self-time column and its unit from a PyTorch table."""
    column = None
    rows: list[OperatorTiming] = []
    units = {'s': 1., 'ms': 1e-3, 'us': 1e-6, 'µs': 1e-6, 'ns': 1e-9}
    for line in operator_summary.splitlines():
        cells = re.split(r'\s{2,}', line.strip())
        if cells and cells[0] == 'Name':
            # CUDA self time is selected when the table includes it.
            names = ('Self CUDA', 'Self GPU', 'Self CPU', 'Self CPU time total')
            column = next((cells.index(name) for name in names if name in cells), None)
            continue
        if column is None or len(cells) <= column or cells[0].startswith('-'):
            continue
        match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(s|ms|us|µs|ns)', cells[column])
        if match is None:
            continue
        rows.append(OperatorTiming(name=cells[0], seconds=float(match[1]) * units[match[2]]))
    top = sorted(rows, key=lambda row: row.seconds, reverse=True)[:10]
    return {row.name: row.seconds for row in top}


def summarize_refresh_timings(directory: Path) -> dict:
    """Fold training-side ANN refresh timings into the cost summary.

    ``FineTunedAnnRefreshCallback`` writes one ``refresh_timings_fold*.json`` per
    fold next to its audit CSVs. Refresh happens during training, in a different
    process from the postprocess report, so it is merged in as its own section
    rather than being re-measured.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    rows: list[dict] = []
    for path in sorted(directory.rglob("refresh_timings_fold*.json")):
        try:
            rows.extend(load_refresh_timings(path))
        except ValueError as exc:
            raise ValueError(f'invalid refresh timing artifact {path}') from exc
    values = [
        float(row["refresh_seconds"]) for row in rows
        if isinstance(row.get("refresh_seconds"), (int, float))
    ]
    if not values:
        return {}
    ordered = sorted(values)
    return {"refresh": {
        "calls": len(values),
        "total_seconds": float(sum(values)),
        "median_seconds": float(ordered[len(ordered) // 2]),
        "max_seconds": float(ordered[-1]),
        "source": "training-side FineTunedAnnRefreshCallback",
        "note": "spans fine-tuned encode, both mining passes and the audit rewrite",
    }}


__all__ = [
    "PERFORMANCE_SCHEMA",
    "REQUIRED_SECTIONS",
    "PerformanceRecorder",
    "self_time_seconds",
    "summarize_profiler_directory",
    "summarize_refresh_timings",
]