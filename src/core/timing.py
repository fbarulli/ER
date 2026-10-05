"""Wall-clock section timing for CPU preparation stages.

Each heavy section prints one `[timing]` line (live on a terminal, captured
in the stage log when run as a subprocess) and is recorded for a structured
JSON dump so runs can be compared when optimizing. The dump path comes from
the ER_TIMING_OUT environment variable (set per stage by
training.prepare_all); without it only the print lines happen, so shared
callers (research lanes, selftests) stay untouched.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, Field


def emit_timing(message: str, *, path: Path | None = None):
    print(message, flush=True)
    destination = path or os.environ.get('ER_TIMING_LOG')
    if destination is None and os.environ.get('ER_TIMING_OUT'):
        destination = Path(os.environ['ER_TIMING_OUT']).with_suffix('.log')
    if destination is not None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('a', encoding='utf-8') as handle:
            handle.write(message + '\n')


class Timing:
    def __init__(self, label: str):
        self.label = label
        self.sections: list[dict] = []
        self._started = self._last = time.perf_counter()

    def mark(self, name: str) -> float:
        """Record the wall time since the previous mark (or construction)."""
        now = time.perf_counter()
        elapsed = now - self._last
        self._last = now
        self.sections.append({"section": name, "seconds": round(elapsed, 3)})
        emit_timing(f"[timing] {self.label} {name}: {elapsed:.3f}s")
        self.dump_if_requested()
        return elapsed

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        status = 'completed'
        emit_timing(f"[timing] {self.label} {name} state=started")
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            elapsed = time.perf_counter() - start
            self._last = time.perf_counter()
            self.sections.append({"section": name, "seconds": round(elapsed, 3), "status": status})
            emit_timing(f"[timing] {self.label} {name} state={status} elapsed_seconds={elapsed:.3f}")
            self.dump_if_requested()

    def dump_if_requested(self) -> Path | None:
        out = os.environ.get("ER_TIMING_OUT")
        if not out:
            return None
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = json.loads(path.read_text()) if path.exists() else {}
        component = {"label": self.label, "sections": self.sections,
                     "total_seconds": round(time.perf_counter() - self._started, 3)}
        components = document.get('components', {})
        components[self.label] = component
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps({**component, 'components': components}, indent=2) + '\n')
        temporary.replace(path)
        return path


class TimingOffender(BaseModel):
    """One timed surface (a stage or a named section inside a stage)."""

    label: str = Field(min_length=1)
    seconds: float = Field(ge=0)


def collect_timing_entries(run_dir: Path, stage_seconds: Mapping[str, float]) -> list[TimingOffender]:
    """Every timed surface of one run: stage totals + their named sections.

    Sections come from each stage's <stage>.timing.json (written by Timing
    through ER_TIMING_OUT); stages without a timing file contribute their
    wall-clock total from stage_seconds only.
    """
    offenders = [TimingOffender(label=f'stage/{name}', seconds=round(seconds, 3))
                 for name, seconds in stage_seconds.items()]
    for timing_path in sorted(Path(run_dir).glob('*.timing.json')):
        stage = timing_path.name.removesuffix('.timing.json')
        try:
            document = json.loads(timing_path.read_text())
        except ValueError:
            continue
        for component in document.get('components', {}).values():
            for section in component.get('sections', []):
                offenders.append(TimingOffender(
                    label=f'{stage}/{component.get("label", stage)}.{section.get("section")}',
                    seconds=float(section.get('seconds', 0.0)),
                ))
    return offenders


def write_offender_report(path: Path, offenders: Sequence[TimingOffender],
                          *, fraction: float = 0.2) -> Path:
    """Rewrite the run's ranked worst-offender report in full.

    One log for the entire dataprep process: regenerated atomically on every
    call so it always reflects the current run, displaying only the top
    `fraction` slowest surfaces; the complete detail stays in timings.json
    and the per-stage timing files.
    """
    ranked = sorted(offenders, key=lambda offender: offender.seconds, reverse=True)
    total = sum(offender.seconds for offender in ranked) or 1.0
    showing = max(1, math.ceil(len(ranked) * fraction)) if ranked else 0
    lines = [
        f"# timing offenders — regenerated {datetime.now(timezone.utc).isoformat()}",
        f"# entries={len(ranked)} total_seconds={total:.3f} "
        f"showing={showing} (top {round(fraction * 100)}%); "
        "full detail: timings.json + <stage>.timing.json",
    ]
    lines.extend(
        f"{rank:3d}. {offender.label:<72.72} {offender.seconds:10.3f}s "
        f"{100 * offender.seconds / total:5.1f}%"
        for rank, offender in enumerate(ranked[:showing], 1)
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(path)
    return path
