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
import os
import time
from contextlib import contextmanager
from pathlib import Path


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
