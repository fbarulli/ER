"""Control-plane observability for the model-agnostic HPO lane.

Three durable, offline-testable surfaces the owner asked for:

* ``TrialEventLog``   — append-only CDC stream of every trial event
  (``trial_events.jsonl``): create/complete/fail/prune/promote rows with the
  trial number, state, value, params and user attrs. A downstream CDC job can
  tail it without touching Postgres.
* ``StudyMirror``     — a local JSONL snapshot of the study's trials, written
  atomically between sessions so a resumed session (or an offline audit) can
  reconstruct the study without the remote RDB.
* ``OfflineTrialLedger`` — the ``hpo_trials.jsonl`` fallback written when the
  shared PostgreSQL RDB is unavailable; it never blocks a trial, and it carries
  enough to re-emit the decision trail later.

``TrialObserver`` composes the three so a worker only calls ``observe(trial)``
and ``flush()``. Nothing here imports optuna/torch/sqlalchemy; trials are plain
objects, so the surfaces are unit-tested on the host.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# CDC event names (the log's vocabulary; one place).
EVENT_CREATE = "trial_created"
EVENT_COMPLETE = "trial_completed"
EVENT_FAIL = "trial_failed"
EVENT_PRUNE = "trial_pruned"
EVENT_PROMOTE = "trial_promoted"


def _trial_state_name(trial):
    state = getattr(trial, "state", None)
    return getattr(state, "name", None) or str(state)


def _trial_value(trial):
    value = getattr(trial, "value", None)
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


class TrialEventLog:
    """Append-only ``trial_events.jsonl`` CDC stream (best-effort, never fatal)."""

    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self._clock = clock
        self._count = 0

    def emit(self, event, *, trial_number=None, state=None, value=None,
             params=None, user_attrs=None, extra=None):
        row = {
            "event": str(event),
            "trial_number": (None if trial_number is None else int(trial_number)),
            "state": state,
            "value": value,
            "params": dict(params or {}),
            "user_attrs": dict(user_attrs or {}),
            "at": float(self._clock()),
        }
        if extra:
            row["extra"] = dict(extra)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            self._count += 1
        except Exception:  # noqa: BLE001,S110 - observability never blocks work
            pass
        return row

    def observe(self, trial, event=None):
        """Emit one row for a trial whose state Optuna has committed."""
        state = _trial_state_name(trial)
        if event is None:
            event = {
                "COMPLETE": EVENT_COMPLETE,
                "FAIL": EVENT_FAIL,
                "PRUNED": EVENT_PRUNE,
            }.get(str(state).upper(), EVENT_CREATE)
        return self.emit(event, trial_number=getattr(trial, "number", None),
                         state=state, value=_trial_value(trial),
                         params=getattr(trial, "params", None),
                         user_attrs=getattr(trial, "user_attrs", None))

    def read(self):
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def count(self):
        return self._count


class StudyMirror:
    """Local, atomically-written JSONL snapshot of the study's trials."""

    def __init__(self, path):
        self.path = Path(path)
        self._rows = []

    def record(self, trial):
        self._rows.append({
            "number": getattr(trial, "number", None),
            "state": _trial_state_name(trial),
            "value": _trial_value(trial),
            "params": dict(getattr(trial, "params", {}) or {}),
            "user_attrs": dict(getattr(trial, "user_attrs", {}) or {}),
        })
        return self

    def rows(self):
        return list(self._rows)

    def flush(self):
        """Write the snapshot atomically (never a partially-written mirror)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        payload = "\n".join(json.dumps(row, sort_keys=True)
                            for row in self._rows)
        temporary.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        os.replace(temporary, self.path)
        return self.path

    def load(self):
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def best(self, direction="maximize"):
        rows = [row for row in self.load()
                if row.get("value") is not None
                and row.get("state") == "COMPLETE"]
        if not rows:
            return None
        if direction == "minimize":
            return min(rows, key=lambda row: row["value"])
        return max(rows, key=lambda row: row["value"])


class OfflineTrialLedger:
    """``hpo_trials.jsonl`` fallback for when the shared RDB is unavailable.

    A best-effort local record of every trial so the decision trail survives an
    offline/single-worker fallback. It is append-only and never blocks a trial.
    """

    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self._clock = clock

    def append(self, *, trial_number, value, params=None, state="COMPLETE",
               user_attrs=None):
        row = {
            "trial_number": int(trial_number),
            "value": value,
            "params": dict(params or {}),
            "state": str(state),
            "user_attrs": dict(user_attrs or {}),
            "at": float(self._clock()),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:  # noqa: BLE001,S110 - fallback never blocks a trial
            pass
        return row

    def observe(self, trial):
        return self.append(
            trial_number=getattr(trial, "number", -1),
            value=_trial_value(trial), params=getattr(trial, "params", None),
            state=_trial_state_name(trial),
            user_attrs=getattr(trial, "user_attrs", None))

    def load(self):
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(
            encoding="utf-8").splitlines() if line.strip()]

    def best(self, direction="maximize"):
        rows = [row for row in self.load()
                if row.get("value") is not None
                and row.get("state") == "COMPLETE"]
        if not rows:
            return None
        if direction == "minimize":
            return min(rows, key=lambda row: row["value"])
        return max(rows, key=lambda row: row["value"])


class TrialObserver:
    """Compose the CDC log, the study mirror and the offline ledger.

    ``observe(trial)`` fans one committed trial out to all three; ``flush()``
    writes the mirror snapshot. Every write is best-effort — observability must
    never fail a trial.
    """

    def __init__(self, root):
        root = Path(root)
        self.events = TrialEventLog(root / "trial_events.jsonl")
        self.mirror = StudyMirror(root / "study_mirror.jsonl")
        self.ledger = OfflineTrialLedger(root / "hpo_trials.jsonl")

    def observe(self, trial, *, promoted=False):
        self.events.observe(
            trial, event=EVENT_PROMOTE if promoted else None)
        self.mirror.record(trial)
        self.ledger.observe(trial)
        return self

    def flush(self):
        try:
            return self.mirror.flush()
        except Exception:  # noqa: BLE001 - best-effort
            return None

    def as_dict(self):
        return {
            "events": str(self.events.path),
            "mirror": str(self.mirror.path),
            "ledger": str(self.ledger.path),
            "events_count": self.events.count(),
        }
